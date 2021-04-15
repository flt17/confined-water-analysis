import math
import numpy as np
import pandas
import scipy
import scipy.signal
import sys
from tqdm.notebook import tqdm

import MDAnalysis.analysis.rdf as mdanalysis_rdf

sys.path.append("../")
from confined_water import hydrogen_bonding
from confined_water import utils
from confined_water import global_variables
from confined_water import free_energy


class KeyNotFound(Exception):
    pass


class VariableNotSet(Exception):
    pass


class UnphysicalValue(Exception):
    pass


class ConfinedWaterSystem:
    """
    Gather computed properties from simulations for easy comparison.

    Attributes:

    Methods:

    """

    def __init__(self, name: str):

        """
        Arguments:
          name (str) :  The name of the instance of the class.
          simulations (dictonary) : Dictionary of all simulations performed on the given system
                                    labelled by user-given names.
        """
        self.name = name
        self.simulations = {}

    def add_simulation(self, simulation_name: str, directory_path: str):
        """
        Initialise instance of Simulation class with given name and directory path and add
        it ot the simulation dictionary.
        Arguments:
            simulation_name (str) : Name which will be used in a dictionary to access the
                                computed properties and raw data.
            directory path (str) :  Path to the simulation directory.
        Returns:

        """

        self.simulations[simulation_name] = Simulation(directory_path)


class Simulation:
    """
    Perform post-processing of a (PI)MD simulation.

    Attributes:

    Methods:

    """

    def __init__(self, directory_path: str):
        """
        Arguments:
            directory path (str) :  Path to the simulation directory.
        """

        self.directory_path = directory_path
        self.time_between_frames = None

        # set system periodicity per default:
        self.set_pbc_dimensions("xyz")

        self.radial_distribution_functions = {}
        self.density_profiles = {}
        self.hydrogen_bonding = []
        self.mean_squared_displacements = {}
        self.diffusion_coefficients = {}
        self.autocorrelation_summed_forces = {}
        self.friction_coefficients = {}
        self.velocity_autocorrelation_function = {}
        self.diffusion_coefficients_via_GK = {}
        self.free_energy_profile = free_energy.FreeEnergyProfile()
        self.tube_radius = 0

        # set default temperature to 330 K:
        self.set_simulation_temperature(330)

    def _set_up_water_topology(
        self,
    ):
        """
        Setup topology for water molecules so that we know which atoms form a water..
        Arguments:

        Returns:

        """

        # start with doing the solid

        solid_atoms = self.position_universes[0].select_atoms("not name O H")

        # all solids as one residuum for each universe
        for count_universe, universe in enumerate(self.position_universes):

            solid_segment = universe.add_Segment(segid=0, segname="Solid", segnum=0)
            liquid_segment = universe.add_Segment(segid=1, segname="Liquid", segnum=0)

            solid_residuum = universe.add_Residue(
                segment=solid_segment,
                resid=0,
                resname="Solid",
                resnum=0,
                icode="",
            )

            universe.atoms[solid_atoms.indices].residues = solid_residuum

        # check if we need to do the same thing for velocities
        if hasattr(self, "velocity_universes"):
            for count_universe, universe in enumerate(self.velocity_universes):

                solid_segment = universe.add_Segment(segid=0, segname="Solid", segnum=0)
                liquid_segment = universe.add_Segment(segid=1, segname="Liquid", segnum=0)

                solid_residuum = universe.add_Residue(
                    segment=solid_segment,
                    resid=0,
                    resname="Solid",
                    resnum=0,
                    icode="",
                )

                universe.atoms[solid_atoms.indices].residues = solid_residuum

        # based on read in trajectory and topology get all oxygen atoms
        oxygen_atoms = self.position_universes[0].select_atoms("name O")

        # get hydrogen_atoms
        hydrogen_atoms = self.position_universes[0].select_atoms("name H")

        # now loop over all oxygen atoms and find hydrogen atoms that fit:
        for count_oxygen, oxygen in enumerate(oxygen_atoms):

            # compute
            vector_to_all_hydrogens = hydrogen_atoms.positions - oxygen.position

            # make sure it satisfies pbc
            vector_to_all_hydrogens_pbc = (
                utils.apply_minimum_image_convention_to_interatomic_vectors(
                    vector_to_all_hydrogens, self.topology.cell, self.pbc_dimensions
                )
            )

            # compute distance and sort arguments
            indices_hydrogens_sorted_by_distance = np.argsort(
                np.linalg.norm(vector_to_all_hydrogens_pbc, axis=1)
            )
            # cluster indices of atoms
            atom_indices_water = np.append(
                hydrogen_atoms.indices[indices_hydrogens_sorted_by_distance[0:2]], oxygen.index
            )

            # now we need to combine them to one residue for each universe
            for count_universe, universe in enumerate(self.position_universes):

                # define residuum for each oxygen/water
                current_residuum = universe.add_Residue(
                    segment=liquid_segment,
                    resid=count_oxygen + 1,
                    resname=f"W{count_oxygen+1}",
                    resnum=count_oxygen + 1,
                    icode="",
                )

                universe.atoms[atom_indices_water].residues = current_residuum

            # check if we need to do the same thing for velocities
            if hasattr(self, "velocity_universes"):
                for count_universe, universe in enumerate(self.velocity_universes):

                    current_residuum = universe.add_Residue(
                        segment=liquid_segment,
                        resid=count_oxygen,
                        resname=f"W{count_oxygen+1}",
                        resnum=count_oxygen,
                        icode="",
                    )

                    universe.atoms[atom_indices_water].residues = current_residuum

    def set_atomic_masses(self, element_mass_dictionary):
        """
        Sets the atomic mass for species given.
        Arguments:
            element_mass_dictionary (dictionary): Dictionary with masses for each element that should be adjusted.
        Returns:
        """

        # check if all species are a subset of species in system
        if not set(element_mass_dictionary.keys()).issubset(self.species_in_system):
            raise KeyNotFound(f"At least on of the species specified is not in the system.")

        # Loop over all entries of dictionary
        for element, mass in element_mass_dictionary.items():

            # Loop over all position universes
            for count_universe, universe in enumerate(self.position_universes):

                universe.atoms[np.where(universe.atoms.names == element)].masses = mass

            # check if we need to do the same thing for velocities
            if hasattr(self, "velocity_universes"):
                for count_universe, universe in enumerate(self.velocity_universes):
                    universe.atoms[np.where(universe.atoms.names == element)].masses = mass

    def read_in_simulation_data(
        self,
        read_positions: bool = True,
        read_velocities: bool = False,
        read_summed_forces: bool = False,
        topology_file_name: str = None,
        trajectory_format: str = "dcd",
    ):

        """
        Setup all selected simulation data.
        Arguments:
            read_positions (bool) : Whether to read in position trajectories.
            read_velocities (bool) : Whether to read in velocity trajectories.
            read_summed_forces (bool) : Whether to read in separately printed summed forces.
            topology_file_name (str) : Name of the topology file (currently only pdb). If not given, first file taken.
            trajectory_format (str) : File format of trajectory, default is dcd.

        Returns:

        """
        # setup topology based on only pdb file in directoy
        path_to_topology = utils.get_path_to_file(self.directory_path, "pdb", topology_file_name)
        self.topology = utils.get_ase_atoms_object(path_to_topology)

        # Read in what needs to be read in (right now via if loop)

        # Trajectory, can be one ore multiple. If more than one, first element is centroid.
        # Functions which compute a specific property will choose which universe is picked in case of PIMD.
        if read_positions:
            universes = utils.get_mdanalysis_universe(
                self.directory_path, "positions", topology_file_name, trajectory_format
            )

            # to make sure self.position_universes is always a list of MDAnalysis Universes
            self.position_universes = (
                universes if isinstance(universes, list) == True else [universes]
            )

            self.species_in_system = np.unique(self.position_universes[0].atoms.names)

        # Velocity trajectory, can be one ore multiple. If more than one, first element is centroid.
        # Functions which compute a specific property will choose which universe is picked in case of PIMD.
        if read_velocities:
            universes = utils.get_mdanalysis_universe(
                self.directory_path, "velocities", topology_file_name, trajectory_format
            )

            # to make sure self.position_universes is always a list of MDAnalysis Universes
            self.velocity_universes = (
                universes if isinstance(universes, list) == True else [universes]
            )

            self.species_in_system = np.unique(self.velocity_universes[0].atoms.names)

        # Summed force (usually for solids), can be one ore multiple. If more than one, they should be averaged for each step.
        # It is quite likely that the timestep between samples is lower than the usual printing frequency.
        if read_summed_forces:
            # see how many summed force files we can find
            # ending is .dat and they should have "nnp-sumforce" in their name if done with CP2K.
            total_summed_force_files = utils.get_path_to_file(
                self.directory_path, "dat", "nnp-sumforce", exact_match=False
            )

            # check if multiple files found, then we need to distinguish further:
            if len(total_summed_force_files) > 1:
                # refine selection taking only correct files (ending on _1.dat):
                # this might be changed when CP2K is updated
                total_summed_force_files = [
                    file for file in total_summed_force_files if "_1.dat" in file
                ]

            # read in summed forces with pandas, if multiple files, they are averaged immediately
            self.summed_forces = np.mean(
                np.asarray(
                    [
                        pandas.read_csv(
                            file, skiprows=1, header=None, delim_whitespace=True
                        ).to_numpy()
                        for file in total_summed_force_files
                    ]
                ),
                axis=0,
            )

            # CP2K saves summed forces in atomic units, i.e. Ha/B.
            # Converte them into eV/A
            self.summed_forces = (
                self.summed_forces
                * global_variables.HARTREE_TO_EV
                / global_variables.BOHR_TO_ANGSTROM
            )

        # now check connectivity of waters and implement scheme for given universes
        if read_positions:
            self._set_up_water_topology()

    def set_simulation_temperature(self, temperature: float):
        """
        Set in temperature at which simulation was performed.
        Arguments:
            temperature (float) : Temperature (in K) at which simulation was performed.

        Returns:

        """

        self.temperature = temperature
        self.free_energy_profile.temperature = temperature

    def set_pbc_dimensions(self, pbc_dimensions: str):
        """
        Set in which direction pbc apply.
        Arguments:
            pbc_dimensions (str) : string of directions in which pbc apply.

        Returns:

        """

        if not global_variables.DIMENSION_DICTIONARY.get(pbc_dimensions):
            raise KeyNotFound(
                f"Specified dimension {pbc_dimensions} is unknown. Possible options are {global_variables.DIMENSION_DICTIONARY.keys()}"
            )
        self.pbc_dimensions = pbc_dimensions

    def set_sampling_times(
        self,
        start_time: int = None,
        end_time: int = None,
        frame_frequency: int = None,
        time_between_frames: float = None,
    ):

        """
        Set times for analysis of trajectories.
        Arguments:
            start_time (int) : Start time for analysis.
            end_time (int) : End time for analysis.
            frame_frequency (int): Take every nth frame only.
            time_between_frames (float): Time (in fs) between two frames in sampled trajectory, e.g. 100 fs.
        Returns:

        """

        self.start_time = start_time if start_time is not None else self.start_time
        self.frame_frequency = (
            frame_frequency if frame_frequency is not None else self.frame_frequency
        )
        self.time_between_frames = (
            time_between_frames if time_between_frames is not None else self.time_between_frames
        )

        total_time = (self.position_universes[0].trajectory.n_frames - 1) * self.time_between_frames
        self.end_time = (
            total_time if end_time == -1 else end_time if end_time is not None else self.end_time
        )

        print(f"SUCCESS: New sampling times.")
        print(f"Start time: \t \t{self.start_time} \t fs")
        print(f"End time: \t \t{self.end_time} \t fs")
        print(f"Time between frames: \t{self.time_between_frames} \t fs")
        print(f"Frame frequency: \t{self.frame_frequency}")

    def _get_sampling_frames(
        self,
        start_time: int = None,
        end_time: int = None,
        frame_frequency: int = None,
        time_between_frames: float = None,
    ):
        """
        Determine sampling frames from given sampling times.
        Arguments:
            start_time (int) : Start time for analysis.
            end_time (int) : End time for analysis.
            frame_frequency (int): Take every nth frame only.
            time_between_frames (float): Time (in fs) between frames. Usually, this is set at the very beginning.
                            Exception applies only to calculation of friction where this is set in the method.
        Returns:
            start_frame (int) : Start frame for analysis.
            end_frame (int) : End frame for analysis.
            frame_frequency (int): Take every nth frame only.

        """
        time_between_frames = (
            time_between_frames if time_between_frames else self.time_between_frames
        )

        start_time = start_time if start_time is not None else self.start_time
        end_time = end_time if end_time is not None else self.end_time

        frame_frequency = int(
            frame_frequency if frame_frequency is not None else self.frame_frequency
        )

        start_frame = int(start_time / time_between_frames)
        end_frame = int(end_time / time_between_frames)

        return start_frame, end_frame, frame_frequency

    def compute_rdf(
        self,
        species_1: str,
        species_2: str,
        spatial_range: float = None,
        spatial_resolution: int = None,
        start_time: int = None,
        end_time: int = None,
        frame_frequency: int = None,
    ):
        """
        Compute radial distribution function for given species. Currently, only
        works for wrapped trajectories.
        Arguments:
            species_1 (str) : Name of species (element) used here.
            species_2 (str) : Name of species (element) used here
            spatial_range (float) : Spatial expansion of the rdf computed (optional)
            spatial_resolution (int) : Number of bins for histogram generated (optional).
            start_time (int) : Start time for analysis (optional).
            end_time (int) : End time for analysis (optional).
            frame_frequency (int): Take every nth frame only (optional).
        Returns:

        """

        # get information about sampling either from given arguments or previously set
        start_frame, end_frame, frame_frequency = self._get_sampling_frames(
            start_time, end_time, frame_frequency
        )

        # set variables important for binning and histogram
        # if no value set for spatial expansion we go up to half the boxlength in z-direction
        spatial_range = (
            spatial_range if spatial_range else self.topology.get_cell_lengths_and_angles()[2] / 2
        )

        print(f"Compute radial distribution function up to a radial distance of {spatial_range}")

        # define default bin width which will be used to determine number of bins if not given
        default_bin_width = 0.02
        spatial_resolution = (
            spatial_resolution if spatial_resolution else int(spatial_range / default_bin_width)
        )

        # determine which position universe are to be used in case of PIMD
        # Thermodynamic properties are based on trajectory of replica
        tmp_position_universes = (
            self.position_universes
            if len(self.position_universes) == 1
            else self.position_universes[1::]
        )

        # check whether chosen species are found in universe
        if species_1 not in self.species_in_system or species_2 not in self.species_in_system:
            raise KeyNotFound(f"At least on of the species specified is not in the system.")

        rdfs_sampled = []
        # loop over all universes

        for count_universe, universe in enumerate(tmp_position_universes):

            # determine MDAnalysis atom groups based on strings for the species provided
            atom_group_1 = universe.select_atoms(f"name {species_1}")
            atom_group_2 = universe.select_atoms(f"name {species_2}")

            # initialise calculation object
            calc = mdanalysis_rdf.InterRDF(
                atom_group_1,
                atom_group_2,
                nbins=spatial_resolution,
                range=[default_bin_width, spatial_range],
            )

            # run calculation
            calc.run(start_frame, end_frame, frame_frequency)

            # append results to list to average if necessary (PIMD)
            rdfs_sampled.append(calc.rdf)

        # average rdfs for PIMDs
        averaged_rdf = np.mean(rdfs_sampled, axis=0)

        # write to class attributes
        name_based_on_species = f"{species_1}-{species_2}"

        self.radial_distribution_functions[name_based_on_species] = [calc.bins, averaged_rdf]

    def compute_water_orientation_profile(
        self,
        start_time: int = None,
        end_time: int = None,
        frame_frequency: int = None,
    ):
        """
        Compute orientation of water molecules in nanotube or on interface, direction will be taken from pbc indices.
        Arguments:

            start_time (int) : Start time for analysis (optional).
            end_time (int) : End time for analysis (optional).
            frame_frequency (int): Take every nth frame only (optional).
            bin_width (flat) : Bin width for angle profile (optional).
        Returns:

        """
        # water orientation only doable for confined water system, not bulk
        if len(self.pbc_dimensions) == 3:
            raise UnphysicalValue(
                "Why would you want to calculate the angular distribution in bulk water?"
            )

        # get information about sampling either from given arguments or previously set
        start_frame, end_frame, frame_frequency = self._get_sampling_frames(
            start_time, end_time, frame_frequency
        )

        # determine which position universe are to be used in case of PIMD
        # Thermodynamic properties are based on trajectory of replica
        tmp_position_universes = (
            self.position_universes
            if len(self.position_universes) == 1
            else self.position_universes[1::]
        )

        orientations_sampled = []

        # Loop over all universes
        for count_universe, universe in enumerate(tmp_position_universes):

            # call function dependent on direction:
            if len(self.pbc_dimensions) == 2:
                (orientations) = self._compute_water_orientation_profile_along_cartesian_axis(
                    universe,
                    start_frame,
                    end_frame,
                    frame_frequency,
                )
            else:
                (orientations) = self._compute_water_orientation_profile_in_radial_direction(
                    universe,
                    start_frame,
                    end_frame,
                    frame_frequency,
                )

            orientations_sampled.append(orientations)

        self.water_orientations = orientations_sampled

    def _compute_water_orientation_profile_along_cartesian_axis(
        self,
        position_universe,
        start_frame: int,
        end_frame: int,
        frame_frequency: int,
    ):
        """
        Compute water orientation profile along non-periodic cartesian axis.
        Arguments:
            position_universe : MDAnalysis universe with trajectory.
            start_frame (int) : Start frame for analysis.
            end_frame (int) : End frame for analysis.
            frame_frequency (int): Take every nth frame only.
        Returns:
            orientations (np.asarray) : Water orientations in the system.

        """

        # get number of water molecules:
        water_atoms = position_universe.select_atoms("resname W*")
        number_of_water_molecules = int(len(water_atoms) / 3)

        # get vector parallel to axis for which we analyse the orientation
        # based on pbc check what direction is investigated
        pbc_dimensions_indices = global_variables.DIMENSION_DICTIONARY.get(self.pbc_dimensions)
        not_pbc_indices = list(set(pbc_dimensions_indices) ^ set([0, 1, 2]))

        # initialise mass profile array
        number_of_frames = len(
            position_universe.trajectory[start_frame:end_frame][::frame_frequency]
        )
        orientations = np.zeros((number_of_water_molecules, number_of_frames))
        # Loop over trajectory
        for count_frames, frames in enumerate(
            tqdm((position_universe.trajectory[start_frame:end_frame])[::frame_frequency])
        ):
            # compute dipole vector for each water molecule
            dipole_moment_vector_all_water = np.asarray(
                [
                    utils.get_dipole_moment_vector_in_water_molecule(
                        water_atoms.select_atoms(f"resname W{water_index+1}"),
                        self.topology,
                        self.pbc_dimensions,
                    )
                    for water_index in np.arange(number_of_water_molecules)
                ]
            )

            # orientation angle is expressed in cos theta
            orientations[:, count_frames] = dipole_moment_vector_all_water[
                :, not_pbc_indices[0]
            ] / np.linalg.norm(dipole_moment_vector_all_water, axis=1)

        return orientations

    def _compute_water_orientation_profile_in_radial_direction(
        self,
        position_universe,
        start_frame: int,
        end_frame: int,
        frame_frequency: int,
    ):
        """
        Compute water orientation profile along radially along periodic cartesian axis.
        Arguments:
            position_universe : MDAnalysis universe with trajectory.
            start_frame (int) : Start frame for analysis.
            end_frame (int) : End frame for analysis.
            frame_frequency (int): Take every nth frame only.
        Returns:
            orientations (np.asarray) : Water orientations in the system.

        """

        # get number of water molecules:
        oxygen_atoms = position_universe.select_atoms("name O")
        number_of_water_molecules = len(oxygen_atoms)

        # get vector parallel to axis for which we analyse the orientation
        # based on pbc check what direction is investigated
        pbc_dimensions_indices = global_variables.DIMENSION_DICTIONARY.get(self.pbc_dimensions)
        not_pbc_indices = list(set(pbc_dimensions_indices) ^ set([0, 1, 2]))

        # initialise mass profile array
        number_of_frames = len(
            position_universe.trajectory[start_frame:end_frame][::frame_frequency]
        )
        orientations = np.zeros((number_of_water_molecules, number_of_frames))
        # Loop over trajectory
        for count_frames, frames in enumerate(
            tqdm((position_universe.trajectory[start_frame:end_frame])[::frame_frequency])
        ):
            # compute dipole vector for each water molecule
            dipole_moment_vector_all_water = np.asarray(
                [
                    utils.get_dipole_moment_vector_in_water_molecule(
                        position_universe.select_atoms(f"resname W{water_index+1}"),
                        self.topology,
                        self.pbc_dimensions,
                    )
                    for water_index in np.arange(number_of_water_molecules)
                ]
            )

            # compute reference coordinate, i.e. center of mass of all atoms
            system_center_of_mass = np.ma.array(position_universe.atoms.center_of_mass())
            system_center_of_mass[pbc_dimensions_indices] = np.ma.masked
            reference_coordinates = system_center_of_mass.compressed()

            # compute radial distance from oxygen to reference
            vectors_2D_oxygens_to_COM = (
                reference_coordinates - oxygen_atoms.positions[:, not_pbc_indices]
            )
            # orientation angle is expressed in cos theta

            orientations[:, count_frames] = (
                np.einsum(
                    "ij,ij->i",
                    dipole_moment_vector_all_water[:, not_pbc_indices],
                    vectors_2D_oxygens_to_COM,
                )
                / np.linalg.norm(dipole_moment_vector_all_water[:, not_pbc_indices], axis=1)
                / np.linalg.norm(vectors_2D_oxygens_to_COM, axis=1)
            )

        return orientations

    def compute_density_profile(
        self,
        species: list,
        direction: str,
        start_time: int = None,
        end_time: int = None,
        frame_frequency: int = None,
        bin_width: float = 0.1,
    ):
        """
        Compute density profile either radially or in a given direction
        Arguments:
            species (list) : Elements considered in the density profile
            direction (str) : Direction in which the profile is computed.
                            Can be either x,y,z or radial.
            start_time (int) : Start time for analysis (optional).
            end_time (int) : End time for analysis (optional).
            frame_frequency (int): Take every nth frame only (optional).
            bin_width (flat) : Bin width for density profile in angstrom (optional).
        Returns:

        """

        # check in which direction/orientation density profile should be calculated
        dictionary_direction = {"x": 0, "y": 1, "z": 2, "radial x": 0, "radial y": 1, "radial z": 2}
        direction_index = dictionary_direction.get(direction)
        if not direction_index:
            raise KeyNotFound(
                f"Specified direction is unknown. Possible options are {dictionary_direction.keys()}"
            )

        # check if all species are a subset of species in system
        if not set(species).issubset(self.species_in_system):
            raise KeyNotFound(f"At least on of the species specified is not in the system.")

        # get information about sampling either from given arguments or previously set
        start_frame, end_frame, frame_frequency = self._get_sampling_frames(
            start_time, end_time, frame_frequency
        )

        # determine which position universe are to be used in case of PIMD
        # Thermodynamic properties are based on trajectory of replica
        tmp_position_universes = (
            self.position_universes
            if len(self.position_universes) == 1
            else self.position_universes[1::]
        )

        # convert list to string for species:
        selected_species_string = " ".join(species)

        density_profiles_sampled = []

        # Loop over all universes
        for count_universe, universe in enumerate(tmp_position_universes):

            # select atoms according to species
            atoms_selected = universe.select_atoms(f"name {selected_species_string}")

            # call function dependent on direction:
            if "radial" not in direction:
                (
                    bins_histogram,
                    density_profile,
                ) = self._compute_density_profile_along_cartesian_axis(
                    universe,
                    atoms_selected,
                    direction_index,
                    start_frame,
                    end_frame,
                    frame_frequency,
                    bin_width,
                )
            else:
                (
                    bins_histogram,
                    density_profile,
                ) = self._compute_density_profile_in_radial_direction(
                    universe,
                    atoms_selected,
                    direction_index,
                    start_frame,
                    end_frame,
                    frame_frequency,
                    bin_width,
                )

            density_profiles_sampled.append(density_profile)

        # average rdfs for PIMDs
        averaged_density_profile = np.mean(density_profiles_sampled, axis=0)

        # write to class attributes
        name_based_on_species_and_direction = f"{selected_species_string} - {direction}"

        self.density_profiles[name_based_on_species_and_direction] = [
            bins_histogram,
            averaged_density_profile,
        ]

    def _compute_density_profile_along_cartesian_axis(
        self,
        position_universe,
        atoms_selected,
        direction: int,
        start_frame: int,
        end_frame: int,
        frame_frequency: int,
        bin_width: float = 0.1,
    ):
        """
        Compute density profile in a given direction along cartesian axis.
        Arguments:
            position_universe : MDAnalysis universe with trajectory.
            atoms_selected (str) : Atoms considered in the density profile.
            direction (int) : Direction in which the profile is computed.
                            x=0, y=1, z=2.
            start_frame (int) : Start frame for analysis.
            end_frame (int) : End frame for analysis.
            frame_frequency (int): Take every nth frame only.
            bin_width (flat) : Bin width for density profile in angstrom.
        Returns:
            bins_histogram (np.array): Bins of the histogram of the density profile.
            density_profile (np.asarray) : Density profile based on the bins.

        """

        # determine reference atoms, here only solid atoms (B,N,C) allowed
        # reference_species = ["B", "N", "C", "Cl", "Na"]
        # reference_species_string = " ".join(reference_species)

        reference_atoms = position_universe.select_atoms("not name O H")

        if len(reference_atoms) == 0:
            raise KeyNotFound(f"Couldn't find a solid phase in this trajectory.")

        # define range and bin width for histogram binning
        # bin range is simply the box length in the given direction
        bin_range = self.topology.get_cell_lengths_and_angles()[direction]
        # for now, bin width is set to a default value of 0.1 angstroms
        bin_width = bin_width
        bins_histogram = np.arange(-bin_range, bin_range, bin_width)

        # initialise mass profile array
        mass_profile = np.zeros(bins_histogram.shape[0])

        # Loop over trajectory
        for count_frames, frames in enumerate(
            tqdm((position_universe.trajectory[start_frame:end_frame])[::frame_frequency])
        ):

            # compute reference coordinate, i.e. center of mass of reference atoms in specified direction
            reference_coordinate = reference_atoms.center_of_mass()[direction]

            # compute coordinates in given direciton for select atoms relative to reference_coordinate
            relative_coordinates_selected_atoms = (
                atoms_selected.positions[:, direction] - reference_coordinate
            )

            # digitize relative positions
            # NOTE: values are assigned to bins according to i <= x[i] < i+1
            digitized_relative_coordinates = np.digitize(
                relative_coordinates_selected_atoms, bins_histogram
            )

            # Compute density profile of frame by looping over all bins and check if theres a match
            bins_occupied_by_atoms = np.unique(digitized_relative_coordinates)
            for occupied_bin in bins_occupied_by_atoms:
                # at this point compute only mass profile, units g
                mass_profile[occupied_bin] += np.sum(
                    np.where(digitized_relative_coordinates == occupied_bin, 1, 0)
                    * atoms_selected.masses
                    / global_variables.AVOGADRO
                )

        # normalise mass profile by number of frames
        mass_profile = mass_profile / (count_frames + 1)

        # compute density profile by dividing each mass bin by its volume in cm^3
        # only possible for orthrombic cells
        volume_per_bin = (
            bin_width
            * np.product(self.topology.get_cell_lengths_and_angles()[0:3])
            / self.topology.get_cell_lengths_and_angles()[direction]
            * 1e-24
        )

        density_profile = mass_profile / volume_per_bin

        return bins_histogram, density_profile

    def _compute_density_profile_in_radial_direction(
        self,
        position_universe,
        atoms_selected,
        direction: int,
        start_frame: int,
        end_frame: int,
        frame_frequency: int,
        bin_width: float = 0.1,
    ):
        """
        Compute density profile in radial direction around
        Arguments:
            position_universe : MDAnalysis universe with trajectory.
            atoms_selected (str) : Atoms considered in the density profile.
            direction (int) : Direction of axis through COM from which
                             the profile is computed radially, x=0, y=1, z=2.
            start_frame (int) : Start frame for analysis.
            end_frame (int) : End frame for analysis.
            frame_frequency (int): Take every nth frame only.
            bin_width (flat) : Bin width for density profile in angstrom.
        Returns:
            bins_histogram (np.array): Bins of the histogram of the density profile.
                                      Note that the bins are based on the center of np.digitize.
            density_profile (np.asarray) : Density profile based on the bins.

        """
        # define range and bin width for histogram binning
        # bin range is simply the half box length along axis orthogonal to direction (squared area)
        bin_range = self.topology.get_cell_lengths_and_angles()[0:3][direction - 1] * 0.5
        # for now, bin width is set to a default value of 0.1 angstroms
        bin_width = bin_width
        bins_histogram = np.arange(0, bin_range, bin_width)

        # initialise mass profile array
        mass_profile = np.zeros(bins_histogram.shape[0])

        # Loop over trajectory
        for count_frames, frames in enumerate(
            tqdm((position_universe.trajectory[start_frame:end_frame])[::frame_frequency])
        ):

            # compute reference coordinate, i.e. center of mass of all atoms
            system_center_of_mass = np.ma.array(position_universe.atoms.center_of_mass())
            system_center_of_mass[direction] = np.ma.masked
            reference_coordinates = system_center_of_mass.compressed()

            # compute radial distances from axis through center of mass in given direciton for selected atoms
            positions_atoms_selected = np.ma.array(atoms_selected.positions)
            positions_atoms_selected[:, direction] = np.ma.masked
            positions_atoms_selected_orthogonal = positions_atoms_selected.compressed().reshape(
                -1, 2
            )

            radial_distances_to_axis = np.linalg.norm(
                positions_atoms_selected_orthogonal - reference_coordinates, axis=1
            )

            # digitize distances
            # NOTE: values are assigned to bins according to i <= x[i] < i+1
            digitized_radial_distances = np.digitize(radial_distances_to_axis, bins_histogram)

            # Compute density profile of frame by looping over all bins and check if theres a match
            bins_occupied_by_atoms = np.unique(digitized_radial_distances)
            for occupied_bin in bins_occupied_by_atoms:
                # at this point compute only mass profile, units g
                mass_profile[occupied_bin] += np.sum(
                    np.where(digitized_radial_distances == occupied_bin, 1, 0)
                    * atoms_selected.masses
                    / global_variables.AVOGADRO
                )

        # normalise mass profile by number of frames
        mass_profile = mass_profile / (count_frames + 1)

        # compute density profile by dividing each mass bin by its volume in cm^3
        # Volume depends on the bin
        # get center of bins and add artificial one for the last entry
        center_of_bins = 0.5 * (bins_histogram[0:-1] + bins_histogram[1::])
        center_of_bins = np.append(center_of_bins, center_of_bins[-1] + bin_width)

        # define volumes for each bin in cm^3
        volume_per_bin = (
            np.pi
            * (center_of_bins[1::] ** 2 - center_of_bins[0:-1] ** 2)
            * self.topology.get_cell_lengths_and_angles()[direction]
            * 1e-24
        )

        volume_per_bin = np.insert(
            volume_per_bin,
            0,
            np.pi
            * center_of_bins[0] ** 2
            * self.topology.get_cell_lengths_and_angles()[direction]
            * 1e-24,
        )

        density_profile = mass_profile / volume_per_bin
        return center_of_bins, density_profile

    def set_up_hydrogen_bonding_analysis(
        self, start_time: int = None, end_time: int = None, frame_frequency: int = None
    ):

        """
        Prepare everything for hydrogen bonding analysis by initialsing instance of
        HydrogenBonding for each position_universe. This will be used to identify
        all hydrogen bonds within the given times.
        Arguments:
            start_time (int) : Start time for analysis (optional).
            end_time (int) : End time for analysis (optional).
            frame_frequency (int): Take every nth frame only (optional).
        Returns:

        """

        # get information about sampling either from given arguments or previously set
        start_frame, end_frame, frame_frequency = self._get_sampling_frames(
            start_time, end_time, frame_frequency
        )

        # determine which position universe are to be used in case of PIMD
        # Thermodynamic properties are based on trajectory of replica
        tmp_position_universes = (
            self.position_universes
            if len(self.position_universes) == 1
            else self.position_universes[1::]
        )

        hydrogen_bonding_objects = []

        # Loop over all universes
        for count_universe, universe in enumerate(tmp_position_universes):

            # create instance of hydrogen_bonding.HydrogenBonding
            hydrogen_bonding_analysis = hydrogen_bonding.HydrogenBonding(universe, self.topology)

            # find all acceptor-donor pairs
            hydrogen_bonding_analysis.find_acceptor_donor_pairs(
                start_frame,
                end_frame,
                frame_frequency,
                self.time_between_frames,
                self.pbc_dimensions,
            )

            hydrogen_bonding_objects.append(hydrogen_bonding_analysis)

        self.hydrogen_bonding = hydrogen_bonding_objects

    def compute_mean_squared_displacement(
        self,
        species: list,
        correlation_time: float = 5000.0,
        number_of_blocks: int = 30,
        start_time: int = None,
        end_time: int = None,
        frame_frequency: int = None,
    ):
        """
        Compute mean squared displacement (MSD) for given species.
        Arguments:
            species (list) : Elements considered for MSD.
            correlation_time (float): Time (in fs) for which we will trace the movement of the atoms.
            number_of_blocks (int): Number of blocks used for block average of MSD.
            start_time (int) : Start time for analysis (optional).
            end_time (int) : End time for analysis (optional).
            frame_frequency (int): Take every nth frame only (optional).
        Returns:

        """

        # check if all species are a subset of species in system
        if not set(species).issubset(self.species_in_system):
            raise KeyNotFound(f"At least on of the species specified is not in the system.")

        # check if all species are a subset of species in system
        if not self.time_between_frames:
            raise VariableNotSet(
                f"Time between frames (in fs) is not specified yet. Please do this via self.set_sampling_times()."
            )

        # get information about sampling either from given arguments or previously set
        start_frame, end_frame, frame_frequency = self._get_sampling_frames(
            start_time, end_time, frame_frequency
        )

        # convert correlation_time to correlation_frames taken into account the time between frames and
        # the frame frequency
        number_of_correlation_frames = int(
            correlation_time / self.time_between_frames / frame_frequency
        )

        # determine which position universe are to be used in case of PIMD
        # Dynamical properties are based on trajectory of centroid!
        tmp_position_universe = self.position_universes[0]

        # check if correlation time can be obtained with current trajectory:
        number_of_samples = int((end_frame - start_frame) / frame_frequency)
        if number_of_correlation_frames >= number_of_samples:
            raise UnphysicalValue(
                f" You want to compute a correlation based on {number_of_correlation_frames} frames."
                f"However, the provided trajectory will only be analysed for {number_of_samples} frames.",
                f" Please adjust your correlation or sampling times or run longer trajectories.",
            )

        # convert list to string for species and select atoms of these species:
        # to get water diffusion we need to look at the movement of the center of mass of the molecule
        # if this is the case we simply need to trace allocate only one thrid of the number of atoms
        selected_species_string = " ".join(species)
        atoms_selected = tmp_position_universe.select_atoms(f"name {selected_species_string}")
        number_of_tracers = (
            int(len(atoms_selected) / 3)
            if selected_species_string == "O H"
            else len(atoms_selected)
        )

        # allocate array for all positions of all selected atoms for all frames sampled
        saved_positions_atoms_selected = np.zeros((number_of_samples, number_of_tracers, 3))

        # allocate array for MSD, length of number_of_correlation_frames
        mean_squared_displacement = np.zeros(number_of_correlation_frames)
        # allocate array for number of samples per correlation frame
        number_of_samples_correlated = np.zeros(number_of_correlation_frames)
        # allocate array for blocks of MSD for statistical error analysis
        mean_squared_displacement_blocks = np.zeros(
            (number_of_blocks, number_of_correlation_frames)
        )
        # define how many samples are evaluated per block
        number_of_samples_per_block = math.ceil(number_of_samples / number_of_blocks)
        index_current_block_used = 0

        # make sure that each block can reach full correlation time
        if number_of_samples_per_block < number_of_correlation_frames:
            raise UnphysicalValue(
                f" Your chosen number of blocks ({number_of_blocks}) is not allowed as:",
                f"samples per block ({number_of_samples_per_block}) < correlation frames {number_of_correlation_frames}.",
                f"Please reduce the number of blocks or run longer trajectories.",
            )

        # Loop over trajectory to sample all positions of selected atoms
        for count_frames, frames in enumerate(
            tqdm((tmp_position_universe.trajectory[start_frame:end_frame])[::frame_frequency])
        ):

            # if water diffusion compute center of mass per water molecule
            # Based on CP2K input this is readily done due to the the H-H-O input structure
            # If the input file has another format this will lead to errorneous results and
            # it would be better to just trace the oxygens.

            # compute center of mass of selected atoms, which will be  substracted afterwards
            center_of_mass_selected_atoms = atoms_selected.center_of_mass()

            if selected_species_string == "O H":
                # compute center of mass of water molecules
                center_of_masses_water_molecules = [
                    atoms_selected[atoms_selected.resids == index_molecule + 1].center_of_mass()
                    for index_molecule in np.arange(number_of_tracers)
                ]
                # center_of_masses_water_molecules = np.asarray(
                #     [
                #         utils.get_center_of_mass_of_atoms_in_accordance_with_MIC(
                #             atoms_selected[3 * index_molecule : 3 * index_molecule + 3],
                #             self.topology,
                #             self.pbc_dimensions,
                #         )
                #         for index_molecule in np.arange(number_of_tracers)
                #     ]
                # )
                saved_positions_atoms_selected[count_frames] = (
                    center_of_masses_water_molecules - center_of_mass_selected_atoms
                )
            else:
                # fill array with positions
                saved_positions_atoms_selected[count_frames] = (
                    atoms_selected.positions - center_of_mass_selected_atoms
                )

        # used the saved positions to now compute MSD
        # first check if water diffusion is calculated, i.e. O-H as species name

        # Loop over saved positions
        for frame, positions_per_frame in enumerate(tqdm(saved_positions_atoms_selected)):

            # compute last frame sampled, i.e. usually frame+correlation frames
            last_correlation_frame = frame + number_of_correlation_frames
            if last_correlation_frame > number_of_samples - 1:
                last_correlation_frame = number_of_samples

            # define variable to save how many frames where used for correlation
            number_of_frames_correlated = last_correlation_frame - frame

            # increment which correlation frames were sampled
            number_of_samples_correlated[0:number_of_frames_correlated] += 1

            # compute how much select atoms have moved within the correlation time
            vectors_atom_movement = (
                saved_positions_atoms_selected[frame:last_correlation_frame]
                - saved_positions_atoms_selected[frame]
            )

            # compute squared_distance of the moved atoms between each frame
            squared_distances_atom_movement = np.square(
                np.linalg.norm(vectors_atom_movement, axis=2)
            )

            # add contributions to array (for correlations frames sampled)
            mean_squared_displacement[0:number_of_frames_correlated] += np.mean(
                squared_distances_atom_movement, axis=1
            )

            # to get insight on the statistical error we compute block averages
            mean_squared_displacement_blocks[
                index_current_block_used, 0:number_of_frames_correlated
            ] += np.mean(squared_distances_atom_movement, axis=1)

            # close block when number of samples per block are reached
            if (
                frame + 1 >= (index_current_block_used + 1) * number_of_samples_per_block
                or frame + 1 == number_of_samples
            ):
                # initialise with 0
                number_of_samples_correlated_per_block = 0
                # check how many samples per frame were taken for this block
                if index_current_block_used == 0:
                    # in first block this corresponds to the global number of samples correlated
                    number_of_samples_correlated_per_block = number_of_samples_correlated
                else:

                    # in all others we just need to get the difference between current and previous global samples
                    number_of_samples_correlated_per_block = (
                        number_of_samples_correlated - previous_global_number_of_samples_correlated
                    )

                # average current block
                mean_squared_displacement_blocks[index_current_block_used, :] = (
                    mean_squared_displacement_blocks[index_current_block_used, :]
                    / number_of_samples_correlated_per_block
                )

                # define previous global number of samples
                previous_global_number_of_samples_correlated = number_of_samples_correlated.copy()

                # increment index to move to next block
                index_current_block_used += 1

        # define time:
        measured_time = (
            np.arange(number_of_correlation_frames) * self.time_between_frames * frame_frequency
        )

        # get average MSD
        average_mean_squared_displacement = mean_squared_displacement / number_of_samples_correlated

        # compute statistical error based on block averags
        std_mean_squared_displacement = np.std(mean_squared_displacement_blocks, axis=0)

        # compute diffusion coefficients
        average_diffusion_coefficient = utils.compute_diffusion_coefficient_based_on_MSD(
            average_mean_squared_displacement, measured_time
        )

        diffusion_coefficient_block = np.asarray(
            [
                utils.compute_diffusion_coefficient_based_on_MSD(MSD_per_block, measured_time)
                for MSD_per_block in mean_squared_displacement_blocks
            ]
        )

        std_diffusion_coefficient = np.std(diffusion_coefficient_block)

        # save all data to dictionary of class
        string_for_dict = f"{selected_species_string} - ct: {correlation_time}"
        self.mean_squared_displacements[string_for_dict] = [
            measured_time,
            average_mean_squared_displacement,
            std_mean_squared_displacement,
            mean_squared_displacement_blocks,
        ]

        self.diffusion_coefficients[string_for_dict] = [
            average_diffusion_coefficient,
            std_diffusion_coefficient,
        ]

    def compute_diffusion_coefficient_via_green_kubo(
        self,
        species: list,
        correlation_time: float = 5000.0,
        number_of_blocks: int = 30,
        units_velocity: str = "a.u.",
        start_time: int = None,
        end_time: int = None,
        frame_frequency: int = None,
    ):
        """
        Compute velocity autocorrelation function (VACF) for given species and returns diffusion
        coefficient in all directions and the total average.
        Arguments:
            species (list) : Elements considered for VACF.
            correlation_time (float): Time (in fs) for which we will trace the movement of the atoms.
            number_of_blocks (int): Number of blocks used for block average of VACF.
            units_velocity (str): In which unit where velocities saved.
            start_time (int) : Start time for analysis (optional).
            end_time (int) : End time for analysis (optional).
            frame_frequency (int): Take every nth frame only (optional).
        Returns:

        """

        # get information about sampling either from given arguments or previously set
        start_frame, end_frame, frame_frequency = self._get_sampling_frames(
            start_time, end_time, frame_frequency
        )

        # convert correlation_time to correlation_frames taken into account the time between frames and
        # the frame frequency
        number_of_correlation_frames = int(
            correlation_time / self.time_between_frames / frame_frequency
        )

        # determine which velocity universes are to be used in case of PIMD
        # Dynamical properties are based on trajectory of centroid!
        tmp_velocity_universe = self.velocity_universes[0]

        # check if correlation time can be obtained with current trajectory:
        number_of_samples = int((end_frame - start_frame) / frame_frequency)
        if number_of_correlation_frames >= number_of_samples:
            raise UnphysicalValue(
                f" You want to compute a correlation based on {number_of_correlation_frames} frames."
                f"However, the provided trajectory will only be analysed for {number_of_samples} frames.",
                f" Please adjust your correlation or sampling times or run longer trajectories.",
            )

        # convert list to string for species and select atoms of these species:
        # to get water diffusion we need to look at the movement of the center of mass of the molecule
        # if this is the case we simply need to trace allocate only one thrid of the number of atoms
        selected_species_string = " ".join(species)
        atoms_selected = tmp_velocity_universe.select_atoms(f"name {selected_species_string}")
        number_of_tracers = (
            int(len(atoms_selected) / 3)
            if selected_species_string == "O H"
            else len(atoms_selected)
        )

        # allocate array for all velocities of all selected atoms for all frames sampled
        saved_velocities_atoms_selected = np.zeros((number_of_samples, number_of_tracers, 3))

        # allocate array for VACF, length of number_of_correlation_frames
        VACF = np.zeros(number_of_correlation_frames)

        # allocate array for number of samples per correlation frame
        number_of_samples_correlated = np.zeros(number_of_correlation_frames)

        # allocate array for blocks of VACF for statistical error analysis
        VACF_block = np.zeros((number_of_blocks, number_of_correlation_frames))

        # define how many samples are evaluated per block
        number_of_samples_per_block = math.ceil(number_of_samples / number_of_blocks)
        index_current_block_used = 0

        # make sure that each block can reach full correlation time
        if number_of_samples_per_block < number_of_correlation_frames:
            raise UnphysicalValue(
                f" Your chosen number of blocks ({number_of_blocks}) is not allowed as:",
                f"samples per block ({number_of_samples_per_block}) < correlation frames {number_of_correlation_frames}.",
                f"Please reduce the number of blocks or run longer trajectories.",
            )

        # Loop over trajectory to sample all positions of selected atoms
        for count_frames, frames in enumerate(
            tqdm((tmp_velocity_universe.trajectory[start_frame:end_frame])[::frame_frequency])
        ):

            # if water diffusion compute center of mass per water molecule
            # Based on CP2K input this is readily done due to the the H-H-O input structure
            # If the input file has another format this will lead to errorneous results and
            # it would be better to just trace the oxygens.

            # compute center of mass of selected atoms, which will be  substracted afterwards
            center_of_mass_selected_atoms = atoms_selected.center_of_mass()

            if selected_species_string == "O H":
                # compute center of mass of water molecules
                center_of_masses_water_molecules = [
                    atoms_selected[atoms_selected.resids == index_molecule + 1].center_of_mass()
                    for index_molecule in np.arange(number_of_tracers)
                ]

                saved_velocities_atoms_selected[count_frames] = (
                    center_of_masses_water_molecules - center_of_mass_selected_atoms
                )
            else:
                # fill array with positions
                saved_velocities_atoms_selected[count_frames] = (
                    atoms_selected.positions - center_of_mass_selected_atoms
                )

        # Loop over saved positions
        for frame, velocities_per_frames in enumerate(tqdm(saved_velocities_atoms_selected)):

            # compute last frame sampled, i.e. usually frame+correlation frames
            last_correlation_frame = frame + number_of_correlation_frames
            if last_correlation_frame > number_of_samples - 1:
                last_correlation_frame = number_of_samples

            # define variable to save how many frames where used for correlation
            number_of_frames_correlated = last_correlation_frame - frame

            # increment which correlation frames were sampled
            number_of_samples_correlated[0:number_of_frames_correlated] += 1

            # compute autocorrelation function per frame, yet in all directions
            VACF_per_frame = np.mean(
                np.sum(
                    saved_velocities_atoms_selected[frame]
                    * saved_velocities_atoms_selected[frame:last_correlation_frame],
                    axis=2,
                ),
                axis=1,
            )

            VACF[0:number_of_frames_correlated] += VACF_per_frame
            # to get insight on the statistical error we compute block averages
            VACF_block[index_current_block_used, 0:number_of_frames_correlated] += VACF_per_frame

            # close block when number of samples per block are reached
            if (
                frame + 1 >= (index_current_block_used + 1) * number_of_samples_per_block
                or frame + 1 == number_of_samples
            ):
                # initialise with 0
                number_of_samples_correlated_per_block = 0
                # check how many samples per frame were taken for this block
                if index_current_block_used == 0:
                    # in first block this corresponds to the global number of samples correlated
                    number_of_samples_correlated_per_block = number_of_samples_correlated
                else:

                    # in all others we just need to get the difference between current and previous global samples
                    number_of_samples_correlated_per_block = (
                        number_of_samples_correlated - previous_global_number_of_samples_correlated
                    )

                # average current block
                VACF_block[index_current_block_used, :] = (
                    VACF_block[index_current_block_used, :] / number_of_samples_correlated_per_block
                )

                # define previous global number of samples
                previous_global_number_of_samples_correlated = number_of_samples_correlated.copy()

                # increment index to move to next block
                index_current_block_used += 1

        # get average autocorrelation
        average_VACF = VACF / number_of_samples_correlated

        # compute statistical error based on block averags
        std_VACF = np.std(VACF_block, axis=0)

        # compute diffusion coefficient from obtained VACF by integration
        # IMPORTANT: the diffusion coefficient will be expressed in m^2/s
        velocity_unit_conversion = (
            global_variables.BOHR_TO_ANGSTROM
            * global_variables.ANGSTROM_TO_METER
            / global_variables.AU_TIME_TO_SECOND
            if units_velocity == "a.u."
            else 1
        )
        # Compute prefactor for unit conversion
        prefactor = velocity_unit_conversion ** 2 * global_variables.FEMTOSECOND_TO_SECOND / 3

        # compute ensemble average of diffusion
        average_diffusion_coefficient = (
            scipy.integrate.cumtrapz(
                average_VACF,
                dx=frame_frequency * self.time_between_frames,
                initial=0.0,
            )
            * prefactor
        )

        # compute diffusion coefficient for each block
        average_diffusion_coefficient_block = (
            scipy.integrate.cumtrapz(
                VACF_block,
                dx=frame_frequency * self.time_between_frames,
                axis=1,
                initial=0.0,
            )
            * prefactor
        )

        # based on these blocks compute std
        std_diffusion_coefficient = np.std(average_diffusion_coefficient_block, axis=0)

        # define time:
        measured_time = (
            np.arange(number_of_correlation_frames) * self.time_between_frames * frame_frequency
        )

        # save all data to dictionary of class
        string_for_dict = f"{selected_species_string} - ct: {correlation_time}"
        self.velocity_autocorrelation_function[string_for_dict] = [
            measured_time,
            average_VACF,
            std_VACF,
        ]

        self.diffusion_coefficients_via_GK[string_for_dict] = [
            measured_time,
            average_diffusion_coefficient,
            std_diffusion_coefficient,
        ]

    def compute_friction_coefficient_via_green_kubo(
        self,
        time_between_frames: float,
        correlation_time: float = 1000.0,
        number_of_blocks: int = 30,
        start_time: int = None,
        end_time: int = None,
        frame_frequency: int = None,
        frame_frequency_surface: int = None,
    ):
        """
        Compute mean friction coefficient lambda via green kubo relation from summed force autocorrelation.
        Arguments:
            time_between_frames (float): Time (in fs) between frames where summed force was measured.
                                        This will substantially vary from the usual printing frequency.
            correlation_time (float) : Time (in fs) to correlate the summed forces, 1000 fs should be usually sufficient.
            start_time (int) : Start time for analysis (optional).
            end_time (int) : End time for analysis (optional).
            frame_frequency (int): Take every nth frame only (optional).
            frame_frequency_surface (int): Take every mth frame only for calculation of radius of tube.
        Returns:
        """

        # get information about sampling either from given arguments or previously set
        start_frame, end_frame, frame_frequency = self._get_sampling_frames(
            start_time, end_time, frame_frequency, time_between_frames
        )
        # convert correlation_time to correlation_frames taken into account the time between frames and
        # the frame frequency
        number_of_correlation_frames = int(correlation_time / time_between_frames / frame_frequency)

        # check if correlation time can be obtained with current trajectory:
        number_of_samples = int((end_frame - start_frame) / frame_frequency)
        if number_of_correlation_frames >= number_of_samples:
            raise UnphysicalValue(
                f" You want to compute a correlation based on {number_of_correlation_frames} frames."
                f"However, the provided trajectory will only be analysed for {number_of_samples} frames.",
                f" Please adjust your correlation or sampling times or run longer trajectories.",
            )

        # get indices in which direction we would like to compute the frictino, usually along one or two axis
        direction_index = global_variables.DIMENSION_DICTIONARY[self.pbc_dimensions]

        # allocate array for MSD, length of number_of_correlation_frames
        autocorrelation_total_summed_forces = np.zeros(number_of_correlation_frames)
        # allocate array for number of samples per correlation frame
        number_of_samples_correlated = np.zeros(number_of_correlation_frames)
        # allocate array for blocks of MSD for statistical error analysis
        autocorrelation_total_summed_forces_block = np.zeros(
            (number_of_blocks, number_of_correlation_frames)
        )

        # define how many samples are evaluated per block
        number_of_samples_per_block = math.ceil(number_of_samples / number_of_blocks)
        index_current_block_used = 0

        # make sure that each block can reach full correlation time
        if number_of_samples_per_block < number_of_correlation_frames:
            raise UnphysicalValue(
                f" Your chosen number of blocks ({number_of_blocks}) is not allowed as:",
                f"samples per block ({number_of_samples_per_block}) < correlation frames {number_of_correlation_frames}.",
                f"Please reduce the number of blocks or run longer trajectories.",
            )

        # determine in which direction the summed force will be correlated
        summed_force_all_directions = self.summed_forces[:, direction_index]

        # Loop over summed forces in the desired direction
        for frame, summed_forces_per_frame in enumerate(
            tqdm((summed_force_all_directions[start_frame:end_frame])[::frame_frequency])
        ):

            # compute last frame sampled, i.e. usually frame+correlation frames
            last_correlation_frame = frame + number_of_correlation_frames
            if last_correlation_frame > number_of_samples - 1:
                last_correlation_frame = number_of_samples

            # define variable to save how many frames where used for correlation
            number_of_frames_correlated = last_correlation_frame - frame

            # increment which correlation frames were sampled
            number_of_samples_correlated[0:number_of_frames_correlated] += 1

            # compute autocorrelation of summed force per frame, for now for each direction separately
            autocorrelation_total_summed_forces_per_frame = np.sum(
                summed_force_all_directions[frame]
                * summed_force_all_directions[frame:last_correlation_frame],
                axis=1,
            )

            # add to variable for ensemble average
            autocorrelation_total_summed_forces[
                0:number_of_frames_correlated
            ] += autocorrelation_total_summed_forces_per_frame

            # to get insight on the statistical error we compute block averages
            autocorrelation_total_summed_forces_block[
                index_current_block_used, 0:number_of_frames_correlated
            ] += autocorrelation_total_summed_forces_per_frame

            # close block when number of samples per block are reached
            if (
                frame + 1 >= (index_current_block_used + 1) * number_of_samples_per_block
                or frame + 1 == number_of_samples
            ):
                # initialise with 0
                number_of_samples_correlated_per_block = 0
                # check how many samples per frame were taken for this block
                if index_current_block_used == 0:
                    # in first block this corresponds to the global number of samples correlated
                    number_of_samples_correlated_per_block = number_of_samples_correlated
                else:

                    # in all others we just need to get the difference between current and previous global samples
                    number_of_samples_correlated_per_block = (
                        number_of_samples_correlated - previous_global_number_of_samples_correlated
                    )

                # average current block
                autocorrelation_total_summed_forces_block[index_current_block_used, :] = (
                    autocorrelation_total_summed_forces_block[index_current_block_used, :]
                    / number_of_samples_correlated_per_block
                )

                # define previous global number of samples
                previous_global_number_of_samples_correlated = number_of_samples_correlated.copy()

                # increment index to move to next block
                index_current_block_used += 1

        # get average autocorrelation
        average_autocorrelation_total_summed_forces = (
            autocorrelation_total_summed_forces / number_of_samples_correlated
        )

        # compute statistical error based on block averags
        std_autocorrelation_total_summed_forces = np.std(
            autocorrelation_total_summed_forces_block, axis=0
        )

        # compute friction from obtained autocorrelation of summed forces by integrating over autcorrelation function
        # IMPORTANT: the friction coefficient lambda will be expressed in N s/m^3
        # first: compute surface area of solid phase:
        surface_area_solid = self._get_surface_area_of_solid_phase(
            direction_index, start_time, end_time, frame_frequency_surface
        )
        # Compute prefactor for unit conversion
        prefactor = (
            (global_variables.EV_TO_JOULE / global_variables.ANGSTROM_TO_METER) ** 2
            / len(direction_index)
            * global_variables.FEMTOSECOND_TO_SECOND
            / global_variables.BOLTZMANN
            / self.temperature
            / surface_area_solid
            / global_variables.ANGSTROM_TO_METER ** 2
        )

        # compute ensemble average of friction
        average_friction_coefficient = (
            scipy.integrate.cumtrapz(
                average_autocorrelation_total_summed_forces,
                dx=frame_frequency * time_between_frames,
                initial=0.0,
            )
            * prefactor
        )

        # compute friction coefficient for each block
        average_friction_coefficient_block = (
            scipy.integrate.cumtrapz(
                autocorrelation_total_summed_forces_block,
                dx=frame_frequency * time_between_frames,
                axis=1,
                initial=0.0,
            )
            * prefactor
        )

        # based on these blocks compute std
        std_friction_coefficient = np.std(average_friction_coefficient_block, axis=0)

        # save all data to dictionary of class
        string_for_dict = f"ct: {correlation_time}"
        self.autocorrelation_summed_forces[string_for_dict] = [
            np.arange(number_of_correlation_frames) * time_between_frames * frame_frequency,
            average_autocorrelation_total_summed_forces,
            std_autocorrelation_total_summed_forces,
        ]

        self.friction_coefficients[string_for_dict] = [
            np.arange(number_of_correlation_frames) * time_between_frames * frame_frequency,
            average_friction_coefficient,
            std_friction_coefficient,
        ]

    def _get_surface_area_of_solid_phase(
        self,
        pbc_dimensions_indices: list,
        start_time: int = None,
        end_time: int = None,
        frame_frequency: int = None,
    ):
        """
        Compute the surface area of the solid phase of the system. This is readily done for sheets as
        the area is simple the product of the box dimensions in the periodic directions. In case of
        nanotubes, however, we need to determine the radius of the tube from the simulation. Obviously,
        this requires a position universe trajectory to be read in previously.
        Arguments:
            pbc_dimensions_indices (list): List of indices of axes being periodic in the system.
            start_time (int) : Start time for analysis (optional).
            end_time (int) : End time for analysis (optional).
            frame_frequency (int): Take every nth frame only (optional).
        Returns:
            surface_area_solid_phase (float): Surface area of the solid phase in A^2.
        """

        # dependent on pbc-dimensions compute surface area
        if len(pbc_dimensions_indices) > 1:
            # assume sheet
            return np.prod(self.topology.get_cell_lengths_and_angles()[pbc_dimensions_indices])

        else:
            # in case we have a tube start we need to compute the radius first

            tube_radius = self.compute_tube_radius(
                pbc_dimensions_indices, start_time, end_time, frame_frequency
            )
            # return surface area: 2*pi*circumference*length
            return (
                2
                * np.pi
                * tube_radius
                * self.topology.get_cell_lengths_and_angles()[pbc_dimensions_indices]
            )

    def compute_tube_radius(
        self,
        pbc_dimensions_indices: list,
        start_time: int = None,
        end_time: int = None,
        frame_frequency: int = None,
    ):
        """
        Compute radius of tube from position universe
        Arguments:
            pbc_dimensions_indices (list): List of indices of axes being periodic in the system.
            start_time (int) : Start time for analysis (optional).
            end_time (int) : End time for analysis (optional).
            frame_frequency (int): Take every nth frame only (optional).
        Returns:
            tube_radius (float): tube radius in A.
        """

        # get information about sampling
        start_frame, end_frame, frame_frequency = self._get_sampling_frames(
            start_time, end_time, frame_frequency
        )

        # determine which position universe are to be used in case of PIMD
        # Thermodynamic properties are based on trajectory of replica
        tmp_position_universes = (
            self.position_universes
            if len(self.position_universes) == 1
            else self.position_universes[1::]
        )

        # loop over all universes
        radii_sampled = []

        for count_universe, universe in enumerate(tmp_position_universes):

            radii_universe = []
            # determine solid atoms, so far only B N C supported
            solid_atoms = universe.select_atoms(f"not name O H")

            # Loop over trajectory, as the radius should converge quickly we take only every 10th frame in comparison
            # to the global settings
            for count_frames, frames in enumerate(
                tqdm((universe.trajectory[start_frame:end_frame])[:: int(frame_frequency)])
            ):

                # determine center of mass:
                system_center_of_mass = np.ma.array(universe.atoms.center_of_mass())

                # use only in 2D (the directions confined)
                system_center_of_mass[pbc_dimensions_indices] = np.ma.masked
                reference_coordinates = system_center_of_mass.compressed()

                # get solid atoms coordinates in 2D
                solid_atoms_positions = np.ma.array(solid_atoms.positions)
                solid_atoms_positions[:, pbc_dimensions_indices] = np.ma.masked
                solid_atoms_positions_confined_directions = (
                    solid_atoms_positions.compressed().reshape(-1, 2)
                )

                # compute radial distance of solid atoms in non-periodic directions from center of mass
                radial_distances_to_axis = np.linalg.norm(
                    solid_atoms_positions_confined_directions - reference_coordinates, axis=1
                )

                radii_universe.append(np.mean(radial_distances_to_axis))

            radii_sampled.append(np.mean(radii_universe))
        # average radius of trajectory
        tube_radius = np.mean(radii_sampled)

        return tube_radius

    def get_water_contact_layer_on_interface(self):
        """
        Compute spatial extension of water contact layer on solid interface.
        Important: This function requires a previous evaluation of the density profile.
        Arguments:
        Returns:
            spatial expansion of the contact layer in angstroms
        """

        # based on pbc check what direction is investigated
        pbc_dimensions_indices = global_variables.DIMENSION_DICTIONARY.get(self.pbc_dimensions)

        not_pbc_dimensions = list(set(self.pbc_dimensions) ^ set("xyz"))
        not_pbc_indices = list(set(pbc_dimensions_indices) ^ set([0, 1, 2]))

        # if bulk, raise error
        if len(pbc_dimensions_indices) == 3:
            raise UnphysicalValue(
                f" You want to compute a contact layer of water for bulk water."
                f" If there is no solid, there won't be a contact layer.",
                f" If you have a solid in your system, please change the pbc dimensions accordingly.",
            )

        # if 2 dimensions
        elif len(pbc_dimensions_indices) == 2:
            string_density_dict = f"O H - {not_pbc_dimensions[0]}"

            # if density profile is not computed raise error
            if not self.density_profiles.get(string_density_dict):
                raise KeyNotFound(
                    f"Couldn't find a density profile with the name {string_density_dict}."
                    f"Make sure you compute the profile first for oxygens and hydrogens."
                )

            # let's start by smoothing the density profile
            smooth_density_profile = scipy.signal.savgol_filter(
                self.density_profiles[string_density_dict][1], 11, 5
            )

            # based on smooth profile, find peaks
            # we multiply the profile by -1 to find minima instead of peaks
            # Use negative to get minima as 'peaks'
            peak_indices, __ = scipy.signal.find_peaks(
                -smooth_density_profile,
                distance=2
                / (
                    self.density_profiles[string_density_dict][0][1]
                    - self.density_profiles[string_density_dict][0][0]
                ),
            )

            # we return now only the second minimum find expressed in the bins of the profile, i.e. in angstroms
            # the second minima is chosen instead of the first, as this is not zero and represents the "end" of
            # the contact layer.
            return self.density_profiles[string_density_dict][0][peak_indices[1]]

        # if 1 dimension
        elif len(pbc_dimensions_indices) == 1:
            string_density_dict = f"O H - radial {self.pbc_dimensions}"

            # if density profile is not computed raise error
            if not self.density_profiles.get(string_density_dict):
                raise KeyNotFound(
                    f"Couldn't find a density profile with the name {string_density_dict}."
                    f"Make sure you compute the profile first for oxygens and hydrogens."
                )
            # let's start by smoothing the density profile
            smooth_density_profile = scipy.signal.savgol_filter(
                self.density_profiles[string_density_dict][1], 11, 5
            )

            # based on smooth profile, find peaks
            # we multiply the profile by -1 to find minima instead of peaks
            # Use negative to get minima as 'peaks'
            peak_indices, __ = scipy.signal.find_peaks(
                -smooth_density_profile,
                distance=2
                / (
                    self.density_profiles[string_density_dict][0][1]
                    - self.density_profiles[string_density_dict][0][0]
                ),
            )

            # we return now only the second minimum find expressed in the bins of the profile, i.e. in angstroms
            # the second minima is chosen instead of the first, as this is not zero and represents the "end" of
            # the contact layer. However, here we have to take the second last element, as the solid is located
            # at larger distances.
            return self.density_profiles[string_density_dict][0][peak_indices[-2]]

    def compute_free_energy_profile(
        self,
        tube_length_in_unit_cells: int = None,
        start_time: int = None,
        end_time: int = None,
        frame_frequency: int = None,
    ):
        """
        Compute the free energy profile of water on top of solid surface.
        Arguments:
            start_time (int) : Start time for analysis (optional).
            end_time (int) : End time for analysis (optional).
            frame_frequency (int): Take every nth frame only (optional).
        Returns:
            surface_area_solid_phase (float): Surface area of the solid phase in A^2.
        """

        # get information about sampling
        start_frame, end_frame, frame_frequency = self._get_sampling_frames(
            start_time, end_time, frame_frequency
        )

        # determine which position universe are to be used in case of PIMD
        # Thermodynamic properties are based on trajectory of replica
        tmp_position_universes = (
            self.position_universes
            if len(self.position_universes) == 1
            else self.position_universes[1::]
        )
        # get periodic directions:
        pbc_dimensions_indices = global_variables.DIMENSION_DICTIONARY.get(self.pbc_dimensions)

        # compute spatial extent of contact layer
        spatial_extent_contact_layer = self.get_water_contact_layer_on_interface()

        # compute average tube radius if tube
        if len(pbc_dimensions_indices) == 1:
            self.tube_radius = self.compute_tube_radius(
                pbc_dimensions_indices, start_time, end_time, frame_frequency
            )

        else:
            self.tube_radius = 0

        # loop over all universes
        for count_universe, universe in enumerate(tmp_position_universes):

            # compute probability of water molecules and solid atoms
            (
                distribution_liquid,
                distribution_solid,
            ) = free_energy.compute_spatial_distribution_of_atoms_on_interface(
                universe,
                self.topology,
                spatial_extent_contact_layer,
                pbc_dimensions_indices,
                start_frame,
                end_frame,
                frame_frequency,
                self.tube_radius,
                tube_length_in_unit_cells,
            )

        # save as attribute of the class instance
        self.free_energy_profile.distribution_liquid = distribution_liquid
        self.free_energy_profile.distribution_solid = distribution_solid

    def prepare_plotting_free_energy_profile(
        self, number_of_bins: int, multiples_of_unit_cell, plot_replica: int = 3
    ):
        """
        Uses the previously computed distributions of atom positions to compute free energy
        and prepare the arrays to be plotted nicely.
        Arguments:
            number_of_bins (int): number of bins used for larger dimension, lower dimensions will be adjusted.
            multiples_of_unit_cell: integer array of periodic replica in 2D.
            plot_replica (int) : Number of replica of the unit cell plotted in 2D.
        """

        # get periodic directions:
        pbc_dimensions_indices = global_variables.DIMENSION_DICTIONARY.get(self.pbc_dimensions)

        # raise error if tube as system but no tube radius given
        if len(pbc_dimensions_indices) == 1 and self.tube_radius == 0:
            raise UnphysicalValue(
                f"Tube radius is set to 0. Please compute the tube radius first which should be done in the free energy routine"
            )

        # do everything in the free energy class
        self.free_energy_profile.prepare_for_plotting(
            self.topology,
            pbc_dimensions_indices,
            number_of_bins,
            multiples_of_unit_cell,
            plot_replica,
            self.tube_radius,
        )
