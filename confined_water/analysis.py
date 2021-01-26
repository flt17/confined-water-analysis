import math
import numpy as np
import pandas
import scipy
import sys
from tqdm.notebook import tqdm

import MDAnalysis.analysis.rdf as mdanalysis_rdf

sys.path.append("../")
from confined_water import hydrogen_bonding
from confined_water import utils
from confined_water import global_variables


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
        self.autocorrelation_summed_forces = {}
        self.friction_coefficients = {}

    def read_in_simulation_data(
        self,
        read_positions: bool = True,
        read_summed_forces: bool = False,
        topology_file_name: str = None,
    ):

        """
        Setup all selected simulation data.
        Arguments:
            read_positions (bool) : Whether to read in position trajectories.
            read_summed_forces (bool) : Whether to read in separately printed summed forces.
            topology_file_name (str) : Name of the topology file (currently only pdb). If not given, first file taken.

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
                self.directory_path, "positions", topology_file_name
            )

            # to make sure self.position_universes is always a list of MDAnalysis Universes
            self.position_universes = (
                universes if isinstance(universes, list) == True else [universes]
            )

            self.species_in_system = np.unique(self.position_universes[0].atoms.names)

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

    def compute_density_profile(
        self,
        species: list,
        direction: str,
        start_time: int = None,
        end_time: int = None,
        frame_frequency: int = None,
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
        Returns:
            bins_histogram (np.array): Bins of the histogram of the density profile.
            density_profile (np.asarray) : Density profile based on the bins.

        """

        # determine reference atoms, here only solid atoms (B,N,C) allowed
        reference_species = ["B", "N", "C"]
        reference_species_string = " ".join(reference_species)

        if not set(self.species_in_system).intersection(reference_species):
            raise KeyNotFound(
                f"Couldn't find a solid phase in this trajectory. Currently only {reference_species_string} are implemented."
            )

        reference_atoms = position_universe.select_atoms(f"name {reference_species_string}")

        # define range and bin width for histogram binning
        # bin range is simply the box length in the given direction
        bin_range = self.topology.get_cell_lengths_and_angles()[direction]
        # for now, bin width is set to a default value of 0.1 angstroms
        bin_width = 0.1
        bins_histogram = np.arange(0, bin_range, bin_width)

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
        Returns:
            bins_histogram (np.array): Bins of the histogram of the density profile.
                                      Note that the bins are based on the center of np.digitize.
            density_profile (np.asarray) : Density profile based on the bins.

        """
        # define range and bin width for histogram binning
        # bin range is simply the half box length along axis orthogonal to direction (squared area)
        bin_range = self.topology.get_cell_lengths_and_angles()[0:3][direction - 1] * 0.5
        # for now, bin width is set to a default value of 0.1 angstroms
        bin_width = 0.1
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

        # number_of_tracers = (
        #     int(len(atoms_selected) / 3)
        #     if selected_species_string == "O H"
        #     else len(atoms_selected)
        # )

        # allocate array for all positions of all selected atoms for all frames sampled
        saved_positions_atoms_selected = np.zeros((number_of_samples, len(atoms_selected), 3))

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
            (tmp_position_universe.trajectory[start_frame:end_frame])[::frame_frequency]
        ):
            # This shouldn't be necessary as we should only use wrapped trajectories
            # Leaving it in as it is cheap and better safe than sorry
            atoms_selected.pack_into_box(
                box=self.topology.get_cell_lengths_and_angles(), inplace=True
            )

            # if water diffusion compute center of mass per water molecule
            # Based on CP2K input this is readily done due to the the H-H-O input structure
            # If the input file has another format this will lead to errorneous results and
            # it would be better to just trace the oxygens.

            # fill array with positions
            saved_positions_atoms_selected[count_frames] = atoms_selected.positions

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

            # apply minimum image convention to these vectors
            vectors_atom_movement_MIC = utils.apply_minimum_image_convention_to_interatomic_vectors(
                vectors_atom_movement, self.topology.cell, self.pbc_dimensions
            )

            # compute squared_distance of the moved atoms between each frame
            squared_distances_atom_movement = np.square(
                np.linalg.norm(vectors_atom_movement_MIC, axis=2)
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

        # get average MSD
        average_mean_squared_displacement = mean_squared_displacement / number_of_samples_correlated

        # compute statistical error based on block averags
        std_mean_squared_displacement = np.std(mean_squared_displacement_blocks, axis=0)

        # save all data to dictionary of class
        string_for_dict = f"{selected_species_string} - ct: {correlation_time}"
        self.mean_squared_displacements[string_for_dict] = [
            np.arange(number_of_correlation_frames) * self.time_between_frames * frame_frequency,
            average_mean_squared_displacement,
            std_mean_squared_displacement,
        ]

    def compute_friction_coefficient_via_green_kubo(
        self,
        time_between_frames: float,
        temperature: float = 330,
        correlation_time: float = 1000.0,
        number_of_blocks: int = 30,
        start_time: int = None,
        end_time: int = None,
        frame_frequency: int = None,
    ):
        """
        Compute mean friction coefficient lambda via green kubo relation from summed force autocorrelation.
        Arguments:
            time_between_frames (float): Time (in fs) between frames where summed force was measured.
                                        This will substantially vary from the usual printing frequency.
            temperature (float) : Simulation temperature in K.
            correlation_time (float) : Time (in fs) to correlate the summed forces, 1000 fs should be usually sufficient.
            start_time (int) : Start time for analysis (optional).
            end_time (int) : End time for analysis (optional).
            frame_frequency (int): Take every nth frame only (optional).
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
        surface_area_solid = self._get_surface_area_of_solid_phase(direction_index)
        # Compute prefactor for unit conversion
        prefactor = (
            (global_variables.EV_TO_JOULE / global_variables.ANGSTROM_TO_METER) ** 2
            / len(direction_index)
            * global_variables.FEMTOSECOND_TO_SECOND
            / global_variables.BOLTZMANN
            / temperature
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
            # in case we have a tube start sampling radius, i.e. distance from atom from center of mass

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
                solid_atoms = universe.select_atoms(f"name B N C")

                # Loop over trajectory, as the radius should converge quickly we take only every 10th frame in comparison
                # to the global settings
                for count_frames, frames in enumerate(
                    tqdm((universe.trajectory[start_frame:end_frame])[:: int(10 * frame_frequency)])
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
            average_radius = np.mean(radii_sampled)
            # return surface area: 2*pi*circumference*length
            return (
                2
                * np.pi
                * average_radius
                * self.topology.get_cell_lengths_and_angles()[pbc_dimensions_indices]
            )
