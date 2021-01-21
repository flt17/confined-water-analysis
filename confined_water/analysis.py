import numpy as np
import sys
from tqdm.notebook import tqdm

import MDAnalysis.analysis.rdf as mdanalysis_rdf

sys.path.append("../")
from confined_water import hydrogen_bonding
from confined_water import utils


# GLOBAL VARIABLES

AVOGADRO = 6.0221409 * 1e23


class KeyNotFound(Exception):
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

        # set system periodicity per default:
        self.set_pbc_dimensions("xyz")

        self.radial_distribution_functions = {}
        self.density_profiles = {}
        self.hydrogen_bonding = []

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

    def set_pbc_dimensions(self, pbc_dimensions: str):
        """
        Set in which direction pbc apply.
        Arguments:
            pbc_dimensions (str) : string of directions in which pbc apply.

        Returns:

        """

        dimension_dictionary = {
            "x": [0],
            "xy": [0, 1],
            "xz": [0, 2],
            "y": [1],
            "yz": [1, 2],
            "z": [2],
            "xyz": [0, 1, 2],
        }

        if not dimension_dictionary.get(pbc_dimensions):
            raise KeyNotFound(
                f"Specified dimension {pbc_dimensions} is unknown. Possible options are {dimension_dictionary.keys()}"
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
        self, start_time: int = None, end_time: int = None, frame_frequency: int = None
    ):
        """
        Determine sampling frames from given sampling times.
        Arguments:
            start_time (int) : Start time for analysis.
            end_time (int) : End time for analysis.
            frame_frequency (int): Take every nth frame only.
        Returns:
            start_frame (int) : Start frame for analysis.
            end_frame (int) : End frame for analysis.
            frame_frequency (int): Take every nth frame only.

        """
        start_time = start_time if start_time is not None else self.start_time
        end_time = end_time if end_time is not None else self.end_time

        frame_frequency = int(
            frame_frequency if frame_frequency is not None else self.frame_frequency
        )

        start_frame = int(start_time / self.time_between_frames)
        end_frame = int(end_time / self.time_between_frames)

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
                    / AVOGADRO
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
                    / AVOGADRO
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
