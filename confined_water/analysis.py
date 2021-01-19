import numpy as np
import sys

import MDAnalysis.analysis.rdf as mdanalysis_rdf

sys.path.append("../")
from confined_water import utils


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

        self.radial_distribution_functions = {}

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
        path_to_topology = utils.get_path_to_file(self.directory_path, "pdb")
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
                universes if isinstance(universes, list) == list else [universes]
            )

    def set_sampling_times(
        self,
        start_time: int = None,
        end_time: int = None,
        frame_frequency: int = None,
        time_between_frames: int = None,
    ):

        """
        Set times for analysis of trajectories.
        Arguments:
            start_time (int) : Start time for analysis.
            end_time (int) : End time for analysis.
            frame_frequency (int): Take every nth frame only.
            time_between_frames (int): Time (in fs) between two frames in sampled trajectory, e.g. 100 fs.
        Returns:

        """

        self.start_time = start_time if start_time is not None else self.start_time
        self.frame_frequency = (
            frame_frequency if frame_frequency is not None else self.frame_frequency
        )
        self.time_between_frames = (
            time_between_frames if time_between_frames is not None else self.time_between_frames
        )

        total_time = (self.position_universes[0].nframes - 1) * self.time_between_frames
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
        Compute radial distribution function for given species.
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
