import sys

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

    def read_in_simulation_data(
        self, read_positions: bool = True, read_summed_forces: bool = False
    ):

        """
        Setup all selected simulation data.
        Arguments:
            read_positions (bool) : Whether to read in position trajectories.
            read_summed_forces (bool) : Whether to read in separately printed summed forces.

        Returns:

        """
        # setup topology based on only pdb file in directoy
        path_to_topology = utils.get_path_to_file(self.directory_path, "pdb")
        self.topology = utils.get_ase_atoms_object(path_to_topology)

        # Read in what needs to be read in (right now via if loop)

        # Trajectory, can be one ore multiple. If more than one, first element is centroid.
        # Functions which compute a specific property will choose which universe is picked in case of PIMD.
        if read_positions:
            universes = utils.get_mdanalysis_universe(self.directory_path, "positions")

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
