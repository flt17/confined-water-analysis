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
        """
        self.name = name


class Simulation:
    """
    Perform post-processing of a (PI)MD simulation.

    Attributes:

    Methods:

    """

    def __init__(self, directory_path):
        """
        Arguments:
            path (str) :  Path to the simulation directory.
        """

        self.directory_path = directory_path

    def read_in_simulation_data(self, read_positions: bool = True):

        """
        Setup all selected simulation data.
        Arguments:
            read_positions (bool) : Whether to read in position trajectories.

        Returns:

        """
        # setup topology based on only pdb file in directoy
        path_to_topology = utils.get_path_to_file(self.directory_path, "pdb")
        self.topology = utils.get_ase_atoms_object(path_to_topology)

        # Read in what needs to be read in (right now via if loop)

        # Trajectory, can be one ore multiple. If more than one, first element is centroid.
        # Functions which compute a specific property will choose which universe is picked in case of PIMD.
        if read_positions:
            self.position_universes = utils.get_mdanalysis_universe(
                self.directory_path, "positions"
            )
