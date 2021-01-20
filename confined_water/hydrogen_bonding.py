from confined_water import analysis
from confined_water import utils


class HydrogenBonding:
    """
    Perform all analysis around hydrogen bonding.

    Attributes:

    Methods:

    """

    def __init__(self, position_universe):

        """
        Arguments:
          simulation (MDAnalysis Universe) :  Simulation used to perform HB analysis.
        """

        self.position_universe = position_universe

    def find_hydrogen_bonds_in_simulation(
        self, start_frame: int, end_frame: int, frame_frequency: int
    ):
        """
        Identify hydrogen bonds and involved atoms in a simulation
        Arguments:
            start_frame (int) : Start frame for analysis.
            end_frame (int) : End frame for analysis.
            frame_frequency (int): Take every nth frame only.
        Returns:

        """

        pass
