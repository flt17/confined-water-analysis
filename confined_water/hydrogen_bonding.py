import numpy as np
from tqdm.notebook import tqdm

from confined_water import analysis
from confined_water import utils


class HydrogenBonding:
    """
    Perform all analysis around hydrogen bonding.
    Currently, only support for O and H in water.

    Attributes:

    Methods:

    """

    def __init__(self, position_universe):

        """
        Arguments:
          simulation (MDAnalysis Universe) :  Simulation used to perform HB analysis.
        """

        self.position_universe = position_universe

    def find_acceptor_donor_pairs(
        self, start_frame: int, end_frame: int, frame_frequency: int, time_between_frames: float
    ):
        """
        Identify hydrogen bonds and involved atoms in a simulation
        Arguments:
            start_frame (int) : Start frame for analysis.
            end_frame (int) : End frame for analysis.
            frame_frequency (int): Take every nth frame only.
            time_between_frames (int): Time (in fs) between two frames in sampled trajectory, e.g. 100 fs.
        Returns:

        """

        # Select atoms relevant for HB bonding
        # Here, this is limited to O and H
        oxygen_atoms = self.position_universe.select_atoms("name O")
        hydrogen_atoms = self.position_universe.select_atoms("name H")

        # Loop over trajectory
        for count_frames, frames in enumerate(
            tqdm((self.position_universe.trajectory[start_frame:end_frame])[::frame_frequency])
        ):

            # wrap relevant atoms back in box, dimensions are defined via topology part of position_universe
            self.position_universe.atoms.pack_into_box(inplace=True)

            # initialise instance of DonorAcceptorPairs which will be used to save all information.
            donor_acceptor_pairs_per_frame = DonorAcceptorPairs()

            # save time of the frame
            donor_acceptor_pairs_per_frame.time = frames.frame * time_between_frames

            # to find all hydrogen bonds we need to find pairs which satisfy three criteria:
            # 1.) oxygen-oxygen 2.) donor(hydrogen)-acceptor(oxygen) 3.) angle between OOH
            # start with: 1.) oxygen-oxygen criterion

            # define all vectors between all oxygens in the system:
            oxygen_oxygen_vectors = (
                oxygen_atoms.positions[np.newaxis, :] - oxygen_atoms.positions[:, np.newaxis]
            )


class DonorAcceptorPairs:
    """
    Gather donor-acceptor pairs in dataframe format for one frame
    of a position_universe.trajectory

    Attributes:
        time: time of the frame analysed.

    Methods:

    """

    def __init__(self):
        self.time = 0
