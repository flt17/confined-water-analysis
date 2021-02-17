import collections
import numpy as np
import pandas
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

    def __init__(self, position_universe, topology):

        """
        Arguments:
          simulation (MDAnalysis Universe) :  Simulation used to perform HB analysis.
        """

        self.position_universe = position_universe
        self.topology = topology

    def find_acceptor_donor_pairs(
        self,
        start_frame: int,
        end_frame: int,
        frame_frequency: int,
        time_between_frames: float,
        pbc_dimensions: str = "xyz",
        oxygen_oxygen_distance: float = 3.5,
        acceptor_hydrogen_distance: float = 2.6,
        angle_OOH: float = 30,
    ):
        """
        Identify hydrogen bonds and involved atoms in a simulation. In case of default settings,
        hydrogen bonds are defined as via geometrical criteria defined by:
        A Luzar and D Chandler, Nature (1996), 379, 55-57.
        Arguments:
            start_frame (int) : Start frame for analysis.
            end_frame (int) : End frame for analysis.
            frame_frequency (int): Take every nth frame only.
            time_between_frames (float): Time (in fs) between two frames in sampled trajectory, e.g. 100 fs.
            pbc_dimension (str): Apply pbcs in these directions.
            oxygen_oxygen_distance (float): Allowed distance (in angstroms) between oxygens to satisfy hydrogen
                                            bonding criterion.
            acceptor_hydrogen_distance (float): Allowed distance (in angstroms) between acceptor and donor hydrogen
                                                to satisfy hydrogen bonding criterion.
            angle_OOH (float): Allowed angle (in degrees) between O-O  and donor-oxygen-hydrogen axis to satisfy
                              hydrogen bonding criterion.

        Returns:

        """

        # Select atoms relevant for HB bonding
        # Here, this is limited to O and H
        oxygen_atoms = self.position_universe.select_atoms("name O")
        hydrogen_atoms = self.position_universe.select_atoms("name H")

        # get number of water molecules in system
        number_of_water_molecules = len(oxygen_atoms)

        hydrogen_bonding_data = []

        # determine minimum index, i.e. is water saved as HHO or OHH
        min_index = np.min([np.min(oxygen_atoms.indices), np.min(hydrogen_atoms.indices)])

        # Loop over trajectory
        for count_frames, frames in enumerate(
            tqdm((self.position_universe.trajectory[start_frame:end_frame])[::frame_frequency])
        ):

            # wrap relevant atoms back in box, dimensions are defined via topology part of position_universe

            self.position_universe.atoms.pack_into_box(
                box=self.topology.get_cell_lengths_and_angles(), inplace=True
            )

            # initialise instance of DonorAcceptorPairs which will be used to save all information.
            donor_acceptor_pairs_per_frame = DonorAcceptorPairs()

            # save time of the frame
            donor_acceptor_pairs_per_frame.time = frames.frame * time_between_frames

            # to find all hydrogen bonds we need to find pairs which satisfy three criteria:
            # 1.) oxygen-oxygen 2.) donor(hydrogen)-acceptor(oxygen) 3.) angle between OOH
            # start with: 1.) oxygen-oxygen criterion

            # define all vectors between all oxygens in the system:
            vectors_oxygen_oxygen = (
                oxygen_atoms.positions[np.newaxis, :] - oxygen_atoms.positions[:, np.newaxis]
            )

            # apply MIC for all oxygen-oxygen pairs
            vectors_oxygen_oxygen_MIC = utils.apply_minimum_image_convention_to_interatomic_vectors(
                vectors_oxygen_oxygen, self.topology.cell, pbc_dimensions
            )

            # get distances based on vectors
            distances_oxygen_oxygen = np.linalg.norm(vectors_oxygen_oxygen_MIC, axis=2)

            # take all oxygen-oxygen pairs which satisfy oxygen_oxygen_distance criterion
            oxygen_oxygen_pairs_crit1 = np.where(
                (distances_oxygen_oxygen <= oxygen_oxygen_distance) & (distances_oxygen_oxygen > 0)
            )

            # create dictionary listing number of possible candidates for each oxygen, double counting not prevented
            dictionary_number_of_oxygens_per_oxygen_crit1 = collections.OrderedDict(
                collections.Counter(oxygen_oxygen_pairs_crit1[0])
            )

            # from dictionary infer indices in oxygen_oxygen_pairs_criterion_1 for each oxygen
            tmp_indices_per_oxygen_crit1 = np.cumsum(
                np.asarray(
                    [
                        dictionary_number_of_oxygens_per_oxygen_crit1.get(i, 0)
                        for i in np.arange(number_of_water_molecules)
                    ]
                )
            )

            # split oxygen-oxygen vectors, distances, and oxygen-oxygen pairs accordingly
            vectors_oxygen_oxygen_crit1_split = np.split(
                vectors_oxygen_oxygen_MIC[oxygen_oxygen_pairs_crit1],
                tmp_indices_per_oxygen_crit1[:-1],
            )
            distances_oxygen_oxygen_crit1_split = np.split(
                distances_oxygen_oxygen[oxygen_oxygen_pairs_crit1],
                tmp_indices_per_oxygen_crit1[:-1],
            )

            oxygen_oxygen_pairs_crit1_split = np.split(
                oxygen_oxygen_pairs_crit1[1], tmp_indices_per_oxygen_crit1[:-1]
            )

            ######################################
            # prepare 2.) donor-acceptor criterion
            ######################################

            # split hydrogen positions according to respective waters
            hydrogen_positions_split = np.split(hydrogen_atoms.positions, number_of_water_molecules)

            # define all vectors from oxygens to covalently bonded hydrogens
            vectors_oxyen_covalent_hydrogens_split = (
                hydrogen_positions_split - oxygen_atoms.positions[:, np.newaxis]
            )

            # apply MIC to all vectors
            vectors_oxyen_covalent_hydrogens_split_MIC = (
                utils.apply_minimum_image_convention_to_interatomic_vectors(
                    vectors_oxyen_covalent_hydrogens_split, self.topology.cell, pbc_dimensions
                )
            )

            # compute distances from hydrogens of donor molecule to all oxygens which satisfy criterion 1
            distances_acceptor_oxygen_hydrogen_crit1_split = [
                (
                    np.linalg.norm(
                        vectors_oxyen_covalent_hydrogens_split_MIC[i]
                        - vectors_oxygen_oxygen_crit1_split[i][:, np.newaxis],
                        axis=2,
                    )
                ).T
                for i in np.arange(number_of_water_molecules)
            ]

            ###############################
            # prepare 3.) OOH-angle criterion
            ###############################

            angles_from_oxygen_donor_to_hydrogen_acceptor_oxygen = [
                np.rad2deg(
                    np.arccos(
                        np.clip(
                            (
                                np.dot(
                                    vectors_oxyen_covalent_hydrogens_split_MIC[i],
                                    vectors_oxygen_oxygen_crit1_split[i].T,
                                )
                                / distances_oxygen_oxygen_crit1_split[i]
                            )
                            / np.linalg.norm(
                                vectors_oxyen_covalent_hydrogens_split_MIC[i], axis=1
                            ).reshape((2, -1)),
                            -1,
                            1,
                        )
                    )
                )
                for i in np.arange(number_of_water_molecules)
            ]

            ###################
            # apply 2.) and 3.)
            ###################

            # get indices of hydrogens and related acceptors oxygens going through all water molecules
            tmp_indices_crit123 = [
                np.where(
                    (angles_from_oxygen_donor_to_hydrogen_acceptor_oxygen[i] < angle_OOH)
                    & (
                        distances_acceptor_oxygen_hydrogen_crit1_split[i]
                        < acceptor_hydrogen_distance
                    )
                )
                for i in np.arange(number_of_water_molecules)
            ]

            ##########################
            # save all pairs to object
            ##########################

            # ids of donors (hydrogens)
            donor_acceptor_pairs_per_frame.hydrogen_donor_ids = np.concatenate(
                [
                    hydrogen_atoms.ids[tmp_indices_crit123[i][0] + i * 2].flatten()
                    for i in np.arange(number_of_water_molecules)
                ]
            )

            # ids of acceptors (oxygens)
            donor_acceptor_pairs_per_frame.oxygen_acceptor_ids = oxygen_atoms.ids[
                np.concatenate(
                    [
                        oxygen_oxygen_pairs_crit1_split[i][tmp_indices_crit123[i][1]].flatten()
                        for i in np.arange(number_of_water_molecules)
                    ]
                )
            ]

            # id of donor water molecule
            donor_acceptor_pairs_per_frame.water_donor_ids = np.asarray(
                (
                    np.concatenate(
                        [
                            hydrogen_atoms.indices[tmp_indices_crit123[i][0] + i * 2].flatten()
                            for i in np.arange(number_of_water_molecules)
                        ]
                    )
                    - min_index
                )
                / 3
                + 1
            ).astype(int)

            # id of acceptor water molecule
            donor_acceptor_pairs_per_frame.water_acceptor_ids = np.asarray(
                (
                    np.asarray(
                        oxygen_atoms.indices[
                            np.concatenate(
                                [
                                    oxygen_oxygen_pairs_crit1_split[i][
                                        tmp_indices_crit123[i][1]
                                    ].flatten()
                                    for i in np.arange(number_of_water_molecules)
                                ]
                            )
                        ]
                    )
                    - min_index
                )
                / 3
                + 1
            )

            # distance between oxygens
            donor_acceptor_pairs_per_frame.oxygen_oxygen_distances = np.concatenate(
                [
                    distances_oxygen_oxygen_crit1_split[i][tmp_indices_crit123[i][1]].flatten()
                    for i in np.arange(number_of_water_molecules)
                ]
            )

            # angles between acceptor oxygen, donor oxygen, and donor hydrogen
            donor_acceptor_pairs_per_frame.angles_OOH = np.concatenate(
                [
                    angles_from_oxygen_donor_to_hydrogen_acceptor_oxygen[i][tmp_indices_crit123[i]]
                    for i in np.arange(number_of_water_molecules)
                ]
            )

            # compute delta distance: r_(O-H)-r(H...O)
            # 1. distances between donor oxygen and hydrogen

            distances_oxyen_covalent_hydrogens = np.concatenate(
                [
                    np.linalg.norm(
                        vectors_oxyen_covalent_hydrogens_split_MIC[i, tmp_indices_crit123[i][0]],
                        axis=1,
                    )
                    for i in np.arange(number_of_water_molecules)
                ]
            )

            # 2. compute acceptor oxygen hydrogen distance
            distances_acceptor_oxygen_H = np.concatenate(
                [
                    np.linalg.norm(
                        vectors_oxyen_covalent_hydrogens_split_MIC[i, tmp_indices_crit123[i][0]]
                        - vectors_oxygen_oxygen_crit1_split[i][tmp_indices_crit123[i][1]],
                        axis=1,
                    )
                    for i in np.arange(number_of_water_molecules)
                ]
            )

            # 3. compute delta_distance
            donor_acceptor_pairs_per_frame.delta_distances = (
                distances_oxyen_covalent_hydrogens - distances_acceptor_oxygen_H
            )

            # save coordinates of hydrogen bond
            # so far, we approxiate the coordinate by oxygen atom
            # could be easily changed to be weighted according to molecular mass with hydrogen
            # breakpoint()
            donor_acceptor_pairs_per_frame.hydrogen_bond_coordinates = np.concatenate(
                [
                    oxygen_atoms[
                        oxygen_oxygen_pairs_crit1_split[i][tmp_indices_crit123[i][1]]
                    ].positions
                    for i in np.arange(number_of_water_molecules)
                ]
            )[:, :]

            # Eventually, append computed array with all this information for one frame
            hydrogen_bonding_data.extend(
                donor_acceptor_pairs_per_frame.gather_information_in_numpy_array()
            )

        # convert collected arrays to pandas dataframe
        hydrogen_bonding_dataframe = pandas.DataFrame(
            hydrogen_bonding_data,
            columns=[
                "Time",
                "Donor ID",
                "Acceptor ID",
                "Donor molecule",
                "Acceptor molecule",
                "Distance between oxygens",
                "Delta distance",
                "angle OOH",
                "Hydrogen bond x",
                "Hydrogen bond y",
                "Hydrogen bond z",
            ],
        )

        hydrogen_bonding_dataframe["Donor ID"] = hydrogen_bonding_dataframe["Donor ID"].astype(int)
        hydrogen_bonding_dataframe["Acceptor ID"] = hydrogen_bonding_dataframe[
            "Acceptor ID"
        ].astype(int)

        hydrogen_bonding_dataframe["Donor molecule"] = hydrogen_bonding_dataframe[
            "Donor molecule"
        ].astype(int)
        hydrogen_bonding_dataframe["Acceptor molecule"] = hydrogen_bonding_dataframe[
            "Acceptor molecule"
        ].astype(int)

        self.dataframe = hydrogen_bonding_dataframe


class DonorAcceptorPairs:
    """
    Gather donor-acceptor pairs in np.array format for one frame
    of a position_universe.trajectory. This will then be used latter
    to generate pandas.DataFrame object.

    Attributes:
        time (float): time of the frame analysed.
        hydrogen_donor_ids (np.array): atom ids of hydrogen donors.
        oxygen_acceptor_ids (np.array): atom ids of oxygen acceptors.
        oxygen_oxygen_distances (np.array): distances between donor oxygen and acceptor oxygen.
        angles_OOH (np.array): angles between acceptor oxygen, donor oxygen, and hydrogen.
        delta_distances (np.array): r_(O-H)-r(H...O) to quantify hydrogen bonding.
        hydrogen_bond_coordinates (np.array) :

    Methods:

    """

    def __init__(self):
        self.time = 0

    def gather_information_in_numpy_array(self):
        """
        Generate np.array based on all information collected and saved in the instance.
        Arguments:

        Returns:
            array (np.array): contains all information in stacked array.
        """

        array = np.vstack(
            [
                np.repeat(self.time, len(self.hydrogen_donor_ids)),
                self.hydrogen_donor_ids,
                self.oxygen_acceptor_ids,
                self.water_donor_ids,
                self.water_acceptor_ids,
                self.oxygen_oxygen_distances,
                self.delta_distances,
                self.angles_OOH,
                self.hydrogen_bond_coordinates[:, 0],
                self.hydrogen_bond_coordinates[:, 1],
                self.hydrogen_bond_coordinates[:, 2],
            ]
        ).T

        return array
