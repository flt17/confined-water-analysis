import collections
import numpy as np
import pandas
from tqdm.notebook import tqdm

from confined_water import analysis
from confined_water import utils
from confined_water import global_variables


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

        # rewind trajectory
        self.position_universe.trajectory[0]

        # Select atoms relevant for HB bonding
        # Here, this is limited to O and H
        oxygen_atoms = self.position_universe.select_atoms("name O")
        hydrogen_atoms = self.position_universe.select_atoms("name H")

        # get number of water molecules in system
        number_of_water_molecules = len(oxygen_atoms)

        hydrogen_bonding_data = []

        # determine minimum index, i.e. is water saved as HHO or OHH
        min_index = np.min(
            [np.min(oxygen_atoms.indices), np.min(hydrogen_atoms.indices)]
        )

        # Loop over trajectory
        for count_frames, frames in enumerate(
            tqdm(
                (self.position_universe.trajectory[start_frame:end_frame])[
                    ::frame_frequency
                ]
            )
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
                oxygen_atoms.positions[np.newaxis, :]
                - oxygen_atoms.positions[:, np.newaxis]
            )

            # apply MIC for all oxygen-oxygen pairs
            vectors_oxygen_oxygen_MIC = (
                utils.apply_minimum_image_convention_to_interatomic_vectors(
                    vectors_oxygen_oxygen, self.topology.cell, pbc_dimensions
                )
            )

            # get distances based on vectors
            distances_oxygen_oxygen = np.linalg.norm(vectors_oxygen_oxygen_MIC, axis=2)

            # take all oxygen-oxygen pairs which satisfy oxygen_oxygen_distance criterion
            oxygen_oxygen_pairs_crit1 = np.where(
                (distances_oxygen_oxygen <= oxygen_oxygen_distance)
                & (distances_oxygen_oxygen > 0)
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
            hydrogen_positions_split = np.split(
                hydrogen_atoms.positions[np.argsort(hydrogen_atoms.resids)],
                number_of_water_molecules,
            )

            # define all vectors from oxygens to covalently bonded hydrogens
            vectors_oxygen_covalent_hydrogens_split = (
                hydrogen_positions_split - oxygen_atoms.positions[:, np.newaxis]
            )

            # apply MIC to all vectors
            vectors_oxygen_covalent_hydrogens_split_MIC = (
                utils.apply_minimum_image_convention_to_interatomic_vectors(
                    vectors_oxygen_covalent_hydrogens_split,
                    self.topology.cell,
                    pbc_dimensions,
                )
            )

            # compute distances from hydrogens of donor molecule to all oxygens which satisfy criterion 1
            distances_acceptor_oxygen_hydrogen_crit1_split = [
                (
                    np.linalg.norm(
                        vectors_oxygen_covalent_hydrogens_split_MIC[i]
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
                                    vectors_oxygen_covalent_hydrogens_split_MIC[i],
                                    vectors_oxygen_oxygen_crit1_split[i].T,
                                )
                                / distances_oxygen_oxygen_crit1_split[i]
                            )
                            / np.linalg.norm(
                                vectors_oxygen_covalent_hydrogens_split_MIC[i], axis=1
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
                    (
                        angles_from_oxygen_donor_to_hydrogen_acceptor_oxygen[i]
                        < angle_OOH
                    )
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
                        oxygen_oxygen_pairs_crit1_split[i][
                            tmp_indices_crit123[i][1]
                        ].flatten()
                        for i in np.arange(number_of_water_molecules)
                    ]
                )
            ]

            # id of donor water molecule
            donor_acceptor_pairs_per_frame.water_donor_ids = np.asarray(
                (
                    np.concatenate(
                        [
                            hydrogen_atoms.indices[
                                tmp_indices_crit123[i][0] + i * 2
                            ].flatten()
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
                    distances_oxygen_oxygen_crit1_split[i][
                        tmp_indices_crit123[i][1]
                    ].flatten()
                    for i in np.arange(number_of_water_molecules)
                ]
            )

            # angles between acceptor oxygen, donor oxygen, and donor hydrogen
            donor_acceptor_pairs_per_frame.angles_OOH = np.concatenate(
                [
                    angles_from_oxygen_donor_to_hydrogen_acceptor_oxygen[i][
                        tmp_indices_crit123[i]
                    ]
                    for i in np.arange(number_of_water_molecules)
                ]
            )

            # compute delta distance: r_(O-H)-r(H...O)
            # 1. distances between donor oxygen and hydrogen

            distances_oxygen_covalent_hydrogens = np.concatenate(
                [
                    np.linalg.norm(
                        vectors_oxygen_covalent_hydrogens_split_MIC[
                            i, tmp_indices_crit123[i][0]
                        ],
                        axis=1,
                    )
                    for i in np.arange(number_of_water_molecules)
                ]
            )

            # 2. compute acceptor oxygen hydrogen distance
            distances_acceptor_oxygen_H = np.concatenate(
                [
                    np.linalg.norm(
                        vectors_oxygen_covalent_hydrogens_split_MIC[
                            i, tmp_indices_crit123[i][0]
                        ]
                        - vectors_oxygen_oxygen_crit1_split[i][
                            tmp_indices_crit123[i][1]
                        ],
                        axis=1,
                    )
                    for i in np.arange(number_of_water_molecules)
                ]
            )

            # 3. compute delta_distance
            donor_acceptor_pairs_per_frame.delta_distances = (
                distances_oxygen_covalent_hydrogens - distances_acceptor_oxygen_H
            )

            # save coordinates of hydrogen bond, first acceptor
            donor_acceptor_pairs_per_frame.acceptor_coordinates = np.concatenate(
                [
                    oxygen_atoms[
                        oxygen_oxygen_pairs_crit1_split[i][tmp_indices_crit123[i][1]]
                    ].positions
                    for i in np.arange(number_of_water_molecules)
                ]
            )[:, :]

            # then donor
            donor_acceptor_pairs_per_frame.donor_coordinates = np.asarray(
                [
                    oxygen_atoms[int(water_mol - 1)].position
                    for water_mol in donor_acceptor_pairs_per_frame.water_donor_ids
                ]
            )

            # Eventually, append computed array with all this information for one frame
            hydrogen_bonding_data.extend(
                donor_acceptor_pairs_per_frame.gather_information_in_numpy_array()
            )

        # convert collected arrays to pandas dataframe
        hydrogen_bonding_dataframe = pandas.DataFrame(
            hydrogen_bonding_data,
            columns=[
                "Time",
                "Donor atom ID",
                "Acceptor atom ID",
                "Donor molecule ID",
                "Acceptor molecule ID",
                "Distance between oxygens",
                "Delta distance",
                "angle OOH",
                "Donor molecule x",
                "Donor molecule y",
                "Donor molecule z",
                "Acceptor molecule x",
                "Acceptor molecule y",
                "Acceptor molecule z",
            ],
        )

        hydrogen_bonding_dataframe["Donor atom ID"] = hydrogen_bonding_dataframe[
            "Donor atom ID"
        ].astype(int)
        hydrogen_bonding_dataframe["Acceptor atom ID"] = hydrogen_bonding_dataframe[
            "Acceptor atom ID"
        ].astype(int)

        hydrogen_bonding_dataframe["Donor molecule ID"] = hydrogen_bonding_dataframe[
            "Donor molecule ID"
        ].astype(int)
        hydrogen_bonding_dataframe["Acceptor molecule ID"] = hydrogen_bonding_dataframe[
            "Acceptor molecule ID"
        ].astype(int)

        self.dataframe = hydrogen_bonding_dataframe

    def heavy_atoms_analysis(
        self,
        start_frame: int,
        end_frame: int,
        frame_frequency: int,
        time_between_frames: float,
        spatial_extent_contact_layer: float,
        pbc_dimensions: str = "xyz",
        oxygen_heavy_distance: float = 3.5,
    ):
        """
        Analyse Heavy atom distances and angles with hydrogens in contact layer.
        Arguments:
            start_frame (int) : Start frame for analysis.
            end_frame (int) : End frame for analysis.
            frame_frequency (int): Take every nth frame only.
            time_between_frames (float): Time (in fs) between two frames in sampled trajectory, e.g. 100 fs.
            spatial_extent_contact_layer (float): How far ranges the water contact layer.
            pbc_dimension (str): Apply pbcs in these directions.
            oxygen_heavy_dtistance (float): Allowed distance (in angstroms) between oxygens to satisfy hydrogen
                                            bonding criterion.

        Returns:

        """

        # define dimensions not periodic, indices
        pbc_indices = global_variables.DIMENSION_DICTIONARY.get(pbc_dimensions)
        not_pbc_indices = list(set(pbc_indices) ^ set([0, 1, 2]))

        # rewind trajectory
        self.position_universe.trajectory[0]

        # this analysis is only for the presence of solid. For bulk use the classic pandas way.
        if len(pbc_indices) == 3:
            raise UnphysicalValue(
                "For bulk water please use find_acceptor_donor_pairs() and create the histogram yourself."
            )

        # start by defining oxygen atoms and solid
        oxygen_atoms = self.position_universe.select_atoms(f"name O")
        solid_atoms = self.position_universe.select_atoms("not name O H")
        heavy_atoms = self.position_universe.select_atoms("not name H")

        # get number of water molecules in system
        number_of_water_molecules = len(oxygen_atoms)

        heavy_atom_data = []

        # Loop over trajectory
        for count_frames, frames in enumerate(
            tqdm(
                (self.position_universe.trajectory[start_frame:end_frame])[
                    ::frame_frequency
                ]
            )
        ):

            # wrap atoms in box
            self.position_universe.atoms.pack_into_box(
                box=self.topology.get_cell_lengths_and_angles(), inplace=True
            )

            # initialise instance of DonorAcceptorPairs which will be used to save all information.
            heavy_atom_pairs_per_frame = HeavyAtomPairs()

            # save time of the frame
            heavy_atom_pairs_per_frame.time = frames.frame * time_between_frames

            # define center of mass of solid now
            solid_COM = solid_atoms.center_of_mass()

            # now compute vector from oxygen atoms from the center axis of the solid
            vector_oxygen_to_solid_COM = oxygen_atoms.positions - solid_COM

            # find oxygen atoms which are within the contact layer
            # distinguish between tube and flat sheet
            if len(pbc_dimensions) == 1:
                # tube
                oxygen_atoms_in_contact_layer = oxygen_atoms[
                    np.where(
                        np.linalg.norm(
                            vector_oxygen_to_solid_COM[:, not_pbc_indices], axis=1
                        )
                        >= spatial_extent_contact_layer
                    )[0]
                ]
            elif len(pbc_dimensions) == 2:

                # flat sheet
                oxygen_atoms_in_contact_layer = oxygen_atoms[
                    np.where(
                        vector_oxygen_to_solid_COM[:, not_pbc_indices]
                        <= spatial_extent_contact_layer
                    )[0]
                ]

            # now get distances between oxygens in contact layer and all heavy atoms
            vectors_oxygen_contact_heavy_atoms = (
                heavy_atoms.positions[np.newaxis, :]
                - oxygen_atoms_in_contact_layer.positions[:, np.newaxis]
            )

            # apply MIC for all oxygen-oxygen pairs
            vectors_contact_heavy_MIC = (
                utils.apply_minimum_image_convention_to_interatomic_vectors(
                    vectors_oxygen_contact_heavy_atoms,
                    self.topology.cell,
                    pbc_dimensions,
                )
            )
            # get distances based on vectors
            distances_contact_heavy = np.linalg.norm(vectors_contact_heavy_MIC, axis=2)

            # take all oxygen-oxygen pairs which satisfy oxygen_oxygen_distance criterion
            contact_heavy_pairs_cutoff = np.where(
                (distances_contact_heavy <= oxygen_heavy_distance)
                & (distances_contact_heavy > 0)
            )

            # create dictionary listing number of possible candidates for each oxygen, double counting not prevented
            dictionary_number_of_heavy_atoms_per_oxygen = collections.OrderedDict(
                collections.Counter(contact_heavy_pairs_cutoff[0])
            )

            # from dictionary infer indices in contact_heavy_pairs_cutoff for each oxygen
            tmp_indices_per_oxygen_in_contact = np.cumsum(
                np.asarray(
                    [
                        dictionary_number_of_heavy_atoms_per_oxygen.get(i, 0)
                        for i in np.arange(len(oxygen_atoms_in_contact_layer))
                    ]
                )
            )

            # split oxygen-heavy vectors, distances, and oxygen-oxygen pairs accordingly
            vectors_contact_heavy_split = np.split(
                vectors_contact_heavy_MIC[contact_heavy_pairs_cutoff],
                tmp_indices_per_oxygen_in_contact[:-1],
            )
            distances_contact_heavy_split = np.split(
                distances_contact_heavy[contact_heavy_pairs_cutoff],
                tmp_indices_per_oxygen_in_contact[:-1],
            )

            contact_heavy_pairs_split = np.split(
                contact_heavy_pairs_cutoff[1], tmp_indices_per_oxygen_in_contact[:-1]
            )

            # now focus on computing angles
            # first find hydrogens of oxygens in contact layer
            resids_contact = " ".join(
                [str(i) for i in oxygen_atoms_in_contact_layer.resids]
            )
            hydrogens_contact = self.position_universe.select_atoms(
                f"name H and resid {resids_contact}"
            )

            # split hydrogen positions according to respective waters
            hydrogen_positions_split = np.split(
                hydrogens_contact.positions[np.argsort(hydrogens_contact.resids)],
                len(oxygen_atoms_in_contact_layer),
            )

            # define all vectors from oxygens to covalently bonded hydrogens
            vectors_oxygen_covalent_hydrogens_split = (
                hydrogen_positions_split
                - oxygen_atoms_in_contact_layer.positions[:, np.newaxis]
            )

            # apply MIC to all vectors
            vectors_oxygen_covalent_hydrogens_split_MIC = (
                utils.apply_minimum_image_convention_to_interatomic_vectors(
                    vectors_oxygen_covalent_hydrogens_split,
                    self.topology.cell,
                    pbc_dimensions,
                )
            )
            # compute angles

            angles_oxygen_hydrogen_heavy = [
                np.rad2deg(
                    np.arccos(
                        np.clip(
                            (
                                np.dot(
                                    vectors_oxygen_covalent_hydrogens_split_MIC[i],
                                    vectors_contact_heavy_split[i].T,
                                )
                                / distances_contact_heavy_split[i]
                            )
                            / np.linalg.norm(
                                vectors_oxygen_covalent_hydrogens_split_MIC[i], axis=1
                            ).reshape((2, -1)),
                            -1,
                            1,
                        )
                    )
                )
                for i in np.arange(len(oxygen_atoms_in_contact_layer))
            ]

            ##########################
            # save all pairs to object
            ##########################

            # id of donor water molecule
            
            heavy_atom_pairs_per_frame.water_ids = np.concatenate([
                np.repeat(oxygen_atoms_in_contact_layer.resids[contact_water] - 1,dictionary_number_of_heavy_atoms_per_oxygen[contact_water])
                for contact_water in np.arange(len(oxygen_atoms_in_contact_layer))
            ])
            

            # heavy atom species
            heavy_atom_pairs_per_frame.heavy_atom_species = np.concatenate(
                [
                    heavy_atoms.names[contact_heavy_pairs_split[i]].flatten()
                    for i in np.arange(len(oxygen_atoms_in_contact_layer))
                ]
            )

            # distance between heavy_atoms
            heavy_atom_pairs_per_frame.heavy_atom_distances = np.concatenate(
                distances_contact_heavy_split
            )

            # distance between hydrogens and heavy atom
            heavy_atom_pairs_per_frame.hydrogens_heavy_atom_distances = np.concatenate(
                [
                    np.linalg.norm(
                        vectors_contact_heavy_split[contact_water][np.newaxis, :]
                        - vectors_oxygen_covalent_hydrogens_split_MIC[contact_water][
                            :, np.newaxis
                        ],
                        axis=2,
                    ).T
                    for contact_water in np.arange(len(oxygen_atoms_in_contact_layer))
                ]
            )
            # heavy_atom_pairs_per_frame.hydrogen_heavy_atom_distances = vectors_contact_heavy_split

            # angles between acceptor oxygen, donor oxygen, and donor hydrogen
            heavy_atom_pairs_per_frame.angles = np.concatenate(
                [
                    (angles_oxygen_hydrogen_heavy[i].T)
                    for i in np.arange(len(oxygen_atoms_in_contact_layer))
                ]
            )

            # Eventually, append computed array with all this information for one frame
            heavy_atom_data.extend(
                heavy_atom_pairs_per_frame.gather_information_in_numpy_array()
            )

        # convert collected arrays to pandas dataframe
        heavy_atom_dataframe = pandas.DataFrame(
            heavy_atom_data,
            columns=[
                "Time",
                "Water molecule index",
                "Species",
                "O-X Distance",
                "H1-X Distance",
                "H2-X Distance",
                "Angle H1",
                "Angle H2",
            ],
        )

        self.heavy_atoms_dataframe = heavy_atom_dataframe


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
                self.donor_coordinates[:, 0],
                self.donor_coordinates[:, 1],
                self.donor_coordinates[:, 2],
                self.acceptor_coordinates[:, 0],
                self.acceptor_coordinates[:, 1],
                self.acceptor_coordinates[:, 2],
            ]
        ).T

        return array


class HeavyAtomPairs:
    """
    Gather heavy atom pairs in np.array format for one frame
    of a position_universe.trajectory. This will then be used latter
    to generate pandas.DataFrame object.

    Attributes:
        heavy_atom_species (np.array): Species of heavy atom.
        heavy_atom_distances (np.array): distances between oxygen and heavy atom.
        angles_sorted (np.array): angles between donor oxygen, heavy atom and hydrogen.

    Methods:

    """

    def __init__(self):
        pass

    def gather_information_in_numpy_array(self):
        """
        Generate np.array based on all information collected and saved in the instance.
        Arguments:

        Returns:
            array (np.array): contains all information in stacked array.
        """
        array = np.vstack(
            [
                np.repeat(self.time, len(self.heavy_atom_species)),
                self.water_ids,
                self.heavy_atom_species,
                self.heavy_atom_distances,
                self.hydrogens_heavy_atom_distances[:, 0],
                self.hydrogens_heavy_atom_distances[:, 1],
                self.angles[:, 0],
                self.angles[:, 1],
            ]
        ).T

        return array
