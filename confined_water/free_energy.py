import numpy as np
import sys
from tqdm.notebook import tqdm

sys.path.append("../")
from confined_water import utils


def compute_atomic_probabilities(
    position_universes,
    topology,
    pbc_indices,
    start_frame: int,
    end_frame: int,
    frame_frequency: int,
):
    """
    Compute probability of atomic positions.
    Arguments:
        position_universes : MDAnalysis universes to be analysed.
        topology : ASE atoms object containing information about topology.
        pbc_indices : Direction indices in which system is periodic
        start_frame (int) : Start frame for analysis.
        end_frame (int) : End frame for analysis.
        frame_frequency (int): Take every nth frame only.

    """

    if len(pbc_indices) == 1:

        # compute probabilities for tube
        return _compute_probabilities_for_system_with_one_periodic_direction(
            position_universes, topology, pbc_indices, start_frame, end_frame, frame_frequency
        )

    else:
        # compute probabilities for flat interface
        return _compute_probabilities_for_system_with_two_periodic_directions(
            position_universes, topology, pbc_indices, start_frame, end_frame, frame_frequency
        )


def _compute_probabilities_for_system_with_one_periodic_direction(
    position_universes,
    topology,
    pbc_indices,
    start_frame: int,
    end_frame: int,
    frame_frequency: int,
):
    """
    Compute probability of atomic positions for 1D systems.
    Arguments:
        position_universes : MDAnalysis universes to be analysed.
        topology : ASE atoms object containing information about topology.
        pbc_indices : Direction indices in which system is periodic
        start_frame (int) : Start frame for analysis.
        end_frame (int) : End frame for analysis.
        frame_frequency (int): Take every nth frame only.

    """
    # define dimensions not periodic, indices
    not_pbc_indices = list(set(pbc_indices) ^ set([0, 1, 2]))
    periodic_vector = np.zeros(3)
    periodic_vector[pbc_indices] = 1

    # loop over all trajectories (only relevant for PIMD):
    for count_universe, universe in enumerate(position_universes):

        # wrap atoms in box
        universe.atoms.pack_into_box(box=topology.get_cell_lengths_and_angles(), inplace=True)

        # start by separating solid atoms from liquid atoms
        solid_atoms = universe.select_atoms("name B N C Na Cl")
        liquid_atoms = universe.select_atoms("name O H")

        # define one "reference atom (ideally in solid phase)"
        # this will serve as our anchor for computing the free energy profile
        anchor_coordinates = solid_atoms[10].position

        # Loop over trajectory
        for count_frames, frames in enumerate(
            tqdm((universe.trajectory[start_frame:end_frame])[::frame_frequency])
        ):
            # wrap atoms in box
            universe.atoms.pack_into_box(box=topology.get_cell_lengths_and_angles(), inplace=True)

            # we start by making the frames translationally and rotationally invariant
            # 1. Translations
            # This is done by computing the translation and substracting it
            translation_from_frame0 = solid_atoms.atoms.positions[10] - anchor_coordinates
            universe.atoms.positions -= translation_from_frame0

            # 2. Rotations (only relevant for nanotubes obviously)

            # to enable an easy rotation, translate atoms to COM
            COM = universe.atoms.center_of_mass()
            universe.atoms.positions -= COM

            # Compute angle between reference atom and axis perpendicular to periodic axis
            angle_anchor_first_axis = np.arccos(
                np.clip(
                    np.dot(universe.atoms.positions[10, not_pbc_indices], np.asarray([1, 0]))
                    / np.linalg.norm(universe.atoms.positions[10, not_pbc_indices]),
                    -1.0,
                    1.0,
                )
            )

            # get rotation matrix for periodic axis and computed angle
            rotation_matrix = utils.rotation_matrix(periodic_vector, -angle_anchor_first_axis)

            # rotate atoms, so that this can be compared
            universe.atoms.positions = np.matmul(rotation_matrix, universe.atoms.positions.T).T

            # translate now back to center of mass
            universe.atoms.positions += COM


def _compute_probabilities_for_system_with_two_periodic_direction(
    position_universes,
    topology,
    pbc_indices,
    start_frame: int,
    end_frame: int,
    frame_frequency: int,
):
    """
    Compute probability of atomic positions for 1D systems.
    Arguments:
        position_universes : MDAnalysis universes to be analysed.
        topology : ASE atoms object containing information about topology.
        pbc_indices : Direction indices in which system is periodic
        start_frame (int) : Start frame for analysis.
        end_frame (int) : End frame for analysis.
        frame_frequency (int): Take every nth frame only.

    """
    # define dimensions not periodic, indices
    not_pbc_indices = list(set(pbc_indices) ^ set([0, 1, 2]))
    periodic_vector = np.zeros(3)
    periodic_vector[pbc_indices] = 1

    # loop over all trajectories (only relevant for PIMD):
    for count_universe, universe in enumerate(position_universes):

        # wrap atoms in box
        universe.atoms.pack_into_box(box=topology.get_cell_lengths_and_angles(), inplace=True)

        # start by separating solid atoms from liquid atoms
        solid_atoms = universe.select_atoms("name B N C Na Cl")

        # approximate water with oxygens here
        liquid_atoms = universe.select_atoms("name O")

        # define one "reference atom (ideally in solid phase)"
        # this will serve as our anchor for computing the free energy profile
        anchor_coordinates = solid_atoms[10].position

        # Loop over trajectory
        for count_frames, frames in enumerate(
            tqdm((universe.trajectory[start_frame:end_frame])[::frame_frequency])
        ):
            # wrap atoms in box
            universe.atoms.pack_into_box(box=topology.get_cell_lengths_and_angles(), inplace=True)

            # we start by making the frames translationally invariant
            # This is done by computing the translation and substracting it
            translation_from_frame0 = solid_atoms.atoms.positions[10] - anchor_coordinates
            universe.atoms.positions -= translation_from_frame0

            # define center of mass of solid now
            solid_COM = solid_atoms.center_of_mass()

            # now compute distance from liquid atoms perpendicular to the center of mass of the solid
            perpendicular_distance_liquid_to_solid = (
                liquid_atoms.positions[:, not_pbc_indices] - solid_COM[not_pbc_indices]
            )
