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

            pass
