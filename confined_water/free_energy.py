import numpy as np
import sys
from tqdm.notebook import tqdm

from joblib import Parallel, delayed
from findpeaks import findpeaks
import pandas as pd

sys.path.append("../")
from confined_water import utils
from confined_water import global_variables
from confined_water import analysis

import dask
import dask.multiprocessing
from multiprocessing.pool import Pool


class FreeEnergyProfile:
    """
    Gather all information/data about Free energy profile

    Attributes:

    Methods:

    """

    def __init__(self):

        self.distribution_solid = []
        self.distribution_liquid = []
        self.temperature = 330

        self.free_energy_profile_liquid = []
        self.solid_atoms_on_free_energy_profile = {}

    def prepare_for_plotting(
        self,
        topology,
        pbc_indices,
        number_of_bins: int,
        multiples_of_unit_cell,
        plot_replica=3,
        tube_radius: float = None,
    ):
        """
        Prepare previously computed distributions of liquid and solid atoms
        for being plotted as free energy profile.
        Arguments:
            topology : ASE atoms object wih topology of system.
            pbc_indices: Direction indices in which system is periodic.
            number_of_bins (int): number of bins used for larger dimension, lower dimensions will be adjusted.
            multiples_of_unit_cell: integer array of periodic replica in 2D.
            plot_replica (int, array of scalar) : Number of replica of the unit cell plotted in 2D. If array first element in x, second in y direction.
            tube_radius (float) : tube radius of tube in A.

        Returns:
        """

        # define dimensions not periodic, indices

        not_pbc_indices = list(set(pbc_indices) ^ set([0, 1, 2]))

        # start by setting bins and ranges for histogram
        # thereby, it is important to distinguish between sheet and tube
        # replica of unit cell in x and y direction
        # note in case of tube y is axial and x is angular
        dim_x = multiples_of_unit_cell[0]
        dim_y = multiples_of_unit_cell[1]

        # define ranges
        range_x = (
            [0, 2 * np.pi * tube_radius]
            if len(pbc_indices) == 1
            else [0, topology.get_cell_lengths_and_angles()[pbc_indices][0]]
        )

        range_y = (
            [0, topology.get_cell_lengths_and_angles()[pbc_indices][0]]
            if len(pbc_indices) == 1
            else [0, topology.get_cell_lengths_and_angles()[pbc_indices][1]]
        )

        # compute ratio of dimensions per unit cell
        ratio_y_to_x = range_y[1] / dim_y / (range_x[1] / dim_x)

        # this gives the number of bins per unit cell in each direction
        if ratio_y_to_x < 1:

            bins_x_unit = number_of_bins
            bins_y_unit = int(np.round(number_of_bins * ratio_y_to_x))

        else:

            bins_x_unit = int(np.round(number_of_bins / ratio_y_to_x))
            bins_y_unit = number_of_bins

        # now create bins for system
        bins_x_total = bins_x_unit * dim_x
        bins_y_total = bins_y_unit * dim_y

        # create now histogram with these options
        # first for liquid
        hist_liquid, xedges_total, yedges_total = np.histogram2d(
            self.distribution_liquid[:, 0],
            self.distribution_liquid[:, 1],
            bins=[bins_x_total, bins_y_total],
            range=[range_x, range_y],
            density=False,
        )

        # project histogram on unit cell
        hist_liquid_projected = self._project_histogram_to_unit_cell(
            hist_liquid, bins_x_unit, bins_y_unit, multiples_of_unit_cell
        )

        # now normalise to get a probablity
        hist_liquid_projected_normalised = hist_liquid_projected / np.sum(
            hist_liquid_projected
        )

        if not (np.array(plot_replica)).shape:
            replica_array = [plot_replica, plot_replica]
        else:
            replica_array = plot_replica
        # now multiply this representation according to given number of replica
        hist_liquid_final = np.tile(
            hist_liquid_projected_normalised, replica_array[::-1]
        )

        # now, finally, compute, free energy
        free_energy_profile_liquid = (
            -self.temperature
            * global_variables.BOLTZMANN
            / global_variables.EV_TO_JOULE
            * np.log(hist_liquid_final / np.max(hist_liquid_final))
        )

        # adapt xedges and yedges accordingly
        xedges_free_energy_liquid = np.linspace(
            0, range_x[1] / dim_x * replica_array[0], bins_x_unit * replica_array[0] + 1
        )

        yedges_free_energy_liquid = np.linspace(
            0, range_y[1] / dim_y * replica_array[1], bins_y_unit * replica_array[1] + 1
        )

        self.free_energy_profile_liquid = [
            free_energy_profile_liquid,
            xedges_free_energy_liquid,
            yedges_free_energy_liquid,
        ]

        # repeat this proceedure in slightly different form for solid
        # define edges for unit cell

        xedges_unit = np.linspace(0, range_x[1] / dim_x, bins_x_unit + 1)

        yedges_unit = np.linspace(0, range_y[1] / dim_y, bins_y_unit + 1)

        # loop over solid dictionary:
        for element, distribution in self.distribution_solid.items():
            hist_solid, __, __ = np.histogram2d(
                distribution[:, 0],
                distribution[:, 1],
                bins=[bins_x_total, bins_y_total],
                range=[range_x, range_y],
                density=False,
            )

            # project histogram on unit cell
            hist_solid_projected = self._project_histogram_to_unit_cell(
                hist_solid, bins_x_unit, bins_y_unit, multiples_of_unit_cell
            )

            # compute number of atoms which should be in unit cell, currently only equal pairs possible
            number_of_atoms_of_element_in_unit_cell = (
                len(np.where(np.asarray(topology.get_chemical_symbols()) == element)[0])
                / dim_y
                / dim_x
            )

            # get positions of peaks
            coordinates_per_element_unit = self._get_solid_coordinates_from_histogram(
                hist_solid_projected,
                xedges_unit,
                yedges_unit,
                number_of_atoms_of_element_in_unit_cell,
            )

            # eventually, create coordinates for plotted replica
            coordinates_per_element = np.concatenate(
                [
                    coordinates_per_element_unit
                    + [
                        x_rep * xedges_unit[-1],
                        y_rep * yedges_unit[-1],
                    ]
                    for x_rep in np.arange(replica_array[0])
                    for y_rep in np.arange(replica_array[1])
                ]
            )

            # save these coordinates to dictionary
            self.solid_atoms_on_free_energy_profile[element] = coordinates_per_element

    def _project_histogram_to_unit_cell(
        self, histogram, bins_x_unit, bins_y_unit, multiples_of_unit_cell
    ):
        """
        Returns histogram projecte to unit cell.
        Arguments:
                histogram: numpy 2D histogram with the distribution of atoms.
                bins_x_unit: number of bins in x direction unit cell.
                bins_y_unit: number of bins in x direction unit cell.
                multiples_of_unit_cell: integer array of periodic replica in 2D.

        Returns:
        """

        # reshape histogram in y-direction first
        histogram_y_reshaped = histogram.reshape(
            histogram.shape[0], multiples_of_unit_cell[1], bins_y_unit
        )

        # now sum over projections
        histogram_y_summed = np.sum(histogram_y_reshaped, axis=1)

        # do the same thing with x direction
        histogram_y_summed = histogram_y_summed.T
        histogram_x_reshaped = histogram_y_summed.reshape(
            histogram_y_summed.shape[0], multiples_of_unit_cell[0], bins_x_unit
        )
        histogram_x_summed = np.sum(histogram_x_reshaped, axis=1)

        return histogram_x_summed

    def _get_solid_coordinates_from_histogram(
        self, hist_solid_projected, xedges_unit, yedges_unit, expected_number_of_atoms
    ):
        """
        Returns coordinates of solid atoms in unit cell for given histogram of element.
        Arguments:
                peak_indices: Indices of peaks found.
                xedges_unit: Binned x range of unit cell.
                yedges_unit: Binned x range of unit cell.
                expected_number_of_atoms: How many atoms are expected for this element in unit cell.

        Returns:
                Coordinates of element in unit cell.
        """
        # Initialise peak finder

        peak_finder = findpeaks(method="mask")

        # repeat histogram to avoid pbc issues
        histogram_repeated = np.tile(hist_solid_projected, [3, 3])

        # compute xedges and yedges
        xedges_supercell = np.linspace(
            0, xedges_unit[-1] * 3, 3 * (len(xedges_unit) - 1) + 1
        )
        yedges_supercell = np.linspace(
            0, yedges_unit[-1] * 3, 3 * (len(yedges_unit) - 1) + 1
        )

        # find peaks in 2D histograms
        peaks_found = peak_finder.fit(histogram_repeated)
        peaks_indices = np.where(peaks_found["Xdetect"])

        # compute preliminary coordinates
        x_coordinates = 0.5 * (
            xedges_supercell[peaks_indices[1]] + xedges_supercell[peaks_indices[1] + 1]
        )
        y_coordinates = 0.5 * (
            yedges_supercell[peaks_indices[0]] + yedges_supercell[peaks_indices[0] + 1]
        )
        preliminary_coordinates = np.column_stack((x_coordinates, y_coordinates))

        # get now coordinates of 2,2 unit cell which has no pbc issues
        unit_cell_peaks_without_pbc_issues = np.where(
            (preliminary_coordinates[:, 0] >= xedges_unit[-1] + xedges_unit[1] * 0.5)
            & (
                preliminary_coordinates[:, 0]
                < 2 * xedges_unit[-1] + xedges_unit[1] * 0.5
            )
            & (preliminary_coordinates[:, 1] >= yedges_unit[-1] + yedges_unit[1] * 0.5)
            & (
                preliminary_coordinates[:, 1]
                < 2 * (yedges_unit[-1]) + yedges_unit[1] * 0.5
            )
        )

        if len(unit_cell_peaks_without_pbc_issues[0]) != expected_number_of_atoms:

            raise analysis.UnphysicalValue(
                f" Severe error: you found more solid atoms in your unit cell than you should."
                f" There are {len(unit_cell_peaks_without_pbc_issues[0])} atoms from the peak analysis."
                f" You should have found {expected_number_of_atoms}."
                f" Using more bins or sampling longer could help."
            )

        solid_coordinates_unit_cell = preliminary_coordinates[
            unit_cell_peaks_without_pbc_issues
        ] - [xedges_unit[-1], yedges_unit[-1]]

        return solid_coordinates_unit_cell


def compute_spatial_distribution_of_atoms_on_interface(
    position_universe,
    topology,
    spatial_extent_contact_layer: float,
    pbc_indices,
    species,
    start_frame: int,
    end_frame: int,
    frame_frequency: int,
    tube_radius: float = None,
    tube_length_in_unit_cells: int = None,
    parallel: bool = False,
    number_of_cores: int = 4,
):
    """
    Compute distribution of atomic positions on interface.
    Arguments:
        position_universes : MDAnalysis universes to be analysed.
        topology : ASE atoms object containing information about topology.
        spatial_extent_contact_layer (float): How far ranges the water contact layer.
        pbc_indices : Direction indices in which system is periodic
        species : Element used to perform analysis (O or H)
        start_frame (int) : Start frame for analysis.
        end_frame (int) : End frame for analysis.
        frame_frequency (int): Take every nth frame only.
        parallel (bool): Whether to compute the free energy profile in parallel.
        number_of_cores (int): If parallel==True, this sets the number of cores used.
    """

    # get all atom types which are not water:
    solid_types = list(set(np.unique(position_universe.atoms.names)) ^ set(["O", "H"]))

    # define how many samples will be taken
    number_of_samples = np.arange(start_frame, end_frame, frame_frequency).shape[0]
    if len(pbc_indices) == 1:

        if parallel == False:

            # compute probabilities for tube
            (
                liquid,
                solid,
            ) = _compute_distribution_for_system_with_one_periodic_direction(
                position_universe,
                topology,
                spatial_extent_contact_layer,
                tube_radius,
                tube_length_in_unit_cells,
                pbc_indices,
                species,
                start_frame,
                end_frame,
                frame_frequency,
            )

        if parallel == True:

            # compute probabilities for tube
            (
                liquid,
                solid,
            ) = _compute_distribution_for_system_with_one_periodic_direction_in_parallel(
                position_universe,
                topology,
                spatial_extent_contact_layer,
                tube_radius,
                tube_length_in_unit_cells,
                pbc_indices,
                species,
                start_frame,
                end_frame,
                frame_frequency,
                number_of_cores,
            )

        n_coords = 2

    else:
        # compute probabilities for flat interface
        liquid, solid = _compute_distribution_for_system_with_two_periodic_directions(
            position_universe,
            topology,
            spatial_extent_contact_layer,
            pbc_indices,
            species,
            start_frame,
            end_frame,
            frame_frequency,
        )

        n_coords = 3

    # before returning distribution assign solid positions to types
    # start by reshaping solid
    solid_reshaped = solid.reshape(number_of_samples, -1, n_coords)

    # re-define solid atoms once more (could be done better)
    solid_atoms = position_universe.select_atoms("not name O H")

    # define dictionary
    dict_solid_per_element = {}

    # now loop over all atom types
    for element in solid_types:

        # get indices
        indices_for_element = np.where(solid_atoms.names == element)

        # positions of element
        positions_for_element = solid_reshaped[:, indices_for_element].reshape(
            -1, n_coords
        )

        dict_solid_per_element[element] = positions_for_element

    return liquid, dict_solid_per_element


def _compute_distribution_for_system_with_one_periodic_direction(
    universe,
    topology,
    spatial_extent_contact_layer: float,
    tube_radius: float,
    tube_length_in_unit_cells: int,
    pbc_indices,
    species,
    start_frame: int,
    end_frame: int,
    frame_frequency: int,
):
    """
    Compute distribution of atomic positions for 1D systems.
    Arguments:
        universe : MDAnalysis universes to be analysed.
        topology : ASE atoms object containing information about topology.
        spatial_extent_contact_layer (float): How far ranges the water contact layer.
        tube_radius (float) : radius of the tube in A.
        tube_length_in_unit_cells (int): multiples of tube unit cell in periodic direction.
        pbc_indices : Direction indices in which system is periodic
        species : Element to perform analysis with.
        start_frame (int) : Start frame for analysis.
        end_frame (int) : End frame for analysis.
        frame_frequency (int): Take every nth frame only.

    """
    # define dimensions not periodic, indices
    not_pbc_indices = list(set(pbc_indices) ^ set([0, 1, 2]))
    periodic_vector = np.zeros(3)
    periodic_vector[pbc_indices] = 1
    # wrap atoms in box
    # universe.atoms.pack_into_box(box=topology.get_cell_lengths_and_angles(), inplace=True)

    # rewind trajectory
    universe.trajectory[0]

    # start by separating solid atoms from liquid atoms
    solid_atoms = universe.select_atoms("not name O H")
    # liquid_atoms = universe.select_atoms(f"name {species}")
    selected_atoms = universe.select_atoms(f"name {species}")
    oxygen_atoms = universe.select_atoms(f"name O")

    # this will serve as our anchor for translation for computing the free energy profile
    anchor_coordinates = solid_atoms.center_of_mass()

    # define reference atoms which will be used to determine rotation
    indices_atoms_anchor_rotation = _get_atom_ids_on_same_tube_axis(
        solid_atoms, tube_length_in_unit_cells, not_pbc_indices
    )

    # compute circumference based on diameter
    tube_circumference = 2 * np.pi * tube_radius

    # define arrays where the coordinates of oxygens and solid atoms will be saved in
    liquid_contact_coord1 = []
    liquid_contact_coord2 = []
    solid_coord1 = []
    solid_coord2 = []

    liquid_contact_coord1_all = []
    liquid_contact_coord2_all = []
    solid_coord1_all = []
    solid_coord2_all = []

    number_of_samples = len(np.arange(start_frame, end_frame, frame_frequency))

    # define solid bins for getting z-dependent COM, add a little bit more for numerical reasons
    bins_COM_solid = np.linspace(0, topology.get_cell_lengths_and_angles()[2] + 1e-3, 7)

    # based on bins get atom groups forming each bin
    # solid_z_resolved = [solid_atoms[(np.where((solid_atoms.positions[:,2] > bins_COM_solid[count]) & (solid_atoms.positions[:,2] <= bins_COM_solid[count+1])))] for count in np.arange(len(bins_COM_solid)-1)]

    # Loop over trajectory
    for count_frames, frames in enumerate(
        tqdm((universe.trajectory[start_frame:end_frame])[::frame_frequency])
    ):

        # we start by making the frames translationally and rotationally invariant
        # 1. Translations
        # This is done by computing the translation and substracting it

        # solid
        translation_from_frame0 = solid_atoms.center_of_mass() - anchor_coordinates
        universe.atoms.positions -= translation_from_frame0

        # 2. Rotations (only relevant for nanotubes obviously)

        # to enable an easy rotation, translate atoms to COM
        COM = universe.atoms.center_of_mass()
        universe.atoms.positions -= COM

        # prepare everything to compute angle between axis and anchor axis
        solid_axis = np.mean(
            solid_atoms.positions[indices_atoms_anchor_rotation][:, not_pbc_indices],
            axis=0,
        )

        # define normed dot product
        normed_dot_product = np.clip(
            np.dot(
                solid_axis,
                np.asarray([1, 0]),
            )
            / np.linalg.norm(solid_axis),
            -1.0,
            1.0,
        )

        # Compute angle between reference atom and axis perpendicular to periodic axis
        # note we need the negative angle if above 180 degrees
        angle_anchor_first_axis = (
            np.arccos(normed_dot_product)
            if solid_axis[1] <= 0
            else -np.arccos(normed_dot_product)
        )

        # get rotation matrix for periodic axis and computed angle
        rotation_matrix = utils.rotation_matrix(
            periodic_vector, +angle_anchor_first_axis
        )

        # rotate atoms, so that this can be compared
        universe.atoms.positions = np.matmul(
            rotation_matrix, universe.atoms.positions.T
        ).T

        # translate now back to center of mass
        universe.atoms.positions += COM

        # wrap atoms in box
        universe.atoms.pack_into_box(
            box=topology.get_cell_lengths_and_angles(), inplace=True
        )

        solid_COM = solid_atoms.center_of_mass()

        #####################################################
        # Having prepared the positions now we have to find:
        # 1. The water molecules within the contact layer
        # 2. Their projection on the surface by taking into account the tube's breathing modes

        # get vectors water molecules (here oxygens) to closest solid neighbor
        vectors_oxygens_to_solid = (
            solid_atoms.positions[np.newaxis, :] - oxygen_atoms.positions[:, np.newaxis]
        )

        # apply MIC for all oxygen-oxygen pairs
        vectors_oxygens_to_solid_MIC = (
            utils.apply_minimum_image_convention_to_interatomic_vectors(
                vectors_oxygens_to_solid, topology.cell, "z"
            )
        )

        # get minimum distances based on vectors
        min_distances_oxygens_to_solid = np.min(
            np.linalg.norm(vectors_oxygens_to_solid_MIC, axis=2), axis=1
        )

        # get resids of water molecules inside contact layer
        # resids_water_contact = oxygen_atoms[np.where(min_distances_oxygens_to_solid < spatial_extent_contact_layer)].resids
        resids_water_contact = "resid " + (" ").join(
            [
                str(resid)
                for resid in oxygen_atoms[
                    np.where(
                        min_distances_oxygens_to_solid < spatial_extent_contact_layer
                    )
                ].resids
            ]
        )

        # get the respective atoms of the species chosen O/H
        selected_atoms_contact = selected_atoms.select_atoms(resids_water_contact)

        # the projection is done in two parts:
        # a. for each water molecule compute z center of mass of the tube based on finite displacements
        # b. We then need to find the solid atom closest which will then serve as radius

        # based on bins get atom groups forming each bin
        solid_z_resolved = [
            solid_atoms[
                (
                    np.where(
                        (solid_atoms.positions[:, 2] > bins_COM_solid[count])
                        & (solid_atoms.positions[:, 2] <= bins_COM_solid[count + 1])
                    )
                )
            ]
            for count in np.arange(len(bins_COM_solid) - 1)
        ]
        indices_digitized_liquid = (
            np.digitize(
                selected_atoms_contact.positions[:, 2], bins_COM_solid, right=True
            )
            - 1
        )

        # get the COM for each z-segment and create array to subtract from positions
        COMs_z_resolved = np.asarray(
            [atom_group.center_of_mass() for atom_group in solid_z_resolved]
        )

        # now get the vectors from the respective center to the atom, we only take the inplane part
        # this is enough to get the angle but we need to correct radius to project it to
        vectors_COM_z_selected_contact = (
            selected_atoms_contact.positions - COMs_z_resolved[indices_digitized_liquid]
        )[:, not_pbc_indices]

        # vectors_COM_z_selected_contact = (
        #         selected_atoms_contact.positions
        #         - solid_COM
        #     )[:,not_pbc_indices]

        # to get the right radius we have to find the atom which lies on the vector from the COMz to
        # the selected atom. To do so, we calculate the angle between the elongated vector and the
        # vector to all solid atoms in the z-segment. For now just, use the atom which is closest as radius
        # get vectors chosen atoms to closest solid neighbor
        # vectors_selected_to_solid = (
        #         solid_atoms.positions[np.newaxis, :]
        #         - selected_atoms_contact.positions[:, np.newaxis]
        #     )

        # # apply MIC for all oxygen-oxygen pairs
        # vectors_selected_to_solid_MIC = (
        #         utils.apply_minimum_image_convention_to_interatomic_vectors(
        #             vectors_selected_to_solid, topology.cell,"z"
        #         )
        #     )

        # # get minimum distances based on vectors
        # min_distances_selected_to_solid = np.min(
        #         np.linalg.norm(vectors_selected_to_solid_MIC, axis=2), axis=1
        #     )

        # atom_dependent_radius = np.linalg.norm(vectors_COM_z_selected_contact,axis=1)+min_distances_selected_to_solid

        # now compute vector from liquid atoms from the center axis of the solid
        # vector_liquid_to_central_axis = liquid_atoms.positions - solid_COM

        # get vectors between each liquid atom and all solid atoms
        # vectors_liquid_to_solid = (
        #         solid_atoms.positions[np.newaxis, :]
        #         - liquid_atoms.positions[:, np.newaxis]
        #     )

        # # apply MIC for all oxygen-oxygen pairs
        # vectors_liquid_to_solid_MIC = (
        #     utils.apply_minimum_image_convention_to_interatomic_vectors(
        #             vectors_liquid_to_solid, topology.cell, "z"
        #         )
        #     )

        # # get minimum distances based on vectors
        # min_distances_liquid_to_solid = np.min(
        #         np.linalg.norm(vectors_liquid_to_solid_MIC, axis=2), axis=1
        #     )

        # liquid_atoms_in_contact_positions = liquid_atoms[
        #     np.where(min_distances_liquid_to_solid <= spatial_extent_contact_layer)
        # ].positions[:, pbc_indices]

        # only choose those atoms which are within contact layer from solid
        # liquid_atoms_in_contact_positions = vector_liquid_to_central_axis[
        #     np.where(
        #         np.linalg.norm(vector_liquid_to_central_axis[:, not_pbc_indices], axis=1)
        #         >= spatial_extent_contact_layer
        #     )
        # ]

        # for chosen atoms compute position in periodic and angular direction
        # periodic is easy
        liquid_contact_coord2 = np.append(
            liquid_contact_coord2,
            selected_atoms_contact.positions[:, 2],
        )

        # the angular coordinate is a bit more tricky
        # start by computing angle from axis
        angles_liquid_contact_central_axis = np.arctan2(
            vectors_COM_z_selected_contact[:, not_pbc_indices[1]],
            vectors_COM_z_selected_contact[:, not_pbc_indices[0]],
        )

        # compute expansion on opened tube (adding pi to get only positive values)
        angular_component_liquid_contact = tube_radius * (
            angles_liquid_contact_central_axis + np.pi
        )
        liquid_contact_coord1 = np.append(
            liquid_contact_coord1, angular_component_liquid_contact
        )

        # do the same thing for solid
        indices_digitized_solid = (
            np.digitize(solid_atoms.positions[:, 2], bins_COM_solid, right=True) - 1
        )

        vectors_COM_z_solid = (
            solid_atoms.positions - COMs_z_resolved[indices_digitized_solid]
        )[:, not_pbc_indices]

        # vectors_COM_z_solid = (
        #         solid_atoms.positions
        #         - solid_COM
        #     )[:,not_pbc_indices]

        solid_coord2 = np.append(solid_coord2, solid_atoms.positions[:, pbc_indices])

        angles_solid_central_axis = np.arctan2(
            vectors_COM_z_solid[:, not_pbc_indices[1]],
            vectors_COM_z_solid[:, not_pbc_indices[0]],
        )

        # angular_component_solid = (
        #     np.linalg.norm(vectors_COM_z_solid,axis=1) * (angles_solid_central_axis + np.pi)
        # )

        angular_component_solid = tube_radius * (angles_solid_central_axis + np.pi)

        solid_coord1 = np.append(solid_coord1, angular_component_solid)

        # making code more efficient, ugly bu useful
        if count_frames % 1000 == 0 or count_frames == number_of_samples - 1:

            # save full arrays to global array and empty local to free memory and speed up loop
            liquid_contact_coord1_all = np.append(
                liquid_contact_coord1_all, liquid_contact_coord1
            )
            liquid_contact_coord1 = []
            liquid_contact_coord2_all = np.append(
                liquid_contact_coord2_all, liquid_contact_coord2
            )
            liquid_contact_coord2 = []
            solid_coord1_all = np.append(solid_coord1_all, solid_coord1)
            solid_coord1 = []
            solid_coord2_all = np.append(solid_coord2_all, solid_coord2)
            solid_coord2 = []

    # stack everything
    liquid_contact_2d = np.column_stack(
        (liquid_contact_coord1_all, liquid_contact_coord2_all)
    )
    solid_2d = np.column_stack((solid_coord1_all, solid_coord2_all))

    return liquid_contact_2d, solid_2d


def _compute_distribution_for_system_with_one_periodic_direction_in_parallel(
    universe,
    topology,
    spatial_extent_contact_layer: float,
    tube_radius: float,
    tube_length_in_unit_cells: int,
    pbc_indices,
    species,
    start_frame: int,
    end_frame: int,
    frame_frequency: int,
    number_of_cores: int,
):
    """
    Compute distribution of atomic positions for 1D systems.
    Arguments:
        universe : MDAnalysis universes to be analysed.
        topology : ASE atoms object containing information about topology.
        spatial_extent_contact_layer (float): How far ranges the water contact layer.
        tube_radius (float) : radius of the tube in A.
        tube_length_in_unit_cells (int): multiples of tube unit cell in periodic direction.
        pbc_indices : Direction indices in which system is periodic
        species : Element to perform analysis with.
        start_frame (int) : Start frame for analysis.
        end_frame (int) : End frame for analysis.
        frame_frequency (int): Take every nth frame only.
        number_of_cores (int): Number of processors the analysis runs on.
    """

    not_pbc_indices = list(set(pbc_indices) ^ set([0, 1, 2]))
    periodic_vector = np.zeros(3)
    periodic_vector[pbc_indices] = 1

    # rewind trajectory
    universe.trajectory[0]

    # start by separating solid atoms from liquid atoms
    solid_atoms = universe.select_atoms("not name O H")
    # liquid_atoms = universe.select_atoms(f"name {species}")
    selected_atoms = universe.select_atoms(f"name {species}")
    oxygen_atoms = universe.select_atoms(f"name O")

    # this will serve as our anchor for translation for computing the free energy profile
    anchor_coordinates = solid_atoms.center_of_mass()

    # define reference atoms which will be used to determine rotation
    indices_atoms_anchor_rotation = _get_atom_ids_on_same_tube_axis(
        solid_atoms, tube_length_in_unit_cells, not_pbc_indices
    )

    # compute circumference based on diameter
    tube_circumference = 2 * np.pi * tube_radius

    number_of_samples = len(np.arange(start_frame, end_frame, frame_frequency))

    # define solid bins for getting z-dependent COM, add a little bit more for numerical reasons
    bins_COM_solid = np.linspace(0, topology.get_cell_lengths_and_angles()[2] + 1e-3, 7)

    # up to here, everything is like in the serial version
    # now we need to split the trajectory into chunks
    # the number of chunks corresponds to the number of cores used

    # get the frames treated on every processor
    frames_per_proc = np.array_split(
        np.arange(start_frame, end_frame + frame_frequency, frame_frequency),
        number_of_cores,
    )
    # get initial and end frame per proc
    start_frame_per_proc = np.array([block[0] for block in frames_per_proc])
    start_frame_per_proc[1::] -= 1
    end_frame_per_proc = np.append(start_frame_per_proc[1::], np.array(end_frame))

    # ------------------------------------
    # dask
    # ------------------------------------
    # dask.config.set(scheduler='processes', pool=Pool(number_of_cores))

    # frames_all = np.arange(start_frame, end_frame + frame_frequency, frame_frequency)
    # job_list = [dask.delayed(_sample_distribution_for_systems_with_one_periodic_direction_per_frame(frame,
    #                                     universe,topology,solid_atoms, selected_atoms, oxygen_atoms,anchor_coordinates,
    #                                     indices_atoms_anchor_rotation,pbc_indices, spatial_extent_contact_layer,tube_radius)) for frame in
    #                                      tqdm(np.arange(start_frame, end_frame + frame_frequency, frame_frequency))]

    # result = dask.compute(job_list,num_workers=number_of_cores)

    # ------------------------------------
    # joblib
    # ------------------------------------
    # now we loop over the chunks, these will be computed in parallel
    with Parallel(n_jobs=number_of_cores, verbose=20, backend="loky") as parallel:
        data = parallel(
            delayed(
                _sample_distribution_for_systems_with_one_periodic_direction_per_processor
            )(
                universe,
                topology,
                spatial_extent_contact_layer,
                anchor_coordinates,
                indices_atoms_anchor_rotation,
                tube_radius,
                tube_length_in_unit_cells,
                pbc_indices,
                species,
                start_frame_per_proc[proc_id],
                end_frame_per_proc[proc_id],
                frame_frequency,
            )
            for proc_id in np.arange(len(start_frame_per_proc))
        )
    liquid = np.concatenate([data[i][0] for i in np.arange(len(data))])
    solid = np.concatenate([data[i][1] for i in np.arange(len(data))])

    return liquid, solid


def _sample_distribution_for_systems_with_one_periodic_direction_per_processor(
    universe,
    topology,
    spatial_extent_contact_layer: float,
    anchor_coordinates,
    indices_atoms_anchor_rotation,
    tube_radius: float,
    tube_length_in_unit_cells: int,
    pbc_indices,
    species,
    start_frame: int,
    end_frame: int,
    frame_frequency: int,
):
    """
    Compute distribution of atomic positions for 1D systems (for parallelism).
    Arguments:
        universe : MDAnalysis universes to be analysed.
        topology : ASE atoms object containing information about topology.
        anchor_coordinates: Reference coordinartes of the solid.
        indices_atoms_anchor_rotation: Atomic indices forming the reference rotation axis.
        spatial_extent_contact_layer (float): How far ranges the water contact layer.
        tube_radius (float) : radius of the tube in A.
        tube_length_in_unit_cells (int): multiples of tube unit cell in periodic direction.
        pbc_indices : Direction indices in which system is periodic
        species : Element to perform analysis with.
        start_frame (int) : Start frame for analysis.
        end_frame (int) : End frame for analysis.
        frame_frequency (int): Take every nth frame only.
    """


    # define dimensions not periodic, indices
    not_pbc_indices = list(set(pbc_indices) ^ set([0, 1, 2]))
    periodic_vector = np.zeros(3)
    periodic_vector[pbc_indices] = 1

    universe.trajectory[start_frame]

    # repeat separating solid atoms from liquid atoms
    solid_atoms = universe.select_atoms("not name O H")
    selected_atoms = universe.select_atoms(f"name {species}")
    oxygen_atoms = universe.select_atoms(f"name O")

    # define arrays where the coordinates of oxygens and solid atoms will be saved in
    liquid_contact_coord1 = []
    liquid_contact_coord2 = []
    solid_coord1 = []
    solid_coord2 = []

    liquid_contact_coord1_all = []
    liquid_contact_coord2_all = []
    solid_coord1_all = []
    solid_coord2_all = []

    number_of_samples = len(np.arange(start_frame, end_frame, frame_frequency))

    solid_COM_all = []

    # define solid bins for getting z-dependent COM, add a little bit more for numerical reasons
    bins_COM_solid = np.linspace(0, topology.get_cell_lengths_and_angles()[2]+1e-3, 7)

    # Loop over trajectory
    for count_frames, frames in enumerate(
         tqdm((universe.trajectory[start_frame:end_frame])[::frame_frequency])
     ):

        # we start by making the frames translationally and rotationally invariant
        # 1. Translations
        # This is done by computing the translation and substracting it
        translation_from_frame0 = solid_atoms.center_of_mass() - anchor_coordinates
        universe.atoms.positions -= translation_from_frame0

        #2. Rotations (only relevant for nanotubes obviously)
        # to enable an easy rotation, translate atoms to COM
        COM = universe.atoms.center_of_mass()
        universe.atoms.positions -= COM

        # prepare everything to compute angle between axis and anchor axis
        solid_axis = np.mean(
             solid_atoms.positions[indices_atoms_anchor_rotation][:, not_pbc_indices],
             axis=0,
        )
        # define normed dot product
        normed_dot_product = np.clip(
            np.dot(
                solid_axis,
                np.asarray([1, 0]),
            )
            / np.linalg.norm(solid_axis),
            -1.0,
            1.0,
        )

        # Compute angle between reference atom and axis perpendicular to periodic axis
        # note we need the negative angle if above 180 degrees
        angle_anchor_first_axis = (
            np.arccos(normed_dot_product) if solid_axis[1] <= 0 else -np.arccos(normed_dot_product)
        )

        # get rotation matrix for periodic axis and computed angle
        rotation_matrix = utils.rotation_matrix(periodic_vector, +angle_anchor_first_axis)

        # rotate atoms, so that this can be compared
        universe.atoms.positions = np.matmul(rotation_matrix, universe.atoms.positions.T).T

        # translate now back to center of mass
        universe.atoms.positions += COM

        # wrap atoms in box
        universe.atoms.pack_into_box(box=topology.get_cell_lengths_and_angles(), inplace=True)
        
        #####################################################
        # Having prepared the positions now we have to find:
        # 1. The water molecules within the contact layer
        # 2. Their projection on the surface by taking into account the tube's breathing modes

        # get vectors water molecules (here oxygens) to closest solid neighbor
        vectors_oxygens_to_solid = (
                solid_atoms.positions[np.newaxis, :]
                - oxygen_atoms.positions[:, np.newaxis]
            )

        # apply MIC for all oxygen-oxygen pairs
        vectors_oxygens_to_solid_MIC = (
                utils.apply_minimum_image_convention_to_interatomic_vectors(
                    vectors_oxygens_to_solid, topology.cell,"z"
                )
            )

        # get minimum distances based on vectors
        min_distances_oxygens_to_solid = np.min(
                np.linalg.norm(vectors_oxygens_to_solid_MIC, axis=2), axis=1
            )

        # get resids of water molecules inside contact layer
        # resids_water_contact = oxygen_atoms[np.where(min_distances_oxygens_to_solid < spatial_extent_contact_layer)].resids
        resids_water_contact = "resid " + (" ").join([str(resid) for resid in oxygen_atoms[np.where(min_distances_oxygens_to_solid < spatial_extent_contact_layer)].resids])

        # get the respective atoms of the species chosen O/H
        selected_atoms_contact = selected_atoms.select_atoms(resids_water_contact)

        # for each water molecule compute z center of mass of the tube based on finite displacements
        # based on bins get atom groups forming each bin
        solid_z_resolved = ([solid_atoms[(np.where((solid_atoms.positions[:,2] > bins_COM_solid[count]) & (solid_atoms.positions[:,2] <= bins_COM_solid[count+1])))] for count in np.arange(len(bins_COM_solid)-1)])
        indices_digitized_liquid= np.digitize(selected_atoms_contact.positions[:,2],bins_COM_solid,right=True)-1

        # get the COM for each z-segment and create array to subtract from positions
        COMs_z_resolved = np.asarray([atom_group.center_of_mass() for atom_group in solid_z_resolved])

        if np.max(indices_digitized_liquid)>5:
            breakpoint()

        # now get the vectors from the respective center to the atom, we only take the inplane part
        # this is enough to get the angle but we need to correct radius to project it to
        vectors_COM_z_selected_contact = (
                selected_atoms_contact.positions
                - COMs_z_resolved[indices_digitized_liquid]
            )[:,not_pbc_indices]


        # for chosen atoms compute position in periodic and angular direction
        # periodic is easy
        liquid_contact_coord2 = np.append(
            liquid_contact_coord2,
            selected_atoms_contact.positions[:,2],
        )

        # the angular coordinate is a bit more tricky
        # start by computing angle from axis
        angles_liquid_contact_central_axis = np.arctan2(
            vectors_COM_z_selected_contact[:, not_pbc_indices[1]],
            vectors_COM_z_selected_contact[:, not_pbc_indices[0]],
        )

        # compute expansion on opened tube (adding pi to get only positive values)
        angular_component_liquid_contact = (
            tube_radius * (angles_liquid_contact_central_axis + np.pi)
        )
        liquid_contact_coord1 = np.append(liquid_contact_coord1, angular_component_liquid_contact)

        # do the same thing for solid
        indices_digitized_solid= np.digitize(solid_atoms.positions[:,2],bins_COM_solid,right=True)-1

        vectors_COM_z_solid = (
                solid_atoms.positions
                - COMs_z_resolved[indices_digitized_solid]
            )[:,not_pbc_indices]

        solid_coord2 = np.append(solid_coord2, solid_atoms.positions[:, pbc_indices])

        angles_solid_central_axis = np.arctan2(
            vectors_COM_z_solid[:, not_pbc_indices[1]],
            vectors_COM_z_solid[:, not_pbc_indices[0]],
        )

        angular_component_solid = (
            tube_radius * (angles_solid_central_axis + np.pi)
        )


        solid_coord1 = np.append(solid_coord1, angular_component_solid)
        # making code more efficient, ugly bu useful
        if count_frames % 1000 == 0 or count_frames == number_of_samples - 1:
            # save full arrays to global array and empty local to free memory and speed up loop
            liquid_contact_coord1_all = np.append(liquid_contact_coord1_all, liquid_contact_coord1)
            liquid_contact_coord1 = []
            liquid_contact_coord2_all = np.append(liquid_contact_coord2_all, liquid_contact_coord2)
            liquid_contact_coord2 = []
            solid_coord1_all = np.append(solid_coord1_all, solid_coord1)
            solid_coord1 = []
            solid_coord2_all = np.append(solid_coord2_all, solid_coord2)
            solid_coord2 = []
    # stack everything
    liquid_contact_2d = np.column_stack((liquid_contact_coord1_all, liquid_contact_coord2_all))
    solid_2d = np.column_stack((solid_coord1_all, solid_coord2_all))
    return liquid_contact_2d, solid_2d

def _sample_distribution_for_systems_with_one_periodic_direction_per_frame(
    frame_index,
    universe,
    topology,
    solid_atoms,
    selected_atoms,
    oxygen_atoms,
    anchor_coordinates,
    indices_atoms_anchor_rotation,
    pbc_indices,
    spatial_extent_contact_layer,
    tube_radius,
):
    """
    Compute distribution of atomic positions for 1D systems (for parallelism).
    Arguments:
        universe : MDAnalysis universes to be analysed.
        topology : ASE atoms object containing information about topology.
        anchor_coordinates: Reference coordinartes of the solid.
        indices_atoms_anchor_rotation: Atomic indices forming the reference rotation axis.
        spatial_extent_contact_layer (float): How far ranges the water contact layer.
        tube_radius (float) : radius of the tube in A.
        tube_length_in_unit_cells (int): multiples of tube unit cell in periodic direction.
        pbc_indices : Direction indices in which system is periodic
        species : Element to perform analysis with.
        start_frame (int) : Start frame for analysis.
        end_frame (int) : End frame for analysis.
        frame_frequency (int): Take every nth frame only.
    """

    # define dimensions not periodic, indices
    not_pbc_indices = list(set(pbc_indices) ^ set([0, 1, 2]))
    periodic_vector = np.zeros(3)
    periodic_vector[pbc_indices] = 1

    universe.trajectory[frame_index]
    solid_atoms.universe.trajectory[frame_index]
    selected_atoms.universe.trajectory[frame_index]
    oxygen_atoms.universe.trajectory[frame_index]

    bins_COM_solid = np.linspace(0, topology.get_cell_lengths_and_angles()[2] + 1e-3, 7)

    # we start by making the frames translationally and rotationally invariant
    # 1. Translations
    # This is done by computing the translation and substracting it
    translation_from_frame0 = solid_atoms.center_of_mass() - anchor_coordinates
    universe.atoms.positions -= translation_from_frame0

    # 2. Rotations (only relevant for nanotubes obviously)

    # to enable an easy rotation, translate atoms to COM
    COM = universe.atoms.center_of_mass()
    universe.atoms.positions -= COM

    # prepare everything to compute angle between axis and anchor axis
    solid_axis = np.mean(
        solid_atoms.positions[indices_atoms_anchor_rotation][:, not_pbc_indices],
        axis=0,
    )

    # define normed dot product
    normed_dot_product = np.clip(
        np.dot(
            solid_axis,
            np.asarray([1, 0]),
        )
        / np.linalg.norm(solid_axis),
        -1.0,
        1.0,
    )

    # Compute angle between reference atom and axis perpendicular to periodic axis
    # note we need the negative angle if above 180 degrees
    angle_anchor_first_axis = (
        np.arccos(normed_dot_product)
        if solid_axis[1] <= 0
        else -np.arccos(normed_dot_product)
    )

    # get rotation matrix for periodic axis and computed angle
    rotation_matrix = utils.rotation_matrix(periodic_vector, +angle_anchor_first_axis)

    # rotate atoms, so that this can be compared
    universe.atoms.positions = np.matmul(rotation_matrix, universe.atoms.positions.T).T

    # translate now back to center of mass
    universe.atoms.positions += COM

    # wrap atoms in box
    universe.atoms.pack_into_box(
        box=topology.get_cell_lengths_and_angles(), inplace=True
    )

    solid_COM = solid_atoms.center_of_mass()

    #####################################################
    # Having prepared the positions now we have to find:
    # 1. The water molecules within the contact layer
    # 2. Their projection on the surface by taking into account the tube's breathing modes

    # get vectors water molecules (here oxygens) to closest solid neighbor
    vectors_oxygens_to_solid = (
        solid_atoms.positions[np.newaxis, :] - oxygen_atoms.positions[:, np.newaxis]
    )

    # apply MIC for all oxygen-oxygen pairs
    vectors_oxygens_to_solid_MIC = (
        utils.apply_minimum_image_convention_to_interatomic_vectors(
            vectors_oxygens_to_solid, topology.cell, "z"
        )
    )

    # get minimum distances based on vectors
    min_distances_oxygens_to_solid = np.min(
        np.linalg.norm(vectors_oxygens_to_solid_MIC, axis=2), axis=1
    )

    # get resids of water molecules inside contact layer
    # resids_water_contact = oxygen_atoms[np.where(min_distances_oxygens_to_solid < spatial_extent_contact_layer)].resids
    resids_water_contact = "resid " + (" ").join(
        [
            str(resid)
            for resid in oxygen_atoms[
                np.where(min_distances_oxygens_to_solid < spatial_extent_contact_layer)
            ].resids
        ]
    )

    # get the respective atoms of the species chosen O/H
    selected_atoms_contact = selected_atoms.select_atoms(resids_water_contact)

    # for each water molecule compute z center of mass of the tube based on finite displacements
    # based on bins get atom groups forming each bin
    solid_z_resolved = [
        solid_atoms[
            (
                np.where(
                    (solid_atoms.positions[:, 2] > bins_COM_solid[count])
                    & (solid_atoms.positions[:, 2] <= bins_COM_solid[count + 1])
                )
            )
        ]
        for count in np.arange(len(bins_COM_solid) - 1)
    ]
    indices_digitized_liquid = (
        np.digitize(selected_atoms_contact.positions[:, 2], bins_COM_solid, right=True)
        - 1
    )

    # get the COM for each z-segment and create array to subtract from positions
    COMs_z_resolved = np.asarray(
        [atom_group.center_of_mass() for atom_group in solid_z_resolved]
    )

    # now get the vectors from the respective center to the atom, we only take the inplane part
    # this is enough to get the angle but we need to correct radius to project it to
    vectors_COM_z_selected_contact = (
        selected_atoms_contact.positions - COMs_z_resolved[indices_digitized_liquid]
    )[:, not_pbc_indices]

    # for chosen atoms compute position in periodic and angular direction
    # periodic is easy
    liquid_contact_coord2 = selected_atoms_contact.positions[:, 2]

    # the angular coordinate is a bit more tricky
    # start by computing angle from axis
    angles_liquid_contact_central_axis = np.arctan2(
        vectors_COM_z_selected_contact[:, not_pbc_indices[1]],
        vectors_COM_z_selected_contact[:, not_pbc_indices[0]],
    )

    # compute expansion on opened tube (adding pi to get only positive values)
    angular_component_liquid_contact = tube_radius * (
        angles_liquid_contact_central_axis + np.pi
    )
    liquid_contact_coord1 = angular_component_liquid_contact

    # do the same thing for solid
    indices_digitized_solid = (
        np.digitize(solid_atoms.positions[:, 2], bins_COM_solid, right=True) - 1
    )

    vectors_COM_z_solid = (
        solid_atoms.positions - COMs_z_resolved[indices_digitized_solid]
    )[:, not_pbc_indices]

    solid_coord2 = solid_atoms.positions[:, pbc_indices]

    angles_solid_central_axis = np.arctan2(
        vectors_COM_z_solid[:, not_pbc_indices[1]],
        vectors_COM_z_solid[:, not_pbc_indices[0]],
    )

    angular_component_solid = tube_radius * (angles_solid_central_axis + np.pi)

    solid_coord1 = angular_component_solid

    # stack everything
    liquid_contact_2d = np.column_stack((liquid_contact_coord1, liquid_contact_coord2))
    solid_2d = np.column_stack((solid_coord1, solid_coord2))

    return liquid_contact_2d, solid_2d


def _sample_distribution_for_systems_with_one_periodic_direction_per_processor2(
    universe,
    topology,
    spatial_extent_contact_layer: float,
    anchor_coordinates,
    indices_atoms_anchor_rotation,
    tube_radius: float,
    tube_length_in_unit_cells: int,
    pbc_indices,
    species,
    start_frame: int,
    end_frame: int,
    frame_frequency: int,
):
    """
    Compute distribution of atomic positions for 1D systems (for parallelism).
    Arguments:
        universe : MDAnalysis universes to be analysed.
        topology : ASE atoms object containing information about topology.
        anchor_coordinates: Reference coordinartes of the solid.
        indices_atoms_anchor_rotation: Atomic indices forming the reference rotation axis.
        spatial_extent_contact_layer (float): How far ranges the water contact layer.
        tube_radius (float) : radius of the tube in A.
        tube_length_in_unit_cells (int): multiples of tube unit cell in periodic direction.
        pbc_indices : Direction indices in which system is periodic
        species : Element to perform analysis with.
        start_frame (int) : Start frame for analysis.
        end_frame (int) : End frame for analysis.
        frame_frequency (int): Take every nth frame only.
    """

    # define dimensions not periodic, indices
    not_pbc_indices = list(set(pbc_indices) ^ set([0, 1, 2]))
    periodic_vector = np.zeros(3)
    periodic_vector[pbc_indices] = 1

    universe.trajectory[start_frame]

    # repeat separating solid atoms from liquid atoms
    solid_atoms = universe.select_atoms("not name O H")
    selected_atoms = universe.select_atoms(f"name {species}")
    oxygen_atoms = universe.select_atoms(f"name O")

    # define arrays where the coordinates of oxygens and solid atoms will be saved in
    liquid_contact_coord1 = []
    liquid_contact_coord2 = []
    solid_coord1 = []
    solid_coord2 = []

    liquid_contact_coord1_all = []
    liquid_contact_coord2_all = []
    solid_coord1_all = []
    solid_coord2_all = []

    number_of_samples = len(np.arange(start_frame, end_frame, frame_frequency))

    solid_COM_all = []

    # define solid bins for getting z-dependent COM, add a little bit more for numerical reasons
    bins_COM_solid = np.linspace(0, topology.get_cell_lengths_and_angles()[2] + 1e-3, 7)

    # Loop over trajectory
    for count_frames, frames in enumerate(
        tqdm((universe.trajectory[start_frame:end_frame])[::frame_frequency])
    ):
        # raw_positions = copy.deepcopy(universe.atoms.positions)
        solid_COM = solid_atoms.center_of_mass()

        # we start by making the frames translationally and rotationally invariant
        # 1. Translations
        # This is done by computing the translation and substracting it
        translation_from_frame0 = solid_COM - anchor_coordinates
        universe.atoms.positions -= translation_from_frame0
        # translated_positions = raw_positions - translation_from_frame0

        # solid_COM_all.append(raw_positions[50])

        # cleaned_positions = _clean_coords(raw_positions, solid_atoms,anchor_coordinates, topology)
        solid_COM_all.append(universe.atoms.positions[200])

        # # 2. Rotations (only relevant for nanotubes obviously)

        # # to enable an easy rotation, translate atoms to COM
        # COM = universe.atoms.center_of_mass()
        # universe.atoms.positions -= COM

        # # prepare everything to compute angle between axis and anchor axis
        # solid_axis = np.mean(
        #     solid_atoms.positions[indices_atoms_anchor_rotation][:, not_pbc_indices],
        #     axis=0,
        # )

        # # define normed dot product
        # normed_dot_product = np.clip(
        #     np.dot(
        #         solid_axis,
        #         np.asarray([1, 0]),
        #     )
        #     / np.linalg.norm(solid_axis),
        #     -1.0,
        #     1.0,
        # )

        # # Compute angle between reference atom and axis perpendicular to periodic axis
        # # note we need the negative angle if above 180 degrees
        # angle_anchor_first_axis = (
        #     np.arccos(normed_dot_product) if solid_axis[1] <= 0 else -np.arccos(normed_dot_product)
        # )

        # # get rotation matrix for periodic axis and computed angle
        # rotation_matrix = utils.rotation_matrix(periodic_vector, +angle_anchor_first_axis)

        # # rotate atoms, so that this can be compared
        # universe.atoms.positions = np.matmul(rotation_matrix, universe.atoms.positions.T).T

        # # translate now back to center of mass
        # universe.atoms.positions += COM

        # # wrap atoms in box
        # universe.atoms.pack_into_box(box=topology.get_cell_lengths_and_angles(), inplace=True)

        # solid_COM = solid_atoms.center_of_mass()

    return solid_COM_all
    #     #####################################################
    #     # Having prepared the positions now we have to find:
    #     # 1. The water molecules within the contact layer
    #     # 2. Their projection on the surface by taking into account the tube's breathing modes

    #     # get vectors water molecules (here oxygens) to closest solid neighbor
    #     vectors_oxygens_to_solid = (
    #             solid_atoms.positions[np.newaxis, :]
    #             - oxygen_atoms.positions[:, np.newaxis]
    #         )

    #     # apply MIC for all oxygen-oxygen pairs
    #     vectors_oxygens_to_solid_MIC = (
    #             utils.apply_minimum_image_convention_to_interatomic_vectors(
    #                 vectors_oxygens_to_solid, topology.cell,"z"
    #             )
    #         )

    #     # get minimum distances based on vectors
    #     min_distances_oxygens_to_solid = np.min(
    #             np.linalg.norm(vectors_oxygens_to_solid_MIC, axis=2), axis=1
    #         )

    #     # get resids of water molecules inside contact layer
    #     # resids_water_contact = oxygen_atoms[np.where(min_distances_oxygens_to_solid < spatial_extent_contact_layer)].resids
    #     resids_water_contact = "resid " + (" ").join([str(resid) for resid in oxygen_atoms[np.where(min_distances_oxygens_to_solid < spatial_extent_contact_layer)].resids])

    #     # get the respective atoms of the species chosen O/H
    #     selected_atoms_contact = selected_atoms.select_atoms(resids_water_contact)

    #     # for each water molecule compute z center of mass of the tube based on finite displacements
    #     # based on bins get atom groups forming each bin
    #     solid_z_resolved = ([solid_atoms[(np.where((solid_atoms.positions[:,2] > bins_COM_solid[count]) & (solid_atoms.positions[:,2] <= bins_COM_solid[count+1])))] for count in np.arange(len(bins_COM_solid)-1)])
    #     indices_digitized_liquid= np.digitize(selected_atoms_contact.positions[:,2],bins_COM_solid,right=True)-1

    #     # get the COM for each z-segment and create array to subtract from positions
    #     COMs_z_resolved = np.asarray([atom_group.center_of_mass() for atom_group in solid_z_resolved])

    #     if np.max(indices_digitized_liquid)>5:
    #         breakpoint()

    #     # now get the vectors from the respective center to the atom, we only take the inplane part
    #     # this is enough to get the angle but we need to correct radius to project it to
    #     vectors_COM_z_selected_contact = (
    #             selected_atoms_contact.positions
    #             - COMs_z_resolved[indices_digitized_liquid]
    #         )[:,not_pbc_indices]

    #     # for chosen atoms compute position in periodic and angular direction
    #     # periodic is easy
    #     liquid_contact_coord2 = np.append(
    #         liquid_contact_coord2,
    #         selected_atoms_contact.positions[:,2],
    #     )

    #     # the angular coordinate is a bit more tricky
    #     # start by computing angle from axis
    #     angles_liquid_contact_central_axis = np.arctan2(
    #         vectors_COM_z_selected_contact[:, not_pbc_indices[1]],
    #         vectors_COM_z_selected_contact[:, not_pbc_indices[0]],
    #     )

    #     # compute expansion on opened tube (adding pi to get only positive values)
    #     angular_component_liquid_contact = (
    #         tube_radius * (angles_liquid_contact_central_axis + np.pi)
    #     )
    #     liquid_contact_coord1 = np.append(liquid_contact_coord1, angular_component_liquid_contact)

    #     # do the same thing for solid
    #     indices_digitized_solid= np.digitize(solid_atoms.positions[:,2],bins_COM_solid,right=True)-1

    #     vectors_COM_z_solid = (
    #             solid_atoms.positions
    #             - COMs_z_resolved[indices_digitized_solid]
    #         )[:,not_pbc_indices]

    #     solid_coord2 = np.append(solid_coord2, solid_atoms.positions[:, pbc_indices])

    #     angles_solid_central_axis = np.arctan2(
    #         vectors_COM_z_solid[:, not_pbc_indices[1]],
    #         vectors_COM_z_solid[:, not_pbc_indices[0]],
    #     )

    #     angular_component_solid = (
    #         tube_radius * (angles_solid_central_axis + np.pi)
    #     )

    #     solid_coord1 = np.append(solid_coord1, angular_component_solid)

    #     # making code more efficient, ugly bu useful
    #     if count_frames % 1000 == 0 or count_frames == number_of_samples - 1:

    #         # save full arrays to global array and empty local to free memory and speed up loop
    #         liquid_contact_coord1_all = np.append(liquid_contact_coord1_all, liquid_contact_coord1)
    #         liquid_contact_coord1 = []
    #         liquid_contact_coord2_all = np.append(liquid_contact_coord2_all, liquid_contact_coord2)
    #         liquid_contact_coord2 = []
    #         solid_coord1_all = np.append(solid_coord1_all, solid_coord1)
    #         solid_coord1 = []
    #         solid_coord2_all = np.append(solid_coord2_all, solid_coord2)
    #         solid_coord2 = []

    # # stack everything
    # liquid_contact_2d = np.column_stack((liquid_contact_coord1_all, liquid_contact_coord2_all))
    # solid_2d = np.column_stack((solid_coord1_all, solid_coord2_all))

    # return liquid_contact_2d, solid_2d


def _compute_distribution_for_system_with_two_periodic_directions(
    universe,
    topology,
    spatial_extent_contact_layer: float,
    pbc_indices,
    species,
    start_frame: int,
    end_frame: int,
    frame_frequency: int,
):
    """
    Compute distribution of atomic positions for 2D systems.
    Arguments:
        position_universes : MDAnalysis universe to be analysed.
        topology : ASE atoms object containing information about topology.
        spatial_extent_contact_layer (float): How far ranges the water contact layer.
        pbc_indices : Direction indices in which system is periodic
        species : element to perform analysis with (O or H)
        start_frame (int) : Start frame for analysis.
        end_frame (int) : End frame for analysis.
        frame_frequency (int): Take every nth frame only.
    Returns:
        liquid_contact_2D: numpy array with positions in periodic directions of
                            oxygens in contact layer
        solid_all: numpy array of all solid atom positions in periodic directions
                     structured by timestep

    """

    # define dimensions not periodic, indices
    not_pbc_indices = list(set(pbc_indices) ^ set([0, 1, 2]))
    periodic_vector = np.zeros(3)
    periodic_vector[pbc_indices] = 1

    # rewind trajectory
    universe.trajectory[0]

    # wrap atoms in box
    # universe.atoms.pack_into_box(box=topology.get_cell_lengths_and_angles(), inplace=True)

    # start by separating solid atoms from liquid atoms
    solid_atoms = universe.select_atoms("not name O H")

    # approximate water with oxygens here
    liquid_atoms = universe.select_atoms(f"name {species}")

    # this will serve as our anchor for computing the free energy profile
    anchor_coordinates = solid_atoms.center_of_mass()

    # define arrays where the coordinates of oxygens and solid atoms will be saved in
    liquid_contact_coord1 = []
    liquid_contact_coord2 = []
    solid_all = []

    liquid_contact_coord1_all = []
    liquid_contact_coord2_all = []
    solid_all_all = []

    number_of_samples = len(np.arange(start_frame, end_frame, frame_frequency))

    # Loop over trajectory
    for count_frames, frames in enumerate(
        tqdm((universe.trajectory[start_frame:end_frame])[::frame_frequency])
    ):
        # wrap atoms in box
        # universe.atoms.pack_into_box(box=topology.get_cell_lengths_and_angles(), inplace=True)

        # we start by making the frames translationally invariant
        # This is done by computing the translation and substracting it
        translation_from_frame0 = solid_atoms.center_of_mass() - anchor_coordinates
        universe.atoms.positions -= translation_from_frame0

        # wrap atoms in box
        universe.atoms.pack_into_box(
            box=topology.get_cell_lengths_and_angles(), inplace=True
        )

        # define center of mass of solid now
        # solid_COM = solid_atoms.center_of_mass()

        # now compute distance from liquid atoms perpendicular to the center of mass of the solid
        # perpendicular_distance_liquid_to_solid = (
        #     liquid_atoms.positions[:, not_pbc_indices] - solid_COM[not_pbc_indices]
        # ).flatten()

        # # only choose those atoms which are within contact layer only 2D
        # liquid_atoms_in_contact_positions = liquid_atoms[
        #     np.where(perpendicular_distance_liquid_to_solid <= spatial_extent_contact_layer)
        # ].positions[:, pbc_indices]

        # now we have to check which atoms are in the contact layer based on the shortest OX distance
        # for this we need to find the closest solid atom for each liquid atom

        # get vectors between each liquid atom and all solid atoms
        vectors_liquid_to_solid = (
            solid_atoms.positions[np.newaxis, :] - liquid_atoms.positions[:, np.newaxis]
        )

        # apply MIC for all oxygen-oxygen pairs
        vectors_liquid_to_solid_MIC = (
            utils.apply_minimum_image_convention_to_interatomic_vectors(
                vectors_liquid_to_solid, topology.cell, "xy"
            )
        )

        # get minimum distances based on vectors
        min_distances_liquid_to_solid = np.min(
            np.linalg.norm(vectors_liquid_to_solid_MIC, axis=2), axis=1
        )

        liquid_atoms_in_contact_positions = liquid_atoms[
            np.where(min_distances_liquid_to_solid <= spatial_extent_contact_layer)
        ].positions[:, pbc_indices]

        # save liquid
        liquid_contact_coord1 = np.append(
            liquid_contact_coord1, liquid_atoms_in_contact_positions[:, pbc_indices[0]]
        )

        liquid_contact_coord2 = np.append(
            liquid_contact_coord2, liquid_atoms_in_contact_positions[:, pbc_indices[1]]
        )

        # save solid
        solid_all = np.append(solid_all, solid_atoms.positions)

        # making code more efficient, ugly but useful
        if count_frames % 1000 == 0 or count_frames == number_of_samples - 1:

            # save full arrays to global array and empty local to free memory and speed up loop
            liquid_contact_coord1_all = np.append(
                liquid_contact_coord1_all, liquid_contact_coord1
            )
            liquid_contact_coord1 = []
            liquid_contact_coord2_all = np.append(
                liquid_contact_coord2_all, liquid_contact_coord2
            )
            liquid_contact_coord2 = []
            solid_all_all = np.append(solid_all_all, solid_all)
            solid_all = []

    # put coords of liquid together
    liquid_contact_2d = np.column_stack(
        (liquid_contact_coord1_all, liquid_contact_coord2_all)
    )

    return liquid_contact_2d, (solid_all_all)


def _get_atom_ids_on_same_tube_axis(
    solid_atoms, tube_length_in_unit_cells: int, not_pbc_indices
):
    """
    Compute axis through atoms of tube parallel to tube axis..
    Arguments:
        solid_atoms: All atoms (including positions) of the tube.
        tube_length_in_unit_cells (int): Length of the tube expressed in multiples of unit cell
        not_pbc_indices: List of ints which are not periodic.
    Returns:
        ids_atoms_on_axis = list of atom ids on the same axis

    """

    # for atom 0 get all indices of atoms which have similar coordinates in the non-pbc directions
    ids_candidate_atoms_on_axis_with_0 = np.argsort(
        np.linalg.norm(
            solid_atoms.positions[:, not_pbc_indices]
            - solid_atoms.positions[0][not_pbc_indices],
            axis=1,
        )
    )

    # return indices closest based on tube length
    return ids_candidate_atoms_on_axis_with_0[0 : 2 * tube_length_in_unit_cells]
