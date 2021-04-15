import glob
import math
import numpy as np
import os
import scipy
from scipy import stats
import sys

import ase
from ase import io
from ase.io import lammpsrun

import MDAnalysis as mdanalysis

sys.path.append("../")
from confined_water import global_variables


class UnableToFindFile(Exception):
    pass


class UndefinedOption(Exception):
    pass


def get_path_to_file(
    directory_path: str, file_suffix: str = None, file_prefix: str = None, exact_match: str = True
):
    """
    Return the path to file based on given directory and suffix and if given prefix.
    Arguments:
        directory_path (str): The path to the directory containing the trajectory.
        file_suffix (str): The general suffix or type of the file.
        file_prefix (str): The general name of the file.
        exact_match (bool): Determines whether file_prefix needs to be matched exactly or partially.
    Returns:
        path_to_file (str): The path to the requested file(s).
    """
    files_in_path = sorted(glob.glob(os.path.join(directory_path, f"*.{file_suffix}")))

    if not files_in_path:
        raise UnableToFindFile(f"No {file_suffix} file found in path {directory_path}.")

    # If prefix of file is given...
    if file_prefix:
        # ... and exact matching is asked we return exactly that file
        if exact_match:
            expected_path = os.path.join(directory_path, file_prefix + f".{file_suffix}")
            return [file for file in files_in_path if file == expected_path][0]
        # If exact_matching is false all files comprising file_prefix in the file name are returned
        else:
            return [file for file in files_in_path if file_prefix in file]

    # Otherwise, we take return all files in alphabetical order.
    else:
        if len(files_in_path) > 1:
            print(f"WARNING: More than one {file_suffix} file found.")
            return files_in_path
        else:
            return files_in_path[0]


def get_ase_atoms_object(pdb_file_path: str):
    """
    Return an ase atoms object used for topology.
    This happens via the previously found pdb-file.
    If this fails, the package will attempt to use
    a lammpstrj file with the same name in the same
    directory.
    Arguments:
        pdb_file_path (str): The path to the pdb file.
    Returns:
        ase_AO (object): The ase atoms object.
    """

    try:
        # Try to use pdb file to generate atoms object
        return ase.io.read(pdb_file_path)

    except KeyError:
        print(
            "WARNING: Could not read pdb file with ASE, maybe you created it with VMD from a lammpstrj file."
        )

        # Get directory path and name we look for
        directory_path = "/".join([i for i in pdb_file_path.split("/")[0:-1]])
        file_prefix = (pdb_file_path.split("/"))[-1].split(".")[0]
        print(f"Looking for lammpstrj files in {directory_path}")

        # Look for lammpstrj files in same directory
        lammpstrj_file = get_path_to_file(directory_path, "lammpstrj", file_prefix)

        print(f"Creating ASE atoms object for first frame of {lammpstrj_file}")
        ase_AO = ase.io.lammpsrun.read_lammps_dump(lammpstrj_file, index=0)

        # setting carbon as element
        ase_AO.set_chemical_symbols(["C"] * ase_AO.get_global_number_of_atoms())

        return ase_AO


def get_mdanalysis_universe(
    directory_path: str,
    universe_type: str = "positions",
    topology_file_prefix: str = None,
    trajectory_format="dcd",
):
    """
    Return an mdanalysis universe object based on a given trajectory. This will be done by
    first finding the correct topology file (e.g. pdb) and coupling it with a dcd file.

    Arguments:
        directory_path (str): The path to the directory in which the simulation was performed.
        universe_type (str): Which kind of universe type (positions, forces, velocities) should
                             be created. So far, only positions are implemented as the velocity and
                             force-dcd files need to be converted.
        topology_file_prefix (str): General name of pdb file. Only exact matching implemented here.
        trajectory_format (str) : File format of trajectory, default is dcd.

    Returns:
        mdanalysis_universe (object): The mdanalsysis universe object(s) dependent on the number of
                                     trajectories found. If PIMD the first element contains the centroid
                                     followed by individual beats of the ring polymer.
    """

    # Link universe_type to file names of respective trajectories via dictionary
    # Currently, only supports default of CP2K
    dictionary_trajectory_files = {"positions": "-pos-", "velocities": "-vel-", "forces": "-frc-"}

    # Look for topology file (only pdb supported) in same directory, only exact matching implemented for prefix
    topology_file = get_path_to_file(directory_path, "pdb", topology_file_prefix)

    print(f"Using the topology from {topology_file}.")

    # Determine trajectory filenames
    trajectory_prefix = dictionary_trajectory_files.get(universe_type)
    if not trajectory_prefix:
        raise UndefinedOption(
            f"Specified {universe_type} is unknown. Possible options are {dictionary_trajectory_files.keys()}"
        )

    # Look for trajectory (dcd) files , could be more than one if PIMD was performed
    trajectory_files = get_path_to_file(
        directory_path, trajectory_format, trajectory_prefix, exact_match=False
    )

    print(f"Creating universes for {len(trajectory_files)} trajectories.")

    if len(trajectory_files) == 1:
        # If only one trajectory file was found, this is a classical simulation
        return mdanalysis.Universe(topology_file, trajectory_files)

    else:
        # If multiple files were found,this is a PIMD simulation
        # In this case we generate multiple Universes, one for each beat
        # for calculating statical and thermodynamical properties as well
        # as the centroid universe which should be used for dynamical properties.

        # This requires that the first file is the centroid trajectory. This is automatically guaranteed
        # by using CP2K to run the PIMD simulations which uses "*centroid*" and "*pos*"/"*vel*"/"*frc*"
        # as part of the file names
        centroid_universe = mdanalysis.Universe(topology_file, trajectory_files[0])
        beat_universes = [
            mdanalysis.Universe(topology_file, traj_file) for traj_file in trajectory_files[1::]
        ]
        return [centroid_universe, *beat_universes]


def apply_minimum_image_convention_to_interatomic_vectors(
    vectors: np.array,
    lattice_vectors: np.array,
    dimension: str = "xyz",
):
    """
    Return vectors which follow the minimum image convention. This function should be mainly
    used in the context of larger scripts when the nearest neighbors or distance criteria are
    needed to compute properties.
    Currently, only implemented for orthorombic cells.

    Arguments:
        vectors (np.array): Vectors between atoms.
        lattice_vectors (np.array): Lattice vectors of simulation box.
        dimension (str) : to speed up calculatio only perform transformation in the periodic direction.


    Returns:
        vectors_MIC: Vectors which are in line with the minimum image convention.
    """
    # implement minimum image convention in x-direction

    if not global_variables.DIMENSION_DICTIONARY.get(dimension):
        raise UndefinedOption(
            f"Specified dimension {dimension} is unknown. Possible options are {global_variables.DIMENSION_DICTIONARY.keys()}"
        )

    for dim in global_variables.DIMENSION_DICTIONARY.get(dimension):

        vectors[
            np.where(np.take(vectors, dim, axis=-1) > lattice_vectors[dim][dim] / 2)
        ] -= lattice_vectors[dim]
        vectors[
            np.where(np.take(vectors, dim, axis=-1) < -lattice_vectors[dim][dim] / 2)
        ] += lattice_vectors[dim]

    vectors_MIC = vectors
    return vectors_MIC


def get_dipole_moment_vector_in_water_molecule(atom_group, topology, dimension: str = "xyz"):

    """
    Return dipole moment vector from oxygen atom to COM of hydrogen atoms of given atom group.
    Arguments:
        atom_group (): Atoms with given coordinates and mass.
        lattice_vectors(ase atoms object): Topology of the system, i.e. cell lengths etc..
        dimension (str) : to speed up calculatio only perform transformation in the periodic direction.
    Returns:
        dipole_moment_vector (np.array): Dipole moment water vector in accordance with pbc.
    """
    # compute center of mass of hydrogen atoms
    COM_hydrogen_atoms = get_center_of_mass_of_atoms_in_accordance_with_MIC(
        atom_group.select_atoms("name H"), topology, dimension
    )

    # now return dipole vector
    return apply_minimum_image_convention_to_interatomic_vectors(
        COM_hydrogen_atoms - atom_group.select_atoms("name O").positions, topology.cell, dimension
    ).flatten()


def get_center_of_mass_of_atoms_in_accordance_with_MIC(
    atom_group,
    topology,
    dimension: str = "xyz",
):
    """
    Return center of mass in passed simulation box for a given atom group.
    Arguments:
        atom_group (): Atoms with given coordinates and mass.
        lattice_vectors(ase atoms object): Topology of the system, i.e. cell lengths etc..
        dimension (str) : to speed up calculatio only perform transformation in the periodic direction.
    Returns:
        center_of_mass_pbc (np.array): Center of mass in accordance with pbc.
    """

    # create copy of atom group
    tmp_atom_group = atom_group.copy()

    # compute vectors from first atom of atom group to remaining atoms
    vectors_first_to_rest = tmp_atom_group.positions - tmp_atom_group[0].position

    # make this vector MIC conform
    vectors_first_to_rest_MIC = apply_minimum_image_convention_to_interatomic_vectors(
        vectors_first_to_rest, topology.cell, dimension
    )

    # compute new positions which will then be used for center of mass
    positions_MIC = tmp_atom_group[0].position + vectors_first_to_rest_MIC

    # save these to the tmp atom group
    tmp_atom_group.positions = positions_MIC

    # compute center of mass via MDAnalysis
    com_MIC_in_box = tmp_atom_group.center_of_mass()

    # wrap COM inside box
    # above cell lengths, only orthorombic
    com_MIC_in_box[
        np.where(com_MIC_in_box > topology.get_cell_lengths_and_angles()[0:3])
    ] -= topology.get_cell_lengths_and_angles()[
        np.where(com_MIC_in_box > topology.get_cell_lengths_and_angles()[0:3])
    ]

    # negative values
    com_MIC_in_box[
        np.where(com_MIC_in_box < np.zeros(3))
    ] += topology.get_cell_lengths_and_angles()[np.where(com_MIC_in_box < np.zeros(3))]

    return com_MIC_in_box


def compute_diffusion_coefficient_based_on_MSD(
    measured_msd: np.array,
    measured_time: np.array,
    start_time_fit: float = None,
    end_time_fit: float = None,
):
    """
    Return diffusion coefficient based on mean squared displacement.
    Arguments:
        measured_msd (np.array): Mean squared displacement measured from trajectory.
        measurement_time (np.array): Time array for which the MSD was measured.
        start_time_fit (float) : Start time (in fs) for linear fit (optional). Default is 20% of time measured.
        end_time_fit (float) : End time (in fs) for linear fit (optional). Default is end of the measurment.
    Returns:
        diffusion_coefficient (float): Diffusion coefficient in m^2/s.
    """

    # determine interval between measurements in fs:
    time_interval_measurements = measured_time[1] - measured_time[0]

    # determine start and end time for fit:
    start_time_fit = start_time_fit if start_time_fit else (0.2 * measured_time[-1])
    end_time_fit = end_time_fit if end_time_fit else measured_time[-1]

    # convert times into frames
    start_frame_fit = math.ceil(start_time_fit / time_interval_measurements)
    end_frame_fit = int(end_time_fit / time_interval_measurements + 1)

    # linear regression to data selected, we are only interested in slope and the quality of the fit.
    fit_slope, __, fit_r_value, __, fit_std_err = scipy.stats.linregress(
        measured_time[start_frame_fit:end_frame_fit], measured_msd[start_frame_fit:end_frame_fit]
    )

    # tell user if fit isn't sufficiently accurate.
    if fit_r_value ** 2 < 0.95:
        print(
            f"WARNING: The linear fit to the mean squared displacement showed an r2 score of {fit_r_value**2}.",
            f"We recommend to increase the run time of the trajectory or change the fitting settings.",
        )

    # return diffusion coefficient based on slope in m^2 /s
    return (
        fit_slope
        / 6
        * global_variables.ANGSTROM_TO_METER ** 2
        / global_variables.FEMTOSECOND_TO_SECOND
    )


def compute_finite_size_correction_for_diffusion_coefficient_of_bulk_fluid(
    topology, fluid: str = "water", temperature: float = 330
):
    """
    Return finite size correction for diffusion coefficient for bulk fluids, e.g. water.
    Arguments:
        topology of the system: ASE atoms object containing information on system dimensions.
        fluid (str): Name of the fluid we want to compute the correction.
        temperature (float): Temperature at which we would like to compute the correction.
    Returns:
        finite_size_correction_diffusion (float): Correction for the finite size of the system in  m^2/s.
    """

    # determine format of simulation box
    box_shape = (
        "cubic"
        if len(np.unique(topology.get_cell_lengths_and_angles()[0:3])) == 1
        and np.all(topology.get_cell_lengths_and_angles()[3::] == 90)
        else ""
    )

    # get prefactor from dictionary
    prefactor = global_variables.PREFACTOR_DIFFUSION_CORRECTION_BASED_ON_BOX_SHAPE_DICTIONARY.get(
        box_shape
    )
    # check that box shape is known, so far only cubic
    if not prefactor:
        raise UndefinedOption(
            f"Specified box shape {box_shape} is unknown.",
            f" Possible options are {global_variables.PREFACTOR_DIFFUSION_CORRECTION_BASED_ON_BOX_SHAPE_DICTIONARY.keys()}",
        )

    # get viscosity of fluid chosen
    viscosity = global_variables.EXP_VISCOSITIES_BULK_FLUIDS.get(fluid)
    if not viscosity:
        raise UndefinedOption(
            f"Specified fluid {fluid} is unknown.",
            f" Possible options are {global_variables.EXP_VISCOSITIES_BULK_FLUIDS.keys()}",
        )

    return (
        prefactor
        * global_variables.BOLTZMANN
        * temperature
        / (
            6
            * np.pi
            * viscosity
            * topology.get_cell_lengths_and_angles()[0]
            * global_variables.ANGSTROM_TO_METER
        )
    )


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    This part of the code was implemented by Patrick Rowe.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )
