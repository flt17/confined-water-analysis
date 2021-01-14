import glob
import os

import ase
from ase import io
from ase.io import lammpsrun

import MDAnalysis as mdanalysis


class UnableToFindFile(Exception):
    pass


class UndefinedOption(Exception):
    pass


def get_path_to_file(directory_path: str, file_suffix=None, file_prefix=None, exact_match=True):
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


def get_mdanalysis_universe(directory_path: str, universe_type="positions"):
    """
    Return an mdanalysis universe object based on a given trajectory. This will be done by
    first finding the correct topology file (e.g. pdb) and coupling it with a dcd file.

    Arguments:
        directory_path (str): The path to the directory in which the simulation was performed.
        universe_type (str): Which kind of universe type (positions, forces, velocities) should
                             be created.

    Returns:
        mdanalysis_universe (object): The mdanalsysis universe object.
    """

    # Link universe_type to file names of respective trajectories via dictionary
    # Currently, only supports default of CP2K
    dictionary_trajectory_files = {"positions": "pos", "velocities": "vel", "forces": "frc"}

    # Look for topology file (only pdb supported) in same directory
    topology_file = get_path_to_file(directory_path, "pdb")

    print(f"Using the topology from {topology_file}.")

    # Determine trajectory filenames
    trajectory_prefix = dictionary_trajectory_files.get(universe_type)
    if not trajectory_prefix:
        raise UndefinedOption(
            f"Specified {universe_type} is unknown. Possible options are {dictionary_trajectory_files.keys()}"
        )

    # Look for trajectory (dcd) files , could be more than one if PIMD was performed
    trajectory_files = get_path_to_file(directory_path, "dcd", trajectory_prefix, exact_match=False)

    print(f"Creating universes for {len(trajectory_files)} trajectories.")

    return mdanalysis.Universe(topology_file, trajectory_files)
