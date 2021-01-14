import glob
import os

import ase
from ase import io
from ase.io import lammpsrun


class UnableToFindFile(Exception):
    pass


def get_path_to_file(directory_path, file_suffix=None, file_prefix=None):
    """
    Return the path to file based on given directory and suffix and if given prefix.
    Arguments:
        directory_path (str): The path to the directory containing the trajectory.
        file_suffix (str): The general suffix or type of the file.
        file_prefix (str): The general name of the file.
    Returns:
        path_to_file: The path to the requested file.
    """
    files_in_path = sorted(glob.glob(os.path.join(directory_path, f"*.{file_suffix}")))

    if not files_in_path:
        raise UnableToFindFile(f"No {file_suffix} file found in path {directory_path}.")

    # If prefix of file is given, we take that file.
    if file_prefix:
        return [file for file in files_in_path if file_prefix + f".{file_suffix}" in file][0]
    # Otherwise, we take the first file given by alphabetical order.
    else:
        if len(files_in_path) > 1:
            file_name_found = files_in_path[0].split("/")[-1]
            print(
                f"WARNING: More than one {file_suffix} file found. Proceeding with {file_name_found}."
            )
        return files_in_path[0]


def get_ase_atoms_object(pdb_file_path):
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
