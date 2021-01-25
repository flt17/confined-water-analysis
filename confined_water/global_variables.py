# Just some global variables

# Scientific constants
AVOGADRO = 6.0221409e23  # unit is 1/mol
BOLTZMANN = 1.38064852e-23  # unit is J/K

# Conversion factors
# Energy
HARTREE_TO_EV = 27.21139664  # unit is eV/Ha
EV_TO_JOULE = 1.602176565e-19  # unit is J/eV

# Distance
BOHR_TO_ANGSTROM = 1.0 / 0.52917726  # unit is A/B
ANGSTROM_TO_METER = 1e-10  # unit is m/A

# Time
FEMTOSECOND_TO_SECOND = 1e-15  # unit is is fs/s


DIMENSION_DICTIONARY = {  # DIMENSION DICTIONARY used to get indices based on direction
    "x": [0],
    "xy": [0, 1],
    "xz": [0, 2],
    "y": [1],
    "yz": [1, 2],
    "z": [2],
    "xyz": [0, 1, 2],
}
