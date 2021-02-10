# Just some global variables

# Scientific constants
AVOGADRO = 6.0221409e23  # unit is 1/mol
BOLTZMANN = 1.38064852e-23  # unit is J/K

# Conversion factors
# Energy
HARTREE_TO_EV = 27.21139664  # unit is eV/Ha
EV_TO_JOULE = 1.602176565e-19  # unit is J/eV

# Distance
BOHR_TO_ANGSTROM = 0.52917726  # unit is A/B
ANGSTROM_TO_METER = 1e-10  # unit is m/A

# Time
FEMTOSECOND_TO_SECOND = 1e-15  # unit is is fs/s
AU_TIME_TO_SECOND = 2.418884309e-17
AU_TIME_TO_FEMTOSECOND = 2.418884309e-2

# Dictionaries
DIMENSION_DICTIONARY = {
    "x": [0],
    "xy": [0, 1],
    "xz": [0, 2],
    "y": [1],
    "yz": [1, 2],
    "z": [2],
    "xyz": [0, 1, 2],
}

PREFACTOR_DIFFUSION_CORRECTION_BASED_ON_BOX_SHAPE_DICTIONARY = {"cubic": 2.837297}

# in Pa*s
EXP_VISCOSITIES_BULK_FLUIDS = {"water": 0.8925e-3}
