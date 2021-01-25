# Just some global variables

AVOGADRO = 6.0221409 * 1e23  # AVOGADRO CONSTANT

# Conversion factors
HARTREE2EV = 27.21139664  # unit is eV/Ha
BOHR2ANGSTROM = 1.0 / 0.52917726  # unit is A/B

DIMENSION_DICTIONARY = {  # DIMENSION DICTIONARY used to get indices based on direction
    "x": [0],
    "xy": [0, 1],
    "xz": [0, 2],
    "y": [1],
    "yz": [1, 2],
    "z": [2],
    "xyz": [0, 1, 2],
}
