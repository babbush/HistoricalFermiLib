import os

# Tolerance to consider number zero.
EQ_TOLERANCE = 1e-12

# Molecular data directory.
THIS_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
DATA_DIRECTORY = os.path.abspath(
    os.path.join(THIS_DIRECTORY, '../..', 'data'))
PSI4_DIRECTORY = os.path.abspath(
    os.path.join(THIS_DIRECTORY, '..', 'psi4tmp'))
