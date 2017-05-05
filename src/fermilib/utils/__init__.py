from ._chemical_series import (make_atomic_ring,
                               make_atomic_lattice,
                               make_atom)

from ._hubbard import fermi_hubbard

from ._jellium import (jellium_model,
                       jordan_wigner_position_jellium,
                       momentum_kinetic_operator,
                       momentum_potential_operator,
                       position_kinetic_operator,
                       position_potential_operator)

from ._molecular_data import MolecularData, periodic_table

from ._operator_utils import eigenspectrum, is_identity, count_qubits

from ._sparse_tools import (expectation,
                            get_density_matrix,
                            get_gap,
                            get_ground_state,
                            is_hermitian,
                            jordan_wigner_sparse,
                            jw_hartree_fock_state,
                            qubit_operator_sparse,
                            sparse_eigenspectrum)

from ._unitary_cc import uccsd_operator
