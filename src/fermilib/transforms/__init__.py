from ._conversion import (get_sparse_operator,
                          get_sparse_operator_term,
                          jordan_wigner_sparse,
                          get_sparse_interaction_operator,
                          get_interaction_rdm,
                          get_interaction_operator,
                          get_fermion_operator,
                          eigenspectrum)  #TODO: we might want to remove this completely
from ._jordan_wigner import jordan_wigner
from ._reverse_jordan_wigner import reverse_jordan_wigner
from ._bravyi_kitaev import bravyi_kitaev
