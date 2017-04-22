"""Jordan-Wigner transform on fermionic operators."""
from __future__ import absolute_import

from fermilib.qubit_operators import QubitOperator
from fermilib.fermion_operators import FermionTerm, FermionOperator
from fermilib.interaction_operators import InteractionOperator

from ._jordan_wigner_term import jordan_wigner_term
from ._jordan_wigner_interaction_op import jordan_wigner_interaction_op


def jordan_wigner(op):
    """Apply the Jordan-Wigner transform the fermionic operator op and
    return qubit operator.

    Returns:
      transformed_operator: An instance of the QubitOperator class.

    Warning:
      The runtime of this method is exponential in the maximum locality
      of the FermionTerms in the original FermionOperator.

    """
    if isinstance(op, FermionTerm):
        op = FermionOperator(op)
    elif isinstance(op, InteractionOperator):
        return jordan_wigner_interaction_op(op)
    if not isinstance(op, FermionOperator):
        raise TypeError("op must be a FermionTerm, FermionOperator, or "
                        "InteractionOperator.")

    transformed_operator = QubitOperator()
    for term in op:
        transformed_operator += jordan_wigner_term(term)
    return transformed_operator
