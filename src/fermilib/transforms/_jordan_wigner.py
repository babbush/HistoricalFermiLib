"""Jordan-Wigner transform on fermionic operators."""
from __future__ import absolute_import

from projectqtemp.ops._qubit_operator import QubitOperator
from fermilib.ops import (FermionOperator,
                          InteractionOperator)

from transforms._jordan_wigner_term import jordan_wigner_term
from transforms._jordan_wigner_interaction_op import (
    jordan_wigner_interaction_op)


def jordan_wigner(op):
    """Apply the Jordan-Wigner transform the fermionic operator op and
    return qubit operator.

    Returns:
      transformed_operator: An instance of the QubitOperator class.

    Warning:
      The runtime of this method is exponential in the maximum locality
      of the original FermionOperator.

    """
    if isinstance(op, InteractionOperator):
        return jordan_wigner_interaction_op(op)

    if not isinstance(op, FermionOperator):
        raise TypeError("op must be a FermionOperator or InteractionOperator.")

    transformed_operator = QubitOperator((), 0.0)
    for term in op.terms:
        transformed_operator += jordan_wigner_term(term, op.terms[term])
    return transformed_operator
