"""Bravyi-Kitaev transform on fermionic operators."""
from __future__ import absolute_import

from fermilib.transforms import FenwickTree
from fermilib.utils import count_qubits

from projectqtemp.ops import QubitOperator


def bravyi_kitaev(operator, n_qubits=None):
    """
    Apply the Bravyi-Kitaev transform and return qubit operator.

    Args:
        operator: A FermionOperator to transform.

    Returns:
        transformed_operator: An instance of the QubitOperator class.

    Raises:
        ValueError: Invalid number of qubits specified.
    """
    # Compute the number of qubits.
    if n_qubits is None:
        n_qubits = count_qubits(operator)
    if n_qubits < count_qubits(operator):
        raise ValueError('Invalid number of qubits specified.')

    # Compute transformed operator.
    transformed_operator = QubitOperator()
    for term in operator.terms:

        # Initialize identity matrix.
        coefficient = operator.terms[term]
        transformed_term = QubitOperator((), coefficient)

        # Build the Fenwick Tree
        fenwick_tree = FenwickTree(n_qubits)

        # Build the Bravyi-Kitaev transformed operators.
        for ladder_operator in term:
            index = ladder_operator[0]

            # Parity set. Set of nodes to apply Z to.
            parity_set = [node.index for node in
                          fenwick_tree.get_parity_set(index)]

            # Update set. Set of ancestors to apply X to.
            ancestors = [node.index for node in
                         fenwick_tree.get_update_set(index)]

            # The C(j) set.
            ancestor_children = [node.index for node in
                                 fenwick_tree.get_remainder_set(index)]

            # Switch between lowering/raising operators.
            d_coefficient = .5j
            if ladder_operator[1]:
                d_coefficient *= -1.

            # The fermion lowering operator is given by
            # a = (c+id)/2 where c, d are the majoranas.
            d_majorana_component = QubitOperator(
                (((ladder_operator[0], 'Y'),) +
                 tuple((index, 'Z') for index in ancestor_children) +
                 tuple((index, 'X') for index in ancestors)),
                d_coefficient)

            c_majorana_component = QubitOperator(
                (((ladder_operator[0], 'X'),) +
                 tuple((index, 'Z') for index in parity_set) +
                 tuple((index, 'X') for index in ancestors)),
                0.5)

            # Update term.
            transformed_term *= c_majorana_component + d_majorana_component
        transformed_operator += transformed_term
    return transformed_operator
