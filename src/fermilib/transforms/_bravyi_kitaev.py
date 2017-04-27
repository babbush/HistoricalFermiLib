"""Bravyi-Kitaev transform on fermionic operators."""
from __future__ import absolute_import

from fermilib.ops import FermionOperator
from fermilib.transforms import FenwickTree

from projectqtemp.ops import QubitOperator


def bravyi_kitaev_term(term, n_qubits=None):
    """
    Apply the Bravyi-Kitaev transform and return qubit operator.

    Note:
        Reference: Operator Locality of Quantum Simulation of Fermionic
            Models (arXiv:1701.07072).
    
    Args:
        term: A fermionic operator to be transformed
        n_qubits: number of qubits in the register TODO (?)

    Returns:
        transformed_term: An instance of the QubitOperator class.

    Warning:
        Likely greedy. At the moment the method gets the node sets for
        each fermionic operator. FenwickNodes are not neccessary in this
        construction, only the indices matter here. This may be optimized
        by removing the unnecessary structure.
    """
    if not isinstance(term, FermionOperator) or len(term.terms) > 1:
        raise ValueError("term must be a single-term FermionOperator.")

    if n_qubits is None:
        n_qubits = term.n_qubits()
    if n_qubits < term.n_qubits():
        raise ValueError('Invalid n_qubits.')

    ops = list(term.terms)[0]
    coeff = term.terms[ops]

    # Build the Fenwick Tree
    fenwick_tree = FenwickTree(n_qubits)

    # Initialize identity matrix.
    transformed_term = QubitOperator((), coeff)

    # Build the Bravyi-Kitaev transformed operators.
    for operator in ops:
        index = operator[0]

        # Parity set. Set of nodes to apply Z to.
        parity_set = [node.index for node in
                      fenwick_tree.get_parity_set(index)]

        # Update set. Set of ancestors to apply X to.
        ancestors = [node.index for node in fenwick_tree.get_update_set(index)]

        # The C(j) set.
        ancestor_children = [node.index for node in
                             fenwick_tree.get_remainder_set(index)]

        # Switch between lowering/raising operators.
        d_coeff = .5j
        if operator[1]:
            d_coeff = -d_coeff

        # The fermion lowering operator is given by
        # a = (c+id)/2 where c, d are the majoranas.
        d_majorana_component = QubitOperator(
            (((operator[0], 'Y'),) +
             tuple((index, 'Z') for index in ancestor_children) +
             tuple((index, 'X') for index in ancestors)),
            d_coeff)

        c_majorana_component = QubitOperator(
            (((operator[0], 'X'),) +
             tuple((index, 'Z') for index in parity_set) +
             tuple((index, 'X') for index in ancestors)),
            0.5)

        transformed_term *= c_majorana_component + d_majorana_component

    return transformed_term


def bravyi_kitaev(op, n_qubits=None):
    """
    Apply the Bravyi-Kitaev transform and return qubit operator.

    Returns:
         transformed_operator: An instance of the QubitOperator class.
    """

    if n_qubits is None:
        n_qubits = op.n_qubits()

    if n_qubits < op.n_qubits():
        raise ValueError('Invalid n_qubits.')

    if isinstance(op, FermionOperator) and len(op.terms) == 1:
        return bravyi_kitaev_term(op, n_qubits)
    transformed_operator = QubitOperator((), 0.0)
    for term in op.terms:
        transformed_operator += bravyi_kitaev_term(
            FermionOperator(term, op.terms[term]), n_qubits)
    return transformed_operator
