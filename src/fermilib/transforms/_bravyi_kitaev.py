"""Bravyi-Kitaev transform on fermionic operators."""
from __future__ import absolute_import

from fermilib.qubit_operators import QubitOperator, QubitTerm
from projectqtemp.ops._fermion_operator import FermionOperator
from fermilib.fenwick_tree import FenwickTree


def bravyi_kitaev_term(term, n_qubits=None):
    """Apply the Bravyi-Kitaev transform and return qubit operator.

    Returns:
      transformed_term: An instance of the QubitOperator class.

    Warning:
      Likely greedy. At the moment the method gets the node sets for
      each fermionic operator. FenwickNodes are not neccessary in this
      construction, only the indices matter here. This may be optimized
      by removing the unnecessary structure.

    Note:
      Reference: Operator Locality of Quantum Simulation of Fermionic
        Models by Havlicek, Troyer, Whitfield (arXiv:1701.07072).
    """
    if not isinstance(term, FermionOperator) or len(term.terms) > 1:
        raise ValueError("term must be a single-term FermionOperator.")

    if n_qubits is None:
        n_qubits = term.n_qubits()
    if not n_qubits or n_qubits < term.n_qubits():
        raise ValueError('Invalid n_qubits.')

    ops = list(term.terms)[0]
    coeff = term.terms[ops]

    # Build the Fenwick Tree
    fenwick_tree = FenwickTree(n_qubits)

    # Initialize identity matrix.
    transformed_term = QubitOperator([QubitTerm([], coeff)])

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
        d_majorana_component = QubitTerm(
            ([(operator[0], 'Y')] +
             [(index, 'Z') for index in ancestor_children] +
             [(index, 'X') for index in ancestors]),
            d_coeff)

        c_majorana_component = QubitTerm(
            ([(operator[0], 'X')] +
             [(index, 'Z') for index in parity_set] +
             [(index, 'X') for index in ancestors]),
            0.5)

        transformed_term *= QubitOperator(
            [c_majorana_component, d_majorana_component])

    return transformed_term


def bravyi_kitaev(op, n_qubits=None):
    """Apply the Bravyi-Kitaev transform and return qubit operator.

    Returns:
         transformed_operator: An instance of the
         QubitOperator class.

    """
    if n_qubits is None:
        n_qubits = op.n_qubits()
    if not n_qubits or n_qubits < op.n_qubits():
        raise ValueError('Invalid n_qubits.')
    if isinstance(op, FermionOperator) and len(op.terms) == 1:
        return bravyi_kitaev_term(op, n_qubits)
    transformed_operator = QubitOperator()
    for term in op.terms:
        transformed_operator += bravyi_kitaev_term(
            FermionOperator(term, op.terms[term]), n_qubits)
    return transformed_operator
