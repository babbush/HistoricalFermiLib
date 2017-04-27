"""Transformations acting on operators and RDMs."""
from __future__ import absolute_import

import copy
import itertools
import numpy

from fermilib.transforms import FenwickTree, jordan_wigner_sparse
from fermilib.ops import (FermionOperator,
                          normal_ordered,
                          number_operator,
                          InteractionOperator,
                          InteractionRDM)

from fermilib.ops._interaction_operator import InteractionOperatorError
from fermilib.ops._sparse_operator import (qubit_term_sparse,
                                           qubit_operator_sparse)
from projectqtemp.ops._qubit_operator import QubitOperator, QubitOperatorError


def get_eigenspectrum(op):
    return get_sparse_operator(op).eigenspectrum()


def get_sparse_operator_term(term, n_qubits=None):
    """Map a single-term QubitOperator to a SparseOperator."""
    if n_qubits is None:
        n_qubits = term.n_qubits()
    if n_qubits == 0:
        raise QubitOperatorError('Invalid n_qubits.')
    if n_qubits < term.n_qubits():
        n_qubits = term.n_qubits()
    return qubit_term_sparse(term, n_qubits)


def get_sparse_operator(op, n_qubits=None):
    """Map a Fermion, Qubit, or InteractionOperator to a SparseOperator."""
    if isinstance(op, InteractionOperator):
        return get_sparse_interaction_operator(op)
    elif isinstance(op, FermionOperator):
        return jordan_wigner_sparse(op, n_qubits)
    elif not isinstance(op, QubitOperator):
        raise TypeError("op must be mappable to SparseOperator.")

    if n_qubits is None:
        n_qubits = op.n_qubits()
    if n_qubits == 0:
        raise QubitOperatorError('Invalid n_qubits.')
    if n_qubits < op.n_qubits():
        n_qubits = op.n_qubits()
    return qubit_operator_sparse(op, n_qubits)


def get_sparse_interaction_operator(iop):
    # TODO: Replace with much faster "direct" routine.
    fermion_operator = get_fermion_operator(iop)
    sparse_operator = jordan_wigner_sparse(fermion_operator)
    return sparse_operator


def get_interaction_rdm(qubit_operator, n_qubits=None):
    """Build a InteractionRDM from measured qubit operators.

    Returns: A InteractionRDM object.
    """
    # to avoid circular import
    from fermilib.transforms import jordan_wigner

    if n_qubits is None:
        n_qubits = qubit_operator.n_qubits()
    if n_qubits == 0:
        raise QubitOperatorError('Invalid n_qubits.')
    if n_qubits < qubit_operator.n_qubits():
        n_qubits = qubit_operator.n_qubits()
    one_rdm = numpy.zeros((n_qubits,) * 2, dtype=complex)
    two_rdm = numpy.zeros((n_qubits,) * 4, dtype=complex)

    # One-RDM.
    for i, j in itertools.product(range(n_qubits), repeat=2):
        transformed_operator = jordan_wigner(FermionOperator(((i, 1), (j, 0))))
        for term, coefficient in transformed_operator.terms.iteritems():
            if term in qubit_operator.terms:
                one_rdm[i, j] += coefficient * qubit_operator.terms[term]

    # Two-RDM.
    for i, j, k, l in itertools.product(range(n_qubits), repeat=4):
        transformed_operator = jordan_wigner(FermionOperator(((i, 1), (j, 1),
                                                              (k, 0), (l, 0))))
        for term, coefficient in transformed_operator.terms.iteritems():
            if term in qubit_operator.terms:
                two_rdm[i, j, k, l] += coefficient * qubit_operator.terms[term]

    return InteractionRDM(one_rdm, two_rdm)


def get_interaction_operator(iop):
    """Convert a 2-body fermionic operator to instance of
    InteractionOperator.

    This function should only be called on fermionic operators which
    consist of only a_p^\dagger a_q and a_p^\dagger a_q^\dagger a_r a_s
    terms. The one-body terms are stored in a matrix, one_body[p, q], and
    the two-body terms are stored in a tensor, two_body[p, q, r, s].

    Returns:
       interaction_operator: An instance of the InteractionOperator class.

    Raises:
        ErrorInteractionOperator: FermionOperator is not a molecular
            operator.

    Warning:
        Even assuming that each creation or annihilation operator appears
        at most a constant number of times in the original operator, the
        runtime of this method is exponential in the number of qubits.

    """
    # Normal order the terms and initialize.
    iop = normal_ordered(iop)
    constant = 0.
    one_body = numpy.zeros((iop.n_qubits(), iop.n_qubits()), complex)
    two_body = numpy.zeros((iop.n_qubits(), iop.n_qubits(),
                            iop.n_qubits(), iop.n_qubits()), complex)

    # Loop through terms and assign to matrix.
    for term in iop.terms:
        coefficient = iop.terms[term]

        # Handle constant shift.
        if len(term) == 0:
            constant = coefficient

        elif len(term) == 2:
            # Handle one-body terms.
            if [operator[1] for operator in term] == [1, 0]:
                p, q = [operator[0] for operator in term]
                one_body[p, q] = coefficient
            else:
                raise InteractionOperatorError('FermionOperator is not a '
                                               'molecular operator.')

        elif len(term) == 4:
            # Handle two-body terms.
            if [operator[1] for operator in term] == [1, 1, 0, 0]:
                p, q, r, s = [operator[0] for operator in term]
                two_body[p, q, r, s] = coefficient
            else:
                raise InteractionOperatorError('FermionOperator is not '
                                               'a molecular operator.')

        else:
            # Handle non-molecular Hamiltonian.
            raise InteractionOperatorError('FermionOperator is not '
                                           'a molecular operator.')

    # Form InteractionOperator and return.
    interaction_operator = InteractionOperator(constant, one_body, two_body)
    return interaction_operator


def get_fermion_operator(iop):
    """
    Output InteractionOperator as an instance of FermionOperator class.

    Returns:
        fermion_operator: An instance of the FermionOperator class.
    """
    # Initialize with identity term.
    fermion_operator = iop.constant * FermionOperator()

    # Add one-body terms.
    for p in range(iop.n_qubits):
        for q in range(iop.n_qubits):
            coefficient = iop[p, q]
            fermion_operator += FermionOperator(((p, 1), (q, 0)), coefficient)

            # Add two-body terms.
            for r in range(iop.n_qubits):
                for s in range(iop.n_qubits):
                    coefficient = iop[p, q, r, s]
                    fermion_operator += FermionOperator(((p, 1), (q, 1),
                                                         (r, 0), (s, 0)),
                                                        coefficient)

    return fermion_operator
