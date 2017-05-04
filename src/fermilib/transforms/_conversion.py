"""Transformations acting on operators and RDMs."""
from __future__ import absolute_import
from future.utils import iteritems

import itertools
import numpy
import copy

from projectqtemp.ops._qubit_operator import QubitOperator, QubitOperatorError

from fermilib.ops import (FermionOperator,
                          normal_ordered,
                          number_operator,
                          InteractionOperator,
                          InteractionRDM)
from fermilib.transforms import FenwickTree
from fermilib.ops._interaction_operator import InteractionOperatorError
from fermilib.ops._sparse_operator import (qubit_term_sparse,
                                           qubit_operator_sparse,
                                           jordan_wigner_operator_sparse)
from fermilib.utils import count_qubits


def get_sparse_operator(operator, n_qubits=None):
    """Map a Fermion, Qubit, or InteractionOperator to a SparseOperator."""
    if isinstance(operator, InteractionOperator):
        return get_sparse_interaction_operator(operator)
    elif isinstance(operator, FermionOperator):
        return jordan_wigner_operator_sparse(operator)
    elif isinstance(operator, QubitOperator):
        if n_qubits is None:
            n_qubits = count_qubits(operator)
        return qubit_operator_sparse(operator, n_qubits)


def get_sparse_interaction_operator(iop):
    # TODO: Replace with much faster "direct" routine.
    fermion_operator = get_fermion_operator(iop)
    sparse_operator = jordan_wigner_operator_sparse(fermion_operator)
    return sparse_operator


def get_interaction_rdm(qubit_operator, n_qubits=None):
    """Build a InteractionRDM from measured qubit operators.

    Returns: A InteractionRDM object.
    """
    # to avoid circular import
    from fermilib.transforms import jordan_wigner

    if n_qubits is None:
        n_qubits = count_qubits(qubit_operator)
    if n_qubits == 0:
        raise QubitOperatorError('Invalid n_qubits.')
    if n_qubits < count_qubits(qubit_operator):
        n_qubits = count_qubits(qubit_operator)
    one_rdm = numpy.zeros((n_qubits,) * 2, dtype=complex)
    two_rdm = numpy.zeros((n_qubits,) * 4, dtype=complex)

    # One-RDM.
    for i, j in itertools.product(range(n_qubits), repeat=2):
        transformed_operator = jordan_wigner(FermionOperator(((i, 1), (j, 0))))
        for term, coefficient in iteritems(transformed_operator.terms):
            if term in qubit_operator.terms:
                one_rdm[i, j] += coefficient * qubit_operator.terms[term]

    # Two-RDM.
    for i, j, k, l in itertools.product(range(n_qubits), repeat=4):
        transformed_operator = jordan_wigner(FermionOperator(((i, 1), (j, 1),
                                                              (k, 0), (l, 0))))
        for term, coefficient in iteritems(transformed_operator.terms):
            if term in qubit_operator.terms:
                two_rdm[i, j, k, l] += coefficient * qubit_operator.terms[term]

    return InteractionRDM(one_rdm, two_rdm)


def get_interaction_operator(fermion_operator, n_qubits=None):
    """Convert a 2-body fermionic operator to instance of
    InteractionOperator.

    This function should only be called on fermionic operators which
    consist of only a_p^\dagger a_q and a_p^\dagger a_q^\dagger a_r a_s
    terms. The one-body terms are stored in a matrix, one_body[p, q], and
    the two-body terms are stored in a tensor, two_body[p, q, r, s].

    Returns:
       interaction_operator: An instance of the InteractionOperator class.

    Raises:
        InteractionOperatorError: FermionOperator is not a molecular
                                  operator.

    Warning:
        Even assuming that each creation or annihilation operator appears
        at most a constant number of times in the original operator, the
        runtime of this method is exponential in the number of qubits.

    """
    if not isinstance(fermion_operator, FermionOperator):
        raise TypeError('fermion_operator must be a FermionOperator.')

    if n_qubits is None:
        n_qubits = count_qubits(fermion_operator)
    if n_qubits < count_qubits(fermion_operator):
        n_qubits = count_qubits(fermion_operator)

    # Normal order the terms and initialize.
    fermion_operator = normal_ordered(fermion_operator)
    constant = 0.
    one_body = numpy.zeros((n_qubits, n_qubits), complex)
    two_body = numpy.zeros((n_qubits, n_qubits,
                            n_qubits, n_qubits), complex)

    # Loop through terms and assign to matrix.
    for term in fermion_operator.terms:
        coefficient = fermion_operator.terms[term]

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


def get_fermion_operator(interaction_operator):
    """
    Output InteractionOperator as an instance of FermionOperator class.

    Returns:
        fermion_operator: An instance of the FermionOperator class.
    """
    # Initialize with identity term.
    fermion_operator = FermionOperator((), interaction_operator.constant)

    # Add one-body terms.
    for p in range(count_qubits(interaction_operator)):
        for q in range(count_qubits(interaction_operator)):
            coefficient = interaction_operator[p, q]
            fermion_operator += FermionOperator(((p, 1), (q, 0)), coefficient)

            # Add two-body terms.
            for r in range(count_qubits(interaction_operator)):
                for s in range(count_qubits(interaction_operator)):
                    coefficient = interaction_operator[p, q, r, s]
    for p in range(interaction_operator.n_qubits):
        for q in range(interaction_operator.n_qubits):
            coefficient = interaction_operator[p, q]
            fermion_operator += FermionOperator(((p, 1), (q, 0)), coefficient)

            # Add two-body terms.
            for r in range(interaction_operator.n_qubits):
                for s in range(interaction_operator.n_qubits):
                    coefficient = interaction_operator[p, q, r, s]
                    fermion_operator += FermionOperator(((p, 1), (q, 1),
                                                         (r, 0), (s, 0)),
                                                        coefficient)

    return fermion_operator
