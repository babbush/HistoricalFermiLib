"""Transformations acting on operators and RDMs."""
from __future__ import absolute_import

import copy
import itertools
import numpy

from fermilib import interaction_rdms
from fermilib.fenwick_tree import FenwickTree
from fermilib.fermion_operators import (FermionTerm, FermionOperator,
                                        fermion_identity, number_operator)
from fermilib.interaction_operators import (InteractionOperator,
                                            InteractionOperatorError)
from fermilib.qubit_operators import (QubitTerm, QubitTermError,
                                      QubitOperator, QubitOperatorError)
from fermilib.sparse_operators import (jordan_wigner_term_sparse,
                                       jordan_wigner_operator_sparse)


def eigenspectrum(op):
    return op.get_sparse_operator().eigenspectrum()


def reverse_jordan_wigner_term(term, n_qubits=None):
    """Transforms a QubitTerm into an instance of FermionOperator using JW.

    Operators are mapped as follows:
    Z_j -> I - 2 a^\dagger_j a_j
    X_j -> (a^\dagger_j + a_j) Z_{j-1} Z_{j-2} .. Z_0
    Y_j -> i (a^\dagger_j - a_j) Z_{j-1} Z_{j-2} .. Z_0

    Args:
      term: the QubitTerm to be transformed.
      n_qubits: the number of qubits term acts on. If not set, defaults
                to the maximum qubit number acted on by term.

    Returns:
      transformed_term: An instance of the FermionOperator class.

    Raises:
      QubitTermError: Invalid operator provided: must be 'X', 'Y' or 'Z'.

    """
    if not isinstance(term, QubitTerm):
        raise TypeError("term must be a QubitTerm.")

    if n_qubits is None:
        n_qubits = term.n_qubits()
    if n_qubits == 0:
        raise QubitTermError('Invalid n_qubits.')
    if n_qubits < term.n_qubits():
        n_qubits = term.n_qubits()

    # Initialize transformed operator.
    identity = fermion_identity()
    transformed_term = FermionOperator(identity)
    working_term = QubitTerm(term.operators, 1.0)

    # Loop through operators.
    if working_term.operators:
        operator = working_term.operators[-1]
        while operator is not None:

            # Handle Pauli Z.
            if operator[1] == 'Z':
                no = number_operator(n_qubits, operator[0], -2.)
                transformed_operator = identity + no

            else:
                raising_term = FermionTerm([(operator[0], 1)])
                lowering_term = FermionTerm([(operator[0], 0)])

                # Handle Pauli X, Y, Z.
                if operator[1] == 'Y':
                    raising_term *= 1j
                    lowering_term *= -1j

                elif operator[1] != 'X':
                    # Raise for invalid operator.
                    raise QubitTermError('Invalid operator provided: '
                                         "must be 'X', 'Y' or 'Z'")

                # Account for the phase terms.
                for j in reversed(range(operator[0])):
                    z_term = QubitTerm(coefficient=1.0,
                                       operators=[(j, 'Z')])
                    z_term *= working_term
                    working_term = copy.deepcopy(z_term)
                transformed_operator = FermionOperator([raising_term,
                                                        lowering_term])
                transformed_operator *= working_term.coefficient
                working_term.coefficient = 1.0

            # Get next non-identity operator acting below the
            # 'working_qubit'.
            working_qubit = operator[0] - 1
            for working_operator in working_term[::-1]:
                if working_operator[0] <= working_qubit:
                    operator = working_operator
                    break
                else:
                    operator = None

            # Multiply term by transformed operator.
            transformed_term *= transformed_operator

    # Account for overall coefficient
    transformed_term *= term.coefficient

    return transformed_term


def reverse_jordan_wigner(op, n_qubits=None):
    """Transforms a QubitTerm or QubitOperator into an instance of
    FermionOperator using the Jordan-Wigner transform.

    Operators are mapped as follows:
    Z_j -> I - 2 a^\dagger_j a_j
    X_j -> (a^\dagger_j + a_j) Z_{j-1} Z_{j-2} .. Z_0
    Y_j -> i (a^\dagger_j - a_j) Z_{j-1} Z_{j-2} .. Z_0

    Args:
      op: the QubitOperator to be transformed.
      n_qubits: the number of qubits term acts on. If not set, defaults
                to the maximum qubit number acted on by term.

    Returns:
      transformed_term: An instance of the FermionOperator class.

    Raises:
      QubitTermError: Invalid operator provided: must be 'X', 'Y' or 'Z'.

    """
    if isinstance(op, QubitTerm):
        op = QubitOperator(op)
    if not isinstance(op, QubitOperator):
        raise TypeError("op must be a QubitOperator.")

    if n_qubits is None:
        n_qubits = op.n_qubits()
    if n_qubits == 0:
        raise QubitTermError('Invalid n_qubits.')
    if n_qubits < op.n_qubits():
        n_qubits = op.n_qubits()
    transformed_operator = FermionOperator()
    for term in op:
        transformed_operator += reverse_jordan_wigner_term(term, n_qubits)
    return transformed_operator


def get_sparse_operator_term(term, n_qubits=None):
    """Map the QubitTerm to a SparseOperator instance."""
    if n_qubits is None:
        n_qubits = term.n_qubits()
    if n_qubits == 0:
        raise QubitTermError('Invalid n_qubits.')
    if n_qubits < term.n_qubits():
        n_qubits = term.n_qubits()
    return qubit_term_sparse(term, n_qubits)


def get_sparse_operator(op, n_qubits=None):
    """Map the QubitOperator to a SparseOperator instance."""
    if n_qubits is None:
        n_qubits = op.n_qubits()
    if n_qubits == 0:
        raise QubitTermError('Invalid n_qubits.')
    if n_qubits < op.n_qubits():
        n_qubits = op.n_qubits()
    return qubit_operator_sparse(op, n_qubits)


def bravyi_kitaev_transform_term(term, n_qubits=None):
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
    if not isinstance(term, FermionTerm):
        raise TypeError("term must be a FermionTerm.")

    if n_qubits is None:
        n_qubits = term.n_qubits()
    if not n_qubits or n_qubits < term.n_qubits():
        raise ValueError('Invalid n_qubits.')

    # Build the Fenwick Tree
    fenwick_tree = FenwickTree(n_qubits)

    # Initialize identity matrix.
    transformed_term = QubitOperator([QubitTerm([], term.coefficient)])

    # Build the Bravyi-Kitaev transformed operators.
    for operator in term:
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


def jordan_wigner_transform_term(term):
    """Jordan-Wigner transform term and return the resulting qubit operator.

    Returns:
      transformed_term: An instance of the QubitOperator class.

    Warning:
      The runtime of this method is exponential in the locality of the
      original FermionTerm.

    """
    # Initialize identity matrix.
    transformed_term = QubitOperator([QubitTerm([], term.coefficient)])

    # Loop through operators, transform and multiply.
    for operator in term:
        z_factors = [(index, 'Z') for index in range(0, operator[0])]

        # Handle identity.
        pauli_x_component = QubitTerm(z_factors + [(operator[0], 'X')], 0.5)
        if operator[1]:
            pauli_y_component = QubitTerm(z_factors +
                                          [(operator[0], 'Y')], -0.5j)
        else:
            pauli_y_component = QubitTerm(z_factors +
                                          [(operator[0], 'Y')], 0.5j)

        transformed_term *= QubitOperator([pauli_x_component,
                                           pauli_y_component])
    return transformed_term


def jordan_wigner_transform_interaction_op(iop):
    """Output InteractionOperator as QubitOperator class under JW
    transform.

    One could accomplish this very easily by first mapping to fermions and
    then mapping to qubits. We skip the middle step for the sake of speed.

    Returns:
      qubit_operator: An instance of the QubitOperator class.

    """
    # Initialize qubit operator.
    qubit_operator = QubitOperator()

    # Add constant.
    qubit_operator += QubitTerm([], iop.constant)

    # Loop through all indices.
    for p in range(iop.n_qubits):
        for q in range(iop.n_qubits):

            # Handle one-body terms.
            coefficient = float(iop[p, q])
            if coefficient and p >= q:
                qubit_operator += coefficient * \
                    iop.jordan_wigner_one_body(p, q)

            # Keep looping for the two-body terms.
            for r in range(iop.n_qubits):
                for s in range(iop.n_qubits):
                    coefficient = float(iop[p, q, r, s])

                    # Skip zero terms.
                    if (not coefficient) or (p == q) or (r == s):
                        continue

                    # Identify and skip one of the complex conjugates.
                    if [p, q, r, s] != [s, r, q, p]:
                        if len(set([p, q, r, s])) == 4:
                            if min(r, s) < min(p, q):
                                continue
                        else:
                            if q < p:
                                continue

                    # Handle the two-body terms.
                    transformed_term = iop.jordan_wigner_two_body(
                        p, q, r, s)
                    transformed_term *= coefficient
                    qubit_operator += transformed_term

    return qubit_operator


def bravyi_kitaev_transform(op, n_qubits=None):
    """Apply the Bravyi-Kitaev transform and return qubit operator.

    Returns:   transformed_operator: An instance of the
    QubitOperator class.

    """
    if n_qubits is None:
        n_qubits = op.n_qubits()
    if not n_qubits or n_qubits < op.n_qubits():
        raise ValueError('Invalid n_qubits.')
    if isinstance(op, FermionTerm):
        return bravyi_kitaev_transform_term(op, n_qubits)
    transformed_operator = QubitOperator()
    for term in op:
        transformed_operator += bravyi_kitaev_transform_term(term, n_qubits)
    return transformed_operator


def jordan_wigner_transform(op):
    """Apply the Jordan-Wigner transform the FermionOperator op and
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
        return jordan_wigner_transform_interaction_op(op)
    if not isinstance(op, FermionOperator):
        raise TypeError("op must be a QubitTerm or QubitOperator.")

    transformed_operator = QubitOperator()
    for term in op:
        transformed_operator += jordan_wigner_transform_term(term)
    return transformed_operator


def jordan_wigner_sparse(op, n_qubits=None):
    """Return a sparse matrix representation of the JW transformed term."""
    if n_qubits is None:
        n_qubits = op.n_qubits()
    if n_qubits == 0:
        raise ValueError('Invalid n_qubits.')
    if n_qubits < op.n_qubits():
        n_qubits = op.n_qubits()

    if isinstance(op, FermionTerm):
        return jordan_wigner_term_sparse(op, n_qubits)
    elif isinstance(op, FermionOperator):
        return jordan_wigner_operator_sparse(op, n_qubits)

    raise TypeError("op should be either a FermionTerm or FermionOperator.")


def get_interaction_rdm(self, n_qubits=None):
    """Build a InteractionRDM from measured qubit operators.

    Returns: A InteractionRDM object.

    """
    if n_qubits is None:
        n_qubits = self.n_qubits()
    if n_qubits == 0:
        raise QubitTermError('Invalid n_qubits.')
    if n_qubits < self.n_qubits():
        n_qubits = self.n_qubits()
    one_rdm = numpy.zeros((n_qubits,) * 2, dtype=complex)
    two_rdm = numpy.zeros((n_qubits,) * 4, dtype=complex)

    # One-RDM.
    for i, j in itertools.product(range(n_qubits), repeat=2):
        transformed_operator = jordan_wigner_transform(FermionTerm([(i, 1),
                                                                    (j, 0)]))
        for term in transformed_operator:
            if tuple(term.operators) in self.terms:
                one_rdm[i, j] += term.coefficient * self[term.operators]

    # Two-RDM.
    for i, j, k, l in itertools.product(range(n_qubits), repeat=4):
        transformed_operator = jordan_wigner_transform(
            FermionTerm([(i, 1), (j, 1), (k, 0), (l, 0)]))
        for term in transformed_operator:
            if tuple(term.operators) in self.terms:
                two_rdm[i, j, k, l] += (term.coefficient *
                                        self[term.operators])

    return interaction_rdms.InteractionRDM(one_rdm, two_rdm)


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
    # Import here to avoid circular dependency.
    # from fermilib import interaction_operators

    # Normal order the terms and initialize.
    iop.normal_order()
    constant = 0.
    one_body = numpy.zeros((iop.n_qubits(), iop.n_qubits()), complex)
    two_body = numpy.zeros((
        iop.n_qubits(), iop.n_qubits(), iop.n_qubits(),
        iop.n_qubits()),
        complex)

    # Loop through terms and assign to matrix.
    for term in iop:
        coefficient = term.coefficient

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
