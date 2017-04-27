"""Reverse Jordan-Wigner transform on QubitOperators."""
from __future__ import absolute_import
import copy

from fermilib.ops import (FermionOperator,
                          number_operator)
from projectqtemp.ops._qubit_operator import QubitOperator, QubitOperatorError


def reverse_jordan_wigner_term(term, n_qubits=None):
    """
    Transforms a single-term QubitOperator into a FermionOperator
    using JW.

    Operators are mapped as follows:
    Z_j -> I - 2 a^\dagger_j a_j
    X_j -> (a^\dagger_j + a_j) Z_{j-1} Z_{j-2} .. Z_0
    Y_j -> i (a^\dagger_j - a_j) Z_{j-1} Z_{j-2} .. Z_0

    Args:
        term: the QubitOperator to be transformed.
        n_qubits: the number of qubits term acts on. If not set, defaults
                  to the maximum qubit number acted on by term.

    Returns:
        transformed_term: An instance of the FermionOperator class.

    Raises:
        QubitOperatorError: Invalid operator provided: must be 'X', 'Y'
                            or 'Z'.

    """
    if not isinstance(term, QubitOperator) or len(term.terms) != 1:
        raise TypeError("term must be a single-term QubitOperator.")

    if n_qubits is None:
        n_qubits = term.n_qubits()

    if n_qubits < term.n_qubits():
        n_qubits = term.n_qubits()

    # Initialize transformed operator.
    transformed_term = FermionOperator()
    working_ops = list(term.terms)[0]
    working_term = QubitOperator(working_ops, 1.0)
    original_coeff = term.terms[working_ops]

    # Loop through operators.
    if working_term.terms:
        ops = list(working_term.terms)[0]
        operator = ops[-1] if ops else None
        while operator is not None:

            # Handle Pauli Z.
            if operator[1] == 'Z':
                no = number_operator(n_qubits, operator[0], -2.)
                transformed_operator = FermionOperator() + no

            else:
                raising_term = FermionOperator(((operator[0], 1),))
                lowering_term = FermionOperator(((operator[0], 0),))

                # Handle Pauli X, Y, Z.
                if operator[1] == 'Y':
                    raising_term *= 1j
                    lowering_term *= -1j

                # Account for the phase terms.
                for j in reversed(range(operator[0])):
                    z_term = QubitOperator(((j, 'Z'),))
                    z_term *= working_term
                    working_term = copy.deepcopy(z_term)
                transformed_operator = raising_term + lowering_term
                working_term_ops = list(working_term.terms)[0]
                working_term_coeff = working_term.terms[working_term_ops]
                transformed_operator *= working_term_coeff
                working_term.terms[working_term_ops] = 1.0

            # Get next non-identity operator acting below the
            # 'working_qubit'.
            working_qubit = operator[0] - 1
            for working_operator in list(working_term.terms)[0][::-1]:
                if working_operator[0] <= working_qubit:
                    operator = working_operator
                    break
                else:
                    operator = None

            # Multiply term by transformed operator.
            transformed_term *= transformed_operator

    # Account for overall coefficient
    transformed_term *= original_coeff

    return transformed_term


def reverse_jordan_wigner(op, n_qubits=None):
    """
    Transforms a QubitOperator into a FermionOperator using the
    Jordan-Wigner transform.

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
        QubitOperatorError: Invalid operator provided: must be 'X', 'Y'
                            or 'Z'.
    """
    if not isinstance(op, QubitOperator):
        raise TypeError("op must be a QubitOperator.")

    if n_qubits is None:
        n_qubits = op.n_qubits()
    if n_qubits < op.n_qubits():
        n_qubits = op.n_qubits()
    transformed_operator = FermionOperator((), 0.0)
    for term in op.terms:
        single_term = QubitOperator(term, op.terms[term])
        transformed_operator += reverse_jordan_wigner_term(single_term,
                                                           n_qubits)
    return transformed_operator
