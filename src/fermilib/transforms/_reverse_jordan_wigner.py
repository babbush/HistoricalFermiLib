"""Reverse Jordan-Wigner transform on QubitOperators."""
from __future__ import absolute_import
import copy

from fermilib.ops import (FermionOperator,
                          number_operator)
from projectqtemp.ops._qubit_operator import QubitOperator, QubitOperatorError


class ReverseJordanWignerError(Exception):
    pass


def reverse_jordan_wigner(qubit_operator, n_qubits=None):
    """
    Transforms a QubitOperator into a FermionOperator using the
    Jordan-Wigner transform.

    Operators are mapped as follows:
    Z_j -> I - 2 a^\dagger_j a_j
    X_j -> (a^\dagger_j + a_j) Z_{j-1} Z_{j-2} .. Z_0
    Y_j -> i (a^\dagger_j - a_j) Z_{j-1} Z_{j-2} .. Z_0

    Args:
        qubit_operator: the QubitOperator to be transformed.
        n_qubits: the number of qubits term acts on. If not set, defaults
                to the maximum qubit number acted on by term.

    Returns:
        transformed_term: An instance of the FermionOperator class.

    Raises:
        TypeError: Input must be a QubitOperator.
        QubitOperatorError: Invalid number of qubits specified.
        QubitOperatorError: Pauli operators must be X, Y or Z.
    """
    if not isinstance(qubit_operator, QubitOperator):
        raise TypeError('Input must be a QubitOperator.')
    if n_qubits is None:
        n_qubits = qubit_operator.n_qubits()
    if n_qubits < qubit_operator.n_qubits():
        raise QubitOperatorError(
            'Invalid number of qubits specified')

    # Loop through terms.
    transformed_operator = FermionOperator()
    for term in qubit_operator.terms:
        transformed_term = FermionOperator(())
        if term:
            working_term = QubitOperator(term)
            pauli_operator = term[-1]
            while pauli_operator is not None:

                # Handle Pauli Z.
                if pauli_operator[1] == 'Z':
                    transformed_pauli = FermionOperator(
                        ()) + number_operator(n_qubits, pauli_operator[0], -2.)

                # Handle Pauli X and Y.
                else:
                    raising_term = FermionOperator(((pauli_operator[0], 1),))
                    lowering_term = FermionOperator(((pauli_operator[0], 0),))
                    if pauli_operator[1] == 'Y':
                        raising_term *= 1.j
                        lowering_term *= -1.j
                    elif pauli_operator[1] != 'X':
                        raise QubitOperatorError(
                            'Pauli operators must be X, Y, or Z')
                    transformed_pauli = raising_term + lowering_term

                    # Account for the phase terms.
                    for j in reversed(range(pauli_operator[0])):
                        z_term = QubitOperator(((j, 'Z'),))
                        working_term = z_term * working_term
                    transformed_pauli *= working_term.terms.values()[0]
                    working_term.terms[list(working_term.terms)[0]] = 1.

                # Get next non-identity operator acting below 'working_qubit'.
                assert len(working_term.terms) == 1
                working_qubit = pauli_operator[0] - 1
                for working_operator in reversed(list(working_term.terms)[0]):
                    if working_operator[0] <= working_qubit:
                        pauli_operator = working_operator
                        break
                    else:
                        pauli_operator = None

                # Multiply term by transformed operator.
                transformed_term *= transformed_pauli

        # Account for overall coefficient
        transformed_term *= qubit_operator.terms[term]
        transformed_operator += transformed_term
    return transformed_operator
