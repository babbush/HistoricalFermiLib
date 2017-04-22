"""Reverse Jordan-Wigner transform on QubitOperators."""
from __future__ import absolute_import
import copy

from projectqtemp.ops._fermion_operator import (FermionOperator,
                                                fermion_identity,
                                                number_operator)
from fermilib.qubit_operators import (QubitTerm, QubitTermError, QubitOperator)


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
    transformed_term = fermion_identity()
    working_term = QubitTerm(term.operators, 1.0)

    # Loop through operators.
    if working_term.operators:
        operator = working_term.operators[-1]
        while operator is not None:

            # Handle Pauli Z.
            if operator[1] == 'Z':
                no = number_operator(n_qubits, operator[0], -2.)
                transformed_operator = fermion_identity() + no

            else:
                raising_term = FermionOperator(((operator[0], 1),))
                lowering_term = FermionOperator(((operator[0], 0),))

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
                transformed_operator = raising_term + lowering_term
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
    transformed_operator = FermionOperator((), 0.0)
    for term in op:
        transformed_operator += reverse_jordan_wigner_term(term, n_qubits)
    return transformed_operator
