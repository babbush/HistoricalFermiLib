"""This files has utilities to read and store qubit Hamiltonians."""
from __future__ import absolute_import

import copy
import itertools

import numpy

from fermilib.local_operators import LocalOperator
from fermilib.local_terms import LocalTerm
from fermilib.sparse_operators import qubit_term_sparse, qubit_operator_sparse


class QubitTermError(Exception):
    pass


class QubitOperatorError(Exception):
    pass


# Define products of all Pauli matrices for symbolic multiplication.
_PAULI_MATRIX_PRODUCTS = {('I', 'I'): (1., 'I'),
                          ('I', 'X'): (1., 'X'),
                          ('X', 'I'): (1., 'X'),
                          ('I', 'Y'): (1., 'Y'),
                          ('Y', 'I'): (1., 'Y'),
                          ('I', 'Z'): (1., 'Z'),
                          ('Z', 'I'): (1., 'Z'),
                          ('X', 'X'): (1., 'I'),
                          ('Y', 'Y'): (1., 'I'),
                          ('Z', 'Z'): (1., 'I'),
                          ('X', 'Y'): (1.j, 'Z'),
                          ('X', 'Z'): (-1.j, 'Y'),
                          ('Y', 'X'): (-1.j, 'Z'),
                          ('Y', 'Z'): (1.j, 'X'),
                          ('Z', 'X'): (1.j, 'Y'),
                          ('Z', 'Y'): (-1.j, 'X')}


def qubit_identity(coefficient=1.):
    return QubitTerm([], coefficient)


class QubitTerm(LocalTerm):
    """Single term of a hamiltonian for a system of spin 1/2 particles or
    qubits.

    A Hamiltonian of qubits can be written as a sum of QubitTerm objects.
    Suppose you have n_qubits = 5 qubits a term of the Hamiltonian could
    be coefficient * X1 Z3 which we call a QubitTerm object. It means
    coefficient * (1 x PauliX x 1 x PauliZ x 1), where x is the tensor
    product, 1 the identity matrix, and the others are Pauli matrices. We
    only allow to apply one single Pauli Matrix to each qubit.

    Note: Always use the abstractions provided here to manipulate the
    .operators attribute. If ignoring this advice, an important thing to
    keep in mind is that the operators list is assumed to be sorted in order
    of the tensor factor on which the operator acts.

    Attributes:
      operators: A sorted list of tuples. The first element of each tuple is an
        int indicating the qubit on which operators acts. The second element
        of each tuple is a string, either 'X', 'Y' or 'Z', indicating what
        acts on that tensor factor. The list is sorted by the first index.
      coefficient: A real or complex floating point number.

    """

    def __init__(self, operators=None, coefficient=1.):
        """Inits PauliTerm.

        Specify to which qubits a Pauli X, Y, or Z is applied. To all not
        specified qubits (numbered 0, 1, ..., n_qubits-1) the identity is
        applied. Only one Pauli Matrix can be applied to each qubit.

        Args:
          coefficient: A real or complex floating point number.
          operators: A sorted list of tuples. The first element of each tuple
            is an int indicating the qubit on which operators acts, starting
            from zero. The second element of each tuple is a string, either
            'X', 'Y' or 'Z', indicating what acts on that tensor factor.
            operators can also be specified by a string of the form 'X0 Z2 Y5',
            indicating an X on qubit 0, Z on qubit 2, and Y on qubit 5.

        Raises:
          QubitTermError: Invalid operators provided to QubitTerm.

        """
        if operators is not None and not isinstance(operators,
                                                    (tuple, list, str)):
            raise ValueError('Operators specified incorrectly.')

        if isinstance(operators, str):
            list_ops = []
            for el in operators.split():
                if len(el) < 2:
                    raise ValueError('Operators specified incorrectly.')
                list_ops.append((int(el[1:]), el[0]))
            operators = list_ops
        super(QubitTerm, self).__init__(operators, coefficient)

        for operator in self:
            tensor_factor, action = operator
            if not isinstance(action, str) or action not in 'XYZ':
                raise ValueError("Invalid action provided: must be string "
                                 "'X', 'Y', or 'Z'.")
            if not (isinstance(tensor_factor, int) and tensor_factor >= 0):
                raise QubitTermError('Invalid tensor factor provided to '
                                     'QubitTerm: must be a non-negative int.')

        # Make sure operators are sorted by tensor factor.
        self.operators.sort(key=lambda operator: operator[0])

    def n_qubits(self):
        highest_qubit = 0
        for operator in self.operators:
            if operator[0] + 1 > highest_qubit:
                highest_qubit = operator[0] + 1
        return highest_qubit

    def __add__(self, addend):
        """Compute self + addend for a QubitTerm.

        Note that we only need to handle the case of adding other qubit terms.

        Args:
          addend: A QubitTerm.

        Returns:
          summand: A new instance of QubitOperator.

        Raises:
          TypeError: Object of invalid type cannot be added to QubitTerm.

        """
        if not issubclass(type(addend), (QubitTerm, QubitOperator)):
            raise TypeError('Cannot add term of invalid type to QubitTerm.')
        return QubitOperator([self]) + addend

    def __imul__(self, multiplier):
        """Multiply terms with scalar or QubitTerm using *=.

        Note that the "self" term is on the left of the multiply sign.

        Args:
          multiplier: Another QubitTerm object.

        """
        # Handle scalars.
        if (isinstance(multiplier, (int, float, complex)) or
                numpy.isscalar(multiplier)):
            self.coefficient *= multiplier
            return self

        # Handle QubitTerms.
        elif issubclass(type(multiplier), QubitTerm):
            # Relabel self * qubit_term as left_term * right_term.
            left_term = self
            right_term = multiplier
            self.coefficient *= multiplier.coefficient

            # Loop through terms and create new sorted list of operators.
            product_operators = []
            left_operator_index = 0
            right_operator_index = 0
            n_operators_left = len(left_term)
            n_operators_right = len(right_term)
            while (left_operator_index < n_operators_left and
                   right_operator_index < n_operators_right):
                (left_qubit, left_matrix) = left_term[left_operator_index]
                (right_qubit, right_matrix) = right_term[right_operator_index]

                # Multiply matrices if tensor factors are the same.
                if left_qubit == right_qubit:
                    left_operator_index += 1
                    right_operator_index += 1
                    (scalar, matrix) = _PAULI_MATRIX_PRODUCTS[(left_matrix,
                                                               right_matrix)]

                    # Add new term.
                    if matrix != 'I':
                        product_operators += [(left_qubit, matrix)]
                        self.coefficient *= scalar

                # If left_qubit > right_qubit, add right_matrix; else, add
                # left_matrix.
                elif left_qubit > right_qubit:
                    product_operators += [(right_qubit, right_matrix)]
                    right_operator_index += 1
                else:
                    product_operators += [(left_qubit, left_matrix)]
                    left_operator_index += 1

            # If either term_index exceeds the number of operators, finish.
            if left_operator_index == n_operators_left:
                product_operators += right_term[right_operator_index::]
            elif right_operator_index == n_operators_right:
                product_operators += left_term[left_operator_index::]

            # We should now have gone through all operators.
            self.operators = product_operators
            return self

    def __str__(self):
        """Return an easy-to-read string representation of the term."""
        string_representation = '{}'.format(self.coefficient)
        if self.operators == []:
            string_representation += ' I'
        for operator in self:
            if operator[1] == 'X':
                string_representation += ' X{}'.format(operator[0])
            elif operator[1] == 'Y':
                string_representation += ' Y{}'.format(operator[0])
            else:
                assert operator[1] == 'Z'
                string_representation += ' Z{}'.format(operator[0])
        return string_representation

    def get_sparse_operator(self, n_qubits=None):
        """Map the QubitTerm to a SparseOperator instance."""
        if n_qubits is None:
            n_qubits = self.n_qubits()
        if n_qubits == 0:
            raise QubitTermError('Invalid n_qubits.')
        if n_qubits < self.n_qubits():
            n_qubits = self.n_qubits()
        return qubit_term_sparse(self, n_qubits)


class QubitOperator(LocalOperator):
    """A collection of QubitTerm objects acting on same number of qubits.

    Note that to be a Hamiltonian which is a hermitian operator, the individual
    QubitTerm objects need to have only real valued coefficients.

    Attributes:
      terms: Dictionary of QubitTerm objects. The dictionary key is
          QubitTerm.key() and the dictionary value is the QubitTerm.

    """

    def __init__(self, terms=None):
        """Init a QubitOperator.

        Args:
          terms: An instance or dictionary or list of QubitTerm objects.

        Raises:
          QubitOperatorError: Invalid QubitTerms provided to QubitOperator.

        """
        super(QubitOperator, self).__init__(terms)

        for term in self:
            if not isinstance(term, QubitTerm):
                raise QubitTermError(
                    'Invalid QubitTerms provided to QubitOperator.')

    def __setitem__(self, operators, coefficient):
        if operators in self:
            self.terms[tuple(operators)].coefficient = coefficient
        else:
            new_term = QubitTerm(operators, coefficient)
            self.terms[tuple(operators)] = new_term

    def get_sparse_operator(self, n_qubits=None):
        if n_qubits is None:
            n_qubits = self.n_qubits()
        if n_qubits == 0:
            raise QubitTermError('Invalid n_qubits.')
        if n_qubits < self.n_qubits():
            n_qubits = self.n_qubits()
        return qubit_operator_sparse(self, n_qubits)

    def expectation(self, qubit_operator):
        """Take the expectation value of self with another qubit operator.

        Args:
          qubit_operator: An instance of the QubitOperator class.

        Returns:
          expectation: A float, giving the expectation value.

        """
        expectation = 0.
        for term in qubit_operator:
            if term.is_identity():
                expectation += term.coefficient + self[term.operators]
            else:
                expectation += term.coefficient * self[term.operators]
        return expectation
