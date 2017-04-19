"""Class to store and transform fermion operators."""
from __future__ import absolute_import

import copy
import numpy

from fermilib.fenwick_tree import FenwickTree
from fermilib.local_terms import LocalTerm, LocalTermError
from fermilib.local_operators import LocalOperator, LocalOperatorError


class FermionTermError(LocalTermError):
    pass


class FermionOperatorError(LocalOperatorError):
    pass


def fermion_identity(coefficient=1.):
    return FermionTerm([], coefficient)


def one_body_term(p, q, coefficient=1.):
    """Return one-body operator which conserves particle number.

    Args:   p, q: The sites between which the hopping occurs.
    coefficient: Optional float giving coefficient of term.

    """
    return FermionTerm([(p, 1), (q, 0)], coefficient)


def two_body_term(p, q, r, s, coefficient=1.):
    """Return two-body operator which conserves particle number.

    Args:   p, q, r, s: The sites between which the hopping occurs.
    coefficient: Optional float giving coefficient of term.

    """
    return FermionTerm([(p, 1), (q, 1), (r, 0), (s, 0)], coefficient)


def number_operator(n_qubits, site=None, coefficient=1.):
    """Return a number operator.

    Args:   n_qubits: An int giving the number of spin-orbitals in the
    system.   site: The site on which to return the number operator.
    If None, return total number operator on all sites.

    """
    if site is None:
        operator = FermionOperator()
        for spin_orbital in range(n_qubits):
            operator += number_operator(n_qubits, spin_orbital)
    else:
        operator = FermionTerm([(site, 1), (site, 0)], coefficient)
    return operator


class FermionTerm(LocalTerm):
    """Stores a single term composed of products of fermionic ladder operators.

    Attributes:
      operators: A list of tuples. The first element of each tuple is an
        int indicating the site on which operators acts. The second element
        of each tuple is boole, indicating whether raising (1) or lowering (0).
      coefficient: A complex valued float giving the term coefficient.

      Example usage:
        Consider the term 6.7 * a_3^\dagger a_1 a_7^\dagger
        This object would have the attributes:
        term.coefficient = 6.7
        term.operators = [(3, 1), (1, 0), (7, 1)]

    """

    def __init__(self, operators=None, coefficient=1.):
        """Init a FermionTerm.

        There are two ways to initialize the FermionTerm a^\dagger_2 a_7
        Way one is to provide the operators list, e.g. [(2, 1), (7, 0)]
        The other way is to provide a string '2^ 7'

        Args:
          operators: A list of tuples. The first element of each tuple is an
              int indicating the site on which operators acts. The second
              element of each tuple is an integer indicating raising (1) or
              lowering (0). Alternatively, a string can be provided.
          coefficient: A complex valued float giving the term coefficient.

        Raises:
          ValueError: Provided incorrect operator in list of operators.
          ValueError: Invalid action provided to FermionTerm. Must be 0
                      (lowering) or 1 (raising).

        """
        if operators is not None and \
                not isinstance(operators, (tuple, list, str)):
            raise ValueError('Operators specified incorrectly.')

        # Parse string input.
        if isinstance(operators, str):
            list_ops = []
            for el in operators.split():
                if el[-1] == '^':
                    list_ops.append((int(el[:-1]), 1))
                else:
                    try:
                        list_ops.append((int(el), 0))
                    except ValueError:
                        raise ValueError(
                            'Invalid action provided to FermionTerm.')
            operators = list_ops

        # Initialize.
        super(FermionTerm, self).__init__(operators, coefficient)

        # Check type.
        for operator in self:
            if not isinstance(operator, tuple):
                raise ValueError(
                    'Provided incorrect operator in list of operators.')
            tensor_factor, action = operator
            if not (isinstance(tensor_factor, int) and tensor_factor >= 0):
                raise ValueError('Invalid tensor factor provided to '
                                 'FermionTerm: must be a non-negative int.')
            if action not in (0, 1):
                raise ValueError('Invalid action provided to FermionTerm. '
                                 'Must be 0 (lowering) or 1 (raising).')

    def n_qubits(self):
        highest_qubit = 0
        for operator in self.operators:
            if operator[0] + 1 > highest_qubit:
                highest_qubit = operator[0] + 1
        return highest_qubit

    def __add__(self, addend):
        """Compute self + addend for a FermionTerm.

        Note that we only need to handle the case of adding other fermionic
        terms or operators.

        Args:
          addend: A FermionTerm or FermionOperator.

        Returns:
          summand: A new instance of FermionOperator.

        Raises:
          TypeError: Object of invalid type cannot be added to FermionTerm.

        """
        if not issubclass(type(addend), (FermionTerm, FermionOperator)):
            raise TypeError('Cannot add term of invalid type to FermionTerm.')

        return FermionOperator(self) + addend

    def __str__(self):
        """Return an easy-to-read string representation of the term."""
        string_representation = '{} ['.format(self.coefficient)
        for operator in self:
            string_representation += str(operator[0]) + '^' * operator[1] + ' '

        if self:
            string_representation = string_representation[:-1]
        string_representation += ']'
        return string_representation

    def hermitian_conjugate(self):
        """Hermitian conjugate this fermionic term."""
        self.coefficient = numpy.conjugate(self.coefficient)
        self.operators.reverse()
        for tensor_factor in range(len(self)):
            self[tensor_factor] = (self[tensor_factor][0],
                                   1 - self[tensor_factor][1])

    def hermitian_conjugated(self):
        """Calculate Hermitian conjugate of fermionic term.

        Returns:   A new FermionTerm object which is the hermitian
        conjugate of this.

        """
        res = copy.deepcopy(self)
        res.hermitian_conjugate()
        return res

    def is_normal_ordered(self):
        """Return whether or not term is in normal order.

        In our convention, normal ordering implies terms are ordered
        from highest tensor factor (on left) to lowest (on right). Also,
        ladder operators come first.

        """
        for i in range(1, len(self)):
            for j in range(i, 0, -1):
                right_operator = self[j]
                left_operator = self[j - 1]
                if right_operator[1] and not left_operator[1]:
                    return False
                elif (right_operator[1] == left_operator[1] and
                      right_operator[0] >= left_operator[0]):
                    return False
        return True

    def normal_ordered(self):
        """Compute and return the normal ordered form of a FermionTerm.

        Not an in-place method.

        In our convention, normal ordering implies terms are ordered
        from highest tensor factor (on left) to lowest (on right).
        Also, ladder operators come first.

        Returns:
          FermionOperator object in normal ordered form.

        Warning:
          Even assuming that each creation or annihilation operator appears
          at most a constant number of times in the original term, the
          runtime of this method is exponential in the number of qubits.

        """
        # Initialize output.
        normal_ordered_operator = FermionOperator()

        # Copy self.
        term = copy.deepcopy(self)

        # Iterate from left to right across operators and reorder to normal
        # form. Swap terms operators into correct position by moving left to
        # right.
        for i in range(1, len(term)):
            for j in range(i, 0, -1):
                right_operator = term[j]
                left_operator = term[j - 1]

                # Swap operators if raising on right and lowering on left.
                if right_operator[1] and not left_operator[1]:
                    term[j - 1] = right_operator
                    term[j] = left_operator
                    term *= -1.

                    # Replace a a^\dagger with 1 - a^\dagger a if indices are
                    # same.
                    if right_operator[0] == left_operator[0]:
                        operators_in_new_term = term[:(j - 1)]
                        operators_in_new_term += term[(j + 1)::]
                        new_term = FermionTerm(operators_in_new_term,
                                               -1. * term.coefficient)

                        # Recursively add the processed new term.
                        normal_ordered_operator += new_term.normal_ordered()

                    # Handle case when operator type is the same.
                elif right_operator[1] == left_operator[1]:

                    # If same two operators are repeated, term evaluates to
                    # zero.
                    if right_operator[0] == left_operator[0]:
                        return normal_ordered_operator

                        # Swap if same ladder type but lower index on left.
                    elif right_operator[0] > left_operator[0]:
                        term[j - 1] = right_operator
                        term[j] = left_operator
                        term *= -1.

        # Add processed term to output and return.
        normal_ordered_operator += term
        return normal_ordered_operator

    def is_molecular_term(self):
        """Query whether term has correct form to be from a molecular.

        Require that term is particle-number conserving (same number of
        raising and lowering operators). Require that term has 0, 2 or 4
        ladder operators. Require that term conserves spin (parity of
        raising operators equals parity of lowering operators).

        """
        if len(self.operators) not in (0, 2, 4):
            return False

        # Make sure term conserves particle number and spin.
        spin = 0
        particles = 0
        for operator in self:
            particles += (-1) ** operator[1]  # add 1 if create, else subtract
            spin += (-1) ** (operator[0] + operator[1])

        return particles == spin == 0


class FermionOperator(LocalOperator):
    """Data structure which stores sums of FermionTerm objects.

    Attributes:   terms: A dictionary of FermionTerm objects.

    """

    def __init__(self, terms=None):
        """Init a FermionOperator.

        Args:
          terms: An instance or dictionary or list of FermionTerm objects.

        Raises:
          FermionOperatorError: Invalid FermionTerms provided to
            FermionOperator.
        """
        super(FermionOperator, self).__init__(terms)
        for term in self:
            if not isinstance(term, FermionTerm):
                raise FermionOperatorError('Invalid FermionTerms provided to '
                                           'FermionOperator.')

    def __setitem__(self, operators, coefficient):
        if operators in self:
            self.terms[tuple(operators)].coefficient = coefficient
        else:
            new_term = FermionTerm(operators, coefficient)
            self.terms[tuple(operators)] = new_term

    def normal_order(self):
        """Normal order this FermionOperator.

        Warning:   Even assuming that each creation or annihilation
        operator appears   at most a constant number of times in the
        original term, the   runtime of this method is exponential in
        the number of qubits.

        """
        self.terms = self.normal_ordered().terms

    def normal_ordered(self):
        """Compute and return the normal ordered form of this FermionOperator.

        Not an in-place method.

        Returns:
          FermionOperator object in normal ordered form.

        Warning:
          Even assuming that each creation or annihilation operator appears
          at most a constant number of times in the original term, the
          runtime of this method is exponential in the number of qubits.

        """
        normal_ordered_operator = FermionOperator()
        for term in self:
            normal_ordered_operator += term.normal_ordered()
        return normal_ordered_operator

    def hermitian_conjugate(self):
        for term in self:
            term.hermitian_conjugate()

    def hermitian_conjugated(self):
        new = copy.deepcopy(self)
        new.hermitian_conjugate()
        return new
