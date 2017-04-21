"""FermionOperator stores a sum of creation and annihilation operators acting
on some set of modes."""
import copy
import numpy as np


class FermionOperatorError(Exception):
    pass


class FermionOperator(object):
    """
    A sum of terms acting on qubits, e.g., 0.5 * '0^ 5' + 0.3 * '1^ 2^'.

    A term is an operator acting on n modes and can be represented as:

    coefficent * local_operator[0] x ... x local_operator[n-1]

    where x is the tensor product. A local operator is creation or
    annihilation operator acting on a single mode. In math notation a
    term is, for example, 0.5 * '0^ 5', which means that a creation
    operator acts on mode 0 and an annihilation operator acts on mode 5,
    while the identity operator acts on all other qubits.

    A FermionOperator represents a sum of terms acting on modes and
    overloads operations for easy manipulation of these objects by the
    user.

    Attributes:
      terms (dict): key: A term represented by a tuple of tuples. Each
                         tuple represents a local operator and is an
                         operator which acts on one mode stored as a
                         tuple. The first element is an integer
                         indicating the mode on which a non-trivial
                         local operator acts and the second element is a
                         string, either '0' indicating annihilation, or
                         '1' indicating creation in that mode.
                         E.g. '2^ 5' is ((2, 1), (5, 0))
                         The tuples are sorted according to the qubit number
                         they act on, starting from 0.
                    value: Coefficient of this term as a (complex) float
    """

    def __init__(self, term=(), coefficient=1.):
        """
        Inits a FermionOperator.

        The init function only allows to initialize one term. Additional
        terms have to be added using += (which is fast) or using + of
        two FermionOperator objects:

        Example:
            .. code-block:: python

            ham = (QubitOperator('0^ 3', 0.5) + 0.5 * QubitOperator('3^ 0'))
            # Equivalently
            ham2 = QubitOperator('0^ 3', 0.5)
            ham2 += 0.5 * QubitOperator('3^ 0')

        Note:
            Adding terms to FermionOperator is faster using += (as this
            is done by in-place addition). Specifying the coefficient in
            the __init__ is faster than by multiplying a QubitOperator
            with a scalar as calls an out-of-place multiplication.

        Args:
            coefficient (complex float, optional): The coefficient of the
                first term of this FermionOperator. Default is 1.0.
            term (tuple of tuples, a string, or optional):
                1) A tuple of tuples. The first element of each tuple is
                   an integer indicating the mode on which a non-trivial
                   local operator acts, starting from zero. The second
                   element of each tuple is an integer, either 1 or 0,
                   indicating whether creation or annihilation acts on
                   that mode.
                2) A string of the form '0^ 2', indicating creation in
                   mode 0 and annihilation in mode 2. The string should
                   be sorted by the mode number.
                3) default will result in identity operations on all
                   modes, which is just an empty tuple '()'.

        Raises:
          FermionOperatorError: Invalid term provided to FermionOperator.
        """
        if term is not None and not isinstance(term, (tuple, str)):
            raise ValueError('Operators specified incorrectly.')
        if not isinstance(coefficient, (int, float, complex)):
            raise ValueError('Coefficient must be scalar.')

        self.terms = {}

        # Parse string input.
        if isinstance(term, str):
            list_ops = []
            for el in term.split():
                if el[-1] == '^':
                    list_ops.append((int(el[:-1]), 1))
                else:
                    try:
                        list_ops.append((int(el), 0))
                    except ValueError:
                        raise ValueError(
                            'Invalid action provided to FermionTerm.')
            self.terms[tuple(list_ops)] = coefficient

        # Tuple input.
        elif isinstance(term, tuple) and len(term) != 0:
            term = list(term)
            self.terms[tuple(term)] = coefficient
        else:
            self.terms[()] = coefficient

        # Check type.
        for term in self.terms:
            for operator in term:
                tensor_factor, action = operator
                if not (isinstance(tensor_factor, int) and tensor_factor >= 0):
                    raise FermionOperatorError('Invalid tensor factor provided'
                                               ' to FermionTerm: must be a '
                                               'non-negative int.')
                if action not in (0, 1):
                    raise ValueError('Invalid action provided to FermionTerm. '
                                     'Must be 0 (lowering) or 1 (raising).')

    def isclose(self, other, rel_tol=1e-12, abs_tol=1e-12):
        """
        Returns True if other (FermionOperator) is close to self.

        Comparison is done for each term individually. Return True
        if the difference between each terms in self and other is
        less than the relative tolerance w.r.t. either other or self
        (symmetric test) or if the difference is less than the absolute
        tolerance.

        Args:
            other(FermionOperator): FermionOperator to compare against.
            rel_tol(float): Relative tolerance, must be greater than 0.0
            abs_tol(float): Absolute tolerance, must be at least 0.0
        """
        # terms which are in both:
        for term in set(self.terms).intersection(set(other.terms)):
            a = self.terms[term]
            b = other.terms[term]
            # math.isclose does this in Python >=3.5
            if not abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol):
                return False
        # terms only in one (compare to 0.0 so only abs_tol)
        for term in set(self.terms).symmetric_difference(set(other.terms)):
            if term in self.terms:
                if not abs(self.terms[term]) <= abs_tol:
                    return False
            elif not abs(other.terms[term]) <= abs_tol:
                return False
        return True

    def __imul__(self, multiplier):
        """
        In-place multiply (*=) terms with scalar or FermionOperator.

        Args:
          multiplier(complex float, or FermionOperator): multiplier
        """
        # Handle scalars.
        if isinstance(multiplier, (int, float, complex)):
            for term in self.terms:
                self.terms[term] *= multiplier
            return self

        # Handle FermionOperator.
        elif isinstance(multiplier, FermionOperator):
            result_terms = dict()
            for left_term in self.terms:
                for right_term in multiplier.terms:
                    new_coefficient = (self.terms[left_term] *
                                       multiplier.terms[right_term])

                    product_operators = left_term + right_term

                    # Add to result dict
                    result_terms[tuple(product_operators)] = new_coefficient
            self.terms = result_terms
            return self
        else:
            raise TypeError('Cannot in-place multiply term of invalid type '
                            'to FermionOperator.')

    def __mul__(self, multiplier):
        """
        Return self * multiplier for a scalar, or a FermionOperator.

        Args:
          multiplier: A scalar, or a FermionOperator.

        Returns:
          product: A FermionOperator.

        Raises:
          TypeError: Invalid type cannot be multiply with FermionOperator.
        """
        if isinstance(multiplier, (int, float, complex, FermionOperator)):
            product = copy.deepcopy(self)
            product *= multiplier
            return product
        else:
            raise TypeError(
                'Object of invalid type cannot multiply with FermionOperator.')

    def __rmul__(self, multiplier):
        """
        Return multiplier * self for a scalar.

        We only define __rmul__ for scalars because the left multiply
        exist for  FermionOperator and left multiply
        is also queried as the default behavior.

        Args:
          multiplier: A scalar to multiply by.

        Returns:
          product: A new instance of FermionOperator.

        Raises:
          TypeError: Object of invalid type cannot multiply FermionOperator.
        """
        if not isinstance(multiplier, (int, float, complex)):
            raise TypeError(
                'Object of invalid type cannot multiply with FermionOperator.')
        return self * multiplier

    def __div__(self, divisor):
        """
        Return self / divisor for a scalar.

        Note:
            This is always floating point division.

        Args:
          divisor: A scalar to divide by.

        Returns:
          A new instance of FermionOperator.

        Raises:
          TypeError: Cannot divide local operator by non-scalar type.

        """
        if not isinstance(divisor, (int, float, complex)):
            raise TypeError('Cannot divide QubitOperator by non-scalar type.')
        return self * (1.0 / divisor)

    def __idiv__(self, divisor):
        self *= (1.0 / divisor)
        return self

    def __iadd__(self, addend):
        """In-place method for += addition of FermionOperator.

        Args:
          addend: A FermionOperator.

        Raises:
          TypeError: Cannot add invalid type.
        """
        if isinstance(addend, FermionOperator):
            for term in addend.terms:
                if term in self.terms:
                    self.terms[term] += addend.terms[term]
                else:
                    self.terms[term] = addend.terms[term]
        else:
            raise TypeError('Cannot add invalid type to FermionOperator.')
        return self

    def __add__(self, addend):
        """ Return self + addend for a FermionOperator. """
        summand = copy.deepcopy(self)
        summand += addend
        return summand

    def __sub__(self, subtrahend):
        """Return self - subtrahend for a FermionOperator."""
        if not isinstance(subtrahend, FermionOperator):
            raise TypeError('Cannot subtract invalid type to FermionOperator.')
        return self + (-1. * subtrahend)

    def __neg__(self):
        return -1 * self

    def __str__(self):
        """Return an easy-to-read string representation."""
        string_rep = ''
        for term in self.terms:
            tmp_string = '{} ['.format(self.terms[term])
            for operator in term:
                if operator[1] == 1:
                    tmp_string += '{}^ '.format(operator[0])
                elif operator[1] == 0:
                    tmp_string += '{} '.format(operator[0])
            string_rep += '{}]\n'.format(tmp_string.strip())
        return string_rep

    def __repr__(self):
        return str(self)
