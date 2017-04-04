"""Base class for representation of various local terms."""
import copy
import local_operators
import numpy

from config import *


# Define error classes.
class LocalTermError(Exception):
  pass


class LocalTerm(object):
  """Represents a term consisting of a product of operators and a coefficient.

  Attributes:
    coefficient: A complex valued float giving the term coefficient.
    operators: A list of site operators representing the term.
  """
  __array_priority__ = 0  # this ensures good behavior with numpy scalars

  def __init__(self, operators=None, coefficient=1.):
    """Inits a LocalTerm.

    Args:
      operators: A list of site operators representing the term.
      coefficient: A complex valued float giving the term's coefficient.

    Raises:
      ValueError: Number of qubits must be a non-negative integer.
      ValueError: Coefficient must be a scalar.
    """
    # Check that coefficient is a scalar.
    if not numpy.isscalar(coefficient):
      raise ValueError('Coefficient must be scalar.')

    # Initialize.
    self.coefficient = coefficient
    if operators is None:
      operators = []
    self.operators = list(operators)

  @classmethod
  def return_class(cls, operators=None, coefficient=1.):
    return cls(operators, coefficient)

  def __eq__(self, other):
    """Overload equality comparison ==.

    Args:
      other: Another LocalTerm which is to be compared with self.

    Returns:
      True or False, whether objects are the same.

    Raises:
      LocalTermError: Cannot compare terms acting on different Hilbert
                      spaces.

    Notes:
      Two LocalTerms are considered equal either if their coefficients
      are within EQ_TOLERANCE of the first and their operators are the
      same, or if both their coefficients are within tolerance of zero.
    """
    # Operators are equal if their coefficients are sufficiently close
    # and they have the same operators, or if they are both close to 0.
    return ((self.operators == other.operators and
             abs(self.coefficient - other.coefficient) <= EQ_TOLERANCE) or
            (abs(self.coefficient) <= EQ_TOLERANCE and
             abs(other.coefficient) <= EQ_TOLERANCE))

  def __ne__(self, other):
    """Overload not equals comparison !=."""
    return not (self == other)

  def __getitem__(self, index):
    try:
      return self.operators[index]
    except IndexError:
      raise LocalTermError('LocalTerm index out of range.')

  def __setitem__(self, index, value):
    try:
      self.operators[index] = value
    except IndexError:
      raise LocalTermError('LocalTerm assignment index out of range.')

  def __delitem__(self, index):
    try:
      del self.operators[index]
    except IndexError:
      raise LocalTermError('LocalTerm deletion index out of range.')

  def __add__(self, addend):
    """Compute self + addend for a LocalTerm or derivative.

    Args:
      addend: A LocalTerm or LocalTerm derivative.

    Returns:
      summand: A new instance of LocalOperator. The reason for returning
               LocalOperator is that there are ambiguities when
               LocalTerms sum to zero and also because it is difficult
               to determine what class the output should be when adding
               together terms which inherit from LocalTerm.

    Raises:
      TypeError: Cannot add term of invalid type to LocalTerm.
    """
    if not issubclass(type(addend),
                      (LocalTerm, local_operators.LocalOperator)):
      raise TypeError('Cannot add term of invalid type to LocalTerm.')

    return local_operators.LocalOperator([self]) + addend

  def __neg__(self):
    return -1 * self

  def __sub__(self, subtrahend):
    """Compute self - subtrahend for a LocalTerm or derivative."""
    return self + (-1. * subtrahend)

  def __isub__(self, subtrahend):
    """Compute self - subtrahend for a LocalTerm or derivative."""
    self += (-1. * subtrahend)
    return self

  def __imul__(self, multiplier):
    """Compute self *= multiplier. Multiplier must be scalar or LocalTerm.

    Note that this is actually an in-place method. Method undefined for
    LocalOperator types on right side of *= because such a
    multiplication would change the type of self.

    Args:
      multiplier: A scalar or LocalTerm.

    Raises:
      TypeError: Can only *= multiply LocalTerm by scalar or LocalTerm.
    """
    # Handle scalars.
    if numpy.isscalar(multiplier):
      self.coefficient *= complex(multiplier)

    elif issubclass(type(multiplier), LocalTerm):
      # Compute product.
      self.coefficient *= multiplier.coefficient
      self.operators += multiplier.operators

    else:
      # Throw exception for wrong type of multiplier.
      raise TypeError('Can only *= multiply LocalTerm by scalar or LocalTerm.')
    return self

  def __mul__(self, multiplier):
    """Compute self * multiplier for scalar, other LocalTerm or LocalOperator.

    Args:
      multiplier: A scalar, LocalTerm or LocalOperator.

    Returns:
      product: A new instance of LocalTerm or LocalOperator.

    Raises:
      TypeError: Object of invalid type cannot multiply LocalTerm.
    """
    # Handle scalars or LocalTerms.
    if (numpy.isscalar(multiplier) or isinstance(multiplier, LocalTerm)):
      product = copy.deepcopy(self)
      product *= multiplier

    # Handle LocalOperator and derivatives.
    elif issubclass(type(multiplier), local_operators.LocalOperator):
      product = multiplier.return_class()
      for term in multiplier:
        product += self * term

    else:
      # Throw exception for unknown type.
      raise TypeError('Object of invalid type cannot multiply LocalTerm.')
    return product

  def __rmul__(self, multiplier):
    """Compute multiplier * self for a scalar.

    We only define __rmul__ for scalars because the left multiply
    should exist for LocalTerms and LocalOperators and left multiply
    is also queried as the default behavior.

    Args:
      multiplier: A scalar to multiply by..

    Returns:
      product: A new instance of LocalTerm.

    Raises:
      TypeError: Object of invalid type cannot multiply LocalTerm.
    """
    if not numpy.isscalar(multiplier):
      raise TypeError('Object of invalid type cannot multiply LocalTerm.')

    product = copy.deepcopy(self)
    product.coefficient *= multiplier
    return product

  def __div__(self, divisor):
    """Compute self / divisor for a scalar.

    Args:
      divisor: A scalar to divide by.

    Returns:
      A new instance of LocalTerm.

    Raises:
      TypeError: Cannot divide local operator by non-scalar type."""
    if not numpy.isscalar(divisor):
      raise TypeError('Cannot divide local operator by non-scalar type.')
    return self * (1.0 / divisor)

  def __idiv__(self, divisor):
    self *= (1.0 / divisor)
    return self

  def __pow__(self, exponent):
    """Exponentiate the LocalTerm.

    Args:
      exponent: An int, giving the exponent with which to raise the term.

    Returns:
      exponentiated_term: The exponentiated term.

    Raises:
      ValueError: Can only raise LocalTerm to positive integer powers.
    """
    if not isinstance(exponent, int) or exponent < 0:
      raise ValueError('Can only raise LocalTerm to positive integer powers.')

    # Initialize identity.
    exponentiated = self.return_class()

    # Handle other exponents.
    for i in range(exponent):
      exponentiated *= self
    return exponentiated

  def __abs__(self):
    term_copy = copy.deepcopy(self)
    term_copy.coefficient = abs(self.coefficient)
    return term_copy

  def __iter__(self):
    return iter(self.operators)

  def __len__(self):
    return len(self.operators)

  def __str__(self):
    return '{} {}'.format(self.coefficient, self.operators)

  def __repr__(self):
    return str(self)

  def is_identity(self):
    return len(self.operators) == 0

  def commutator(self, term):
    """Evaluate commutator of self with LocalTerm or LocalOperator.

    Args:
      term: Despite the name, this is either a LocalTerm or LocalOperator.

    Returns:
      LocalOperator giving [self, term] = self * term - term * self.
    """
    return self * term - term * self
