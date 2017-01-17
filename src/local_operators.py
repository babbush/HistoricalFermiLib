"""Base class for representation of various local operators."""
import local_terms
import numpy
import copy


# Define error class.
class LocalOperatorError(Exception):
  pass


class LocalOperator(object):
  """A collection of LocalTerm objects acting on same number of qubits.

  Attributes:
    _tolerance: A float, the minimum absolute value below which term is zero.
    _n_qubits: An int giving the number of qubits in simulated Hilbert space.
    terms: Dictionary of LocalTerm objects.
  """
  def __init__(self, n_qubits, terms=None, tolerance=1e-12):
    """Inits a LocalOperator object.

    Args:
      n_qubits: An int giving the number of qubits in simulated Hilbert 
                space.
      terms: Dictionary or list of LocalTerm objects, or LocalTerm 
             object.
      tolerance: A float giving the minimum absolute value below which a
                 term is zero.

    Raises:
      TypeError: Invalid terms provided to initialization.
      ValueError: Number of qubits needs to be a positive integer.
    """
    # Check that n_qubits is an integer.
    if not isinstance(n_qubits, int) or n_qubits < 1:
      raise ValueError('Number of qubits must be a positive integer.')
    
    self._tolerance = tolerance
    self._n_qubits = n_qubits
    if terms is None:
      self.terms = {}
    elif isinstance(terms, dict):
      self.terms = copy.deepcopy(terms)
    elif isinstance(terms, list):
      self.terms = {}
      for term in terms:
        self += local_terms.LocalTerm(term.n_qubits, term.coefficient,
                                      term.operators)
    elif isinstance(terms, local_terms.LocalTerm):
      self.terms = {}
      self += local_terms.LocalTerm(terms.n_qubits, terms.coefficient,
                                    terms.operators)
    else:
      raise TypeError('Invalid terms provided to initialization.')

  @classmethod
  def return_class(cls, n_qubits, terms=None):
    return cls(n_qubits, terms)

  # The methods below stop users from changing _n_qubits.
  @property
  def n_qubits(self):
    return self._n_qubits

  @n_qubits.setter
  def n_qubits(self, n_qubits):
    if hasattr(self, '_n_qubits'):
      raise LocalOperatorError(
          'Do not change the size of Hilbert space on which terms act.')

  def __eq__(self, other):
    """Compare operators to see if they are the same."""
    if self.n_qubits != other.n_qubits:
      raise LocalOperatorError(
          'Cannot compare operators acting on different Hilbert spaces.')
    if len(self) != len(other):
      return False
    for term in self:
      difference = term.coefficient - other[term.operators]
      if abs(difference) > self._tolerance:
        return False
    return True

  def __ne__(self, operator):
    return not (self == operator)

  def __contains__(self, operators):
    return tuple(operators) in self.terms

  def __getitem__(self, operators):
    if tuple(operators) in self:
      return self.terms[tuple(operators)].coefficient
    return 0.

  # As its coded now, __setitem__ must be rewritten for every child class.
  def __setitem__(self, operators, coefficient):
    if operators in self:
      self.terms[tuple(operators)].coefficient = coefficient
    else:
      # TODO: Find better solution than using call to LocalTerm here.
      new_term = local_terms.LocalTerm(self.n_qubits, coefficient, operators)
      self.terms[tuple(operators)] = new_term

  def __delitem__(self, operators):
    if not operators in self:
      raise LocalOperatorError(
        'operators {} not in LocalOperator'.format(operators))
    del self.terms[tuple(operators)]

  def __iadd__(self, addend):
    """In-place method for += addition of LocalTerm or LocalOperator.

    Args:
      addend: A LocalTerm or LocalOperator.

    Raises:
      LocalOperatorError: Cannot add terms acting on different Hilbert spaces.
      TypeError: Cannot add term of invalid type to LocalOperator.
    """
    # Handle LocalTerms.
    if issubclass(type(addend), local_terms.LocalTerm):

      # Make sure number of qubits is the same.
      if self.n_qubits != addend.n_qubits:
        raise LocalOperatorError(
            'Cannot add terms acting on different Hilbert spaces.')

      # Compute new coefficient and update self.terms.
      new_coefficient = self[tuple(addend.operators)] + addend.coefficient
      if abs(new_coefficient) > self._tolerance:
        self[addend.operators] = new_coefficient
      elif addend.operators in self:
        del self[addend.operators]
      return self

    elif issubclass(type(addend), LocalOperator):
      # Handle LocalOperators.
      for term in addend:
        self += term
      return self

    else:
      # Throw exception for unknown type.
      raise TypeError(
          'Cannot add term of invalid type to LocalOperator.')

  def __isub__(self, subtrahend):
    """Compute self - subtrahend for a LocalTerm or LocalOperator."""
    self += (-1. * subtrahend)
    return self

  def __add__(self, addend):
    """Add a LocalTerm or LocalOperator.

    Args:
      addend: A LocalTerm or LocalOperator.

    Returns:
      summand: The sum given by self + addend.

    Raises:
      TypeError: Cannot add term of invalid type of LocalOperator.
    """
    # Copy self.
    summand = copy.deepcopy(self)

    # Handle addition of single LocalTerm.
    if issubclass(type(addend), local_terms.LocalTerm):
      summand += addend

    elif issubclass(type(addend), LocalOperator):
      # Handle addition of local operators.
      for term in addend:
        summand += term

    else:
      # Throw exception for unknown type.
      raise TypeError(
          'Object of invalid type cannot multiply LocalTerm')

    # Return.
    return summand

  def __sub__(self, subtrahend):
    """Compute self - subtrahend for a LocalTerm or LocalOperator."""
    return self + (-1. * subtrahend)

  def __imul__(self, multiplier):
    """Compute self *= multiplier.

    Note that this is only actually an in place method when multiplier
    is a scalar. Otherwise, is is necessary to change all of the keys
    of the dictionary.

    Args:
      multiplier: A scalar, LocalTerm or LocalOperator.

    Raises:
      TypeError: Invalid typed object cannot multiply LocalOperator.
      LocalOperatorError: Cannot multiply terms on different Hilbert spaces.
    """
    # Handle scalars.
    if (isinstance(multiplier, (int, float, complex)) or
       numpy.isscalar(multiplier)):
      for term in self:
        term.coefficient *= complex(multiplier)
      return self

    # Handle LocalTerms. Note that it is necessary to make new dictioanry.
    elif issubclass(type(multiplier), local_terms.LocalTerm):
      new_operator = self.return_class(self._n_qubits)
      for term in self:
        term *= multiplier
        new_operator += term
      self.terms = new_operator.terms
      return self

    # Handle LocalOperators. It is necessary to make new dictionary.
    elif issubclass(type(multiplier), LocalOperator):
      new_operator = self.return_class(self._n_qubits)
      for left_term in self:
        for right_term in multiplier:
          new_operator += left_term * right_term
      self.terms = new_operator.terms
      return self

    else:
      # Throw exception for wrong type of multiplier.
      raise TypeError('Invalid typed object cannot multiply LocalOperator.')

  def __mul__(self, multiplier):
    """Compute self * multiplier for scalar, other LocalTerm or LocalOperator.

    Args:
      multiplier: A scalar, LocalTerm or LocalOperator.

    Returns:
      product: A new instance of LocalOperator.
    """
    product = copy.deepcopy(self)
    product *= multiplier
    return product

  def __rmul__(self, multiplier):
    """Compute multiplier * self for a scalar.

    We only define __rmul__ for scalars because the left multiply
    should exist for LocalTerms and LocalOperators and left multiply
    is also queried as the default behavior.

    Args:
      multiplier: A scalar.

    Returns:
      product: A new instance of LocalOperator.

    Raises:
      TypeError: Invalid typed object cannot multiply LocalOperator.
    """
    if (isinstance(multiplier, (int, float, complex)) or
       numpy.isscalar(multiplier)):
      return self * multiplier
    else:
      raise TypeError(
          'Invalid typed object cannot multiply LocalOperator.')
    
  def __pow__(self, exponent):
    """Exponentiate the LocalOperator.

    Args:
      exponent: An int, giving the exponent with which to raise the operator.

    Returns:
      The exponentiated operator.

    Raises:
      ValueError: Can only raise LocalOperator to positive integer powers.
    """
    if not isinstance(exponent, int) or exponent < 0:
      raise ValueError('Can only raise LocalTerm to positive integer powers.')
    
    identity_term = local_terms.LocalTerm(self.n_qubits, 1.0)
    res = LocalOperator(self.n_qubits, identity_term)
    
    for i in xrange(exponent):
      res *= self
      
    return res

  def __abs__(self):
    operator_copy = copy.deepcopy(self)
    for term in operator_copy:
      term.coefficient = abs(term.coefficient)
    return operator_copy

  def list_coefficients(self):
    return [term.coefficient for term in self]

  def list_terms(self):
    return list(self.terms.values())

  def __iter__(self):
    return iter(self.terms.values())

  def __len__(self):
    return len(self.terms)

  def __str__(self):
    return ''.join('{}\n'.format(term) for term in self)
