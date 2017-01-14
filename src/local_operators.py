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
      n_qubits: An int giving the number of qubits in simulated Hilbert space.
      terms: Dictionary or list of LocalTerm objects.
      tolerance: A float giving the minimum absolute value below which a term
                 is zero.

    Raises:
      TypeError: Invalid terms provided to initialization.
    """
    self._tolerance = tolerance
    self._n_qubits = n_qubits
    if terms is None:
      self.terms = {}
    elif isinstance(terms, dict):
      self.terms = terms
    elif isinstance(terms, list):
      self.terms = {}
      for term in terms:
        self += term
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
    del self.terms[tuple(operators)]

  def __iadd__(self, addend):
    """In-place method for += addition of LocalTerm or LocalOperator.

    Args:
      addend: A LocalTerm or LocalOperator.

    Raises:
      LocalOperatorError: Cannot add terms acting on different Hilbert spaces.
      LocalOperatorError: Cannot add term of invalid type to LocalOperator.
    """
    # Handle LocalTerms.
    if issubclass(type(addend), local_terms.LocalTerm):

      # Make sure number of qubits is the same.
      if self._n_qubits != addend._n_qubits:
        raise LocalOperatorError(
            'Cannot add terms acting on different Hilbert spaces.')

      # Compute new coefficient and update self.terms.
      new_coefficient = self[addend.operators] + addend.coefficient
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
      raise LocalOperatorError(
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
      LocalOperatorError: Cannot add term of invalid type of LocalOperator.
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
      raise LocalOperatorError(
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
      LocalOperatorError: Invalid typed object cannot multiply LocalOperator.
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
      raise ErrorLocalTerm(
          'Invalid typed object cannot multiply LocalOperator.')

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
      product: A new instance of LocalTerm.

    Raises:
      LocalOperatorError: Invalid typed object cannot multiply LocalOperator.
    """
    if (isinstance(multiplier, (int, float, complex)) or
       numpy.isscalar(multiplier)):
      return self * multiplier
    else:
      raise LocalOperatorError(
          'Invalid typed object cannot multiply LocalOperator.')

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

if __name__ == '__main__':
  n_qubits = 5
  coefficient_a = 6.7j
  coefficient_b = -88.
  coefficient_c = 2.
  operators_a = [1, 2, 3, 4]
  operators_b = [1, 2]
  operators_c = [0, 3, 4]
  term_a = local_terms.LocalTerm(
    n_qubits, coefficient_a, operators_a)
  term_b = local_terms.LocalTerm(
    n_qubits, coefficient_b, operators_b)
  term_c = local_terms.LocalTerm(
    n_qubits, coefficient_c, operators_c)  
  
  operator_a = LocalOperator(n_qubits, [term_a])
  operator_bc = LocalOperator(n_qubits, [term_b, term_c])
  operator_abc = LocalOperator(n_qubits, [term_a, term_b, term_c])
  
  print operator_a
  print operator_bc
  print operator_abc
  print operator_a + operator_bc
  print operator_a * operator_bc
  
  print operator_bc * operator_bc
  
  print operator_a.terms
  print operator_a.terms.values()[0]