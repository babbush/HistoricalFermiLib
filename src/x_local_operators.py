"""Base class for representation of various local operators.
"""
import x_local_terms as local_terms
import copy


class ErrorLocalOperator(Exception):
  pass


class LocalOperator(object):
  """A collection of LocalTerm objects acting on same number of qubits.

  Attributes:
    _tolerance: A float, the minimum absolute value below which term is zero.
    _n_qubits: An int giving the number of qubits in simulated Hilbert space.
    terms: Dictionary of LocalTerm objects.
  """
  def __init__(self, n_qubits, terms=None):
    """Inits a LocalOperator object.

    Args:
      n_qubits: An int giving the number of qubits in simulated Hilbert space.
      terms: Dictionary of LocalTerm objects.

    Raises:
      ErrorLocalOperator: Invalid terms provided to initialization.
    """
    self._tolerance = 1e-12
    self._n_qubits = n_qubits
    if terms is None:
      self.terms = {}
    elif isinstance(terms, dict):
      self.terms = terms
    elif isinstance(terms, list):
      self.terms = {}
      for term in terms:
        # TODO: Figure out why self += term does not work but line below does.
        self[term.operators] = term.coefficient
    else:
      raise ErrorLocalOperator('Invalid terms provided to initialization.')

  # The methods below stop users from changing _n_qubits.
  @property
  def n_qubits(self):
    return self._n_qubits

  @n_qubits.setter
  def n_qubits(self, n_qubits):
    if hasattr(self, '_n_qubits'):
      raise ErrorLocalTerm(
          'Do not change the size of Hilbert space on which terms act.')

  def __eq__(self, operator):
    """Compare operators to see if they are the same."""
    if self._n_qubits != operator._n_qubits:
      raise ErrorLocalOperator(
          'Cannot compare operators acting on different Hilbert spaces.')
    if len(self) != len(operator):
      return False
    for term in self:
      if self[term.operators] != operator[term.operators]:
        return False
    return True

  def __ne__(self, operator):
    return not (self == operator)

  def __contains__(self, operators):
    if tuple(operators) in self.terms:
      return True
    else:
      return False

  def __getitem__(self, operators):
    if operators in self:
      return self.terms[tuple(operators)].coefficient
    else:
      return 0.

  def __setitem__(self, operators, coefficient):
    if operators in self:
      self.terms[tuple(operators)].coefficient = coefficient
    else:
      new_term = local_terms.LocalTerm(self.n_qubits, coefficient, operators)
      self.terms[tuple(operators)] = new_term

  def __delitem__(self, operators):
    del self.terms[tuple(operators)]

  def __add__(self, addend):
    """Add a LocalTerm or LocalOperator.

    Args:
      addend: A LocalTerm or LocalOperator.

    Returns:
      summand: The sum given by self + addend.

    Raises:
      ErrorLocalOperator: Cannot add term of invalid type of LocalOperator.
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
      raise ErrorLocalOperator(
          'Object of invalid type cannot multiply LocalTerm')

    # Return.
    return summand

  def __sub__(self, subtrahend):
    """Compute self - subtrahend for a LocalTerm or LocalOperator."""
    return self + (-1. * subtrahend)

  def __iadd__(self, addend):
    """In-place method for += addition of LocalTerm or LocalOperator.

    Args:
      addend: A LocalTerm or LocalOperator.

    Raises:
      ErrorLocalOperator: Cannot add terms acting on different Hilbert spaces.
      ErrorLocalOperator: Cannot add term of invalid type to LocalOperator.
    """
    # Handle LocalTerms.
    if issubclass(type(addend), local_terms.LocalTerm):

      # Make sure number of qubits is the same.
      if self._n_qubits != addend._n_qubits:
        raise ErrorLocalOperator(
            'Cannot add terms acting on different Hilbert spaces.')

      # Compute new coefficient and update self.terms.
      new_coefficient = self[addend.operators] + addend.coefficient
      if abs(new_coefficient) > self._tolerance:
        self[addend.operators] = new_coefficient
      else:
        del self[addend.operators]
      return self

    elif issubclass(type(addend), LocalOperator):
      # Handle LocalOperators.
      for term in addend:
        self += term
      return self

    else:
      # Throw exception for unknown type.
      raise ErrorLocalOperator(
          'Cannot add term of invalid type to LocalOperator.')

  def __isub__(self, subtrahend):
    """Compute self - subtrahend for a LocalTerm or LocalOperator."""
    self += (-1. * subtrahend)
    return self

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
      ErrorLocalOperator: Invalid typed object cannot multiply LocalOperator.
    """
    if isinstance(multiplier, (int, long, float, complex)):
      return self * multiplier
    else:
      raise ErrorLocalOperator(
          'Invalid typed object cannot multiply LocalOperator.')

  def __imul__(self, multiplier):
    """Compute self *= multiplier.

    Note that this is only actually an in place method when multiplier
    is a scalar. Otherwise, is is necessary to change all of the keys
    of the dictionary.

    Args:
      multiplier: A scalar, LocalTerm or LocalOperator.

    Raises:
      ErrorLocalOperator: Invalid typed object cannot multiply LocalOperator.
      ErrorLocalOperator: Cannot multiply terms on different Hilbert spaces.
    """
    # Handle scalars.
    if isinstance(multiplier, (int, long, float, complex)):
      for term in self:
        term.coefficient *= complex(multiplier)
      return self

    elif issubclass(type(multiplier), local_operators.LocalTerm):
      # Handle LocalTerms. Note that it is necessary to make new dictioanry.
      new_operator = LocalOperator(self.n_qubits)
      for term in self:
        term *= multiplier
        new_operator += term
      self.terms = new_operator.terms
      return self

    elif issubclass(type(multiplier), LocalOperator):
      # Handle LocalOperators. It is necessary to make new dictionary.
      new_operator = LocalOperator(self.n_qubits)
      for left_term in self:
        for right_term in multiplier:
          new_operator += left_term * right_term
      self.terms = new_operator.terms
      return self

    else:
      # Throw exception for wrong type of multiplier.
      raise ErrorLocalTerm(
          'Invalid typed object cannot multiply LocalOperator.')

  def list_terms(self):
    return self.terms.values()

  def __iter__(self):
    return self.terms.itervalues()

  def __len__(self):
    return len(self.terms)

  def __str__(self):
    return ''.join('{}\n'.format(term) for term in self)
