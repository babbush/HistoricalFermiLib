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
    self._n_qubits = _n_qubits
    if terms is None:
      self.terms = {}
    elif isinstance(terms, dict):
      self.terms = terms
    elif isinstance(terms, list):
      self.terms = {}
      for term in terms:
        self += term
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
      if self[term] != operator[term]:
        return False
    return True

  def __ne__(self, operator):
    return not (self == operator)

  def __contains__(self, term):
    # TODO: Notice if line below doesn't require "hash".
    if term in self.terms:
      return True
    else:
      return False

  def __getitem__(self, term):
    if term in self:
      return self.terms[hash(term)].coefficient
    else:
      return 0.

  def __setitem__(self, term, coefficient):
    if term not in self:
      self.terms[hash(term)] = copy.deepcopy(term)
    self.terms[hash(term)].coefficient = coefficient

  def __delitem__(self, term):
    del self.terms[hash(term)]

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
    if issubclass(addend, local_terms.LocalTerm):
      summand += addend

    elif:
      # Handle addition of local operators.
      for term in addend:
        summand += term

    else:
      # Throw exception for unknown type.
      raise ErrorLocalOperator(
          'Object of invalid type cannot multiply LocalTerm')

    # Return.
    return summand

  def __iadd__(self, addend):
    """In-place method for += addition of LocalTerm or LocalOperator.

    Args:
      addend: A LocalTerm or LocalOperator.

    Raises:
      ErrorLocalOperator: Cannot add terms acting on different Hilbert spaces.
      ErrorLocalOperator: Cannot add term of invalid type to LocalOperator.
    """
    # Handle LocalTerms.
    if issubclass(addend, local_terms.LocalTerm):

      # Make sure number of qubits is the same.
      if self._n_qubits != addend._n_qubits:
        raise ErrorLocalOperator(
            'Cannot add terms acting on different Hilbert spaces.')

      # Compute new coefficient and update self.
      new_coefficient = self[addend] + addend.coefficient
      if abs(new_coefficient) > self._tolerance:
        self[addend] = new_coefficient
      else:
        del self[addend]

    elif issubclass(addend, LocalOperator):
      # Handle LocalOperators.
      for term in addend:
        self += term

    else:
      # Throw exception for unknown type.
      raise ErrorLocalOperator(
          'Cannot add term of invalid type to LocalOperator.')

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
        term.coefficient *= multiplier

    elif issubclass(multiplier, local_operators.LocalTerm):
      # Handle LocalTerms. Note that it is necessary to make new dictioanry.
      new_operator = LocalOperator(self.n_qubits)
      for term in self:
        term *= multiplier
        new_operator += term
      self.terms = new_operator.terms

    elif issubclass(multiplier, LocalOperator):
      # Handle LocalOperators. It is necessary to make new dictionary.
      new_operator = LocalOperator(self.n_qubits)
      for left_term in self:
        for right_term in multiplier:
          new_operator += left_term * right_term
      self.terms = new_operator.terms

    else:
      # Throw exception for wrong type of multiplier.
      raise ErrorLocalTerm(
          'Invalid typed object cannot multiply LocalOperator.')

  def __iter__(self):
    return self.terms.itervalues()

  def __len__(self):
    return len(self.terms)

  def __str__(self):
    return ''.join('{}\n'.format(term) for term in self)
