"""Base class for representation of various local terms."""
import x_local_operators as local_operators
import copy


# Define error classes.
class ErrorLocalTerm(Exception):
  pass


class LocalTerm(object):
  """Represents a term consisting of a product of operators and a coefficient.

  Attributes:
    _tolerance: A float, the minimum absolute value below which term is zero.
    _n_qubits: An int giving the number of qubits in simulated Hilbert space.
    coefficient: A complex valued float giving the term coefficient.
    operators: A list of site operators representing the term.
  """
  def __init__(self, n_qubits, coefficient=0., operators=None):
    """Inits a LocalTerm.

    Args:
      n_qubits: An int giving the number of qubits in simulated Hilbert space.
      coefficient: A complex valued float giving the term coefficient.
      operators: A list of site operators representing the term.

    Raises:
      ErrorLocalTerm: Number of qubits needs to be an integer.
    """
    # Check that n_qubits is an integer.
    if not isinstance(n_qubits, int):
      raise ErrorLocalTerm('Number of qubits needs to be an integer.')

    # Initialize.
    self._tolerance = 1e-12
    self._n_qubits = n_qubits
    self.coefficient = coefficient
    if operators is None:
      self.operators = []
    else:
      self.operators = operators

  @classmethod
  def return_class(cls, n_qubits, coefficient=0, operators=None):
    return cls(n_qubits, coefficient, operators)

  # The methods below stop users from changing _n_qubits.
  @property
  def n_qubits(self):
    return self._n_qubits

  @n_qubits.setter
  def n_qubits(self, n_qubits):
    if hasattr(self, '_n_qubits'):
      raise ErrorLocalTerm(
          'Do not change the size of Hilbert space on which terms act.')

  def __eq__(self, local_term):
    """Overload equality comparison == to interact with standard library.

    Args:
      local_term: Another LocalTerm which is to be compared with self.

    Returns:
      True or False, whether objects are the same.

    Raises:
      ErrorLocalTerm: Cannot compare terms acting on different Hilbert spaces.
    """
    if self._n_qubits != local_term._n_qubits:
      raise ErrorLocalTerm(
          'Cannot compare terms acting on different Hilbert spaces.')
    elif abs(self.coefficient - local_term.coefficient) > self._tolerance:
      return False
    elif self.operators != local_term.operators:
      return False
    else:
      return True

  def __ne__(self, local_term):
    """Overload not equals comparison != to interact with standard library."""
    return not (self == local_term)

  def __getitem__(self, index):
    return self.operators[index]

  def __setitem__(self, index, value):
    self.operators[index] = value

  def __delitem__(self, index):
    del self.operators[index]

  def __add__(self, addend):
    """Compute self + addend for a LocalOperator or derivative.

    Note that we will not allow one to add together two LocalTerms.
    The reason is because there are ambiguities when LocalTerms sum
    to zero and also because it is difficult to determine what class
    the output should be when adding together terms which inherit from
    LocalTerm.

    Args:
      addend: A LocalOperator or LocalOperator derivative.

    Returns:
      summand: A new instance of LocalOperator.

    Raises:
      ErrorLocalTerm: Can only add LocalOperator type to LocalTerm type.
      ErrorLocalTerm: Object of invalid type cannot be added to LocalTerm.
      ErrorLocalTerm: Cannot add terms acting on different Hilbert spaces.
    """
    # Handle LocalTerms.
    if issubclass(type(addend), LocalTerm):
      raise ErrorLocalTerm(
          'Can only add LocalOperator type to LocalTerm type.')

    # Handle LocalOperators.
    elif issubclass(type(addend), local_operators.LocalOperator):
      summand = addend + self

    else:
      # Throw exception for unknown type.
      raise ErrorLocalTerm(
          'Cannot add term of invalid type to LocalTerm.')

    # Return the summand.
    return summand

  def __sub__(self, subtrahend):
    """Compute self - subtrahend for a LocalTerm or LocalOperator."""
    return self + (-1. * subtrahend)

  def __imul__(self, multiplier):
    """Compute self *= multiplier. Multiplier must be scalar or LocalTerm.

    Note that this is actually an in-place method. Method undefined for
    LocalOperator types on right side of *= because such a multiplication
    would change the type of self.

    Args:
      multiplier: A scalar or LocalTerm.

    Raises:
      ErrorLocalTerm: Cannot multiply terms acting on different Hilbert spaces.
      ErrorLocalTerm: Can only *= multiply LocalTerm by scalar or LocalTerm.
    """
    # Handle scalars.
    if isinstance(multiplier, (int, long, float, complex)):
      self.coefficient *= complex(multiplier)
      return self

    elif issubclass(type(multiplier), LocalTerm):
      # Handle LocalTerms. Make sure number of qubits is the same.
      if self._n_qubits != multiplier._n_qubits:
        raise ErrorLocalTerm(
            'Cannot multiply terms acting on different Hilbert spaces.')

      # Compute product.
      self.coefficient *= multiplier.coefficient
      self.operators += multiplier.operators
      return self

    else:
      # Throw exception for wrong type of multiplier.
      raise ErrorLocalTerm(
          'Can only *= multiply LocalTerm by scalar or LocalTerm.')

  def __mul__(self, multiplier):
    """Compute self * multiplier for scalar, other LocalTerm or LocalOperator.

    Args:
      multiplier: A scalar, LocalTerm or LocalOperator.

    Returns:
      product: A new instance of LocalTerm or LocalOperator.

    Raises:
      ErrorLocalTerm: Object of invalid type cannot multiply LocalTerm.
      ErrorLocalTerm: Cannot multiply terms acting on different Hilbert spaces.
    """
    # Handle scalars or LocalTerms.
    if isinstance(multiplier, (int, long, float, complex, LocalTerm)):
      product = copy.deepcopy(self)
      product *= multiplier

    # Handle LocalOperator and derivatives.
    elif issubclass(type(multiplier), local_operators.LocalOperator):
      product = multiplier.return_class(self._n_qubits)
      for term in multiplier:
        product += self * term

    else:
      # Throw exception for unknown type.
      raise ErrorLocalTerm(
          'Object of invalid type cannot multiply LocalTerm.')

    # Return the product.
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
      ErrorLocalTerm: Object of invalid type cannot multiply LocalTerm.
    """
    if isinstance(multiplier, (int, long, float, complex)):
      product = copy.deepcopy(self)
      product.coefficient *= multiplier
      return product
    else:
      raise ErrorLocalTerm(
          'Object of invalid type cannot multiply LocalTerm.')

  def __iter__(self):
    return iter(self.operators)

  def __len__(self):
    return len(self.operators)

  def __str__(self):
    return '{} {}'.format(self.coefficient, self.operators)
