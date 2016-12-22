"""Base classes for representation of various local operator types.
"""
import copy


# Set the tolerance below which a coefficient is regarded as zero.
_TOLERANCE = 1e-12


# Define error classes.
class ErrorLocalTerm(Exception):
  pass


class ErrorLocalOperator(Exception):
  pass


class LocalTerm(object):
  """Represents a term consisting of a product of operators and a coefficient.

  Attributes:
    n_qubits: An int giving the number of qubits in simulated Hilbert space.
    coefficient: A complex valued float giving the term coefficient.
    operators: A list of site operators representing the term.
  """
  def __init__(self, n_qubits, coefficient=0., operators=None):
    """Inits a LocalTerm.

    Args:
      n_qubits: An int giving the number of qubits in simulated Hilbert space.
      coefficient: A complex valued float giving the term coefficient.
      operators: A list of site operators representing the term.
    """
    self.n_qubits = n_qubits
    self.coefficient = coefficient
    if operators is None:
      self.operators = []
    else:
      self.operators = operators

  def __eq__(self, local_term):
    """Overload equality comparison == to interact with standard library.

    Args:
      local_term: Another LocalTerm which is to be compared with self.

    Returns:
      True or False, whether objects are the same.

    Raises:
      ErrorLocalTerm: Cannot compare terms acting on different Hilbert spaces.
    """
    if self.n_qubits != local_term.n_qubits:
      raise ErrorLocalTerm(
          'Cannot compare terms acting on different Hilbert spaces.')
    elif abs(self.coefficient - local_term.coefficient) > _TOLERANCE:
      return False
    elif self.operators != local_term.operators:
      return False
    else:
      return True

  def __ne__(self, local_term):
    """Overload not equals comparison != to interact with standard library."""
    return not (self == local_term)

  def __mul__(self, multiplier):
    """Compute self * multiplier for scalar, other LocalTerm or LocalOperator.

    Args:
      multiplier: A scalar, LocalTerm or LocalOperator.

    Returns:
      product: A new instance of LocalTerm which gives the product.

    Raises:
      ErrorLocalTerm: Object of invalid type cannot multiply LocalTerm.
      ErrorLocalTerm: Cannot multiply terms acting on different Hilbert spaces.
    """
    # Handle scalars.
    if isinstance(multiplier, (int, long, float, complex)):
      product = copy.deepcopy(self)
      product.coefficient *= multiplier

    # Handle LocalTerms.
    elif issubclass(multiplier, LocalTerm):

      # Make sure number of qubits is the same.
      if self.n_qubits != multiplier.n_qubits:
        raise ErrorLocalTerm(
            'Cannot multiply terms acting on different Hilbert spaces.')

      # Compute product.
      product = copy.deepcopy(self)
      product.coefficient *= multiplier.coefficient
      product.operators += multiplier.operators

    # Handle LocalOperators.
    elif issubclass(multiplier, LocalOperator):

      # Loop through multiplier terms to perform multiply.
      product = LocalOperator(self.n_qubits)
      for term in multiplier:
        product += self * term

    else:
      # Throw exception for unknown type.
      raise ErrorLocalTerm(
          'Object of invalid type cannot multiply LocalTerm')

    # Return the product.
    return product

  def __imul__(self, multiplier):
    """Compute self *= multiplier. Multiplier must be scalar or LocalTerm.

    Args:
      multiplier: A scalar or LocalTerm.

    Raises:
      ErrorLocalTerm: Cannot multiply terms acting on different Hilbert spaces.
      ErrorLocalTerm: Can only *= multiply LocalTerm by scalar or LocalTerm.
    """
    # Handle scalars.
    if isinstance(multiplier, (int, long, float, complex)):
      self.coefficient *= multiplier

    # Handle LocalTerms.
    elif issubclass(multiplier, LocalTerm):

      # Make sure number of qubits is the same.
      if self.n_qubits != multiplier.n_qubits:
        raise ErrorLocalTerm(
            'Cannot multiply terms acting on different Hilbert spaces.')

      # Compute product.
      self.coefficient *= multiplier.coefficient
      self.operators += multiplier.operators

    else:
      # Throw exception for wrong type of multiplier.
      raise ErrorLocalTerm(
          'Can only *= multiply LocalTerm by scalar or LocalTerm.')

  def __add__(self, addend):
    """Compute self + addend for scalar, other LocalTerm or LocalOperator.

    Args:
      addend: A scalar, LocalTerm or LocalOperator.

    Returns:
      summand: A new instance of LocalOperator which gives the sum.

    Raises:
      ErrorLocalTerm: Object of invalid type cannot be added to LocalTerm.
      ErrorLocalTerm: Cannot add terms acting on different Hilbert spaces.
    """
    # Handle scalars.
    if isinstance(addend, (int, long, float, complex)):
      identity = LocalTerm(self.n_qubits, addend)
      summand = self + identity

    # Handle LocalTerms.
    elif issubclass(addend, LocalTerm):

      # Make sure number of qubits is the same.
      if self.n_qubits != addend.n_qubits:
        raise ErrorLocalTerm(
            'Cannot add terms acting on different Hilbert spaces.')

      # Compute addition.
      if self.operators == addend.operators:
        summand = copy.deepcopy(self)
        summand.coefficient += addend.coefficient

      else:
        summand = LocalOperator(self.n_qubits,
                                [copy.deepcopy(self), copy.deepcopy(addend)])

    # Handle LocalOperators.
    elif issubclass(addend, LocalOperator):

      # Make sure number of qubits is the same.
      if self.n_qubits != addend.n_qubits:
        raise ErrorLocalTerm(
            'Cannot add terms acting on different Hilbert spaces.')

      # Loop through addend and add.
      summand = LocalOperator(self.n_qubits)
      for term in addend:
        summand += self + term

    else:
      # Throw exception for unknown type.
      raise ErrorLocalTerm(
          'Object of invalid type cannot multiply LocalTerm')

    # Return the summand.
    return summand


  def __hash__(self):
    """Returns a hashable unique key representing operators.

    Note that this method is essential to the operation of the LocalOperators
    class which uses a python dictionary to store LocalTerm objects.
    """
    return hash(tuple(self.operators))

  def __str__(self):
    string_representation = '{} {}'.format(
        self.coefficient, tuple(self.operators))
    return string_representation

  def __iter__(self):
    return iter(self.operators)

  def __len__(self):
    return len(self.operators)


class LocalOperator(object):
  """A collection of LocalOperator objects acting on same number of qubits.

  Attributes:
    n_qubits: An int giving the number of qubits in simulated Hilbert space.
    terms: Dictionary of LocalTerm objects. The dictionary key is
        LocalTerm.key() and the dictionary value is the LocalTerm.
  """
  def __init__(self, n_qubits, terms=None):
    """Inits a LocalOperator object.

    Args:
      n_qubits: An int giving the number of qubits in simulated Hilbert space.
      terms: Dictionary of LocalTerm objects. The dictionary key is
          LocalTerm.key() and the dictionary value is the LocalTerm.

    Raises:
      ErrorLocalOperator: Invalid terms provided to initialization.
    """
    self.n_qubits = n_qubits
    if terms is None:
      self.terms = {}
    elif isinstance(terms, dict):
      self.terms = terms
    elif isinstance(terms, list):
      self.terms = {}
      self += terms
    else:
      raise ErrorLocalOperator('Invalid terms provided to initialization.')

  def __iter__(self):
    return self.terms.itervalues()

  def __iadd__(self, added_term):
    """In-place add terms to self.

    Args:
      added_term: A scalar (meaning add identity with given scale),
          LocalTerm, LocalOperator or list of LocalTerms.

    Raises:
      ErrorLocalTerm: Cannot add terms which act on different Hilbert spaces.
    """
    # Handle LocalTerms.
    if issubclass(added_term, LocalTerm):
      if added_term in self:
        new_coefficient = self[added_term] + added_term.coefficient
        if abs(new_coefficient) < _TOLERANCE:
          self.remove_term(added_term)
        else:
          s


    # Handle scalars.
    if isinstance(added_term, (complex, int, long, float)):
      identity = LocalTerm(self.n_qubits,

  def __add__(self, added_term):
    """Add a scalar (i.e. identity), LocalTerm or LocalOperator.

    Args:
      added_term: A scalar (meaning add identity, LocalTerm or LocalOperator.

    Returns:
      sum: The sum of the added terms.

    Raises:
      ErrorLocalTerm: Cannot add terms which act on different Hilbert spaces.
    """
    # Handle scalars.
    if isinstance(added_term, (complex, int, long, float)):




    # Make sure terms act on same number of qubits.
    if self.n_qubits != new_term.n_qubits:
      raise ErrorLocalOperator(
          'Cannot add terms which act on different Hilbert spaces.')

    # Add term.
    term_key = new_term.key()
    if term_key in self.terms:
      new_coefficient = (self.terms[term_key].coefficient +
                         new_term.coefficient)
      if abs(new_coefficient) < _TOLERANCE:
        del self.terms[term_key]
      else:
        new_term.coefficient = new_coefficient
        self.terms[term_key] = new_term
    else:
      self.terms[term_key] = new_term

  def add_terms_list(self, terms_list):
    for new_term in terms_list:
      self.add_term(new_term)

  def add_operator(self, new_operator):
    for new_term in new_operator.iter_terms():
      self.add_term(new_term)

  def multiply_by_scalar(self, scalar):
    """Multiplies all terms by a scalar."""
    for term in self.iter_terms():
      term.coefficient *= scalar

  def multiply_by_term(self, new_term):
    """Multiplies the LocalOperator by a new LocalTerm.

    Args:
      new_term: LocalTerm object.
    """
    new_operator = LocalOperator(self.n_qubits)
    for term in self.iter_terms():
      term.multiply_by_term(new_term)
      new_operator.add_term(term)
    self.terms = new_operator.terms

  def multiply_by_operator(self, new_operator):
    """Multiplies two LocalOperators together.

    Args:
      new_operator: LocalOperator which will multiply self.
    """
    product_operator = LocalOperator(self.n_qubits)
    for term in self.iter_terms():
      for new_term in new_operator.iter_terms():
        cloned_term = copy.deepcopy(term)
        cloned_term.multiply_by_term(new_term)
        product_operator.add_term(cloned_term)
    self.terms = product_operator.terms

  def list_coefficients(self):
    """Return the coefficients of all the terms in the operator

    Returns:
      A list of complex floats giving the operator term coefficients.
    """
    coefficients = [term.coefficient for term in self.iter_terms()]
    return coefficients

  def print_operator(self):
    for term in self.iter_terms():
      print(term.__str__())

  def count_terms(self):
    return len(self.terms)

  def __eq__(self, operator):
    """Compare operators to see if they are the same."""
    if self.n_qubits != operator.n_qubits:
      raise ErrorLocalOperator(
          'Cannot compare operators acting on different Hilbert spaces.')
    if len(self.terms) != len(operator.terms):
      return False
    for term in self.iter_terms():
      if term.key() in operator.terms:
        if term == operator.terms[term.key()]:
          continue
      return False
    return True

  def __ne__(self, operator):
    return not (self == operator)

  def remove_term(self, operators):
    if isinstance(operators, list):
      operators = tuple(operators)
    if operators in self.terms:
      del self.terms[operators]

  def look_up_coefficient(self, operators):
    """Given operators list, look up coefficient."""
    if isinstance(operators, list):
      operators = tuple(operators)
    if operators in self.terms:
      term = self.terms[operators]
      return term.coefficient
    else:
      return 0.

  def __call__(self, operators):
    """Provide a very easy way of looking up term coefficients."""
    # TODO: Perhaps it would be best to overload slice with __getitem__.
    return self.look_up_coefficient(operators)

  def list_terms(self):
    return self.terms.values()

  def list_coefficients(self):
    coefficients = [term.coefficient for term in self.iter_terms()]
    return coefficients
