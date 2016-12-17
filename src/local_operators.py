"""Base classes for representation of various local operator types.
"""

# Set the tolerance below which a coefficient is regarded as zero.
_TOLERANCE = 1e-15


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
    """
    if self.n_qubits != local_term.n_qubits:
      return False
    elif abs(self.coefficient - local_term.coefficient) > _TOLERANCE:
      return False
    elif self.operators != local_term.operators:
      return False
    else:
      return True

  def __ne__(self, local_term):
    """Overload not equals comparison != to interact with standard library."""
    return not (self == local_term)

  def multiply_by_term(self, local_term):
    """Multiplies a self by another LocalTerm (new one is on right)."""
    self.coefficient *= local_term.coefficient
    self.operators += local_term.operators

  def key(self):
    """Returns a hashable unique key representing operators."""
    return tuple(self.operators)

  def __str__(self):
    string_representation = '{} {}'.format(
        self.coefficient, self.key())
    return string_representation


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
      self.add_terms_list(terms)
    else:
      raise ErrorLocalOperator('Invalid terms provided to initialization.')

  def list_terms(self):
    return self.terms.values()

  def list_coefficients(self):
    coefficients = [term.coefficient for term in self.iter_terms()]
    return coefficients

  def iter_terms(self):
    return self.terms.itervalues()

  def add_term(self, new_term):
    """Add another LocalTerm to the LocalOperator.

    Args:
      new_term: LocalTerm object. It is added to the LocalOperator.

    Raises:
      ErrorLocalTerm: Cannot add terms which act on different Hilbert spaces.
    """
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
      print type(new_term)
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
        term.multiply_by_term(new_term)
        product_operator.add_term(term)
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
    if self.count_terms() != operator.count_terms():
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
    return self.look_up_coefficient(operators)
