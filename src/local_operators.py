# Set the tolerance regarding a coefficient as zero.
_TOLERANCE = 1e-15

class LocalTerm(object):
  """Represents a term consisting of a product of operators and a coefficient.

  Attributes:
    n_qubits: An int giving the number of qubits in the space term acts.
    coefficient: A complex valued float giving the term coefficient.
    operators: A list of operators representing term.
  """
  def __init__(self, n_qubits, coefficient=0., operators=None):
    """Inits a LocalTerm.

    Args:
      n_qubits: An int giving the number of spin-orbitals in the system.
      coefficient: A complex valued float giving the term coefficient.
      operators: A list of tuples. The first element of each tuple is an
        int indicating the site on which operators acts. The second element
        of each tuple is boole, indicating whether raising (1) or lowering (0).
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
      local_term: Another LocalTerm.

    Returns:
      True or False, whether terms are the same (without normal ordering).
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
    """Multiplies a fermionic term by another one (new one is on right)."""
    self.coefficient *= local_term.coefficient
    self.operators += local_term.operators

  def get_hermitian_conjugate(self):
    """Calculate hermitian conjugate.

    Returns:
      New LocalTerm object which is the hermitian conjugate.
    """
    conjugate_coefficient = numpy.conjugate(self.coefficient)
    conjugate_operators = [(operator[0], not operator[1]) for operator in
                           self.operators[::-1]]
    hermitian_conjugate = LocalTerm(self.n_qubits,
                                      conjugate_coefficient,
                                      conjugate_operators)
    return hermitian_conjugate

  def is_normal_ordered(self):
    """Function to return whether or not term is in normal order."""
    for i in range(1, len(self.operators)):
      for j in range(i, 0, -1):
        right_operator = self.operators[j]
        left_operator = self.operators[j - 1]
        if right_operator[1] and not left_operator[1]:
          return False
        elif (right_operator[1] == left_operator[1] and
              right_operator[0] > left_operator[0]):
          return False
    return True

  def return_normal_order(self):
    """Compute and return the normal ordered form of operator.

    Returns:
      normal_ordered_operator: FermionOperator object which is the
        normal ordered form.
    """
    # Initialize output.
    normal_ordered_operator = FermionOperator(self.n_qubits)

    # Copy self.
    term = copy.copy(self)

    # Iterate from left to right across operators and reorder to normal form.
    # Swap terms operators into correct position by moving left to right.
    for i in range(1, len(term.operators)):
      for j in range(i, 0, -1):
        right_operator = term.operators[j]
        left_operator = term.operators[j - 1]

        # Swap operators if raising on right and lowering on left.
        if right_operator[1] and not left_operator[1]:
          term.operators[j - 1] = right_operator
          term.operators[j] = left_operator
          term.coefficient *= -1.

          # Replace a a^\dagger with 1 - a^\dagger a if indices are same.
          if right_operator[0] == left_operator[0]:
            operators_in_new_term = term.operators[:(j - 1)]
            operators_in_new_term += term.operators[(j + 1)::]
            new_term = LocalTerm(term.n_qubits,
                                   -1. * term.coefficient,
                                   operators_in_new_term)

            # Recursively add the processed new term.
            normal_ordered_operator.add_operator(
                new_term.return_normal_order())

        # Also swap if same ladder type but lower index on left.
        elif (right_operator[1] == left_operator[1] and
              right_operator[0] > left_operator[0]):
          term.operators[j - 1] = right_operator
          term.operators[j] = left_operator
          term.coefficient *= -1.

    # Add processed term to output and return.
    normal_ordered_operator.add_term(term)
    return normal_ordered_operator

  def jordan_wigner_transform(self):
    """Apply the Jordan-Wigner transform and return qubit operator.

    Returns:
      transformed_term: An instance of the QubitOperator class.
    """
    transformed_term = qubit_operators.QubitOperator(self.n_qubits, [
        qubit_operators.QubitTerm(self.n_qubits)])
    for operator in self.operators:
      ladder_operator = jordan_wigner_ladder(
          self.n_qubits, operator[0], operator[1])
      transformed_term.multiply_by_operator(ladder_operator)
    transformed_term.multiply_by_scalar(self.coefficient)
    return transformed_term

  def __str__(self):
    """Return an easy-to-read string representation of the term."""
    string_representation = '{} ('.format(self.coefficient)
    for operator in self.operators:
      if operator[1]:
        string_representation += '{}+ '.format(operator[0])
      else:
        string_representation += '{} '.format(operator[0])
    n_characters = len(string_representation)
    if self.operators:
      string_representation = string_representation[:(n_characters - 1)]
    string_representation += ')'
    return string_representation

  def key(self):
    """Returns a hashable unique key representing operators"""
    return tuple(self.operators)

class LocalOperator(object):
  """A collection of LocalOperator objects acting on same number of qubits.

  Note that to be a Hamiltonian which is a hermitian operator, the individual
  QubitTerm objects need to have only real valued coefficients.

  Attributes:
    n_qubits: The number of qubits on which the operator acts.
    terms: Dictionary of QubitTerm objects. The dictionary key is
        QubitTerm.key() and the dictionary value is the QubitTerm.
  """

  def __init__(self, n_qubits, terms=None):
    """Inits QubitHamiltonian.

    Args:
      n_qubits: The number of qubits the operator acts on.
      terms: A python list of QubitTerm terms or a dictionary
          of QubitTerm objects with keys of QubitTerm.key() and
          values of the QubitTerm. If None, initialize empty dict.

    Raises:
      ErrorQubitOperator: Invalid terms provided to initialization.
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
      raise ErrorQubitOperator('Invalid terms provided to initialization.')

  def list_terms(self):
    return self.terms.values()

  def iter_terms(self):
    return self.terms.itervalues()

  def add_term(self, new_term):
    """Add another QubitTerm to hamiltonian.

    If hamiltonian already has this term, then the coefficients are merged.

    Args:
      new_term: QubitTerm object. It is added to the Hamiltonian.

    Raises:
      ErrorQubitOperator: Not allowed to add this term.
    """
    # Make sure terms act on same number of qubits.
    if self.n_qubits != new_term.n_qubits:
      raise ErrorQubitOperator(
          'Cannot add terms which act on different Hilbert spaces')

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

  def add_terms_list(self, list_terms):
    for new_term in self.iter_terms():
      self.add_term(new_term)

  def add_operator(self, new_operator):
    for new_term in new_operator.iter_terms():
      self.add_term(new_term)

  def multiply_by_term(self, new_term):
    """Multiplies the QubitOperator by a new QubitTerm.

    Args:
      new_term: QubitTerm object.
    """
    new_operator = QubitOperator(self.n_qubits)
    for term in self.iter_terms():
      new_operator.add_term(multiply_qubit_terms(term, new_term))
    self.terms = new_operator.terms

  def multiply_by_operator(self, new_operator):
    """Multiplies two QubitOperators together.

    Args:
      new_operator: QubitOperator which will multiply self.
    """
    product_operator = QubitOperator(self.n_qubits)
    for term in self.iter_terms():
      for new_term in new_operator.iter_terms():
        product_operator.add_term(multiply_qubit_terms(term, new_term))
    self.terms = product_operator.terms

  def multiply_by_scalar(self, scalar):
    """Multiplies all terms by a scalar."""
    for term in self.iter_terms():
      term.coefficient *= scalar

  def get_coefficients(self):
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

  def get_coefficients(self):
    coefficients = [term.coefficient for term in self.iter_terms()]
    return coefficients

  def __eq__(self, operator):
    if self.count_terms() != operator.count_terms():
      return False
    for term in self.iter_terms():
      if term.key() in operator.terms:
        if term == operator[term.key()]:
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

  def reverse_jordan_wigner(self):
    transformed_operator = fermion_operators.FermionOperator(self.n_qubits)
    for term in self.iter_terms():
      transformed_operator.add_operator(term.reverse_jordan_wigner())
    return transformed_operator

  def to_sparse_matrix(self):
    hilbert_dimension = 2 ** self.n_qubits
    matrix_form = scipy.sparse.csc_matrix(
        (hilbert_dimension, hilbert_dimension), dtype=complex)
    for term in self.terms:
      matrix_form = matrix_form + term.to_sparse_matrix()
    return matrix_form
    return coefficients

  def expectation(self, qubit_operator):
    """Take the expectation value of self with another qubit operator.

    Args:
      qubit_operator: An instance of the QubitOperator class.

    Returns:
      expectation: A float, giving the expectation value.
    """
    expectation = 0.
    for term in self.iter_terms():
      complement = qubit_operator.look_up_coefficient(term.operators)
      expectation += term.coefficient * complement
    return expectation
