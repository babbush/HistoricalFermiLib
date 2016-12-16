"""Class to store and transform fermion operators.
"""
import pauli_data
import numpy
import copy


# Set tolerance for a term to be considered zero.
_TOLERANCE = 1e-14


class ErrorJordanWigner(Exception):
  pass


class ErrorMolecularOperator(Exception):
  pass


class ErrorFermionOperator(Exception):
  pass


def jordan_wigner_ladder(n_qubits, site, ladder_type):
  """Transforms a single fermion raising or lowering operator.

  Args:
    n_qubits: Int, the number of qubits.
    site: Int, the site on which operator acts.
    ladder_type: Boole indicating whether raising (1) or lowering (0).

  Returns:
    transformed_operator: An instance of the QubitOperator class.
  """
  pauli_x_component = pauli_data.PauliString(
      n_qubits, 0.5,
      [(site, 'X')] + [(index, 'Z') for index in range(site - 1, -1, -1)])
  if ladder_type:
    pauli_y_component = pauli_data.PauliString(
        n_qubits, -0.5j,
        [(site, 'Y')] + [(index, 'Z') for index in range(site - 1, -1, -1)])
  else:
    pauli_y_component = pauli_data.PauliString(
        n_qubits, 0.5j,
        [(site, 'Y')] + [(index, 'Z') for index in range(site - 1, -1, -1)])
  transformed_operator = pauli_data.QubitOperator(
      n_qubits, [pauli_x_component, pauli_y_component])
  return transformed_operator


def number_operator(n_sites, site=None, coefficient=1.):
  """Return a number operator.

  Args:
    n_sites: An int giving the number of spin-orbitals in the system.
    site: The site on which to return the number operator.
      If None, return number operator on all sites.
  """
  if site is None:
    operator = FermionOperator(n_sites)
    for spin_orbital in range(n_sites):
      operator.add_term(number_operator(n_sites, spin_orbital))
  else:
    operator = FermionTerm(n_sites, coefficient, [(site, 1), (site, 0)])
  return operator


class FermionTerm(object):
  """Stores a single term composed of products of fermionic ladder operators.

  Attributes:
    n_sites: An int giving the number of spin-orbitals in the system.
    coefficient: A complex valued float giving the term coefficient.
    operators: A list of tuples. The first element of each tuple is an
      int indicating the site on which operators acts. The second element
      of each tuple is boole, indicating whether raising (1) or lowering (0).

    Example usage:
      Consider the term 6.7 * a_3^\dagger a_1 a_7^\dagger
      This object would have the attributes:
      term.coefficient = 6.7
      term.operators = [(3, 1), (1, 0), (7, 1)]
  """
  def __init__(self, n_sites, coefficient=0., operators=None):
    """Inits a FermionTerm.

    Args:
      n_sites: An int giving the number of spin-orbitals in the system.
      coefficient: A complex valued float giving the term coefficient.
      operators: A list of tuples. The first element of each tuple is an
        int indicating the site on which operators acts. The second element
        of each tuple is boole, indicating whether raising (1) or lowering (0).
    """
    self.n_sites = n_sites
    self.coefficient = coefficient
    if operators is None:
      self.operators = []
    else:
      self.operators = operators

  def __eq__(self, fermion_term):
    """Overload equality comparison == to interact with python standard library.
    Args:
      fermion_term: Another FermionTerm.

    Returns:
      True or False, whether terms are the same (without normal ordering).
    """
    if self.n_qubits != fermion_term.n_qubits:
      return False
    elif abs(self.coefficient - fermion_term.coefficient) < _TOLERANCE:
      return False
    elif self.operators != fermion_term.operators:
      return False
    else:
      return True

  def __ne__(self, fermion_term):
    """Overload not equals comparison != to interact with python standard library"""
    return not (self == fermion_term)

  def multiply_by_term(self, fermion_term):
    """Multiplies a fermionic term by another one (new one is on right)."""
    self.coefficient *= fermion_term.coefficient
    self.operators += fermion_term.operators

  def get_hermitian_conjugate(self):
    """Calculate hermitian conjugate.

    Returns:
      New FermionTerm object which is the hermitian conjugate.
    """
    conjugate_coefficient = numpy.conjugate(self.coefficient)
    conjugate_operators = [(operator[0], not operator[1]) for operator in
                           self.operators[::-1]]
    hermitian_conjugate = FermionTerm(self.n_sites,
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
    normal_ordered_operator = FermionOperator(self.n_sites)

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
            new_term = FermionTerm(term.n_sites,
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
    transformed_term = pauli_data.QubitOperator(self.n_sites, [
        pauli_data.PauliString(self.n_sites)])
    for operator in self.operators:
      ladder_operator = jordan_wigner_ladder(
          self.n_sites, operator[0], operator[1])
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


class FermionOperator(object):
  """Data structure which stores sums of FermionTerm objects.

  Attributes:
    n_sites: An int giving the number of fermionic modes.
    terms: A dictionary of FermionTerm objects. The key is given as
        FermionTerm.key() and the value is the FermionTerm.
  """
  def __init__(self, n_sites, terms=None):
    """Init a FermionOperator object.

    Args:
      n_sites: The number of sites in the fermion lattice.
      terms: This can be either a python list of FermionTerm objects
          or a dictionary of FermionTerm objects with the keys given as
          FermionTerm.key() and the value is the FermionTerm.
          If None, then initialize empty FermionOperator.

    Raises:
      ErrorFermionOperator: Invalid terms provided to initialization.
    """
    self.n_sites = n_sites
    if terms is None:
      self.terms = {}
    elif isinstance(terms, dict):
      self.terms = terms
    elif isinstance(terms, list):
      self.terms = {}
      self.add_terms_list(terms)
    else:
      raise ErrorFermionOperator('Invalid terms provided to initialization.')

  def list_terms(self):
    return self.terms.values()

  def iter_terms(self):
    return self.terms.itervalues()

  def add_term(self, new_term):
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

  def normal_order(self):
    normal_ordered_operator = FermionOperator(self.n_sites)
    for old_term in self.iter_terms():
      new_operator = old_term.return_normal_order()
      normal_ordered_operator.add_operator(new_operator)
    self.terms = normal_ordered_operator.terms

  def jordan_wigner_transform(self):
    transformed_operator = pauli_data.QubitOperator(self.n_sites)
    for term in self.iter_terms():
      transformed_term = term.jordan_wigner_transform()
      transformed_operator.add_operator(transformed_term)
    return transformed_operator

  def multiply_by_scalar(self, scalar):
    for term in self.iter_terms():
      term.coefficient *= scalar

  def multiply_by_term(self, new_term):
    for term in self.iter_terms():
      term.multiply_by_term(new_term)

  def multiply_by_operator(self, operator):
    new_operator = FermionOperator(self.n_sites)
    for term_a in self.iter_terms():
      for term_b in operator.iter_terms():
        new_term = copy.deepcopy(term_a)
        new_term.multiply_by_term(term_b)
        new_operator.add_term(new_term)
    self.terms = new_operator.terms

  def print_operator(self):
    for term in self.iter_terms():
      print(term.__str__())

  def bravyi_kitaev_transform(self):
    # TODO Jarrod.
    return None

  def look_up_coefficient(self, operators):
    """Given operators list, look up coefficient."""
    if isinstance(operators, list):
      operators = tuple(operators)
    if operators in self.terms:
      term = self.terms[operators]
      return term.coefficient
    else:
      return 0.

  def to_molecular_operator(self):
    """Convert a 2-body fermionic operator to a molecular operator matrix.

    This function should only be called on fermionic operators which consist
    of only a_p^\dagger a_q and a_p^\dagger a_q^\dagger a_r a_s terms.
    The one-body terms are stored in a matrix, one_body[p, q], and the
    two-body terms are stored in a tensor, two_body[p, q, r, s].

    Returns:
      constant: The coefficient of identity (float)
      one_body: The N x N numpy array of floats giving one-body terms.
      two_body: The N x N x N x N numpy array of floats giving two-body terms.

    Raises:
      ErrorMolecularOperator: FermionOperator is not a molecular operator.
    """
    # Normal order the terms and initialize.
    self.normal_order()
    constant = 0.
    one_body = numpy.zeros((self.n_sites, self.n_sites), float)
    two_body = numpy.zeros((
        self.n_sites, self.n_sites, self.n_sites, self.n_sites), float)

    # Loop through terms and assign to matrix.
    for term in self.iter_terms():
      coefficient = term.coefficient

      # Handle constant shift.
      if not len(term.operators):
        constant = coefficient

      elif len(term.operators) == 2:
        # Handle one-body terms.
        if [operator[1] for operator in term.operators] == [1, 0]:
          p, q = [operator[0] for operator in term.operators]
          one_body[p, q] = coefficient
        else:
          raise ErrorMolecularOperator(
              'FermionOperator is not a molecular operator.')

      elif len(term.operators) == 4:
        # Handle two-body terms.
        if [operator[1] for operator in term.operators] == [1, 1, 0, 0]:
          p, q, r, s = [operator[0] for operator in term.operators]
          two_body[p, q, r, s] = coefficient
        else:
          raise ErrorMolecularOperator(
              'FermionOperator is not a molecular operator.')

      else:
        # Handle non-molecular Hamiltonian.
        raise ErrorMolecularOperator(
            'FermionOperator is not a molecular operator.')

    # Return.
    return constant, one_body, two_body

  def count_terms(self):
    return len(self.terms)

  def get_coefficients(self):
    coefficients = [term.coefficient for term in self.iter_terms()]
    return coefficients

  def __eq__(self, operator):
    self.normal_order()
    operator.normal_order()
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

  def expectation(self, reduced_one_body, reduced_two_body):
    """Take the expectation value of self with a one- and two- body operator

    Args:
      reduced_one_body: N x N numpy array representing the reduced one-
        electron density matrix
      reduced_two_body: N x N x N x N numpy array representing the two-
        electron reduced density matrix

    Returns:
      expectation: A float, giving the expectation value.
    """

    self.normal_order()
    expectation = 0.

    for term in self.terms:
      if (len(term.operators) == 2):
        reduced_value = reduced_one_body[term.operators[0][0],
                                         term.operators[1][0]]
      elif (len(term.operators) == 4):
        reduced_value = reduced_two_body[term.operators[0][0],
                                         term.operators[1][0],
                                         term.operators[2][0],
                                         term.operators[3][0]]
      elif (len(term.operators) == 0):
        reduced_value = 1.0
      else:
        raise ErrorMolecularOperator(
          'FermionOperator is not a molecular operator.')

      expectation += term.coefficient * reduced_value
    return expectation
