"""Class to store and transform fermionic operators.
"""
import pauli_data
import numpy
import copy


class ErrorJordanWigner(Exception):
  pass


class ErrorMolecularOperator(Exception):
  pass


def jordan_wigner_ladder(n_qubits, site, ladder_type):
  """Transforms a single fermionic raising or lowering operator.

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
    operator = FermionicOperator(n_sites)
    for spin_orbital in range(n_sites):
      operator.add_term(number_operator(n_sites, spin_orbital))
  else:
    operator = FermionicTerm(n_sites, coefficient, [(site, 1), (site, 0)])
  return operator


class FermionicTerm(object):
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
    """Inits a fermionic operator.

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

  def is_identical_term(self, fermionic_term):
    """Queries whether term is the same as another fermionic_term.

    Args:
      fermionic_term: Another FermionicTerm.

    Returns:
      True or False, whether terms are the same (without normal ordering).
    """
    return self.operators == fermionic_term.operators

  def multiply_by_term(self, fermionic_term):
    """Multiplies a fermionic term by another one (new one is on right)."""
    self.coefficient *= fermionic_term.coefficient
    self.operators += fermionic_term.operators

  def get_hermitian_conjugate(self):
    """Calculate hermitian conjugate.

    Returns:
      New FermionicTerm object which is the hermitian conjugate.
    """
    conjugate_coefficient = numpy.conjugate(self.coefficient)
    conjugate_operators = [(operator[0], not operator[1]) for operator in
                           self.operators[::-1]]
    hermitian_conjugate = FermionicTerm(self.n_sites,
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
      normal_ordered_operator: FermionicOperator object which is the
        normal ordered form.
    """
    # Initialize output.
    normal_ordered_operator = FermionicOperator(self.n_sites)

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
            new_term = FermionicTerm(term.n_sites,
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


class FermionicOperator(object):
  """Data structure which stores sums of FermionicTerm objects.

  Attributes:
    n_sites: An int giving the number of fermionic modes.
    terms: A list of FermionicTerm objects.
  """
  def __init__(self, n_sites, terms=None):
    self.n_sites = n_sites
    if terms is None:
      self.terms = []
    else:
      self.terms = terms

  def add_raw_term(self, coefficient, operators):
    new_term = FermionicTerm(self.n_sites, coefficient, operators)
    self.add_term(new_term)

  def add_term(self, new_term, tolerance=1e-14):
    for term_number, old_term in enumerate(self.terms):
      if old_term.is_identical_term(new_term):
        # Delete the term entirely if addition cancels it out.
        if abs(old_term.coefficient + new_term.coefficient) < tolerance:
          del self.terms[term_number]
        else:
          old_term.coefficient += new_term.coefficient
        return
    self.terms += [new_term]

  def add_operator(self, new_operator):
    for new_term in new_operator.terms:
      self.add_term(new_term)

  def normal_order(self):
    normal_ordered_operator = FermionicOperator(self.n_sites)
    for old_term in self.terms:
      new_operator = old_term.return_normal_order()
      normal_ordered_operator.add_operator(new_operator)
    self.terms = normal_ordered_operator.terms

  def jordan_wigner_transform(self):
    transformed_operator = pauli_data.QubitOperator(self.n_sites)
    for term in self.terms:
      transformed_term = term.jordan_wigner_transform()
      transformed_operator.add_operator(transformed_term)
    return transformed_operator

  def multiply_by_scalar(self, scalar):
    for term in self.terms:
      term.coefficient *= scalar

  def multiply_by_term(self, term):
    for term in self.terms:
      term.multiply_by_term(term)

  def multiply_by_operator(self, operator):
    new_operator = FermionicOperator(self.n_sites)
    for term_a in self.terms:
      for term_b in operator.terms:
        new_term = copy.deepcopy(term_a)
        new_term.multiply_by_term(term_b)
        new_operator.add_term(new_term)
    self.terms = new_operator.terms

  def print_operator(self):
    for term in self.terms:
      print(term.__str__())

  def parity_transform(self):
    # TODO
    return None

  def bravyi_kitaev_transform(self):
    # TODO
    return None

  def look_up_coefficient(self, operators):
    fermionic_term = FermionicTerm(self.n_sites, 1., operators)
    for term in self.terms:
      if term.is_identical_term(fermionic_term):
        return term.coefficient
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
      ErrorMolecularOperator: FermionicOperator is not a molecular operator.
    """
    # Normal order the terms and initialize.
    self.normal_order()
    constant = 0.
    one_body = numpy.zeros((self.n_sites, self.n_sites), float)
    two_body = numpy.zeros((
        self.n_sites, self.n_sites, self.n_sites, self.n_sites), float)

    # Loop through terms and assign to matrix.
    for term in self.terms:
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
              'FermionicOperator is not a molecular operator.')

      elif len(term.operators) == 4:
        # Handle two-body terms.
        if [operator[1] for operator in term.operators] == [1, 1, 0, 0]:
          p, q, r, s = [operator[0] for operator in term.operators]
          two_body[p, q, r, s] = coefficient
        else:
          raise ErrorMolecularOperator(
              'FermionicOperator is not a molecular operator.')

      else:
        # Handle non-molecular Hamiltonian.
        raise ErrorMolecularOperator(
            'FermionicOperator is not a molecular operator.')

    # Return.
    return constant, one_body, two_body

  def count_terms(self):
    return len(self.terms)

  def get_coefficients(self):
    coefficients = [term.coefficient for term in self.terms]
    return coefficients

  def is_identical_operator(self, operator, tolerance=1e-15):
    self.normal_order()
    operator.normal_order()
    if self.count_terms() != operator.count_terms():
      return False
    for term in self.terms:
      difference = abs(term.coefficient -
                       operator.look_up_coefficient(term.operators))
      if difference > tolerance:
        return False
    return True

  def remove_term(self, operators):
    for term_number, term in enumerate(self.terms):
      if term.operators == operators:
        del self.terms[term_number]

  def expectation(self, fermionic_operator):
    """Take the expectation value of self with another fermionic operator.

    Args:
      fermionic_operator: An instance of the FermionicOperator class.

    Returns:
      expectation: A float, giving the expectation value.
    """
    expectation = 0.
    self.normal_order()
    fermionic_operator.normal_order()
    for term in self.terms:
      complement = fermionic_operator.look_up_coefficient(term.operators)
      expectation += term.coefficient * complement
    return expectation
