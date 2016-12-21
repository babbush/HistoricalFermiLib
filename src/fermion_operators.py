"""Class to store and transform fermion operators.
"""
import molecular_operators
import local_operators
import qubit_operators
import numpy
import copy


class ErrorJordanWigner(Exception):
  pass


class ErrorMolecularOperator(Exception):
  pass


class ErrorFermionOperator(Exception):
  pass


def number_operator(n_qubits, site=None, coefficient=1.):
  """Return a number operator.

  Args:
    n_qubits: An int giving the number of spin-orbitals in the system.
    site: The site on which to return the number operator.
      If None, return number operator on all sites.
  """
  if site is None:
    operator = FermionOperator(n_qubits)
    for spin_orbital in range(n_qubits):
      operator.add_term(number_operator(n_qubits, spin_orbital))
  else:
    operator = FermionTerm(n_qubits, coefficient, [(site, 1), (site, 0)])
  return operator


class FermionTerm(local_operators.LocalTerm):
  """Stores a single term composed of products of fermionic ladder operators.

  Attributes:
    n_qubits: An int giving the number of spin-orbitals in the system.
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

  def get_hermitian_conjugate(self):
    """Calculate hermitian conjugate.

    Returns:
      New FermionTerm object which is the hermitian conjugate.
    """
    conjugate_coefficient = numpy.conjugate(self.coefficient)
    conjugate_operators = [(operator[0], not operator[1]) for operator in
                           self.operators[::-1]]
    hermitian_conjugate = FermionTerm(self.n_qubits,
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
            new_term = FermionTerm(term.n_qubits,
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
    # Initialize identity matrix.
    transformed_term = qubit_operators.QubitOperator(
        self.n_qubits, [qubit_operators.QubitTerm(self.n_qubits,
                                                  self.coefficient)])
    # Loop through operators, transform and multiply.
    for operator in self.operators:

      # Handle identity.
      pauli_x_component = qubit_operators.QubitTerm(
          self.n_qubits, 0.5,
          [(operator[0], 'X')] +
          [(index, 'Z') for index in range(operator[0] - 1, -1, -1)])
      if operator[1]:
        pauli_y_component = qubit_operators.QubitTerm(
            self.n_qubits, -0.5j,
            [(operator[0], 'Y')] +
            [(index, 'Z') for index in range(operator[0] - 1, -1, -1)])
      else:
        pauli_y_component = qubit_operators.QubitTerm(
            self.n_qubits, 0.5j,
            [(operator[0], 'Y')] +
            [(index, 'Z') for index in range(operator[0] - 1, -1, -1)])
      transformed_operator = qubit_operators.QubitOperator(
          self.n_qubits, [pauli_x_component, pauli_y_component])
      transformed_term.multiply_by_operator(transformed_operator)
    return transformed_term


class FermionOperator(local_operators.LocalOperator):
  """Data structure which stores sums of FermionTerm objects.

  Attributes:
    n_qubits: An int giving the number of spin-orbitals in the system.
    terms: A dictionary of FermionTerm objects. The key is given as
        FermionTerm.key() and the value is the FermionTerm.
  """
  def normal_order(self):
    normal_ordered_operator = FermionOperator(self.n_qubits)
    for old_term in self.iter_terms():
      new_operator = old_term.return_normal_order()
      normal_ordered_operator.add_operator(new_operator)

    # Remove terms that are zero after normal ordering.
    # TODO Jarrod: Remove such terms during the normal ordering process.
    normal_ordered_operator.remove_zero_terms()
    self.terms = normal_ordered_operator.terms

  def remove_zero_terms(self):
    """Remove terms that would equate to zero, e.g. a_i a_i"""
    terms_to_remove = []
    for term in self.iter_terms():

      # Remove terms with zero expectation value.
      for i in range(1, len(term.operators)):
        if (term.operators[i - 1] == term.operators[i]):
          terms_to_remove.append(term.key())
          break

    # Remove terms.
    for key in terms_to_remove:
      del self.terms[key]

  def jordan_wigner_transform(self):
    transformed_operator = qubit_operators.QubitOperator(self.n_qubits)
    for term in self.iter_terms():
      transformed_term = term.jordan_wigner_transform()
      transformed_operator.add_operator(transformed_term)
    return transformed_operator

  def bravyi_kitaev_transform(self):
    # TODO Jarrod.
    return None

  def get_molecular_operator(self):
    """Convert a 2-body fermionic operator to instance of MolecularOperator.

    This function should only be called on fermionic operators which consist
    of only a_p^\dagger a_q and a_p^\dagger a_q^\dagger a_r a_s terms.
    The one-body terms are stored in a matrix, one_body[p, q], and the
    two-body terms are stored in a tensor, two_body[p, q, r, s].

    Returns:
      molecular_operator: An instance of the MolecularOperator class.

    Raises:
      ErrorMolecularOperator: FermionOperator is not a molecular operator.
    """
    # Normal order the terms and initialize.
    self.normal_order()
    constant = 0.
    one_body = numpy.zeros((self.n_qubits, self.n_qubits), float)
    two_body = numpy.zeros((
        self.n_qubits, self.n_qubits, self.n_qubits, self.n_qubits), float)

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

    # Form MolecularOperator and return.
    molecular_operator = molecular_operators.MolecularOperator(
        constant, one_body, two_body)
    return molecular_operator
