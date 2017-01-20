"""Class to store and transform fermion operators.
"""
import molecular_operators
import qubit_operators
import local_operators
import local_terms
import numpy
import copy


class JordanWignerError(Exception):
  pass


class FermionTermError(Exception):
  pass


class FermionOperatorError(Exception):
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
      operator += number_operator(n_qubits, spin_orbital)
  else:
    operator = FermionTerm(n_qubits, coefficient, [(site, 1), (site, 0)])
  return operator


class FermionTerm(local_terms.LocalTerm):
  """Stores a single term composed of products of fermionic ladder operators.

  Attributes:
    _n_qubits: An int giving the number of spin-orbitals in the system.
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
  def __init__(self, n_qubits, coefficient=0., operators=None):
    """Init a FermionTerm.

    Args:
      n_qubits: Int, the number of qubits in the system.
      coefficient: A complex valued float giving the term coefficient.
      operators: A list of tuples. The first element of each tuple is an
          int indicating the site on which operators acts. The second element
          of each tuple is boole, indicating raising (1) or lowering (0).

    Raises:
      FermionTermError: Invalid operators provided to FermionTerm.
    """
    super(FermionTerm, self).__init__(n_qubits, coefficient, operators)
    for operator in self:
      if isinstance(operator, tuple):
        tensor_factor, action = operator
        if ((action == 1 or action == 0) and
           (isinstance(tensor_factor, int) and tensor_factor < n_qubits)):
          continue
      raise FermionTermError('Invalid operators provided to FermionTerm.')

  def __add__(self, addend):
    """Compute self + addend for a FermionTerm.

    Note that we only need to handle the case of adding other fermion terms.

    Args:
      addend: A FermionTerm.

    Returns:
      summand: A new instance of FermionOperator.

    Raises:
      TypeError: Object of invalid type cannot be added to FermionTerm.
      FermionTermError: Cannot add terms acting on different Hilbert spaces.
    """
    if not issubclass(type(addend),
                      (FermionTerm, FermionOperator)):
      raise TypeError('Cannot add term of invalid type to FermionTerm.')

    if not self._n_qubits == addend._n_qubits:
      raise FermionTermError(
        'Cannot add terms acting on different Hilbert spaces.')

    return FermionOperator(self._n_qubits, [self]) + addend

  def __str__(self):
    """Return an easy-to-read string representation of the term."""
    string_representation = '{} ('.format(self.coefficient)
    for operator in self:
      if operator[1]:
        string_representation += '{}+ '.format(operator[0])
      else:
        string_representation += '{} '.format(operator[0])
    n_characters = len(string_representation)
    if self:
      string_representation = string_representation[:(n_characters - 1)]
    string_representation += ')'
    return string_representation

  def get_hermitian_conjugate(self):
    """Calculate hermitian conjugate.

    Returns:
      New FermionTerm object which is the hermitian conjugate.
    """
    conjugate_coefficient = numpy.conjugate(self.coefficient)
    conjugate_operators = [(operator[0], not operator[1]) for
                           operator in self[::-1]]
    hermitian_conjugate = FermionTerm(self._n_qubits,
                                      conjugate_coefficient,
                                      conjugate_operators)
    return hermitian_conjugate

  def is_normal_ordered(self):
    """Function to return whether or not term is in normal order."""
    for i in range(1, len(self)):
      for j in range(i, 0, -1):
        right_operator = self[j]
        left_operator = self[j - 1]
        if right_operator[1] and not left_operator[1]:
          return False
        elif (right_operator[1] == left_operator[1] and
              right_operator[0] >= left_operator[0]):
          return False
    return True

  def return_normal_order(self):
    """Compute and return the normal ordered form of a FermionTerm.

    Not an in-place method.

    Returns:
      normal_ordered_operator: FermionOperator object which is the
          normal ordered form.
    """
    # Initialize output.
    normal_ordered_operator = FermionOperator(self._n_qubits)

    # Copy self.
    term = copy.deepcopy(self)

    # Iterate from left to right across operators and reorder to normal form.
    # Swap terms operators into correct position by moving left to right.
    for i in range(1, len(term)):
      for j in range(i, 0, -1):
        right_operator = term[j]
        left_operator = term[j - 1]

        # Swap operators if raising on right and lowering on left.
        if right_operator[1] and not left_operator[1]:
          term[j - 1] = right_operator
          term[j] = left_operator
          term *= -1.

          # Replace a a^\dagger with 1 - a^\dagger a if indices are same.
          if right_operator[0] == left_operator[0]:
            operators_in_new_term = term[:(j - 1)]
            operators_in_new_term += term[(j + 1)::]
            new_term = FermionTerm(term._n_qubits,
                                   -1. * term.coefficient,
                                   operators_in_new_term)

            # Recursively add the processed new term.
            normal_ordered_operator += new_term.return_normal_order()

          # Handle case when operator type is the same.
        elif right_operator[1] == left_operator[1]:

          # If same two operators are repeated, term evaluates to zero.
          if right_operator[0] == left_operator[0]:
            return normal_ordered_operator

            # Swap if same ladder type but lower index on left.
          elif right_operator[0] > left_operator[0]:
            term[j - 1] = right_operator
            term[j] = left_operator
            term *= -1.

    # Add processed term to output and return.
    normal_ordered_operator += term
    return normal_ordered_operator

  def bravyi_kitaev_transform(self):
    # TODO Jarrod.
    return None

  def jordan_wigner_transform(self):
    """Apply the Jordan-Wigner transform and return qubit operator.

    Returns:
      transformed_term: An instance of the QubitOperator class.
    """
    # Initialize identity matrix.
    transformed_term = qubit_operators.QubitOperator(
        self._n_qubits, [qubit_operators.QubitTerm(self._n_qubits,
                                                   self.coefficient)])
    # Loop through operators, transform and multiply.
    for operator in self:

      # Handle identity.
      pauli_x_component = qubit_operators.QubitTerm(
          self._n_qubits, 0.5,
          [(operator[0], 'X')] +
          [(index, 'Z') for index in range(operator[0] - 1, -1, -1)])
      if operator[1]:
        pauli_y_component = qubit_operators.QubitTerm(
            self._n_qubits, -0.5j,
            [(operator[0], 'Y')] +
            [(index, 'Z') for index in range(operator[0] - 1, -1, -1)])
      else:
        pauli_y_component = qubit_operators.QubitTerm(
            self._n_qubits, 0.5j,
            [(operator[0], 'Y')] +
            [(index, 'Z') for index in range(operator[0] - 1, -1, -1)])
      transformed_term *= qubit_operators.QubitOperator(
          self._n_qubits, [pauli_x_component, pauli_y_component])
    return transformed_term


class FermionOperator(local_operators.LocalOperator):
  """Data structure which stores sums of FermionTerm objects.

  Attributes:
    _n_qubits: An int giving the number of spin-orbitals in the system.
    terms: A dictionary of FermionTerm objects.
  """
  def __init__(self, n_qubits, terms=None):
    """Init a FermionOperator.

    Args:
      n_qubits: Int, the number of qubits in the system.
      terms: Dictionary or list of FermionTerm objects.

    Raises:
      FermionOperatorError: Invalid FermionTerms provided to FermionOperator.
    """
    super(FermionOperator, self).__init__(n_qubits, terms)
    for term in self:
      if isinstance(term, FermionTerm) and term._n_qubits == n_qubits:
          continue
      raise FermionTermError(
          'Invalid FermionTerms provided to FermionOperator.')

  def __setitem__(self, operators, coefficient):
    if operators in self:
      self.terms[tuple(operators)].coefficient = coefficient
    else:
      new_term = FermionTerm(self.n_qubits, coefficient, operators)
      self.terms[tuple(operators)] = new_term

  def normal_order(self):
    normal_ordered_operator = FermionOperator(self._n_qubits)
    for term in self:
      normal_ordered_operator += term.return_normal_order()
    self.terms = normal_ordered_operator.terms

  def jordan_wigner_transform(self):
    transformed_operator = qubit_operators.QubitOperator(self._n_qubits)
    for term in self:
      transformed_operator += term.jordan_wigner_transform()
    return transformed_operator

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
    one_body = numpy.zeros((self._n_qubits, self._n_qubits), complex)
    two_body = numpy.zeros((
        self._n_qubits, self._n_qubits, self._n_qubits, self._n_qubits),
        complex)

    # Loop through terms and assign to matrix.
    for term in self:
      coefficient = term.coefficient

      # Handle constant shift.
      if not len(term):
        constant = coefficient

      elif len(term) == 2:
        # Handle one-body terms.
        if [operator[1] for operator in term] == [1, 0]:
          p, q = [operator[0] for operator in term]
          one_body[p, q] = coefficient
        else:
          raise ErrorMolecularOperator(
              'FermionOperator is not a molecular operator.')

      elif len(term) == 4:
        # Handle two-body terms.
        if [operator[1] for operator in term] == [1, 1, 0, 0]:
          p, q, r, s = [operator[0] for operator in term]
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
