"""This files has utilities to read and store qubit hamiltonians.
"""
import copy
import scipy
import scipy.sparse
import fermionic_data
import sparse_operators


class ErrorPauliString(Exception):
  pass


class ErrorQubitOperator(Exception):
  pass


# Define products of all Pauli matrices for symbolic multiplication.
_PAULI_MATRIX_PRODUCTS = {('I', 'I'): (1., 'I'),
                          ('I', 'X'): (1., 'X'),
                          ('X', 'I'): (1., 'X'),
                          ('I', 'Y'): (1., 'Y'),
                          ('Y', 'I'): (1., 'Y'),
                          ('I', 'Z'): (1., 'Z'),
                          ('Z', 'I'): (1., 'Z'),
                          ('X', 'X'): (1., 'I'),
                          ('Y', 'Y'): (1., 'I'),
                          ('Z', 'Z'): (1., 'I'),
                          ('X', 'Y'): (1.j, 'Z'),
                          ('X', 'Z'): (-1.j, 'Y'),
                          ('Y', 'X'): (-1.j, 'Z'),
                          ('Y', 'Z'): (1.j, 'X'),
                          ('Z', 'X'): (1.j, 'Y'),
                          ('Z', 'Y'): (-1.j, 'X')}


def reverse_jordan_wigner_pauli(n_qubits, tensor_factor, operator):
  """Transforms a single pauli operator into an instance of FermionicOperator.

  Args:
    n_qubits: Int, the number of qubits.
    tensor_factor: Int, the tensor factor on which operator acts.
    operator: String, either 'X', 'Y' or 'Z'.

  Returns:
    transformed_operator: An instance of the FermionicOperator class.

  Raises:
    ErrorPauliString: Invalid operator provided: must be 'X', 'Y' or 'Z'.
  """
  if operator == 'Z':
    # Handle Pauli Z.
    identity = fermionic_data.FermionicTerm(n_qubits, 1.)
    number_operator = fermionic_data.FermionicTerm(
        n_qubits, -2., [(tensor_factor, 1), (tensor_factor, 0)])
    transformed_operator = fermionic_data.FermionicOperator(
        n_qubits, [identity, number_operator])

  else:
    if operator == 'X':
      # Handle Pauli X.
      raising_term = fermionic_data.FermionicTerm(
          n_qubits, 1., [(tensor_factor, 1)])
      lowering_term = fermionic_data.FermionicTerm(
          n_qubits, 1., [(tensor_factor, 0)])

    elif operator == 'Y':
      # Handle Pauli Y.
      raising_term = fermionic_data.FermionicTerm(
          n_qubits, 1.j, [(tensor_factor, 1)])
      lowering_term = fermionic_data.FermionicTerm(
          n_qubits, -1.j, [(tensor_factor, 0)])

    else:
      # Raise for invalid operator.
      raise ErrorPauliString(
          "Invalid operator provided: must be 'X', 'Y' or 'Z'")

    # Account for the phase terms.
    transformed_operator = fermionic_data.FermionicOperator(
        n_qubits, [raising_term, lowering_term])
    for qubit in range(tensor_factor - 1, -1, -1):
      transformed_operator.multiply_by_operator(
          reverse_jordan_wigner_pauli(n_qubits, qubit, 'Z'))

  # Return.
  return transformed_operator


def multiply_pauli_strings(pauli_string_a, pauli_string_b):
  """Multiply together pauli_string_a and pauli_string_b from left to right.

  Args:
    pauli_string_a: A PauliString object.
    pauli_string_b: Another PauliString object.

  Returns:
    product_string: The PauliString pauli_string_a * pauli_string_b.

  Raises:
    ErrorPauliString: Not same number of qubits in each term.
  """
  if pauli_string_a.n_qubits != pauli_string_b.n_qubits:
    raise ErrorPauliString(
        'Attempting to multiply terms acting on different Hilbert spaces.')

  # Loop through terms and create new sorted list of operators.
  product_coefficient = pauli_string_a.coefficient * \
      pauli_string_b.coefficient
  product_operators = []
  term_index_a = 0
  term_index_b = 0
  n_terms_a = len(pauli_string_a.operators)
  n_terms_b = len(pauli_string_b.operators)
  while term_index_a < n_terms_a and term_index_b < n_terms_b:
    (tensor_factor_a, matrix_a) = pauli_string_a.operators[term_index_a]
    (tensor_factor_b, matrix_b) = pauli_string_b.operators[term_index_b]

    # Multiply matrices if tensor factors are the same.
    if tensor_factor_a == tensor_factor_b:
      (scalar, matrix) = _PAULI_MATRIX_PRODUCTS[(matrix_a, matrix_b)]
      term_index_a += 1
      term_index_b += 1

      # Add new term.
      if matrix != 'I':
        product_operators += [(tensor_factor_a, matrix)]
        product_coefficient *= scalar

    # If tensor_factor_a > tensor_factor_b, add matrix_b; else, add matrix_a.
    elif tensor_factor_a > tensor_factor_b:
      product_operators += [(tensor_factor_b, matrix_b)]
      term_index_b += 1
    else:
      term_index_a += 1
      product_operators += [(tensor_factor_a, matrix_a)]

  # If either term_index exceeds the number operators, finish.
  if term_index_a == n_terms_a:
    product_operators += pauli_string_b.operators[term_index_b::]
  elif term_index_b == n_terms_b:
    product_operators += pauli_string_a.operators[term_index_a::]

  # We should now have gone through all operators. Create the new PauliString.
  product_string = PauliString(pauli_string_a.n_qubits, product_coefficient,
                               product_operators)
  return product_string


class PauliString(object):
  """Single term of a hamiltonian for a system of spin 1/2 particles or qubits.

  A hamiltonian of qubits can be written as a sum of PauliString objects.
  Suppose you have n_qubits = 5 qubits a term of the hamiltonian
  could be coefficient * X1 Z3 which we call a PauliString object. It means
  coefficient *(1 x PauliX x 1 x PauliZ x 1),
  where x is the tensor product, 1 the identity matrix, and the others are
  Pauli matrices. We only allow to apply one single Pauli Matrix to each qubit.

  Note: We assume in this class that indices start from 0 to n_qubits - 1.

  Attributes:
    n_qubits: The total number of qubits in the system.
    coefficient: A real or complex floating point number.
    operators: A sorted list of tuples. The first element of each tuple is an
      int indicating the qubit on which operators acts. The second element
      of each tuple is a string, either 'X', 'Y' or 'Z', indicating what
      acts on that tensor factor. The list is sorted by the first index.
  """
  def __init__(self, n_qubits, coefficient=1., operators=None):
    """Inits PauliTerm.

    Specify to which qubits a Pauli X, Y, or Z is applied. To all not
    specified qubits (numbered 0, 1, ..., n_qubits-1) the identity is applied.
    Only one Pauli Matrix can be applied to each qubit.

    Args:
      n_qubits: The total number of qubits in the system.
      coefficient: A real or complex floating point number.
      operators: A sorted list of tuples. The first element of each tuple is an
        int indicating the qubit on which operators acts. The second element
        of each tuple is a string, either 'X', 'Y' or 'Z', indicating what
        acts on that tensor factor.

    Raises:
      ErrorPauliString: Wrong input.
    """
    # Check that n_qubits is an integer.
    if not isinstance(n_qubits, int):
      raise ErrorPauliString('Number of qubits needs to be an integer.')

    # Initialize attributes.
    self.n_qubits = n_qubits
    self.coefficient = coefficient
    if operators is None or operators == []:
      self.operators = []
    else:
      self.operators = sorted(operators, key=lambda operator: operator[0])

      # Make sure each term has proper number of qubits.
      if max(self.operators, key=lambda operator: operator[0])[0] >= n_qubits:
        raise ErrorPauliString('Operators acting outside of n_qubit space.')

  def is_identical_string(self, pauli_string):
    """Compare operators with another PauliString object.

    Args:
      pauli_string: Another PauliString object.

    Returns:
      Boole. True if PauliTerm.operators is equal.

    Raises:
      ErrorPauliString: Not same number of qubits in each term.
    """
    if self.n_qubits != pauli_string.n_qubits:
      raise ErrorPauliString(
          'Comparing terms acting on different Hilbert spaces.')
    if self.operators == pauli_string.operators:
      return True
    else:
      return False

  def multiply_by_string(self, pauli_string):
    """Multiply operators with another PauliString object.

    Note that the "self" term is on the left of the multiply sign.

    Args:
      pauli_string: Another PauliString object.

    Raises:
      ErrorPauliString: Not same number of qubits in each term.
    """
    product_string = multiply_pauli_strings(self, pauli_string)
    self.coefficient = product_string.coefficient
    self.operators = product_string.operators

  def reverse_jordan_wigner(self):
    """Map Pauli term back to fermionic operators."""
    identity = fermionic_data.FermionicTerm(self.n_qubits, self.coefficient)
    transformed_operator = fermionic_data.FermionicOperator(
        self.n_qubits, [identity])
    if self.operators:
      for operator in self.operators:
        transformed_operator.multiply_by_operator(
            reverse_jordan_wigner_pauli(
                self.n_qubits, operator[0], operator[1]))
    return transformed_operator

  def __str__(self):
    """Return an easy-to-read string representation of the term."""
    string_representation = '{}'.format(self.coefficient)
    if self.operators == []:
      string_representation += ' I'
    for operator in self.operators:
      if operator[1] == 'X':
        string_representation += ' X{}'.format(operator[0])
      elif operator[1] == 'Y':
        string_representation += ' Y{}'.format(operator[0])
      else:
        assert operator[1] == 'Z'
        string_representation += ' Z{}'.format(operator[0])
    return string_representation

  def to_sparse_matrix(self):
    tensor_factor = 0
    matrix_form = self.coefficient
    for operator in self.operators:

      # Grow space for missing identity operators.
      if operator[0] > tensor_factor:
        identity_qubits = operator[0] - tensor_factor
        identity = scipy.sparse.identity(
            2 ** identity_qubits, dtype=complex, format='csc')
        matrix_form = scipy.sparse.kron(matrix_form, identity, 'csc')

      # Kronecker product the operator.
      matrix_form = scipy.sparse.kron(
          matrix_form, sparse_operators._PAULI_MATRIX_MAP[operator[1]], 'csc')
      tensor_factor = operator[0] + 1

    # Grow space at end of string unless operator acted on final qubit.
    if tensor_factor < self.n_qubits or not self.operators:
      identity_qubits = self.n_qubits - tensor_factor
      identity = scipy.sparse.identity(
          2 ** identity_qubits, dtype=complex, format='csc')
      matrix_form = scipy.sparse.kron(matrix_form, identity, 'csc')
    return matrix_form


class QubitOperator(object):
  """A collection of PauliString objects acting on same number of qubits.

  Note that to be a Hamiltonian which is a hermitian operator, the individual
  PauliString objects need to have only real valued coefficients.

  Attributes:
    n_qubits: The number of qubits on which the operator acts.
    terms: List of PauliString objects.
  """

  def __init__(self, n_qubits, terms=None):
    """Inits QubitHamiltonian.

    Args:
      n_qubits: The number of qubits the operator acts on.
      terms: A python list of PauliString terms.
    """
    self.n_qubits = n_qubits
    if terms is None:
      self.terms = []
    else:
      self.terms = terms

  def add_term(self, new_term, tolerance=1e-14):
    """Add another PauliString to hamiltonian.

    If hamiltonian already has this term, then the coefficients are merged.

    Args:
      new_term: PauliString object. It is added to the Hamiltonian.
      tolerance: The tolerance with which to consider terms zero.

    Raises:
      ErrorQubitOperator: Not allowed to add this term.
    """
    # Make sure terms act on same number of qubits.
    if self.terms:
      if self.terms[0].n_qubits != new_term.n_qubits:
        raise ErrorQubitOperator(
            'Cannot add terms which act on different Hilbert spaces')

    # Add term.
    for term_number, old_term in enumerate(self.terms):
      if old_term.is_identical_string(new_term):
        # Delete the term entirely if addition cancels it out.
        if abs(old_term.coefficient + new_term.coefficient) < tolerance:
          del self.terms[term_number]
        else:
          old_term.coefficient += new_term.coefficient
        return
    self.terms.append(new_term)

  def multiply_by_term(self, new_term):
    """Multiplies the QubitOperator by a new PauliString.

    Args:
      new_term: PauliString object.
    """
    new_operator = QubitOperator(self.n_qubits)
    for term in self.terms:
      new_operator.add_term(multiply_pauli_strings(term, new_term))
    self.terms = new_operator.terms

  def add_operator(self, new_operator):
    """Adds two QubitOperators together.

    Args:
      new_operator: QubitOperator which will be added to self.
    """
    for new_term in new_operator.terms:
      self.add_term(new_term)

  def multiply_by_operator(self, new_operator):
    """Multiplies two QubitOperators together.

    Args:
      new_operator: QubitOperator which will multiply self.
    """
    product_operator = QubitOperator(self.n_qubits)
    for term in self.terms:
      for new_term in new_operator.terms:
        product_operator.add_term(multiply_pauli_strings(term, new_term))
    self.terms = product_operator.terms

  def multiply_by_scalar(self, scalar):
    """Multiplies all terms by a scalar."""
    for term in self.terms:
      term.coefficient *= scalar

  def get_coefficients(self):
    """Return the coefficients of all the terms in the operator

    Returns:
      A list of complex floats giving the operator term coefficients.
    """
    coefficients = [term.coefficient for term in self.terms]
    return coefficients

  def print_operator(self):
    for term in self.terms:
      print(term.__str__())

  def look_up_coefficient(self, operators):
    if self.terms:
      n_qubits = self.n_qubits
      pauli_string = PauliString(n_qubits, 1., operators)
      for term in self.terms:
        if term.is_identical_string(pauli_string):
          return term.coefficient
    return 0.

  def is_identical_operator(self, operator, tolerance=1e-15):
    if self.count_terms() != operator.count_terms():
      return False
    for term in self.terms:
      difference = abs(term.coefficient -
                       operator.look_up_coefficient(term.operators))
      if difference > tolerance:
        return False
    return True

  def reverse_jordan_wigner(self):
    transformed_operator = fermionic_data.FermionicOperator(self.n_qubits)
    for term in self.terms:
      transformed_operator.add_operator(term.reverse_jordan_wigner())
    return transformed_operator

  def to_sparse_matrix(self):
    hilbert_dimension = 2 ** self.n_qubits
    matrix_form = scipy.sparse.csc_matrix(
        (hilbert_dimension, hilbert_dimension), dtype=complex)
    for term in self.terms:
      matrix_form = matrix_form + term.to_sparse_matrix()
    return matrix_form

  def count_terms(self):
    return len(self.terms)

  def get_coefficients(self):
    coefficients = [term.coefficient for term in self.terms]
    return coefficients

  def remove_term(self, operators):
    for term_number, term in enumerate(self.terms):
      if term.operators == operators:
        del self.terms[term_number]

  def expectation(self, qubit_operator):
    """Take the expectation value of self with another qubit operator.

    Args:
      qubit_operator: An instance of the QubitOperator class.

    Returns:
      expectation: A float, giving the expectation value.
    """
    expectation = 0.
    for term in self.terms:
      complement = qubit_operator.look_up_coefficient(term.operators)
      expectation += term.coefficient * complement
    return expectation
