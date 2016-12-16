"""This files has utilities to read and store qubit hamiltonians.
"""
import copy
import scipy
import scipy.sparse
import local_operators
import sparse_operators
import fermion_operators


class ErrorQubitTerm(Exception):
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
  """Transforms a single pauli operator into an instance of FermionOperator.

  Args:
    n_qubits: Int, the number of qubits.
    tensor_factor: Int, the tensor factor on which operator acts.
    operator: String, either 'X', 'Y' or 'Z'.

  Returns:
    transformed_operator: An instance of the FermionOperator class.

  Raises:
    ErrorQubitTerm: Invalid operator provided: must be 'X', 'Y' or 'Z'.
  """
  if operator == 'Z':
    # Handle Pauli Z.
    identity = fermion_operators.FermionTerm(n_qubits, 1.)
    number_operator = fermion_operators.FermionTerm(
        n_qubits, -2., [(tensor_factor, 1), (tensor_factor, 0)])
    transformed_operator = fermion_operators.FermionOperator(
        n_qubits, [identity, number_operator])

  else:
    if operator == 'X':
      # Handle Pauli X.
      raising_term = fermion_operators.FermionTerm(
          n_qubits, 1., [(tensor_factor, 1)])
      lowering_term = fermion_operators.FermionTerm(
          n_qubits, 1., [(tensor_factor, 0)])

    elif operator == 'Y':
      # Handle Pauli Y.
      raising_term = fermion_operators.FermionTerm(
          n_qubits, 1.j, [(tensor_factor, 1)])
      lowering_term = fermion_operators.FermionTerm(
          n_qubits, -1.j, [(tensor_factor, 0)])

    else:
      # Raise for invalid operator.
      raise ErrorQubitTerm(
          "Invalid operator provided: must be 'X', 'Y' or 'Z'")

    # Account for the phase terms.
    transformed_operator = fermion_operators.FermionOperator(
        n_qubits, [raising_term, lowering_term])
    for qubit in range(tensor_factor - 1, -1, -1):
      transformed_operator.multiply_by_operator(
          reverse_jordan_wigner_pauli(n_qubits, qubit, 'Z'))

  # Return.
  return transformed_operator


def multiply_qubit_terms(qubit_term_a, qubit_term_b):
  """Multiply together qubit_term_a and qubit_term_b from left to right.

  Args:
    qubit_term_a: A QubitTerm object.
    qubit_term_b: Another QubitTerm object.

  Returns:
    product_string: The QubitTerm qubit_term_a * qubit_term_b.

  Raises:
    ErrorQubitTerm: Not same number of qubits in each term.
  """
  if qubit_term_a.n_qubits != qubit_term_b.n_qubits:
    raise ErrorQubitTerm(
        'Attempting to multiply terms acting on different Hilbert spaces.')

  # Loop through terms and create new sorted list of operators.
  product_coefficient = qubit_term_a.coefficient * \
      qubit_term_b.coefficient
  product_operators = []
  term_index_a = 0
  term_index_b = 0
  n_terms_a = len(qubit_term_a.operators)
  n_terms_b = len(qubit_term_b.operators)
  while term_index_a < n_terms_a and term_index_b < n_terms_b:
    (tensor_factor_a, matrix_a) = qubit_term_a.operators[term_index_a]
    (tensor_factor_b, matrix_b) = qubit_term_b.operators[term_index_b]

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
    product_operators += qubit_term_b.operators[term_index_b::]
  elif term_index_b == n_terms_b:
    product_operators += qubit_term_a.operators[term_index_a::]

  # We should now have gone through all operators. Create the new QubitTerm.
  product_string = QubitTerm(qubit_term_a.n_qubits, product_coefficient,
                             product_operators)
  return product_string


class QubitTerm(local_operators.LocalTerm):
  """Single term of a hamiltonian for a system of spin 1/2 particles or qubits.

  A hamiltonian of qubits can be written as a sum of QubitTerm objects.
  Suppose you have n_qubits = 5 qubits a term of the hamiltonian
  could be coefficient * X1 Z3 which we call a QubitTerm object. It means
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
      ErrorQubitTerm: Wrong input.
    """
    # Check that n_qubits is an integer.
    if not isinstance(n_qubits, int):
      raise ErrorQubitTerm('Number of qubits needs to be an integer.')

    # Initialize attributes.
    self.n_qubits = n_qubits
    self.coefficient = coefficient
    if operators is None or operators == []:
      self.operators = []
    else:
      self.operators = sorted(operators, key=lambda operator: operator[0])

      # Make sure each term has proper number of qubits.
      if max(self.operators, key=lambda operator: operator[0])[0] >= n_qubits:
        raise ErrorQubitTerm('Operators acting outside of n_qubit space.')

  def multiply_by_term(self, qubit_term):
    """Multiply operators with another QubitTerm object.

    Note that the "self" term is on the left of the multiply sign.

    Args:
      qubit_term: Another QubitTerm object.

    Raises:
      ErrorQubitTerm: Not same number of qubits in each term.
    """
    product_term = multiply_qubit_terms(self, qubit_term)
    self.coefficient = product_term.coefficient
    self.operators = product_term.operators

  def reverse_jordan_wigner(self):
    """Map QubitTerm back to FermionOperator."""
    identity = fermion_operators.FermionTerm(
        self.n_qubits, self.coefficient)
    transformed_operator = fermion_operators.FermionOperator(
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


class QubitOperator(local_operators.LocalOperator):
  """A collection of QubitTerm objects acting on same number of qubits.

  Note that to be a Hamiltonian which is a hermitian operator, the individual
  QubitTerm objects need to have only real valued coefficients.

  Attributes:
    n_qubits: The number of qubits on which the operator acts.
    terms: Dictionary of QubitTerm objects. The dictionary key is
        QubitTerm.key() and the dictionary value is the QubitTerm.
  """
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
