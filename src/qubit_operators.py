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


class QubitTerm(local_operators.LocalTerm):
  """Single term of a hamiltonian for a system of spin 1/2 particles or qubits.

  A hamiltonian of qubits can be written as a sum of QubitTerm objects.
  Suppose you have n_qubits = 5 qubits a term of the hamiltonian
  could be coefficient * X1 Z3 which we call a QubitTerm object. It means
  coefficient *(1 x PauliX x 1 x PauliZ x 1),
  where x is the tensor product, 1 the identity matrix, and the others are
  Pauli matrices. We only allow to apply one single Pauli Matrix to each qubit.

  Note 1: We assume in this class that indices start from 0 to n_qubits - 1.
  Note 2: Always use the abstractions provided here to manipulate the
      .operators attribute. If ignoring this advice, an important thing to
      keep in mind is that the operators list is assumed to be sorted in order
      of the tensor factor on which the operator acts.

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
    # Make sure terms act on same Hilbert space.
    if self.n_qubits != qubit_term.n_qubits:
      raise ErrorQubitTerm(
          'Attempting to multiply terms acting on different Hilbert spaces.')

    # Relabel self * qubit_term as left_term * right_term.
    left_term = self
    right_term = qubit_term
    product_coefficient = left_term.coefficient * right_term.coefficient

    # Loop through terms and create new sorted list of operators.
    product_operators = []
    left_operator_index = 0
    right_operator_index = 0
    n_operators_left = len(left_term.operators)
    n_operators_right = len(right_term.operators)
    while (left_operator_index < n_operators_left and
           right_operator_index < n_operators_right):
      (left_qubit, left_matrix) = left_term.operators[left_operator_index]
      (right_qubit, right_matrix) = right_term.operators[right_operator_index]

      # Multiply matrices if tensor factors are the same.
      if left_qubit == right_qubit:
        (scalar, matrix) = _PAULI_MATRIX_PRODUCTS[(left_matrix, right_matrix)]
        left_operator_index += 1
        right_operator_index += 1

        # Add new term.
        if matrix != 'I':
          product_operators += [(left_qubit, matrix)]
          product_coefficient *= scalar

      # If left_qubit > right_qubit, add right_matrix; else, add left_matrix.
      elif left_qubit > right_qubit:
        product_operators += [(right_qubit, right_matrix)]
        right_operator_index += 1
      else:
        product_operators += [(left_qubit, left_matrix)]
        left_operator_index += 1

    # If either term_index exceeds the number of operators, finish.
    if left_operator_index == n_operators_left:
      product_operators += right_term.operators[right_operator_index::]
    elif right_operator_index == n_operators_right:
      product_operators += left_term.operators[left_operator_index::]

    # We should now have gone through all operators. Update self.
    self.coefficient = product_coefficient
    self.operators = product_operators

  def reverse_jordan_wigner(self):
    """Transforms a QubitTerm into an instance of FermionOperator using JW.

    Operators are mapped as follows:
    Z_j -> I - 2 a^\dagger_j a_j
    X_j -> (a^\dagger_j + a_j) Z_{j-1} Z_{j-2} .. Z_0
    Y_j -> i (a^\dagger_j - a_j) Z_{j-1} Z_{j-2} .. Z_0

    Returns:
      transformed_term: An instance of the FermionOperator class.

    Raises:
      ErrorQubitTerm: Invalid operator provided: must be 'X', 'Y' or 'Z'.
    """
    # Initialize transformed operator.
    identity = fermion_operators.FermionTerm(
        self.n_qubits, self.coefficient)
    transformed_term = fermion_operators.FermionOperator(
        self.n_qubits, [identity])

    # Loop through operators.
    if self.operators:
      for operator in self.operators:

        # Handle Pauli Z.
        if operator[1] == 'Z':
          identity = fermion_operators.FermionTerm(self.n_qubits, 1.)
          number_operator = fermion_operators.FermionTerm(
              self.n_qubits, -2., [(operator[0], 1), (operator[0], 0)])
          transformed_operator = fermion_operators.FermionOperator(
              self.n_qubits, [identity, number_operator])

        else:
          # Handle Pauli X.
          if operator[1] == 'X':
            raising_term = fermion_operators.FermionTerm(
                self.n_qubits, 1., [(operator[0], 1)])
            lowering_term = fermion_operators.FermionTerm(
                self.n_qubits, 1., [(operator[0], 0)])

          elif operator[1] == 'Y':
            # Handle Pauli Y.
            raising_term = fermion_operators.FermionTerm(
                self.n_qubits, 1.j, [(operator[0], 1)])
            lowering_term = fermion_operators.FermionTerm(
                self.n_qubits, -1.j, [(operator[0], 0)])

          else:
            # Raise for invalid operator.
            raise ErrorQubitTerm(
                "Invalid operator provided: must be 'X', 'Y' or 'Z'")

          # Account for the phase terms.
          transformed_operator = fermion_operators.FermionOperator(
              self.n_qubits, [raising_term, lowering_term])
          for qubit in range(operator[0] - 1, -1, -1):
            identity = fermion_operators.FermionTerm(self.n_qubits, 1.)
            number_operator = fermion_operators.FermionTerm(
                self.n_qubits, -2., [(qubit, 1), (qubit, 0)])
            transformed_operator.multiply_by_operator(
                fermion_operators.FermionOperator(
                    self.n_qubits, [identity, number_operator]))

        # Multiply term by transformed operator.
        transformed_term.multiply_by_operator(transformed_operator)

    # Return.
    return transformed_term

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

  def get_sparse_matrix(self):
    """Map the QubitTerm to a scipy.sparse.csc matrix."""
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

  def get_sparse_matrix(self):
    hilbert_dimension = 2 ** self.n_qubits
    matrix_form = scipy.sparse.csc_matrix(
        (hilbert_dimension, hilbert_dimension), dtype=complex)
    for term in self.iter_terms():
      matrix_form = matrix_form + term.get_sparse_matrix()
    return matrix_form

  def expectation(self, qubit_operator):
    """Take the expectation value of self with another qubit operator.

    Args:
      qubit_operator: An instance of the QubitOperator class.

    Returns:
      expectation: A float, giving the expectation value.
    """
    expectation = 0.
    for term in self.iter_terms():
      expectation += term.coefficient * qubit_operator(term.operators)
    return expectation
