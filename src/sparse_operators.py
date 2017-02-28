"""This module provides functions which are useful for the study of fermions.
"""
import numpy
import scipy
import itertools
import scipy.misc
import scipy.sparse
import scipy.sparse.linalg
import numpy.linalg


# Define error class.
class SparseOperatorError(Exception):
  pass


# Make global definitions.
_IDENTITY_CSC = scipy.sparse.identity(2, format='csr', dtype=complex)
_PAULI_X_CSC = scipy.sparse.csc_matrix([[0., 1.], [1., 0.]], dtype=complex)
_PAULI_Y_CSC = scipy.sparse.csc_matrix([[0., -1.j], [1.j, 0.]], dtype=complex)
_PAULI_Z_CSC = scipy.sparse.csc_matrix([[1., 0.], [0., -1.]], dtype=complex)
_Q_RAISE_CSC = (_PAULI_X_CSC - 1.j * _PAULI_Y_CSC) / 2.
_Q_LOWER_CSC = (_PAULI_X_CSC + 1.j * _PAULI_Y_CSC) / 2.
_PAULI_MATRIX_MAP = {'I': _IDENTITY_CSC, 'X': _PAULI_X_CSC,
                     'Y': _PAULI_Y_CSC, 'Z': _PAULI_Z_CSC}


# Function to product Hartree-Fock state in Jordan-Wigner representation.
def jw_hartree_fock_state(n_electrons, n_orbitals):
  occupied = scipy.sparse.csr_matrix([[0], [1]], dtype=float)
  psi = 1.
  unoccupied = scipy.sparse.csr_matrix([[1], [0]], dtype=float)
  for orbital in range(n_electrons):
    psi = scipy.sparse.kron(psi, occupied, 'csr')
  for orbital in range(n_orbitals - n_electrons):
    psi = scipy.sparse.kron(psi, unoccupied, 'csr')
  return psi


def get_density_matrix(states, probabilities):
  n_qubits = states[0].shape[0]
  density_matrix = scipy.sparse.csc_matrix((n_qubits, n_qubits), dtype=complex)
  for state, probability in zip(states, probabilities):
    density_matrix = density_matrix + probability * state * state.getH()
  return density_matrix


# The functions below help to quickly perform Kronecker products.
def wrapped_kronecker(operator_1, operator_2):
  """Return the Kronecker product of two sparse.csc_matrix operators."""
  return scipy.sparse.kron(operator_1, operator_2, 'csc')


def kronecker_operators(*args):
  """Return the Kronecker product of multiple sparse.csc_matrix operators."""
  return reduce(wrapped_kronecker, *args)


# The functions below help to make common sparse operators.
def sparse_identity(n_qubits):
  return SparseOperator(scipy.sparse.identity(
      2 ** n_qubits, dtype=complex, format='csc'))


def jordan_wigner_ladder_sparse(n_qubits, tensor_factor, ladder_type):
  """Make a matrix representation of a fermion ladder operator.

  Args:
    index: This is a nonzero integer. The integer indicates the tensor
      factor and the sign indicates raising or lowering.
    n_qubits(int): Number qubits in the system Hilbert space.

  Returns:
    The corresponding SparseOperator.
  """
  identities = [scipy.sparse.identity(
      2 ** tensor_factor, dtype=complex, format='csc')]
  parities = (n_qubits - tensor_factor - 1) * [_PAULI_Z_CSC]
  if ladder_type:
    operator = kronecker_operators(identities + [_Q_RAISE_CSC] + parities)
  else:
    operator = kronecker_operators(identities + [_Q_LOWER_CSC] + parities)
  return SparseOperator(operator)


def jordan_wigner_term_sparse(fermion_term):
  """Initialize a SparseOperator from a FermionTerm.

  Args:
    fermion_term(FermionTerm): instance of the FermionTerm class.

  Returns:
    The corresponding SparseOperator.
  """
  sparse_operator = sparse_identity(n_qubits)
  for ladder_operator in fermion_term:
    sparse_operator = sparse_operator * jordan_wigner_ladder_sparse(
        n_qubits, ladder_operator[0], ladder_operator[1])
  return fermion_term.coefficient * sparse_operator


def jordan_wigner_operator_sparse(fermion_operator):
  """Initialize a SparseOperator from a FermionOperator.

  Args:
    fermion_operator(FermionOperator): instance of the FermionOperator class.

  Returns:
    The corresponding SparseOperator.
  """
  # Create a list of raising and lowering operators for each orbital.
  jw_operators = []
  for tensor_factor in range(fermion_operator.n_qubits):
    jw_operators += [(jordan_wigner_ladder_sparse(
                      fermion_operator.n_qubits, tensor_factor, 0),
                      jordan_wigner_ladder_sparse(
                      fermion_operator.n_qubits, tensor_factor, 1))]

  # Construct the SparseOperator.
  n_hilbert = 2 ** fermion_operator.n_qubits
  values_list = [[]]
  row_list = [[]]
  column_list = [[]]
  for term in fermion_operator:
    sparse_term = term.coefficient * sparse_identity(fermion_operator.n_qubits)
    for ladder_operator in term:
      sparse_term = sparse_term * jw_operators[
          ladder_operator[0]][ladder_operator[1]]

    # Extract triplets from sparse_term.
    sparse_term.matrix = sparse_term.matrix.tocoo(copy=False)
    values_list.append(sparse_term.matrix.data)
    (row, column) = sparse_term.matrix.nonzero()
    row_list.append(row)
    column_list.append(column)

  values_list = numpy.concatenate(values_list)
  row_list = numpy.concatenate(row_list)
  column_list = numpy.concatenate(column_list)
  operator = SparseOperator(scipy.sparse.coo_matrix((
      values_list, (row_list, column_list)),
      shape=(n_hilbert, n_hilbert)).tocsc(copy=False))
  operator.matrix.eliminate_zeros()
  return operator


def qubit_term_sparse(qubit_term):
  """Initialize a SparseOperator from a QubitTerm.

  Args:
    qubit_term(QubitTerm): instance of the QubitTerm class.

  Returns:
    The corresponding SparseOperator.
  """
  operators = [qubit_term.coefficient]
  tensor_factor = 0
  for operator in qubit_term:

    # Grow space for missing identity operators.
    if operator[0] > tensor_factor:
      identity_qubits = operator[0] - tensor_factor
      identity = scipy.sparse.identity(
          2 ** identity_qubits, dtype=complex, format='csc')
      operators += [identity]

    # Add actual operator to the list.
    operators += [_PAULI_MATRIX_MAP[operator[1]]]
    tensor_factor = operator[0] + 1

  # Grow space at end of string unless operator acted on final qubit.
  if tensor_factor < qubit_term.n_qubits or not qubit_term.operators:
    identity_qubits = qubit_term.n_qubits - tensor_factor
    identity = scipy.sparse.identity(2 ** identity_qubits,
                                     dtype=complex, format='csc')
    operators += [identity]

  # Make matrix and return SparseOperator.
  matrix = kronecker_operators(operators)
  return SparseOperator(matrix)


def qubit_operator_sparse(qubit_operator):
  """Initialize a SparseOperator from a QubitOperator.

  Args:
    qubit_operator(QubitOperator): instance of the QubitOperator class.

  Returns:
    The corresponding SparseOperator.
  """
  # Construct the SparseOperator.
  n_hilbert = 2 ** qubit_operator.n_qubits
  values_list = [[]]
  row_list = [[]]
  column_list = [[]]
  for qubit_term in qubit_operator:
    sparse_term = qubit_term_sparse(qubit_term)
    sparse_term.matrix = sparse_term.matrix.tocoo(copy=False)

    # Extract triplets from sparse_term.
    values_list.append(sparse_term.matrix.data)
    (row, column) = sparse_term.matrix.nonzero()
    row_list.append(row)
    column_list.append(column)

  values_list = numpy.concatenate(values_list)
  row_list = numpy.concatenate(row_list)
  column_list = numpy.concatenate(column_list)
  operator = SparseOperator(scipy.sparse.coo_matrix((
      values_list, (row_list, column_list)),
      shape=(n_hilbert, n_hilbert)).tocsc(copy=False))
  operator.matrix.eliminate_zeros()
  return operator


class SparseOperator(object):
  """ Class to represent sparse operators.

  This class represents operators (usually Hermitian or Unitary) in a
  scipy.sparse.csc_matrix representation. The custom class gives convenient
  access to commonly used methods such as "diagonalize", "get_ground_state",
  "get_hartree_fock_state" and more.

  Attributes:
    matrix(scipy.sparse.csc_matrix): The sparse matrix.
    n_qubits(int): Number qubits in the system Hilbert space.
  """
  def __init__(self, matrix):
    """Construct a sparse operator.

    Args:
      matrix(scipy.sparse.csc_matrix): The sparse matrix.
    """
    self.matrix = matrix

  def __iter__(self):
    return iter(self.matrix)

  def __repr__(self):
    return repr(self.matrix)

  def __str__(self):
    return str(self.matrix)

  def __eq__(self, other):
    return self.matrix == other

  def __ne__(self, other):
    return self.matrix != other

  def __abs__(self):
    return SparseOperator(abs(self.matrix))

  def __add__(self, other):
    if isinstance(other, SparseOperator):
      other = other.matrix
    return SparseOperator(self.matrix + other)

  def __radd__(self, other):
    if isinstance(other, SparseOperator):
      other = other.matrix
    return SparseOperator(other + self.matrix)

  def __sub__(self, other):
    if isinstance(other, SparseOperator):
      other = other.matrix
    return SparseOperator(self.matrix - other)

  def __rsub__(self, other):
    if isinstance(other, SparseOperator):
      other = other.matrix
    return SparseOperator(other - self.matrix)

  def __mul__(self, other):
    if isinstance(other, SparseOperator):
      other = other.matrix
    return SparseOperator(self.matrix * other)

  def __rmul__(self, other):
    if isinstance(other, SparseOperator):
      other = other.matrix
    return SparseOperator(other * self.matrix)

  def __div__(self, other):
    return SparseOperator(self.matrix / other)

  def __neg__(self):
    return SparseOperator(-self.matrix)

  def __pow__(self, other):
    return SparseOpoerator(self.matrix ** other)

  def __len__(self):
    return self.matrix.shape[0]

  def n_qubits(self):
    return numpy.rint(numpy.log2(len(self)))

  def conjugated(self):
    return SparseOperator(self.matrix.getH())

  def to_dense(self):
    return self.matrix.todense()

  def eliminate_zeros(self):
    self.matrix.eliminate_zeros()

  def is_hermitian(self, tolerance=1e-12):
    """Test if matrix is Hermitian.

    Args:
      tolerance: a float giving the allowed Hermitian discrepancy.
        Default value is 1e-12.

    Returns:
      Boole indicating whether matrix passes test.
    """
    difference = self - self.conjugated()
    if difference.matrix.nnz:
      discrepancy = max(map(abs, difference.matrix.data))
      if discrepancy > tolerance:
        return False
    return True

  def get_ground_state(self):
    """Compute lowest eigenvalue and eigenstate.

    Returns:
      eigenvalue: The lowest eigenvalue, a float.
      eigenstate: The lowest eigenstate in scipy.sparse csc format.
    """
    if self.is_hermitian():
      values, vectors = scipy.sparse.linalg.eigsh(
          self.matrix, 2, which="SA", maxiter=1e7)
    else:
      values, vectors = scipy.sparse.linalg.eigs(
          self.matrix, 2, which="SA", maxiter=1e7)
    eigenstate = scipy.sparse.csc_matrix(vectors[:, 0])
    eigenvalue = values[0]
    return eigenvalue, eigenstate.getH()

  def get_eigenspectrum(self):
    """Perform a dense diagonalization.

    Returns:
      eigenspectrum: The lowest eigenvalues in a numpy array.
    """
    dense_operator = self.to_dense()
    if self.is_hermitian():
      eigenspectrum = numpy.linalg.eigvalsh(dense_operator)
    else:
      eigenspectrum = numpy.linalg.eigvals(dense_operator)
    return eigenspectrum

  def expectation(self, state):
    """Compute expectation value of operator with a state.

    Args:
      state_vector: scipy.sparse.csc vector representing a pure state,
        or, a scipy.sparse.csc matrix representing a density matrix.

    Returns:
      A real float giving expectation value.

    Raises:
      SparseOperatorError: Input state has invalid format.
    """
    # Handle density matrix.
    if state.shape == (len(self), len(self)):
      product = state * self.matrix
      expectation = numpy.sum(product.diagonal())

    elif state.shape == (len(self), 1):
      # Handle state vector.
      expectation = state.getH() * self.matrix * state
      expectation = expectation[0, 0]

    else:
      # Handle exception.
      raise SparseOperatorError('Input state has invalid format.')

    # Return.
    return expectation

  def get_gap(self):
    """Compute gap between lowest eigenvalue and first excited state.

    Returns:
      A real float giving eigenvalue gap.
    """
    values, _ = scipy.sparse.linalg.eigsh(
        self.matrix, 2, which="SA", maxiter=1e7)
    gap = abs(values[1] - values[0])
    return gap
