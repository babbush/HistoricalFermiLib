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
_IDENTITY_CSC = scipy.sparse.identity(2, format="csr", dtype=complex)
_PAULI_X_CSC = scipy.sparse.csc_matrix([[0., 1.], [1., 0.]], dtype=complex)
_PAULI_Y_CSC = scipy.sparse.csc_matrix([[0., -1.j], [1.j, 0.]], dtype=complex)
_PAULI_Z_CSC = scipy.sparse.csc_matrix([[1., 0.], [0., -1.]], dtype=complex)
_Q_RAISE_CSC = (_PAULI_X_CSC - 1.j * _PAULI_Y_CSC) / 2.
_Q_LOWER_CSC = (_PAULI_X_CSC + 1.j * _PAULI_Y_CSC) / 2.
_PAULI_MATRIX_MAP = {'I': _IDENTITY_CSC, 'X': _PAULI_X_CSC,
                     'Y': _PAULI_Y_CSC, 'Z': _PAULI_Z_CSC}


class SparseLadderOperators(object):
  """Class to store sparse creation and annihiliation operators

  This class is a convenient numerical intermediate that stores creation and
  annihilation operators on the full Hilbert space of n_qubits so they may be
  easily accessed without requiring additional Kronecker products, which are
  quite expensive due to the method by which they allocate memory.

  Attributes:
    n_qubits(int): Number of qubits the system acts on
    sparse_type(str): Scipy sparse matrix format, e.g. csc
    operators(list): List of sparse operators in the full Hilbert space such
      that a creation (annihilation) operator on index i is stored at
      operators[2 * i + 1] (operators[2 * i]).
  """

  def __init__(self, n_qubits, sparse_type="csc"):
    """Initialize the operator stores

    Args:
      n_qubits(int): Number of qubits in the system
      sparse_type(str): Type of scipy sparse matrix to use in storage
    """
    self.n_qubits = n_qubits
    self.sparse_type = sparse_type
    self.initialize_operators()

  def get_operator(self, index, type):
    """Retrieve the sparse operator at index i of given type

    Args:
      index(int): index the operator acts on
      type(int): 1 for creation, 0 for annihiliation
    """
    return self.operators[2 * index + type]

  def initialize_operators(self):
    """Build all possible creation and annihilation operators for system."""
    self.operators = []

    def wrap_kron(operator_1, operator_2):
      return scipy.sparse.kron(operator_1, operator_2, self.sparse_type)

    for i in range(self.n_qubits):
      for type in range(2):
        term_matrix = \
            reduce(wrap_kron,
                   ([_IDENTITY_CSC for _ in range(i)] +
                    [_Q_RAISE_CSC if type == 1 else _Q_LOWER_CSC] +
                    [_PAULI_Z_CSC for _ in
                       range(self.n_qubits - i - 1)]))
        self.operators.append(term_matrix)


def is_hermitian(matrix, tolerance=1e-12):
  """Test if matrix is Hermitian.

  Args:
    matrix: the operator to test in scipy.sparse 'csc' format.
    tolerance: a float giving the allowed Hermitian discrepancy.
      Default value is 1e-12.

  Returns:
    Boole indicating whether matrix passes test.
  """
  conjugate = matrix.getH()
  difference = matrix - conjugate
  if difference.nnz:
    discrepancy = max(map(abs, difference.data))
    if discrepancy > tolerance:
      return False
  return True


def get_ground_state(operator):
  """Compute lowest eigenstate and eigenvalue.

  Args:
    operator: A scipy.sparse csc operator to diagonalize.

  Returns:
    eigenstate: The lowest eigenstate in scipy.sparse csc format.
    eigenvalue: The lowest eigenvalue, a float.
  """
  values, vectors = scipy.sparse.linalg.eigsh(
      operator, 2, which="SA", maxiter=1e7)
  eigenstate = scipy.sparse.csc_matrix(vectors[:, 0])
  eigenvalue = values[0]
  return eigenstate.getH(), eigenvalue


def get_eigenspectrum(operator):
  """Perform a dense diagonalization.

  Args:
    operators: A scipy.sparse csc operator to diagonalize.

  Returns:
    eigenspectrum: The lowest eigenvalues in a numpy array.
  """
  dense_operator = operator.todense()
  eigenspectrum = numpy.linalg.eigvalsh(dense_operator)
  return eigenspectrum


def expectation(operator, state):
  """Compute expectation value of an operator with a state.

  Args:
    operator: scipy.sparse csc operator.
    state_vector: scipy.sparse.csc vector representing a pure state,
      or, a scipy.sparse.csc matrix representing a density matrix.

  Returns:
    A real float giving expectation value.

  Raises:
    SparseOperatorError: Input state has invalid format.
  """
  n_qubits = operator.shape[0]

  # Handle density matrix.
  if state.shape == (n_qubits, n_qubits):
    product = state * operator
    expectation = numpy.sum(product.diagonal())

  elif state.shape == (n_qubits, 1):
    # Handle state vector.
    expectation = state.getH() * operator * state
    expectation = expectation[0, 0]

  else:
    # Handle exception.
    raise SparseOperatorError('Input state has invalid format.')

  # Return.
  return expectation


def hartree_fock_state(n_electrons, n_orbitals):
  occupied = scipy.sparse.csr_matrix([[0], [1]], dtype=float)
  psi = 1.
  unoccupied = scipy.sparse.csr_matrix([[1], [0]], dtype=float)
  for orbital in range(n_electrons):
    psi = scipy.sparse.kron(psi, occupied, 'csr')
  for orbital in range(n_orbitals - n_electrons):
    psi = scipy.sparse.kron(psi, unoccupied, 'csr')
  return psi


def get_gap(operator):
  """Compute gap between lowest eigenvalue and first excited state.

  Args:
    operator: A scipy.sparse csc operator to diagonalize for gap.

  Returns:
    A real float giving eigenvalue gap.
  """
  values, _ = scipy.sparse.linalg.eigsh(operator, 2, which="SA", maxiter=1e7)
  gap = abs(values[1] - values[0])
  return gap


def get_density_matrix(states, probabilities):
  n_qubits = states[0].shape[0]
  density_matrix = scipy.sparse.csc_matrix((n_qubits, n_qubits), dtype=complex)
  for state, probability in zip(states, probabilities):
    density_matrix = density_matrix + probability * state * state.getH()
  return density_matrix


def get_determinants(n_orbitals, n_electrons):
  """Generate an array of all states on n_orbitals with n_fermions.

  Args:
    n_orbitals: an int giving the number of qubits.
    n_electrons: an int giving the number of electrons:

  Returns:
    A numpy array where each row is a state on n_orbitals with n_fermions.
  """

  # Initialize vector of states.
  n_configurations = int(numpy.rint(scipy.misc.comb(n_orbitals, n_electrons)))
  states = numpy.zeros((n_configurations, n_orbitals), int)

  # Loop over valid states.
  for i, occupied in enumerate(itertools.combinations(range(n_orbitals),
                                                      r=n_electrons)):
    states[i, occupied] = 1
  return states


def configuration_projector(n_orbitals, n_electrons):
  """Construct projector into an n_electron manifold.

  Args:
    n_orbitals: This int gives the number of qubits in the Hilbert space.
    n_electrons: This int gives the number manifold in which to project.

  Returns:
    A projector matrix is scipy.sparse 'csc' format.
  """
  # Initialize projector computation.
  states = get_determinants(n_orbitals, n_electrons)
  unoccupied = scipy.sparse.csc_matrix([[1], [0]], dtype=int)
  occupied = scipy.sparse.csc_matrix([[0], [1]], dtype=int)

  # Construct projector.
  projector = 0
  for state in states:

    # Construct computational basis state in Hilbert space.
    ket = 1.
    for qubit in state:
      if qubit:
        ket = scipy.sparse.kron(ket, occupied, "csc")
      else:
        ket = scipy.sparse.kron(ket, unoccupied, "csc")

    # Add to projector.
    density = ket * ket.getH()
    projector = projector + density
  return projector

def restrict_particle_manifold(operator, n_electrons):
  """Restrict the support of an operator to a fixed particle-number manifold.

  Args:
    operator: A scipy sparse.csc matrix.
    n_electrons: The particle sector to restrict to.

  Returns:
    effective_operator: The effective operator which restricts to the correct
        particle-number manifold.
  """
  n_orbitals = int(numpy.rint(numpy.log2(operator.shape[0])))
  projector = configuration_projector(n_orbitals, n_electrons)
  effective_operator = projector.getH() * operator * projector
  return effective_operator
