"""This module provides functions which are useful for the study of fermions.
"""
import numpy
import scipy
import itertools
import scipy.misc
import scipy.sparse
import scipy.sparse.linalg


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
    eigenvalue: The lowest eigenvalue, a float.
    eigenstate: The lowest eigenstate in scipy.sparse csc format.
  """
  values, vectors = scipy.sparse.linalg.eigsh(
      operator, 2, which="SA", maxiter=1e7)
  eigenstate = scipy.sparse.csc_matrix(vectors[:, 0])
  eigenvalue = values[0]
  return eigenstate.getH(), eigenvalue


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
    ket = 1
    for qubit in state:
      if qubit:
        ket = scipy.sparse.kron(ket, occupied, "csc")
      else:
        ket = scipy.sparse.kron(ket, unoccupied, "csc")

    # Add to projector.
    density = ket * ket.getH()
    projector = projector + density
  return projector
