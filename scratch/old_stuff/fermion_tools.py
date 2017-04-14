"""This module provides functions which are useful for the study of fermions.
"""

import numpy
import scipy
import itertools
import scipy.misc
import scipy.sparse
import scipy.sparse.linalg


# Make global definitions.
_IDENTITY_CSC = scipy.sparse.identity(2, format="csr", dtype=complex)
_PAULI_X_CSC = scipy.sparse.csc_matrix([[0., 1.], [1., 0.]], dtype=complex)
_PAULI_Y_CSC = scipy.sparse.csc_matrix([[0., -1.j], [1.j, 0.]], dtype=complex)
_PAULI_Z_CSC = scipy.sparse.csc_matrix([[1., 0.], [0., -1.]], dtype=complex)
_Q_RAISE_CSC = (_PAULI_X_CSC - 1.j * _PAULI_Y_CSC) / 2.
_Q_LOWER_CSC = (_PAULI_X_CSC + 1.j * _PAULI_Y_CSC) / 2.


def JordanWignerTerm(index, n_qubits):
  """Make a matrix representation of a fermion operator.

  Args:
    index: This is a nonzero integer. The integer indicates the tensor
      factor and the sign indicates raising or lowering.
    n_qubits: This int gives the number of qubits in the Hilbert space.

  Returns:
    The corresponding fermion operator. This is a scipy csc sparse matrix.
  """
  # Construct fermionic operator.
  qubit = abs(index)
  operator = 1
  for _ in xrange(qubit - 1):
    operator = scipy.sparse.kron(operator, _IDENTITY_CSC, "csc")
  if index > 0:
    operator = scipy.sparse.kron(operator, _Q_RAISE_CSC, "csc")
  else:
    operator = scipy.sparse.kron(operator, _Q_LOWER_CSC, "csc")
  for _ in xrange(n_qubits - qubit):
    operator = scipy.sparse.kron(operator, _PAULI_Z_CSC, "csc")
  assert scipy.sparse.isspmatrix_csc(operator)
  return operator


def GetJordanWignerTerms(n_qubits):
  """Make a dictionary of the fermionic operators for up to n_qubits.

  Args:
    n_qubits: This int gives the number of qubits in the Hilbert space.

  Returns:
    A dictionary having keys corresponding to a fermionic raising
    or lowering operator. The raising operators are positive ints
    and the lower operators are negative ints. The magnitude of the
    key represents the tensor factor on which the term acts. The value
    associated with the key is a scipy sparse CSC matrix representing
    the operator.
  """
  jw_terms = {}
  for index in range(-n_qubits, n_qubits + 1):
    if index:
      jw_terms[index] = JordanWignerTerm(index, n_qubits)
  return jw_terms


def MatrixForm(coefficient, term, jw_terms):
  """Given a coefficient and symbolic fermionic term, return matrix.

  Args:
    coefficient: A float giving the coefficient of the operator.
    term: a fermionic term represented symbolically as a list of nonzero ints.
    jw_terms: a python dictionary which indexes the operators. This is passed
      to function in order to avoid recomputing the operators at each call.

  Returns:
    The corresponding operator. This is a csc scipy sparse matrix.
  """
  operator = coefficient
  for index in term:
    operator = operator * jw_terms[index]
  return operator


def GetDeterminants(n_qubits, n_electrons):
  """Generate an array of all states on n_qubits with n_fermions.

  Args:
    n_qubits: an int giving the number of qubits.
    n_electrons: an int giving the number of electrons:

  Returns:
    A numpy array where each row is a state on n_qubits with n_fermions.
  """

  # Initialize vector of states.
  n_configurations = int(numpy.rint(scipy.misc.comb(n_qubits, n_electrons)))
  states = numpy.zeros((n_configurations, n_qubits), int)

  # Loop over valid states.
  for i, occupied in enumerate(
      itertools.combinations(range(n_qubits), r=n_electrons)):
    states[i, occupied] = 1
  return states


def ConfigurationProjector(n_qubits, n_electrons):
  """Construct projector into an n_electron manifold.

  Args:
    n_qubits: This int gives the number of qubits in the Hilbert space.
    n_electrons: This int gives the number manifold in which to project.

  Returns:
    A projector matrix is scipy.sparse 'csc' format.
  """
  # Initialize projector computation.
  states = GetDeterminants(n_qubits, n_electrons)
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


def NumberOperator(n_qubits):
  """Compute number operator.

  Args:
    n_qubits: This int gives the number of qubits in the Hilbert space.

  Returns:
    A number operator matrix in scipy.sparse 'csc' format.
  """
  number_operator = 0
  jw_terms = GetJordanWignerTerms(n_qubits)
  for tensor_factor in range(1, n_qubits + 1):
    term = [tensor_factor, -tensor_factor]
    operator = MatrixForm(1, term, jw_terms)
    number_operator = number_operator + operator
  return number_operator


def IsHermitian(matrix, tolerance=1e-12):
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
      print "Hermitian discrepancy = %s." % repr(discrepancy)
      return False
  return True


def Expectation(operator, state_vector):
  """Compute expectation value of an operator with a state.

  Args:
    operator: scipy.sparse csc operator.
    state_vector: scipy.sparse.csc vector.

  Returns:
    A real float giving expectation value.
  """
  state_vector = scipy.sparse.csc_matrix(state_vector)
  operator = scipy.sparse.csc_matrix(operator)
  expectation = state_vector.getH() * operator * state_vector
  assert expectation.get_shape() == (1, 1)
  return expectation[0, 0]


def SparseDiagonalize(operator):
  """Compute and save information about lowest eigenvalue.

  Args:
    operator: A scipy.sparse csc operator to diagonalize.

  Returns:
    A real float giving lowest eigenvalue.
  """
  values, vectors = scipy.sparse.linalg.eigsh(
      operator, 4, which="SA", maxiter=1e7)
  eigenstate = scipy.sparse.csc_matrix(vectors[:, 0])
  eigenvalue = values[0]
  return eigenvalue, eigenstate.getH()


def GetGap(operator):
  """Compute gap between lowest eigenvalue and first excited state.

  Args:
    operator: A scipy.sparse csc operator to diagonalize for gap.

  Returns:
    A real float giving eigenvalue gap.
  """
  values, _ = scipy.sparse.linalg.eigsh(operator, 4, which="SA", maxiter=1e7)
  gap = abs(values[1] - values[0])
  return gap


def OccupancyOperator(index, n_qubits):
  """Operator to measure orbital occupancy.

  Args:
    index: An int giving site to study.
    n_qubits: An int giving the number of spin-orbitals in system.

  Returns:
    A scipy.sparse csc matrix operator.
  """
  jw_terms = GetJordanWignerTerms(n_qubits)
  term = [index, -index]
  operator = MatrixForm(1., term, jw_terms)
  return operator


def AntiFerromagneticOrderOperator(index, n_qubits):
  """Operator to measure antiferromagnetic order from one site.

  Args:
    index: An int giving site to study.
    n_qubits: An int giving number of qubits in the system.

  Returns:
    A scipy.sparse csc matrix operator.
  """
  jw_terms = GetJordanWignerTerms(n_qubits)
  term = [1, -1, index, -index]
  operator = MatrixForm(1., term, jw_terms)
  return operator
