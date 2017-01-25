"""This file has useful utilities for numerically emulating VQE circuits.
"""

import numpy
import scipy
from scipy import sparse as sps
import scipy.sparse.linalg

import fermionic_utils
import hamiltonian_utils
import ucc_ansatz


IDENTITY_SP = sps.identity(2, format="csr", dtype=complex)
PAULI_X_SP = sps.csr_matrix([[0., 1.], [1., 0.]], dtype=complex)
PAULI_Y_SP = sps.csr_matrix([[0., -1.j], [1.j, 0.]], dtype=complex)
PAULI_Z_SP = sps.csr_matrix([[1., 0.], [0., -1.]], dtype=complex)


class ErrorWavefunction(Exception):
  pass


class ErrorMatrix(Exception):
  pass


def get_pauli_string_matrix(pauli_string):
  """Builds sparse matrix of a PauliString.

  Args:
    pauli_string: PauliString class object

  Returns:
    matrix representation as A_max x ... x A_1 x A_0 where A is a Pauli matrix.
    matrix is a complex scipy csr matrix.
  """
  #TODO(dsteiger): Test if it works faster by building backwards and using bsr
  num_qubits = pauli_string.num_qubits
  coefficient = pauli_string.coefficient
  x_ids = pauli_string.pauli_x
  y_ids = pauli_string.pauli_y
  z_ids = pauli_string.pauli_z
  # Generate list of tuples for all non trivial (i.e. not identity) pauli
  # factors. First element of each tuple is the qubit index and the second
  # element is 1 for PauliX, 2 for PauliY, 3 for PauliZ
  # Example: pauli =  [(2,1), (4,3)] corresponds to pauli term X2Z4
  non_trivial_pauli = sorted(
      zip(x_ids, len(x_ids) * [1]) + zip(y_ids, len(y_ids) * [2]) + zip(
          z_ids, len(z_ids) * [3]))
  matrix = sps.identity(1, format="csr", dtype=complex) * coefficient
  previous_index = -1
  for index, pauli in non_trivial_pauli:
    if previous_index + 1 != index:
      # Apply identity to qubits inbetween
      id_matrix = sps.identity(2**(index - previous_index - 1))
      matrix = sps.kron(id_matrix, matrix, format="csr")
    if pauli == 1:
      matrix = sps.kron(PAULI_X_SP, matrix, format="csr")
    elif pauli == 2:
      matrix = sps.kron(PAULI_Y_SP, matrix, format="csr")
    else:
      matrix = sps.kron(PAULI_Z_SP, matrix, format="csr")
    previous_index = index
  # Apply identity to remaining qubits
  remaining = num_qubits - previous_index - 1
  id_matrix = sps.identity(2**remaining, format="csr", dtype=complex)
  matrix = sps.kron(id_matrix, matrix, format="csr")
  return matrix


def get_qubit_hamiltonian_matrix(qubit_hamiltonian):
  """Builds sparse matrix of a QubitHamiltonian.

  Args:
    qubit_hamiltonian: QubitHamiltonian class object

  Returns:
    matrix representation using a complex scipy csr matrix
  """
  size = 2**qubit_hamiltonian.terms[0].num_qubits
  matrix = sps.csr_matrix((size, size), dtype=complex)
  for term in qubit_hamiltonian.terms:
    matrix = matrix + get_pauli_string_matrix(term)
  return matrix


def get_exp_pauli_string_matrix(pauli_string, theta=1.0):
  """Calculate unitary matrix exp(-i theta pauli_string).

  Returns matrix corresponding to exp(-i theta pauli_string.coefficient "X1Y3")
  In order for the resulting operator/matrix to be unitary,
  theta * pauli_string.coefficient must be real. By convention theta is real and
  hence the pauli_string needs to have a real coefficient in order for the
  output matrix to be unitary.
  Both dimension of the matrix are 2**pauli_string.num_qubits.

  Args:
    pauli_string: PauliString class object
    theta: real coefficient

  Returns:
    complex scipy csr matrix
  """
  # Correct result could be achieved with:
  # matrix = get_pauli_string_matrix(pauli_string) * theta * -1.0j
  # scipy.sparse.linalg.expm(matrix)
  # But that is orders of magnitude slower than this method:
  # Suppose an operator A satisfies A^2 = 1, then
  # exp(1.0j*theta*A) = cos(theta) * identity + 1.0j sin(theta) * A
  # see Nielsen and Chuang Exercise 4.2. This equation can be proven
  # by using the taylor expansion of exp() and using A^2=1
  pauli_coefficient = pauli_string.coefficient
  tmp_pauli = hamiltonian_utils.PauliString(pauli_string.num_qubits, 1,
                                            pauli_string.pauli_x,
                                            pauli_string.pauli_y,
                                            pauli_string.pauli_z)
  identity = sps.identity(2**pauli_string.num_qubits, format="csr",
                          dtype=complex)
  pauli_matrix = get_pauli_string_matrix(tmp_pauli)
  return (scipy.cos(-1 * theta * pauli_coefficient) * identity +
          1.0j * scipy.sin(-1 * theta * pauli_coefficient) * pauli_matrix)


def calculate_exact_ground_state(hamiltonian_matrix):
  """Calculates the exact ground state energy + wavefunction of a hamiltonian.

  Args:
    hamiltonian_matrix: Sparse scipy matrix representing the hamiltonian

  Returns:
    (e0, v0) where e0 is a float (the ground state energy) and v0 is
    numpy.ndarray with shape (2**N,) which represents the ground state
    wavefunction. N is the number of qubits.
  """
  num_eigenvalues = 1
  v0 = numpy.ones(hamiltonian_matrix.shape[0], dtype="complex")
  eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(hamiltonian_matrix,
                                                        k=num_eigenvalues,
                                                        which="SA", tol=1e-9,
                                                        maxiter=20000, v0=v0)
  return (eigenvalues[0], eigenvectors[:, 0])


def get_expectation_value(hermitian_matrix, wavefunction, normalize=False):
  """Calculates the expecation value of a hermitian operator.

  Args:
    hermitian_matrix: Sparse scipy matrix representing a hermitian operator.
    wavefunction: np.ndarray with dimension (2**N, )
    normalize: Bool. Default False. In this case the measurement result is
               divided by the norm of the wavefunction in order to account
               for small numerical errors which changed the norm of the
               wavefunction.

  Returns:
    float. Energy of this wavefunction

  Raises:
    ErrorWavefunction: If input wavefunction has wrong format
    ErrorMatrix: Non hermitian matrix supplied
  """
  if not (isinstance(wavefunction, numpy.ndarray) and wavefunction.ndim == 1):
    raise ErrorWavefunction("Wrong wavefunction format")
  energy = numpy.vdot(wavefunction, hermitian_matrix.dot(wavefunction))
  if energy.imag > 10e-6:
    raise ErrorMatrix("hermitian_matrix is not hermitian")
  if normalize:
    norm_square = calculate_overlap(wavefunction, wavefunction)
    return energy.real / float(norm_square)
  return energy.real


def calculate_overlap(wavefunction1, wavefunction2):
  """Calculates the overlap of two wavefunctions (N qubits).

  Args:
    wavefunction1: numpy.ndarray with shape (2**N,)
    wavefunction2: numpy.ndarray with shape (2**N,)

  Returns:
    float. Absolute value of the inner product between the two wavefunctions
  """
  return abs(numpy.vdot(wavefunction1, wavefunction2))


def get_hf_wavefunction(num_electrons, num_qubits):
  """Returns the Hartree Fock wavefunction.

  We assume that the qubits in the hamiltonian are numbered according to the
  energy of the corresponding spin orbital. Lowest qubits have lowest energy.

  Args:
    num_electrons: integer. Specifies number of electrons in the desired state.
    num_qubits: integer. Specifies the number of qubits which is equal to the
                number of spin orbitals.

  Returns:
    numpy.ndarray with shape (2**num_qubits,)
  """
  hf_ground_state = (num_qubits - num_electrons) * "0" + num_electrons * "1"
  index = int(hf_ground_state, 2)
  wavefunction = numpy.zeros(2**num_qubits)
  wavefunction[index] = 1
  return wavefunction


def apply_ucc_ansatz(ansatz, wavefunction):
  """Applies the UCC operator to a HF wavefunction.

  Note: The application order of the terms matter and is not predefined.

  Args:
    ansatz: UccAnsatz object.
    wavefunction: numpy.ndarray with shape (2**num_qubits,), where N is the
                  number of qubits.
  Returns:
    numpy.ndarray with shape (2**num_qubits,) which is the wavefunction after
    applying the ucc ansatz.
  """
  for theta, _, hamiltonian  in ansatz.ucc_terms:
    for pauli_term in hamiltonian.terms:
      matrix = get_exp_pauli_string_matrix(pauli_term, theta)
      wavefunction = matrix.dot(wavefunction)
  return wavefunction


def get_cc_operator(filename, num_qubits, num_electrons, closed_shell):
  """Returns coupled cluster operator T as a sparse matrix.

  The energy equation of the coupled cluster ansatz is given by
  E := < HF | e**(-T) H e**(T) | HF > or
  E := < HF | H e**(T) | HF >
  where | HF > is the Hartree Fock ground state. The first equation is
  used to calculate the cc amplitudes using the concept of a similarity
  transformed hamiltonian. The second equation is equal to the first equation
  due to how the normalization is chosen. It is cheaper to calculate so use
  that one.

  Args:
    filename: path to cc amplitude inputfile
    num_qubits: integer. Number of spin orbitals or qubits
    num_electrons: integer. Number of electrons.
    closed_shell: bool. If system has a closed shell of electrons.

  Returns:
    complex scipy csr matrix of the T operator.
  """
  cc_amplitudes = ucc_ansatz.read_cc_file(filename)
  ucc_amplitude_dict = ucc_ansatz.convert_cc_amplitudes_to_ucc(cc_amplitudes,
                                                               num_electrons,
                                                               closed_shell)
  pauli_terms = []
  for key in ucc_amplitude_dict:
    fermion_op = fermionic_utils.FermionicOperator(ucc_amplitude_dict[key],
                                                   list(key))
    pauli_terms += fermionic_utils.jordan_wigner_transform(fermion_op,
                                                           num_qubits)
  cc_operator_matrix = sps.csr_matrix((2**num_qubits, 2**num_qubits),
                                      dtype=complex)
  for pauli_term in pauli_terms:
    cc_operator_matrix = cc_operator_matrix + get_pauli_string_matrix(pauli_term)
  return cc_operator_matrix


def calculate_cc_energy(filename_ccamp, num_qubits, num_electrons, closed_shell,
                        hamiltonian_matrix):
  """Calculates the coupled cluster energy from the amplitudes.

  The energy equation of the coupled cluster ansatz is given by
  E := < HF | e**(-T) H e**(T) | HF > or
  E := < HF | H e**(T) | HF >
  where | HF > is the Hartree Fock ground state. The first equation is
  used to calculate the cc amplitudes using the concept of a similarity
  transformed hamiltonian. The second equation is equal to the first equation
  due to how the normalization is chosen. It is cheaper to calculate so we used
  that one.

  Args:
    filename_ccamp: path to cc amplitude inputfile
    num_qubits: integer. Number of spin orbitals or qubits
    num_electrons: integer. Number of electrons.
    closed_shell: bool. If system has a closed shell of electrons.
    hamiltonian_matrix: scipy sparse matrix representing the Hamiltonian H

  Returns:
    flaoting point number. Energy of the coupled cluster ansatz.
  """
  t_matrix = get_cc_operator(filename_ccamp, num_qubits, num_electrons,
                             closed_shell)
  exp_t_matrix = sps.linalg.expm(t_matrix.tocsc())
  operator = hamiltonian_matrix * exp_t_matrix
  hf_wavefunction = get_hf_wavefunction(num_electrons, num_qubits)
  cc_energy = get_expectation_value(operator, hf_wavefunction)
  return cc_energy
