"""Module to construct unitary coupled cluster (UCC) operators with jellium."""

import fermion_operators
import itertools
import numpy
import scipy
import scipy.sparse
import scipy.sparse.linalg


def simple_initial_state(n_qubits, n_fermions):
  """Create an intitial state populated with n_fermions in JW representation

  Args:
    n_qubits(int): Number of qubits
    n_fermions(int): Number of fermions to be populated

  Returns:
    numpy array representing state with correct number of fermions
  """
  zero_state = numpy.array([[1.0], [0.0]], dtype=complex)
  one_state = numpy.array([[0.0], [1.0]], dtype=complex)

  qubit_state = reduce(numpy.kron,
                       (one_state, ) * n_fermions +
                       (zero_state, ) * (n_qubits - n_fermions))
  return qubit_state


def jellium_uccsd_amplitude_count(n_qubits, n_fermions):
  """Compute the number of amplitudes in a spin-conserving uccsd operator

  Args:
    n_qubits(int): Number of qubits in the system, or possible orbitals
    n_fermions(int): Number of fermions present in the sytem

  Returns:
    (int) Number of amplitudes that determine a uccsd state given a single
       reference and spin-conserving excitations.
  """
  n_occupied = int(numpy.ceil(n_fermions / 2.))
  n_virtual = n_qubits / 2 - n_occupied

  n_t1 = n_occupied * n_virtual
  n_t2 = n_t1 ** 2

  return (n_t1 + n_t2)


def jellium_uccsd(n_qubits, n_fermions, amplitudes):
  """Build spin conserving UCCSD generator from amplitudes

   Args:
     n_qubits(int): Number of qubits
     n_fermions(int): Number of fermions
     amplitudes - Amplitude list, t1 followed by t2

   Returns:
     fermionic_operator storing the cluster operator
   """

  n_occupied = int(numpy.ceil(n_fermions / 2.))
  n_virtual = n_qubits / 2 - n_occupied

  n_t1 = n_occupied * n_virtual
  n_t2 = n_t1 ** 2

  t1 = amplitudes[:n_t1]
  t2 = amplitudes[n_t1:]

  def t1_ind(i, j):
    return i * n_occupied + j

  def t2_ind(i, j, k, l):
    return i * n_occupied * n_virtual * n_occupied + \
        j * n_virtual * n_occupied + \
        k * n_occupied + \
        l
  cluster_operator = fermion_operators.FermionOperator(n_qubits)

  # Singles
  for i in range(n_virtual):
    for j in range(n_occupied):
      for s1 in range(2):
        cluster_term = fermion_operators.\
            FermionTerm(n_qubits, t1[t1_ind(i, j)],
                        [(2 * (i + n_occupied) + s1, 1),
                         (2 * j + s1, 0)])
        cluster_operator += cluster_term
        cluster_operator += -1. * cluster_term.hermitian_conjugated()

  # Doubles
  for i in range(n_virtual):
    for j in range(n_occupied):
      for s1 in range(2):
        for k in range(n_virtual):
          for l in range(n_occupied):
            for s2 in range(2):
              cluster_term = fermion_operators. \
                  FermionTerm(n_qubits, t2[t2_ind(i, j, k, l)],
                              [(2 * (i + n_occupied) + s1, 1),
                               (2 * j + s1, 0),
                               (2 * (k + n_occupied) + s2, 1),
                               (2 * l + s2, 0)])
              cluster_operator += cluster_term
              cluster_operator += -1. * cluster_term.hermitian_conjugated()

  return cluster_operator


def jellium_uccsd_energy(n_qubits, n_fermions, initial_state,
                         qubit_hamiltonian, amplitudes):
  """Compute the energy of a uccsd state

  Args:
    n_qubits(int): Number of qubits in the system
    n_fermions(int): Number of fermions present in the system
    initial_state(ndarray): Initial state in qubit representation for cluster
      operator to act on
    qubit_hamiltonian(LinearOperator): Sparse qubit representation of system
      Hamiltonian used to determine the energy
    amplitudes(ndarray): Coupled cluster ampltidues, T1 followed by T2

  Returns:
    (float): Expected energy of the coupled cluster state.
  """
  uccsd_operator = jellium_uccsd(n_qubits, n_fermions, amplitudes)
  uccsd_qubit = uccsd_operator.jordan_wigner_transform().get_sparse_matrix()
  final_state = scipy.sparse.linalg.expm_multiply(uccsd_qubit, initial_state)
  energy = numpy.real(numpy.dot(numpy.conj(final_state.T),
                      qubit_hamiltonian.dot(final_state))[0, 0])
  print("Current Energy: {}".format(energy))
  return energy
