"""Tests for emulation_utils.
"""

import os
import unittest

import numpy as np
import scipy.sparse

import emulation_utils
import hamiltonian_utils


_BASE_PATH = "testdata/"


def get_test_filename(name):
  return os.path.join(_BASE_PATH, name)


class EmulationUtilsTest(unittest.TestCase):

  def setUp(self):
    self.x = np.array([[0, 1], [1, 0]], dtype=complex)
    self.y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    self.z = np.array([[1, 0], [0, -1]], dtype=complex)
    self.identity = np.array([[1, 0], [0, 1]], dtype=complex)

  def test_pauli_matrices(self):
    self.assertTrue(np.allclose(emulation_utils.PAULI_X_SP.toarray(), self.x))
    self.assertTrue(np.allclose(emulation_utils.PAULI_Y_SP.toarray(), self.y))
    self.assertTrue(np.allclose(emulation_utils.PAULI_Z_SP.toarray(), self.z))

  def test_get_pauli_string_matrix(self):
    # Test that identity is added on second tensor factor.
    # Also tests that "0" is not thrown away.
    term1 = hamiltonian_utils.init_pauli_string_from_string(2, 1.0, "X0")
    term1_matrix = emulation_utils.get_pauli_string_matrix(term1)
    correct_term1 = np.kron(self.identity, self.x)
    self.assertTrue(np.allclose(term1_matrix.toarray(), correct_term1))
    # Test that identity is added on first tensor factor.
    term2 = hamiltonian_utils.init_pauli_string_from_string(2, 0.5, "X1")
    term2_matrix = emulation_utils.get_pauli_string_matrix(term2)
    correct_term2 = np.kron(self.x, self.identity) * 0.5
    self.assertTrue(np.allclose(term2_matrix.toarray(), correct_term2))
    # Test that identities are added before, in the middle and after.
    # And test that X, Y, Z factors are correctly applied.
    term3 = hamiltonian_utils.init_pauli_string_from_string(7, 0.5j, "X1Y2Z4")
    term3_matrix = emulation_utils.get_pauli_string_matrix(term3)
    correct_term3 = np.kron(self.identity, np.kron(self.identity, self.z))
    correct_term3 = np.kron(correct_term3, np.kron(self.identity, self.y))
    correct_term3 = np.kron(correct_term3, self.x)
    correct_term3 = np.kron(correct_term3, self.identity) * 0.5j
    self.assertEqual(term3_matrix.toarray().shape, (128, 128))
    self.assertTrue(np.allclose(term3_matrix.toarray(), correct_term3))

  def test_get_qubit_hamiltonian_matrix(self):
    term1 = hamiltonian_utils.init_pauli_string_from_string(2, 1.0, "X0")
    term1_matrix = emulation_utils.get_pauli_string_matrix(term1)
    term2 = hamiltonian_utils.init_pauli_string_from_string(2, 0.5, "X1")
    term2_matrix = emulation_utils.get_pauli_string_matrix(term2)
    hamiltonian = hamiltonian_utils.QubitHamiltonian()
    hamiltonian.add_term(term1)
    hamiltonian.add_term(term2)
    correct_matrix = term1_matrix + term2_matrix
    test = emulation_utils.get_qubit_hamiltonian_matrix(hamiltonian)
    self.assertTrue(np.allclose(correct_matrix.toarray(), test.toarray()))

  def test_get_exp_pauli_string_matrix(self):
    pauli_c = 1.5
    theta = 0.5
    term = hamiltonian_utils.init_pauli_string_from_string(2, pauli_c, "Z1")
    value1 = np.exp(theta * pauli_c * -1.0j)
    value2 = np.exp(theta * pauli_c * 1.0j)
    correct_matrix = np.array(
        [[value1, 0, 0, 0], [0, value1, 0, 0], [0, 0, value2, 0],
         [0, 0, 0, value2]],
        dtype=complex)
    matrix = emulation_utils.get_exp_pauli_string_matrix(term, theta)
    self.assertTrue(np.allclose(correct_matrix, matrix.toarray()))

  def test_calculate_exact_ground_state(self):
    term = hamiltonian_utils.init_pauli_string_from_string(2, 2.0, "Z0Z1")
    matrix = emulation_utils.get_pauli_string_matrix(term)
    e0, v0 = emulation_utils.calculate_exact_ground_state(matrix)
    self.assertAlmostEqual(-2, e0)
    self.assertTrue(isinstance(v0, np.ndarray))
    self.assertEqual(v0.ndim, 1)

  def test_get_expectation_value(self):
    matrix = scipy.sparse.csr_matrix([[1, 0], [0, 2]], dtype="complex")
    wavefunction = np.array([1, -2.0j])
    energy = emulation_utils.get_expectation_value(matrix, wavefunction)
    self.assertEqual(9, energy)
    # Test if normalization of wavefunction is correctly implemented
    energy_normalized = emulation_utils.get_expectation_value(matrix,
                                                              wavefunction,
                                                              True)
    self.assertAlmostEqual(9./5., energy_normalized)
    with self.assertRaises(emulation_utils.ErrorWavefunction):
      wrong = np.array([[1], [2]])
      energy = emulation_utils.get_expectation_value(matrix, wrong)
    with self.assertRaises(emulation_utils.ErrorMatrix):
      matrix = scipy.sparse.csr_matrix([[0, 1], [2, 0]], dtype="complex")
      energy = emulation_utils.get_expectation_value(matrix, wavefunction)

  def test_calculate_overlap(self):
    wavefunction1 = np.array([1, -2.0j])
    wavefunction2 = np.array([1, -2.0j])
    overlap = emulation_utils.calculate_overlap(wavefunction1, wavefunction2)
    self.assertEqual(overlap, 5)

  def test_get_hf_wavefunction(self):
    num_electrons = 3
    num_qubits = 7
    hf_wavefunction = emulation_utils.get_hf_wavefunction(num_electrons,
                                                          num_qubits)
    self.assertEqual(hf_wavefunction.shape[0], 2**7)
    for ii in range(2**7):
      if ii != 7:  # corresponds to 111 in binary
        self.assertAlmostEqual(hf_wavefunction[ii], 0)
      else:
        self.assertAlmostEqual(hf_wavefunction[ii], 1)

  def test_coupled_cluster(self):
    hamiltonian_file = get_test_filename("H4_sto-3g_JW_8.ham")
    cc_amplitude_file = get_test_filename("H4_sto-3g.ccamp")
    num_qubits = 8
    num_electrons = 4
    closed_shell = True
    hamiltonian = hamiltonian_utils.read_hamiltonian_file(hamiltonian_file,
                                                          num_qubits)
    h_matrix = emulation_utils.get_qubit_hamiltonian_matrix(hamiltonian)
    cc_energy = emulation_utils.calculate_cc_energy(cc_amplitude_file,
                                                    num_qubits, num_electrons,
                                                    closed_shell, h_matrix)
    self.assertAlmostEqual(cc_energy, -2.14510622767)


if __name__ == "__main__":
  unittest.main()
