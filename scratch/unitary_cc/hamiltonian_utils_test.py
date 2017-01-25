"""Tests for hamiltonian_utils.
"""

import os

import unittest

import hamiltonian_utils

_BASE_PATH = "testdata/"


def get_test_filename(name):
  return os.path.join(_BASE_PATH, name)


class PauliString(unittest.TestCase):

  def test_correct_input(self):
    # Test that PauliString is correctly initialized.
    term = hamiltonian_utils.PauliString(12, 0.5, [2, 4], [6], [11])
    self.assertAlmostEqual(term.coefficient, 0.5)
    self.assertEqual(term.num_qubits, 12)
    self.assertEqual(len(term.pauli_x), 2)
    self.assertEqual(term.pauli_x[0], 2)
    self.assertEqual(term.pauli_x[1], 4)
    self.assertEqual(len(term.pauli_y), 1)
    self.assertEqual(term.pauli_y[0], 6)
    self.assertEqual(len(term.pauli_z), 1)
    self.assertEqual(term.pauli_z[0], 11)

  def test_correct_identity(self):
    # Test the special case of initializing the identity operator.
    term = hamiltonian_utils.PauliString(10, 1.0, [], [], [])
    self.assertEqual(term.num_qubits, 10)
    self.assertAlmostEqual(term.coefficient, 1.0)
    self.assertEqual(term.pauli_x, [])
    self.assertEqual(term.pauli_y, [])
    self.assertEqual(term.pauli_z, [])

  def test_input_errors(self):
    # Test that input errors are detected.
    with self.assertRaises(hamiltonian_utils.ErrorPauliString):
      hamiltonian_utils.PauliString(10.1, 0.5, [], [], [])
    with self.assertRaises(hamiltonian_utils.ErrorPauliString):
      hamiltonian_utils.PauliString(10, 0.5, [10], ["a"], [])
    with self.assertRaises(hamiltonian_utils.ErrorPauliString):
      hamiltonian_utils.PauliString(10, 0.5, [1, 2], [4, 2], [])
    with self.assertRaises(hamiltonian_utils.ErrorPauliString):
      hamiltonian_utils.PauliString(10, 0.5, [0], [10], [])

  def test_is_identical_pauli(self):
    # Test is_identical function.
    term1 = hamiltonian_utils.PauliString(12, 0.5, [2, 4], [6], [11])
    term2 = hamiltonian_utils.PauliString(12, 0.7, [4, 2], [6], [11])
    term3 = hamiltonian_utils.PauliString(12, 0.5, [2, 4, 7], [6], [11])
    self.assertTrue(term1.is_identical_pauli(term2))
    self.assertFalse(term1.is_identical_pauli(term3))
    with self.assertRaises(hamiltonian_utils.ErrorPauliString):
      term4 = hamiltonian_utils.PauliString(11, 0.1, [], [], [])
      term1.is_identical_pauli(term4)


class InitPauliStringFromString(unittest.TestCase):

  def test_correct_input_from_string(self):
    term = hamiltonian_utils.init_pauli_string_from_string(12, 0.5, "X2Y6X4Z11")
    correct = hamiltonian_utils.PauliString(12, 0.5, [2, 4], [6], [11])
    self.assertTrue(term.is_identical_pauli(correct))
    self.assertAlmostEqual(term.coefficient, correct.coefficient)
    term2 = hamiltonian_utils.init_pauli_string_from_string(12, 0.7, "I")
    correct2 = hamiltonian_utils.PauliString(12, 0.7, [], [], [])
    self.assertTrue(term2.is_identical_pauli(correct2))
    self.assertAlmostEqual(term2.coefficient, correct2.coefficient)

  def test_undefined_pauli_matrix_from_string(self):
    with self.assertRaises(hamiltonian_utils.ErrorPauliString):
      hamiltonian_utils.init_pauli_string_from_string(5, 0.5, "X2X4A10")
    with self.assertRaises(hamiltonian_utils.ErrorPauliString):
      hamiltonian_utils.init_pauli_string_from_string(5, 0.5, "X2X")


class InitPauliStringFromStringList(unittest.TestCase):

  def test_correct_input_from_string(self):
    term = hamiltonian_utils.init_pauli_string_from_string_list(
        4, 0.5, ["X", "I", "Y", "Z"])
    correct = hamiltonian_utils.PauliString(4, 0.5, [0], [2], [3])
    self.assertTrue(term.is_identical_pauli(correct))
    self.assertAlmostEqual(term.coefficient, correct.coefficient)

  def test_wrong_input(self):
    # Unknown Operator
    with self.assertRaises(hamiltonian_utils.ErrorPauliString):
      hamiltonian_utils.init_pauli_string_from_string_list(1, 0.5, ["A"])
    # Wrong number of qubits specified
    with self.assertRaises(hamiltonian_utils.ErrorPauliString):
      hamiltonian_utils.init_pauli_string_from_string_list(2, 0.5, ["X"])


class QubitHamiltonian(unittest.TestCase):

  def test_init(self):
    h = hamiltonian_utils.QubitHamiltonian()
    self.assertFalse(h.terms)
    term = hamiltonian_utils.PauliString(10, 0.5, [0], [9], [1])
    h2 = hamiltonian_utils.QubitHamiltonian(term)
    self.assertTrue(h2.terms[0].is_identical_pauli(term))
    self.assertAlmostEqual(h2.terms[0].coefficient, term.coefficient)
    # Test that coefficient must be real
    with self.assertRaises(hamiltonian_utils.ErrorQubitHamiltonian):
      imaginary_term = hamiltonian_utils.PauliString(4, -1.0j, [], [], [])
      hamiltonian_utils.QubitHamiltonian(imaginary_term)

  def test_add_term(self):
    h = hamiltonian_utils.QubitHamiltonian()
    term = hamiltonian_utils.PauliString(10, 0.5, [0], [9], [1])
    h.add_term(term)
    self.assertTrue(h.terms[0].is_identical_pauli(term))
    self.assertAlmostEqual(h.terms[0].coefficient, term.coefficient)
    self.assertEqual(len(h.terms), 1)
    # Different number of qubits not allowed
    with self.assertRaises(hamiltonian_utils.ErrorQubitHamiltonian):
      wrong_term = hamiltonian_utils.PauliString(8, 0.5, [], [], [])
      h.add_term(wrong_term)
    # Coefficient must be real
    with self.assertRaises(hamiltonian_utils.ErrorQubitHamiltonian):
      wrong_term2 = hamiltonian_utils.PauliString(8, -1.j, [], [], [])
      h.add_term(wrong_term2)


class ReadHamiltonianTest(unittest.TestCase):

  def test_read_hamiltonian_file(self):
    filename = get_test_filename("HRing2_sto-3g_JW_4.ham")
    hamiltonian = hamiltonian_utils.read_hamiltonian_file(filename, 4)
    self.assertEqual(len(hamiltonian.terms), 15)
    term0 = hamiltonian_utils.init_pauli_string_from_string(4, 0.449271256438,
                                                            "I")
    self.assertTrue(hamiltonian.terms[0].is_identical_pauli(term0))
    self.assertAlmostEqual(hamiltonian.terms[0].coefficient, term0.coefficient)
    term2 = hamiltonian_utils.init_pauli_string_from_string(4, 0.0419600475319,
                                                            "X3Y2Y1X0")
    self.assertTrue(hamiltonian.terms[2].is_identical_pauli(term2))
    self.assertAlmostEqual(hamiltonian.terms[2].coefficient, term2.coefficient)


if __name__ == "__main__":
  unittest.main()
