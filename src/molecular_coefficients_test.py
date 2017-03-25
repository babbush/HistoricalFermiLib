"""Tests for molecular_coefficients.py"""
from molecular_coefficients import MolecularCoefficients
import itertools
import unittest
import numpy


class MolecularCoefficientsTest(unittest.TestCase):

  def setUp(self):
    self.n_qubits = 2
    self.constant = 23

  def test_rotate_basis_identical(self):
    rotation_matrix_identical = numpy.zeros((self.n_qubits, self.n_qubits))
    rotation_matrix_identical[0, 0] = 1
    rotation_matrix_identical[1, 1] = 1

    one_body = numpy.zeros((self.n_qubits, self.n_qubits))
    two_body = numpy.zeros((self.n_qubits, self.n_qubits,
                            self.n_qubits, self.n_qubits))
    i = 0
    j = 0
    for p in range(self.n_qubits):
      for q in range(self.n_qubits):
        one_body[p, q] = i
        i = i + 1
        for r in range(self.n_qubits):
          for s in range(self.n_qubits):
            two_body[p, q, r, s] = j
            j = j + 1
    molecular_coefficients = MolecularCoefficients(self.constant,
                                                   one_body, two_body)
    want_molecular_coefficients = MolecularCoefficients(self.constant,
                                                        one_body, two_body)

    molecular_coefficients.rotate_basis(rotation_matrix_identical)
    self.assertEqual(molecular_coefficients, want_molecular_coefficients)

  def test_rotate_basis_reverse(self):
    rotation_matrix_reverse = numpy.zeros((self.n_qubits, self.n_qubits))
    rotation_matrix_reverse[0, 1] = 1
    rotation_matrix_reverse[1, 0] = 1

    one_body = numpy.zeros((self.n_qubits, self.n_qubits))
    two_body = numpy.zeros((self.n_qubits, self.n_qubits,
                            self.n_qubits, self.n_qubits))
    one_body_reverse = numpy.zeros((self.n_qubits, self.n_qubits))
    two_body_reverse = numpy.zeros((self.n_qubits, self.n_qubits,
                                    self.n_qubits, self.n_qubits))
    i = 0
    j = 0
    i_reverse = pow(self.n_qubits, 2) - 1
    j_reverse = pow(self.n_qubits, 4) - 1
    for p in range(self.n_qubits):
      for q in range(self.n_qubits):
        one_body[p, q] = i
        i = i + 1
        one_body_reverse[p, q] = i_reverse
        i_reverse = i_reverse - 1
        for r in range(self.n_qubits):
          for s in range(self.n_qubits):
            two_body[p, q, r, s] = j
            j = j + 1
            two_body_reverse[p, q, r, s] = j_reverse
            j_reverse = j_reverse - 1
    molecular_coefficients = MolecularCoefficients(self.constant,
                                                   one_body, two_body)
    want_molecular_coefficients = MolecularCoefficients(self.constant,
                                                        one_body_reverse,
                                                        two_body_reverse)
    molecular_coefficients.rotate_basis(rotation_matrix_reverse)
    self.assertEqual(molecular_coefficients, want_molecular_coefficients)


# Test.
if __name__ == "__main__":
  unittest.main()
