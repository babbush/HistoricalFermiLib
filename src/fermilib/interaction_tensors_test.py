"""Tests for interaction_tensors.py"""
from __future__ import absolute_import

import unittest

import numpy

from fermilib.interaction_tensors import InteractionTensor


class InteractionTensorTest(unittest.TestCase):

  def setUp(self):
    self.n_qubits = 2
    self.constant = 23.0

  def test_iter_and_str(self):
    one_body = numpy.zeros((self.n_qubits, self.n_qubits))
    two_body = numpy.zeros((self.n_qubits, self.n_qubits,
                            self.n_qubits, self.n_qubits))
    one_body[0, 1] = 11.0
    two_body[0, 1, 1, 0] = 22.0
    interaction_tensor = InteractionTensor(self.constant,
                                           one_body, two_body)
    want_str = '[] 23.0\n[0 1] 11.0\n[0 1 1 0] 22.0\n'
    self.assertEqual(interaction_tensor.__str__(), want_str)

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
    interaction_tensor = InteractionTensor(self.constant,
                                           one_body, two_body)
    want_interaction_tensor = InteractionTensor(self.constant,
                                                one_body, two_body)

    interaction_tensor.rotate_basis(rotation_matrix_identical)
    self.assertEqual(interaction_tensor, want_interaction_tensor)

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
    interaction_tensor = InteractionTensor(self.constant,
                                           one_body, two_body)
    want_interaction_tensor = InteractionTensor(self.constant,
                                                one_body_reverse,
                                                two_body_reverse)
    interaction_tensor.rotate_basis(rotation_matrix_reverse)
    self.assertEqual(interaction_tensor, want_interaction_tensor)


# Test.
if __name__ == "__main__":
  unittest.main()
