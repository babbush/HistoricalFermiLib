"""Tests for molecular_operators.py"""
from molecular_operators import MolecularOperator, MolecularOperatorError
import fermion_operators
import itertools
import unittest
import numpy


class MolecularOperatorsTest(unittest.TestCase):

  def setUp(self):
    self.n_qubits = 5
    self.constant = 0.
    self.one_body = numpy.zeros((self.n_qubits, self.n_qubits), float)
    self.two_body = numpy.zeros((self.n_qubits, self.n_qubits,
                                 self.n_qubits, self.n_qubits), float)
    self.molecular_operator = MolecularOperator(self.constant, self.one_body,
                                                self.two_body)

  def test_jordan_wigner_one_body(self):

    # Make sure it agrees with FermionTerm.jordan_wigner_transform().
    for p in range(self.n_qubits):
      for q in range(self.n_qubits):

        # Get test qubit operator.
        test_operator = self.molecular_operator.jordan_wigner_one_body(p, q)

        # Get correct qubit operator.
        fermion_term = fermion_operators.FermionTerm(
            [(p, 1), (q, 0)], 1.)
        correct_operator = fermion_term.jordan_wigner_transform()
        hermitian_conjugate = fermion_term.hermitian_conjugated()
        if fermion_term != hermitian_conjugate:
          correct_operator += hermitian_conjugate.jordan_wigner_transform()

        # Make sure its correct.
        self.assertEqual(test_operator, correct_operator)

  def test_jordan_wigner_two_body(self):

    # Make sure it agrees with FermionTerm.jordan_wigner_transform().
    for p in range(self.n_qubits):
      for q in range(self.n_qubits):
        for r in range(self.n_qubits):
          for s in range(self.n_qubits):

            # Get test qubit operator.
            test_operator = self.molecular_operator.jordan_wigner_two_body(
                p, q, r, s)

            # Get correct qubit operator.
            fermion_term = fermion_operators.FermionTerm(
                [(p, 1), (q, 1), (r, 0), (s, 0)], 1.)
            correct_operator = fermion_term.jordan_wigner_transform()
            hermitian_conjugate = fermion_term.hermitian_conjugated()
            if fermion_term != hermitian_conjugate:
              correct_operator += hermitian_conjugate.jordan_wigner_transform()

            # Make sure its correct.
            self.assertEqual(test_operator, correct_operator)

  def test_four_point_iter(self):
    constant = 100.0
    one_body = numpy.zeros((self.n_qubits, self.n_qubits), float)
    two_body = numpy.zeros((self.n_qubits, self.n_qubits,
                            self.n_qubits, self.n_qubits), float)
    one_body[1, 1] = 10.0
    one_body[2, 3] = 11.0
    one_body[3, 2] = 11.0
    two_body[1, 2, 3, 4] = 12.0
    two_body[2, 1, 4, 3] = 12.0
    two_body[3, 4, 1, 2] = 12.0
    two_body[4, 3, 2, 1] = 12.0
    molecular_operator = MolecularOperator(constant, one_body, two_body)

    want_str = '100.0\n10.0\n11.0\n12.0\n'
    got_str = ''
    for key in molecular_operator.four_point_iter():
      got_str += '{}\n'.format(molecular_operator[key])
    self.assertEqual(want_str, got_str)

  def test_eight_point_iter(self):
    constant = 100.0
    one_body = numpy.zeros((self.n_qubits, self.n_qubits), float)
    two_body = numpy.zeros((self.n_qubits, self.n_qubits,
                            self.n_qubits, self.n_qubits), float)
    one_body[1, 1] = 10.0
    one_body[2, 3] = 11.0
    one_body[3, 2] = 11.0
    two_body[1, 2, 3, 4] = 12.0
    two_body[2, 1, 4, 3] = 12.0
    two_body[3, 4, 1, 2] = 12.0
    two_body[4, 3, 2, 1] = 12.0
    two_body[1, 4, 3, 2] = 12.0
    two_body[2, 3, 4, 1] = 12.0
    two_body[3, 2, 1, 4] = 12.0
    two_body[4, 1, 2, 3] = 12.0
    molecular_operator = MolecularOperator(constant, one_body, two_body)

    want_str = '100.0\n10.0\n11.0\n12.0\n'
    got_str = ''
    for key in molecular_operator.eight_point_iter():
      got_str += '{}\n'.format(molecular_operator[key])
    self.assertEqual(want_str, got_str)


# Test.
if __name__ == "__main__":
  unittest.main()
