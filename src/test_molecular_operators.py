"""Tests for molecular_operators.py"""
import molecular_operators
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
    self.molecular_operator = molecular_operators.MolecularOperator(
        self.constant, self.one_body, self.two_body)

  def test_jordan_wigner_one_body(self):

    # Make sure it agrees with FermionTerm.jordan_wigner_transform().
    for p in range(self.n_qubits):
      for q in range(self.n_qubits):

        # Get test qubit operator.
        test_operator = self.molecular_operator.jordan_wigner_one_body(
            self.n_qubits, p, q)

        # Get correct qubit operator.
        fermion_term = fermion_operators.FermionTerm(
            self.n_qubits, 1., [(p, 1), (q, 0)])
        correct_operator = fermion_term.jordan_wigner_transform()
        hermitian_conjugate = fermion_term.get_hermitian_conjugate()
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
                self.n_qubits, p, q, r, s)

            # Get correct qubit operator.
            fermion_term = fermion_operators.FermionTerm(
                self.n_qubits, 1., [(p, 1), (q, 1), (r, 0), (s, 0)])
            correct_operator = fermion_term.jordan_wigner_transform()
            hermitian_conjugate = fermion_term.get_hermitian_conjugate()
            if fermion_term != hermitian_conjugate:
              correct_operator += hermitian_conjugate.jordan_wigner_transform()

            # Make sure its correct.
            self.assertEqual(test_operator, correct_operator)


# Test.
if __name__ == "__main__":
  unittest.main()
