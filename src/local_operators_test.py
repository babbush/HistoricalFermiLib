"""Tests for local_operators.py"""
import local_operators
import unittest


class LocalTermsTest(unittest.TestCase):

  def setUp(self):
    self.n_qubits = 5
    self.coefficient_a = 6.7j
    self.coefficient_b = -88.
    self.operators_a = [1, 2, 3, 4]
    self.operators_b = [1, 2]
    self.term_a = local_operators.LocalTerm(
        self.n_qubits, self.coefficient_a, self.operators_a)
    self.term_b = local_operators.LocalTerm(
        self.n_qubits, self.coefficient_b, self.operators_b)

  def test_init_local_term(self):
    self.assertEqual(len(self.term_a.operators), 4)
    self.assertEqual(self.term_a.operators, self.operators_a)
    self.assertEqual(self.term_a.coefficient, self.coefficient_a)

  def test_local_term_comparisons(self):
    self.assertTrue(self.term_a == self.term_a)
    self.assertFalse(self.term_a == self.term_b)
    self.assertTrue(self.term_a != self.term_b)
    self.assertFalse(self.term_a != self.term_a)

  def test_local_term_multiply(self):
    self.term_a.multiply_by_term(self.term_b)
    self.assertTrue(self.term_a.operators,
                    self.operators_a + self.operators_b)
    self.assertAlmostEqual(self.term_a.coefficient,
                           self.coefficient_a * self.coefficient_b)

  def test_local_term_key(self):
    self.assertEqual(self.term_a.key(), tuple(self.operators_a))


class LocalOperators(unittest.TestCase):

  def setUp(self):
    self.n_qubits = 5
    self.coefficient_a = 6.7j
    self.coefficient_b = -88.
    self.coefficient_c = 2.
    self.operators_a = [1, 2, 3, 4]
    self.operators_b = [1, 2]
    self.operators_c = [0, 3, 4]
    self.term_a = local_operators.LocalTerm(
        self.n_qubits, self.coefficient_a, self.operators_a)
    self.term_b = local_operators.LocalTerm(
        self.n_qubits, self.coefficient_b, self.operators_b)
    self.term_c = local_operators.LocalTerm(
        self.n_qubits, self.coefficient_c, self.operators_c)
    self.local_operator_a = local_operators.LocalOperator(
        self.n_qubits, [self.term_a])
    self.local_operator_bc = local_operators.LocalOperator(
        self.n_qubits, [self.term_b, self.term_c])
    self.local_operator_abc = local_operators.LocalOperator(
        self.n_qubits, [self.term_a, self.term_b, self.term_c])

  def test_init_local_operators(self):
    pass


if __name__ == '__main__':
  unittest.main()
