"""Tests for local_operators.py"""
import local_operators
import local_terms
import unittest


class LocalOperatorsTest(unittest.TestCase):

  def setUp(self):
    self.n_qubits = 5
    self.coefficient_a = 6.7j
    self.coefficient_b = -88.
    self.coefficient_c = 2.
    self.operators_a = [1, 2, 3, 4]
    self.operators_b = [1, 2]
    self.operators_c = [0, 3, 4]
    self.term_a = local_terms.LocalTerm(
        self.n_qubits, self.coefficient_a, self.operators_a)
    self.term_b = local_terms.LocalTerm(
        self.n_qubits, self.coefficient_b, self.operators_b)
    self.term_c = local_terms.LocalTerm(
        self.n_qubits, self.coefficient_c, self.operators_c)
    self.operator_a = local_operators.LocalOperator(
        self.n_qubits, [self.term_a])
    self.operator_bc = local_operators.LocalOperator(
        self.n_qubits, [self.term_b, self.term_c])
    self.operator_abc = local_operators.LocalOperator(
        self.n_qubits, [self.term_a, self.term_b, self.term_c])

  def test_init(self):
    self.assertEqual(
        len(self.operator_a.terms), len(self.operator_a))
    with self.assertRaises(local_operators.LocalOperatorError):
      self.operator_a.n_qubits = 2

  def test_comparisons(self):
    self.assertTrue(self.operator_a == self.operator_a)
    self.assertFalse(self.operator_a == self.operator_bc)
    self.assertTrue(self.operator_a != self.operator_bc)
    self.assertFalse(self.operator_a != self.operator_a)

  def test_addition(self):
    new_term = self.operator_a + self.operator_bc
    self.assertTrue(new_term == self.operator_abc)

    new_term -= self.operator_a
    self.assertTrue(new_term == self.operator_bc)

    new_term += self.operator_a
    self.assertTrue(new_term == self.operator_abc)

    new_term = self.operator_abc + self.operator_abc + self.operator_abc
    for term in new_term:
      self.assertAlmostEqual(term.coefficient,
                             3. * self.operator_abc[term.operators])

    new_term = self.operator_abc + self.operator_abc - self.operator_abc
    self.assertEqual(self.operator_abc, new_term)

    self.assertTrue(self.operator_a + self.term_a ==
                    self.term_a + self.operator_a)

    self.assertTrue(self.operator_a - self.term_a ==
                    self.term_a - self.operator_a)

  def test_multiplication(self):
    new_operator = self.operator_abc * self.operator_abc
    new_term = self.term_a * self.term_a
    self.assertTrue(new_term.coefficient ==
                    new_operator[new_term.operators])

    self.assertTrue(3.2 * new_operator == new_operator * 3.2)

    self.operator_abc *= self.operator_abc
    self.assertTrue(new_term.coefficient ==
                    self.operator_abc[new_term.operators])

  def test_abs(self):
    new_operator = abs(self.operator_abc)
    for term in new_operator:
      self.assertTrue(term.coefficient > 0.)


if __name__ == '__main__':
  unittest.main()
