"""Tests for local_terms.py"""
import local_terms
import unittest


class LocalTermsTest(unittest.TestCase):

  def setUp(self):
    self.n_qubits = 5
    self.coefficient_a = -2.
    self.coefficient_b = 8.
    self.operators_a = [0, 1, 2, 3, 4]
    self.operators_b = [1, 2]
    self.term_a = local_terms.LocalTerm(
        self.n_qubits, self.coefficient_a, self.operators_a)
    self.term_b = local_terms.LocalTerm(
        self.n_qubits, self.coefficient_b, self.operators_b)

  def test_init(self):
    self.assertEqual(len(self.term_a), 5)
    self.assertEqual(self.term_a.operators, self.operators_a)
    self.assertAlmostEqual(self.term_a.coefficient, self.coefficient_a)
    with self.assertRaises(local_terms.ErrorLocalTerm):
      self.term_a.n_qubits = 2

  def test_comparisons(self):
    self.assertTrue(self.term_a == self.term_a)
    self.assertFalse(self.term_a == self.term_b)
    self.assertTrue(self.term_a != self.term_b)
    self.assertFalse(self.term_a != self.term_a)

  def test_slicing(self):
    for i in range(len(self.term_a)):
      self.assertEqual(self.term_a[i], i)
      self.term_a[i] += 1
    for i in range(len(self.term_a)):
      self.assertEqual(self.term_a[i], i + 1)

  def test_multiplication(self):
    new_term = 7. * self.term_a
    self.assertTrue(3. * new_term == new_term * 3.)
    self.assertAlmostEqual(7. * self.term_a.coefficient, new_term.coefficient)
    self.assertEqual(self.term_a.operators, new_term.operators)

    self.term_a *= 7.
    self.assertTrue(self.term_a == new_term)

    new_term = self.term_a * self.term_a
    self.assertAlmostEqual(self.term_a.coefficient ** 2.,
                           new_term.coefficient)
    self.assertEqual(2 * self.term_a.operators, new_term.operators)

    new_term = self.term_a * self.term_a * self.term_a
    self.assertAlmostEqual(self.term_a.coefficient ** 3.,
                           new_term.coefficient)
    self.assertEqual(3 * self.term_a.operators, new_term.operators)

  def test_iter(self):
    for i, operator in enumerate(self.term_a):
      self.assertEqual(self.term_a.operators[i], operator)

  def test_abs(self):
    abs_term_a = abs(self.term_a)
    self.assertAlmostEqual(abs(self.term_a.coefficient),
                           abs_term_a.coefficient)

  def test_pow(self):
    squared = self.term_a ** 2
    self.assertEqual(squared, self.term_a * self.term_a)


if __name__ == '__main__':
  unittest.main()
