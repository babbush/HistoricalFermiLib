"""Tests for local_terms.py"""
from __future__ import absolute_import

import unittest

import numpy

from fermilib import local_terms
from fermilib import local_operators


class LocalTermsTest(unittest.TestCase):

  def setUp(self):
    self.coefficient_a = -2.
    self.coefficient_b = 8.
    self.operators_a = [0, 1, 2, 3, 4]
    self.operators_b = [1, 2]
    self.term_a = local_terms.LocalTerm(self.operators_a, self.coefficient_a)
    self.term_b = local_terms.LocalTerm(self.operators_b, self.coefficient_b)

    self.identity = local_terms.LocalTerm([], 1.0)

  def test_init(self):
    self.assertEqual(self.term_a.operators, self.operators_a)
    self.assertEqual(self.term_a.coefficient, self.coefficient_a)

  def test_init_list_protection(self):
    arr = []
    self.term1 = local_terms.LocalTerm(operators=[], coefficient=1)
    arr.append(3)
    self.assertEqual(self.term1.operators, [])

  def test_eq(self):
    self.assertTrue(self.term_a == self.term_a)
    self.assertFalse(self.term_a == self.term_b)

  def test_eq_within_tol_same_ops(self):
    self.term1 = local_terms.LocalTerm(operators=[], coefficient=1)
    self.term2 = local_terms.LocalTerm([], coefficient=(1+9e-13))
    self.assertEqual(self.term1, self.term2)

  def test_eq_within_tol_diff_ops(self):
    self.term1 = local_terms.LocalTerm(operators=[1], coefficient=9e-13)
    self.term2 = local_terms.LocalTerm(operators=[2], coefficient=7e-13)
    self.assertEqual(self.term1, self.term2)

  def test_neq(self):
    self.assertTrue(self.term_a != self.term_b)
    self.assertFalse(self.term_a != self.term_a)

  def test_slicing(self):
    for i in range(len(self.term_a)):
      self.assertEqual(self.term_a[i], i)

  def test_slicing_set(self):
    for i in range(len(self.term_a)):
      self.term_a[i] += 1
    for i in range(len(self.term_a)):
      self.assertEqual(self.term_a[i], i + 1)

  def test_set_not_in(self):
    term1 = local_terms.LocalTerm(operators=[1], coefficient=1)
    with self.assertRaises(local_terms.LocalTermError):
      term1[2] = 2

  def test_get_not_in(self):
    with self.assertRaises(local_terms.LocalTermError):
      self.term_a[11]

  def test_del_not_in(self):
    term1 = local_terms.LocalTerm(operators=range(10), coefficient=1)
    with self.assertRaises(local_terms.LocalTermError):
      del term1[10]

  def test_slicing_del(self):
    term1 = local_terms.LocalTerm(operators=range(10), coefficient=1)
    del term1[3:6]
    self.assertEqual(term1.operators, [0, 1, 2, 6, 7, 8, 9])

  def test_add_localterms(self):
    self.assertEqual(self.term_a + self.term_b,
                     local_operators.LocalOperator([self.term_a, self.term_b]))

  def test_add_localterms_reverse(self):
    self.assertEqual(self.term_b + self.term_a,
                     local_operators.LocalOperator([self.term_a, self.term_b]))

  def test_add_localterms_error(self):
    with self.assertRaises(TypeError):
      self.term_a + 1

  def test_sub(self):
    self.assertEqual(self.term_a - self.term_b,
                     local_operators.LocalOperator(
                         [self.term_a, -self.term_b]))

  def test_sub_cancel(self):
    self.assertEqual(self.term_a - self.term_a,
                     local_operators.LocalOperator([]))

  def test_neg(self):
    expected = local_terms.LocalTerm(self.operators_a, -self.coefficient_a)
    self.assertEqual(-self.term_a, expected)

  def test_rmul(self):
    self.term = self.term_a * -3.
    expected = local_terms.LocalTerm(self.operators_a,
                                     self.coefficient_a * -3.)
    self.assertEqual(self.term, expected)

  def test_lmul_rmul(self):
    new_term = 7. * self.term_a
    self.assertEqual(3. * new_term, new_term * 3.)
    self.assertEqual(7. * self.term_a.coefficient, new_term.coefficient)
    self.assertEqual(self.term_a.operators, new_term.operators)

  def test_imul_scalar(self):
    self.term = self.term_a * 1
    self.term *= -3.+2j
    expected = local_terms.LocalTerm(self.operators_a,
                                     self.coefficient_a * (-3.+2j))
    self.assertEqual(self.term, expected)

  def test_imul_localterm(self):
    expected_coeff = self.term_a.coefficient * self.term_b.coefficient
    expected_ops = self.term_a.operators + self.term_b.operators
    expected = local_terms.LocalTerm(expected_ops, expected_coeff)

    self.term_a *= self.term_b
    self.assertEqual(self.term_a, expected)

  def test_mul_by_scalarzero(self):
    term1 = self.term_a * 0
    expected = local_terms.LocalTerm(self.term_a.operators, 0)
    self.assertEqual(term1, expected)

  def test_mul_by_localtermzero(self):
    term0 = local_terms.LocalTerm([], 0)
    term0d = self.term_a * term0
    self.assertEqual(term0d, term0)

  def test_mul_by_self(self):
    new_term = self.term_a * self.term_a
    self.assertEqual(self.term_a.coefficient ** 2.,
                     new_term.coefficient)
    self.assertEqual(2 * self.term_a.operators, new_term.operators)

  def test_lmul_identity(self):
    self.assertEqual(self.term_b, self.identity * self.term_b)

  def test_rmul_identity(self):
    self.assertEqual(self.term_b, self.term_b * self.identity)

  def test_mul_by_multiple_of_identity(self):
    self.assertEqual(3.0 * self.term_a, (3.0 * self.identity) * self.term_a)

  def test_mul_triple(self):
    new_term = self.term_a * self.term_a * self.term_a
    self.assertEqual(self.term_a.coefficient ** 3.,
                     new_term.coefficient)
    self.assertEqual(3 * self.term_a.operators, new_term.operators)

  def test_mul_scalar(self):
    expected = local_terms.LocalTerm(self.operators_a,
                                     self.coefficient_a * (-3.+2j))
    self.assertEqual(self.term_a * (-3. + 2j), expected)

  def test_mul_npfloat64(self):
    self.assertEqual(self.term_b * numpy.float64(2.303),
                     self.term_b * 2.303)
    self.assertEqual(numpy.float64(2.303) * self.term_b,
                     self.term_b * 2.303)

  def test_mul_npfloat128(self):
    self.assertEqual(self.term_b * numpy.float128(2.303),
                     self.term_b * 2.303)
    self.assertEqual(numpy.float128(2.303) * self.term_b,
                     self.term_b * 2.303)

  def test_mul_scalar_commute(self):
    self.assertEqual(self.term_a * -3.7j, -3.7j * self.term_a)

  def test_mul_localterm(self):
    term_ab = self.term_a * self.term_b

    expected_coeff = self.term_a.coefficient * self.term_b.coefficient
    expected_ops = self.term_a.operators + self.term_b.operators

    expected = local_terms.LocalTerm(expected_ops, expected_coeff)
    self.assertEqual(term_ab, expected)

  def test_div(self):
    new_term = self.term_a / 3
    self.assertEqual(new_term.coefficient, self.term_a.coefficient / 3)
    self.assertEqual(new_term.operators, self.term_a.operators)

  def test_idiv(self):
    self.term_a /= 2
    self.assertEqual(self.term_a.coefficient, self.coefficient_a / 2)
    self.assertEqual(self.term_a.operators, self.operators_a)

  def test_pow_square(self):
    squared = self.term_a ** 2
    expected = local_terms.LocalTerm(self.operators_a + self.operators_a,
                                     self.coefficient_a ** 2)
    self.assertEqual(squared, self.term_a * self.term_a)
    self.assertEqual(squared, expected)

  def test_pow_zero(self):
    zerod = self.term_a ** 0
    expected = local_terms.LocalTerm()
    self.assertEqual(zerod, expected)

  def test_pow_one(self):
    self.assertEqual(self.term_a, self.term_a ** 1)

  def test_pow_neg_error(self):
    with self.assertRaises(ValueError):
      self.term_a ** -1

  def test_pow_nonint_error(self):
    with self.assertRaises(ValueError):
      self.term_a ** 0.5

  def test_pow_high(self):
    high = self.term_a ** 100
    expected = local_terms.LocalTerm(self.operators_a * 100,
                                     self.term_a.coefficient ** 100)
    self.assertEqual(high, expected)

  def test_iter(self):
    for i, operator in enumerate(self.term_a):
      self.assertEqual(self.term_a.operators[i], operator)

  def test_abs(self):
    abs_term_a = abs(self.term_a)
    self.assertEqual(abs(self.term_a.coefficient),
                     abs_term_a.coefficient)

  def test_abs_complex(self):
    term1 = local_terms.LocalTerm([], 2. + 3j)
    self.assertEqual(abs(term1).coefficient, abs(term1.coefficient))

  def test_len(self):
    self.assertEqual(len(self.term_a), 5)
    self.assertEqual(len(self.term_b), 2)

  def test_str(self):
    self.assertEqual(str(self.term_a), "-2.0 [0, 1, 2, 3, 4]")
    self.assertEqual(str(self.term_b), "8.0 [1, 2]")

  def test_str_complex(self):
    term1 = local_terms.LocalTerm([], -2. - 3j)
    self.assertEqual(str(term1), str(-2. - 3j) + ' ' + str([]))

if __name__ == '__main__':
  unittest.main()
