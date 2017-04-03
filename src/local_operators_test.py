"""Tests for local_operators.py"""
import local_operators
import local_terms
import unittest
import numpy


class LocalOperatorsTest(unittest.TestCase):

  def setUp(self):
    self.coefficient_a = 6.7j
    self.coefficient_b = -88.
    self.coefficient_c = 2.

    self.operators_a = [1, 2, 3, 4]
    self.operators_b = [1, 2]
    self.operators_c = [0, 3, 4]

    self.term_a = local_terms.LocalTerm(self.operators_a, self.coefficient_a)
    self.term_b = local_terms.LocalTerm(self.operators_b, self.coefficient_b)
    self.term_c = local_terms.LocalTerm(self.operators_c, self.coefficient_c)

    self.operator_a = local_operators.LocalOperator(self.term_a)
    self.operator_bc = local_operators.LocalOperator([self.term_b,
                                                      self.term_c])
    self.operator_abc = local_operators.LocalOperator([self.term_a,
                                                       self.term_b,
                                                       self.term_c])

  def test_init_list(self):
    self.assertEqual(self.coefficient_a,
                     self.operator_a[tuple(self.operators_a)])
    self.assertEqual(self.term_a, list(self.operator_a.terms.values())[0])
    self.assertEqual(self.coefficient_b,
                     self.operator_abc[self.operators_b])
    self.assertEqual(0.0, self.operator_abc[(1, 2, 9)])
    self.assertEqual(len(self.operator_a), 1)
    self.assertEqual(len(self.operator_abc), 3)

  def test_init_dict(self):
    d = {}
    d[(1, 2, 3, 4)] = self.term_a
    d[(0, 3, 4)] = self.term_c
    op_ac = local_operators.LocalOperator(d)
    self.assertEqual(len(op_ac), 2)
    self.assertEqual(self.coefficient_a,
                     op_ac[tuple(self.operators_a)])
    self.assertEqual(self.coefficient_c,
                     op_ac[tuple(self.operators_c)])
    self.assertEqual(0.0,
                     op_ac[tuple(self.operators_b)])

  def test_init_localterm(self):
    self.assertEqual(self.operator_a,
                     local_operators.LocalOperator(self.term_a))

  def test_init_badterm(self):
    with self.assertRaises(TypeError):
      local_operators.LocalOperator(1)

  def test_init_list_protection(self):
    self.coeff1 = 2.j-3
    self.operators1 = [6, 7, 8, 11]
    self.term1 = local_terms.LocalTerm(self.operators1, self.coeff1)

    self.operator1 = local_operators.LocalOperator([self.term1])
    self.operators1.append(12)

    expected_term = local_terms.LocalTerm(self.operators1[:-1], self.coeff1,)
    expected_op = local_operators.LocalOperator(expected_term)
    self.assertEqual(self.operator1, expected_op)

  def test_init_dict_protection(self):
    d = {}
    d[(1, 2, 3, 4)] = self.term_a
    d[(0, 3, 4)] = self.term_c
    op_ac = local_operators.LocalOperator(d)
    self.assertEqual(len(op_ac), 2)

    # add a new element to the old dictionary
    d[tuple(self.operators_b)] = self.term_b

    self.assertEqual(self.coefficient_a,
                     op_ac[tuple(self.operators_a)])
    self.assertEqual(self.coefficient_c,
                     op_ac[tuple(self.operators_c)])
    self.assertEqual(0.0, op_ac[tuple(self.operators_b)])

  def test_eq(self):
    self.assertTrue(self.operator_a == self.operator_a)
    self.assertFalse(self.operator_a == self.operator_bc)

  def test_neq(self):
    self.assertTrue(self.operator_a != self.operator_bc)
    self.assertFalse(self.operator_a != self.operator_a)

  def test_add(self):
    new_term = self.operator_a + self.operator_bc
    self.assertEqual(new_term, self.operator_abc)

  def test_iadd(self):
    self.operator_bc += self.operator_a
    self.assertEqual(self.operator_bc, self.operator_abc)

  def test_add3(self):
    new_term = self.operator_abc + self.operator_abc + self.operator_abc
    for term in new_term:
      self.assertEqual(term.coefficient,
                       3. * self.operator_abc[term.operators])

  def test_isub(self):
    self.operator_abc -= self.operator_a
    self.assertEqual(self.operator_abc, self.operator_bc)

  def test_sub_cancel(self):
    new_term = self.operator_abc - self.operator_abc
    expected = local_operators.LocalOperator([])
    self.assertEqual(expected, new_term)

  def test_add_localterm(self):
    self.assertEqual(self.operator_a + self.term_a,
                     self.term_a + self.operator_a)

  def test_sub_localterm_cancel(self):
    self.assertEqual(self.operator_a - self.term_a,
                     self.term_a - self.operator_a)
    expected = local_operators.LocalOperator([])
    self.assertEqual(self.operator_a - self.term_a, expected)

  def test_neg(self):
    term = local_terms.LocalTerm(self.operators_a, -self.coefficient_a)
    expected = local_operators.LocalOperator(term)
    self.assertEqual(-self.operator_a, expected)

  def test_mul(self):
    new_operator = self.operator_abc * self.operator_abc
    new_a_term = self.term_a * self.term_a
    new_b_term = self.term_b * self.term_b
    new_c_term = self.term_c * self.term_c
    self.assertEqual(self.coefficient_a ** 2,
                     new_operator[(self.term_a * self.term_a).operators])
    self.assertEqual(self.coefficient_a * self.coefficient_b,
                     new_operator[(self.term_a * self.term_b).operators])
    self.assertEqual(self.coefficient_a * self.coefficient_b,
                     new_operator[(self.term_b * self.term_a).operators])

  def test_mul_by_zero_localterm(self):
    zero_term = local_terms.LocalTerm([1], 0.0)
    zero_op = local_operators.LocalOperator(zero_term)
    self.assertEqual(self.operator_abc * zero_term, zero_op)

  def test_mul_by_zero_op(self):
    zero_term = local_terms.LocalTerm([1], 0.0)
    zero_op = local_operators.LocalOperator(zero_term)
    self.assertEqual(self.operator_abc * zero_op, zero_op)

  def test_mul_by_identity_term(self):
    identity_term = local_terms.LocalTerm()
    self.assertEqual(self.operator_abc * identity_term, self.operator_abc)

  def test_mul_by_identity_op(self):
    identity_term = local_terms.LocalTerm()
    identity_op = local_operators.LocalOperator(identity_term)
    self.assertEqual(self.operator_abc * identity_op, self.operator_abc)

  def test_mul_npfloat64(self):
    self.assertEqual(self.operator_abc * numpy.float64(2.303),
                     self.operator_abc * 2.303)
    self.assertEqual(numpy.float64(2.303) * self.operator_abc,
                     self.operator_abc * 2.303)

  def test_mul_npfloat128(self):
    self.assertEqual(self.operator_abc * numpy.float128(2.303),
                     self.operator_abc * 2.303)
    self.assertEqual(numpy.float128(2.303) * self.operator_abc,
                     self.operator_abc * 2.303)

  def test_mul_scalar_commute(self):
    self.assertEqual(3.2 * self.operator_abc, self.operator_abc * 3.2)

  def test_imul_localterm(self):
    self.operator_abc *= self.term_a
    self.assertEqual(self.operator_abc[(self.term_a * self.term_a).operators],
                     self.coefficient_a ** 2)
    self.assertEqual(self.operator_abc[(self.term_a * self.term_b).operators],
                     0.0)
    self.assertEqual(self.operator_abc[(self.term_b * self.term_a).operators],
                     self.coefficient_a * self.coefficient_b)
    self.assertEqual(self.operator_abc[(self.term_c * self.term_a).operators],
                     self.coefficient_a * self.coefficient_c)
    self.assertEqual(self.operator_abc[self.operators_a], 0.0)
    self.assertEqual(self.operator_abc[self.operators_b], 0.0)

  def test_imul_scalar(self):
    self.operator_a *= 3
    self.assertEqual(self.operator_a[self.operators_a], 3 * self.coefficient_a)

  def test_imul_op(self):
    new_term = self.term_a * self.term_a
    self.operator_abc *= self.operator_abc
    self.assertEqual((self.term_a * self.term_a).coefficient,
                     self.operator_abc[new_term.operators])

  def test_div(self):
    new_op = self.operator_bc / 3
    self.assertEqual(new_op, self.operator_bc * (1.0 / 3.0))

  def test_idiv(self):
    self.operator_bc /= 2
    self.assertEqual(self.operator_bc[self.term_b], self.coefficient_b / 2)
    self.assertEqual(self.operator_bc[self.term_c], self.coefficient_c / 2)

  def test_abs(self):
    new_operator = abs(self.operator_abc)
    for term in new_operator:
      self.assertEqual(term.coefficient, abs(self.operator_abc[term]))

  def test_len(self):
    self.assertEqual(len(self.operator_a.terms), len(self.operator_a))
    self.assertEqual(len(self.operator_a), 1)
    self.assertEqual(len(self.operator_abc), 3)

  def test_len_add_same(self):
    self.assertEqual(len(self.operator_abc + self.operator_abc), 3)

  def test_len_cancel(self):
    self.assertEqual(len(self.operator_bc), 2)
    self.assertEqual(len(self.operator_bc - self.operator_bc), 0)

  def test_contains_true(self):
    self.assertTrue(self.operators_a in self.operator_abc)
    self.assertTrue(self.operators_b in self.operator_abc)

  def test_contains_false(self):
    self.assertFalse(self.operators_a in self.operator_bc)

  def test_pow_sq(self):
    self.assertEqual(self.operator_abc ** 2,
                     self.operator_abc * self.operator_abc)

  def test_pow_zero(self):
    identity_term = local_terms.LocalTerm()
    identity_op = local_operators.LocalOperator(identity_term)
    self.assertEqual(self.operator_abc ** 0, identity_op)

  def test_str(self):
    self.assertEqual(str(self.operator_abc),
                     "-88.0 [1, 2]\n2.0 [0, 3, 4]\n6.7j [1, 2, 3, 4]\n")

  def test_str_zero(self):
    self.assertEqual('0', str(local_operators.LocalOperator()))

  def test_contains(self):
    self.assertFalse((1, 2, 9) in self.operator_abc)
    self.assertTrue(self.operators_a in self.operator_abc)
    self.assertTrue(self.operators_b in self.operator_abc)
    self.assertTrue(self.operators_c in self.operator_abc)

  def test_get(self):
    self.assertEqual(self.coefficient_a, self.operator_abc[self.operators_a])
    self.assertEqual(self.coefficient_b, self.operator_abc[self.operators_b])
    self.assertEqual(self.coefficient_c, self.operator_abc[self.operators_c])
    self.assertEqual(0.0, self.operator_abc[(1, 2, 9)])

  def test_set(self):
    self.operator_abc[self.operators_a] = 2.37
    result = self.operator_abc.terms[tuple(self.operators_a)].coefficient
    self.assertEqual(result, 2.37)
    self.assertEqual(self.operator_abc[self.operators_a], 2.37)

  def test_set_new(self):
    self.operator_bc[self.operators_a] = self.coefficient_a
    self.assertEqual(self.operator_bc, self.operator_abc)

  def test_del(self):
    del self.operator_abc[self.operators_a]
    self.assertEqual(0.0, self.operator_abc[self.operators_a])
    self.assertEqual(len(self.operator_abc), 2)

  def test_del_not_in_term(self):
    with self.assertRaises(local_operators.LocalOperatorError):
      del self.operator_a[self.operators_b]

  def test_list_coeffs(self):
    orig_set = set([self.coefficient_a, self.coefficient_b,
                    self.coefficient_c])
    true_set = set(self.operator_abc.list_coefficients())
    self.assertEqual(orig_set, true_set)

  def test_list_ops(self):
    expected = [self.term_a, self.term_b, self.term_c]
    actual = self.operator_abc.list_terms()

    for term in expected:
      self.assertTrue(term in actual)
    for term in actual:
      self.assertTrue(term in expected)
      self.assertEqual(self.operator_abc[term], term.coefficient)

if __name__ == '__main__':
  unittest.main()
