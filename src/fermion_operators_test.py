"""Tests for fermion_operators.py"""
from fermion_operators import (fermion_identity, number_operator,
                               FermionTerm, FermionOperator,
                               FermionTermError, FermionOperatorError,
                               JordanWignerError)
from qubit_operators import qubit_identity, QubitTerm, QubitOperator
import unittest
import numpy
import local_terms
import local_operators


class NumberOperatorsTest(unittest.TestCase):

  def setUp(self):
    self.n_qubits = 5

  def test_number_operator(self):
    n3 = number_operator(self.n_qubits, 3)
    self.assertEqual(n3, FermionTerm(self.n_qubits, 1., [(3, 1), (3, 0)]))

  def test_number_operator_total(self):
    total_n = number_operator(self.n_qubits)
    self.assertEqual(len(total_n.terms), self.n_qubits)
    for qubit in range(self.n_qubits):
      operators = [(qubit, 1), (qubit, 0)]
      self.assertEqual(total_n[operators], 1.)

  def test_number_operator_above_range(self):
    with self.assertRaises(ValueError):
      n5 = number_operator(self.n_qubits, 5)

  def test_number_operator_below_range(self):
    with self.assertRaises(ValueError):
      n_m1 = number_operator(self.n_qubits, -1)


class FermionTermsTest(unittest.TestCase):

  def setUp(self):
    self.n_qubits = 5
    self.coefficient_a = 6.7j
    self.coefficient_b = -88.
    self.operators_a = [(3, 1), (1, 0), (4, 1)]
    self.operators_b = [(2, 0), (4, 0), (2, 1)]
    self.term_a = FermionTerm(self.n_qubits, self.coefficient_a,
                              self.operators_a)
    self.term_b = FermionTerm(self.n_qubits, self.coefficient_b,
                              self.operators_b)
    self.normal_ordered_a = FermionTerm(self.n_qubits, self.coefficient_a,
                                        [(4, 1), (3, 1), (1, 0)])
    self.normal_ordered_b1 = FermionTerm(self.n_qubits, -self.coefficient_b,
                                         [(4, 0)])
    self.normal_ordered_b2 = FermionTerm(self.n_qubits, -self.coefficient_b,
                                         [(2, 1), (4, 0), (2, 0)])

    self.operator_a = FermionOperator(self.n_qubits, self.term_a)
    self.operator_b = FermionOperator(self.n_qubits, self.term_b)
    self.operator_ab = FermionOperator(self.n_qubits,
                                       [self.term_a, self.term_b])

    self.operator_a_normal = FermionOperator(self.n_qubits,
                                             self.normal_ordered_a)
    self.operator_b_normal = FermionOperator(self.n_qubits,
                                             [self.normal_ordered_b1,
                                              self.normal_ordered_b2])

  def test_init(self):
    self.assertEqual(self.term_a.n_qubits, self.n_qubits)
    self.assertEqual(self.term_a.coefficient, self.coefficient_a)
    self.assertEqual(self.term_a.operators, self.operators_a)

    self.assertEqual(self.term_b.n_qubits, self.n_qubits)
    self.assertEqual(self.term_b.coefficient, self.coefficient_b)
    self.assertEqual(self.term_b.operators, self.operators_b)

  def test_init_bad(self):
    with self.assertRaises(ValueError):
      term = FermionTerm(4, 1, [(1, 2)])

  def test_change_nqubits_error(self):
    with self.assertRaises(local_terms.LocalTermError):
      self.term_a.n_qubits = 2

  def test_add_fermionterm(self):
    self.assertEqual(self.term_a + self.term_b, self.operator_ab)

  def test_sub_fermionterm(self):
    expected = FermionOperator(self.n_qubits, [self.term_a, -self.term_b])
    self.assertEqual(self.term_a - self.term_b, expected)

  def test_sub_cancel(self):
    expected = FermionOperator(self.n_qubits)
    self.assertEqual(self.term_b - self.term_b, expected)

  def test_sub_fermionop(self):
    expected = FermionOperator(self.n_qubits, -self.term_b)
    self.assertEqual(self.term_a - self.operator_ab, expected)

  def test_eq(self):
    self.assertTrue(self.term_a == self.term_a)
    self.assertFalse(self.term_a == self.term_b)

  def test_eq_within_tol_same_ops(self):
    self.term1 = FermionTerm(2, 1, [(1, 1)])
    self.term2 = FermionTerm(2, (1+9e-13), [(1, 1)])
    self.assertEqual(self.term1, self.term2)

  def test_eq_within_tol_diff_ops(self):
    self.term1 = FermionTerm(2, 9e-13, [(1, 1)])
    self.term2 = FermionTerm(2, 7e-13, [(1, 0)])
    self.assertEqual(self.term1, self.term2)

  def test_neq(self):
    self.assertTrue(self.term_a != self.term_b)
    self.assertFalse(self.term_a != self.term_a)

  def test_neq_outside_tol(self):
    self.term1 = FermionTerm(2, 1, [(1, 1)])
    self.term2 = FermionTerm(2, (1+9e-12), [(1, 1)])
    self.assertNotEqual(self.term1, self.term2)
    self.assertFalse(self.term1 == self.term2)

  def test_eq_different_nqubits_error(self):
    self.term1 = FermionTerm(1, coefficient=1)
    self.term2 = FermionTerm(2, coefficient=1)
    with self.assertRaises(local_terms.LocalTermError):
      self.term1 == self.term2

  def test_slicing(self):
    self.assertEqual(self.term_a[0], (3, 1))
    self.assertEqual(self.term_a[1], (1, 0))
    self.assertEqual(self.term_a[2], (4, 1))

  def test_slicing_set(self):
    for i in range(len(self.term_a)):
      self.term_a[i] = (self.term_a[i][0], 1 - self.term_a[i][1])

    self.assertEqual(self.term_a[0], (3, 0))
    self.assertEqual(self.term_a[1], (1, 1))
    self.assertEqual(self.term_a[2], (4, 0))

  def test_set_not_in(self):
    term1 = FermionTerm(5, coefficient=1, operators=[(1, 1)])
    with self.assertRaises(local_terms.LocalTermError):
      term1[2] = [(2, 1)]

  def test_get_not_in(self):
    with self.assertRaises(local_terms.LocalTermError):
      self.term_a[11]

  def test_del_not_in(self):
    term1 = FermionTerm(5, coefficient=1, operators=[(1, 1)])
    with self.assertRaises(local_terms.LocalTermError):
      del term1[10]

  def test_slicing_del(self):
    term1 = FermionTerm(10, 1, [(i, 1) for i in range(10)])
    del term1[3:6]
    self.assertEqual(term1.operators,
                     ([(i, 1) for i in range(3)] +
                      [(i, 1) for i in range(6, 10)]))

  def test_add_fermionterms(self):
    self.assertEqual(self.term_a + self.term_b,
                     FermionOperator(self.n_qubits,
                                     [self.term_a, self.term_b]))

  def test_add_localterms_reverse(self):
    self.assertEqual(self.term_b + self.term_a,
                     FermionOperator(self.n_qubits,
                                     [self.term_a, self.term_b]))

  def test_add_localterms_error(self):
    with self.assertRaises(TypeError):
      self.term_a + 1

  def test_add_different_nqubits_error(self):
    self.term1 = FermionTerm(5, 1)
    self.term2 = FermionTerm(2, 1)
    with self.assertRaises(local_terms.LocalTermError):
      self.term1 + self.term2

  def test_add_terms(self):
    sum_terms = self.term_a + self.term_b
    diff_terms = self.term_a - self.term_b
    self.assertEqual(2. * self.term_a + self.term_b - self.term_b,
                     sum_terms + diff_terms)
    self.assertIsInstance(sum_terms, FermionOperator)
    self.assertIsInstance(diff_terms, FermionOperator)

  def test_iadd(self):
    self.term_a += self.term_b
    self.assertEqual(self.term_a, self.operator_ab)

  def test_sub(self):
    self.assertEqual(self.term_a - self.term_b,
                     FermionOperator(self.n_qubits,
                                     [self.term_a, -self.term_b]))

  def test_sub_cancel(self):
    self.assertEqual(self.term_a - self.term_a,
                     FermionOperator(self.n_qubits))

  def test_neg(self):
    expected = FermionTerm(self.n_qubits, -self.coefficient_a,
                           self.operators_a)
    self.assertEqual(-self.term_a, expected)

  def test_rmul(self):
    term = self.term_a * -3.
    expected = FermionTerm(self.n_qubits, self.coefficient_a * -3.,
                           self.operators_a)
    self.assertEqual(term, expected)

  def test_lmul_rmul(self):
    term = 7. * self.term_a
    self.assertEqual(3. * term, term * 3.)
    self.assertEqual(7. * self.term_a.coefficient, term.coefficient)
    self.assertEqual(self.term_a.operators, term.operators)

  def test_imul_scalar(self):
    term = self.term_a * 1
    term *= -3.+2j
    expected = FermionTerm(self.n_qubits, self.coefficient_a * (-3.+2j),
                           self.operators_a)
    self.assertEqual(term, expected)

  def test_imul_localterm(self):
    expected_coeff = self.term_a.coefficient * self.term_b.coefficient
    expected_ops = self.term_a.operators + self.term_b.operators
    expected = FermionTerm(self.n_qubits, expected_coeff, expected_ops)

    self.term_a *= self.term_b
    self.assertEqual(self.term_a, expected)

  def test_mul_by_scalarzero(self):
    term1 = self.term_a * 0
    expected = FermionTerm(self.n_qubits, 0, self.term_a.operators)
    self.assertEqual(term1, expected)

  def test_mul_by_fermiontermzero(self):
    term0 = FermionTerm(self.n_qubits, 0)
    term0d = self.term_a * term0
    self.assertEqual(term0d, term0)

  def test_mul_by_self(self):
    new_term = self.term_a * self.term_a
    self.assertEqual(self.term_a.coefficient ** 2.,
                     new_term.coefficient)
    self.assertEqual(2 * self.term_a.operators, new_term.operators)

  def test_lmul_identity(self):
    self.assertEqual(self.term_b,
                     fermion_identity(self.n_qubits) * self.term_b)

  def test_rmul_identity(self):
    self.assertEqual(self.term_b,
                     self.term_b * fermion_identity(self.n_qubits))

  def test_mul_by_multiple_of_identity(self):
    self.assertEqual(3.0 * self.term_a,
                     (3.0 * fermion_identity(self.n_qubits)) * self.term_a)

  def test_mul_triple(self):
    new_term = self.term_a * self.term_a * self.term_a
    self.assertEqual(self.term_a.coefficient ** 3.,
                     new_term.coefficient)
    self.assertEqual(3 * self.term_a.operators, new_term.operators)

  def test_mul_scalar(self):
    expected = local_terms.LocalTerm(self.n_qubits,
                                     self.coefficient_a * (-3.+2j),
                                     self.operators_a)
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
    self.assertEqual(3.2 * self.term_a, self.term_a * 3.2)

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
    expected = FermionTerm(self.n_qubits,
                           self.coefficient_a ** 2,
                           self.operators_a + self.operators_a)
    self.assertEqual(squared, self.term_a * self.term_a)
    self.assertEqual(squared, expected)

  def test_pow_zero(self):
    zerod = self.term_a ** 0
    expected = FermionTerm(self.n_qubits, 1.0)
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
    high = self.term_a ** 10
    expected = FermionTerm(self.n_qubits, self.term_a.coefficient ** 10,
                           self.operators_a * 10)
    self.assertEqual(high.n_qubits, expected.n_qubits)
    self.assertAlmostEqual(expected.coefficient, high.coefficient)
    self.assertEqual(high.operators, expected.operators)

  def test_abs_complex(self):
    term = FermionTerm(3, 2. + 3j, [(0, 1), (1, 0)])
    self.assertEqual(abs(term).coefficient, abs(term.coefficient))

  def test_len(self):
    term = FermionTerm(3, 2. + 3j, [(0, 0), (1, 1)])
    self.assertEqual(len(term), 2)
    self.assertEqual(len(self.term_a), 3)
    self.assertEqual(len(self.term_b), 3)

  def test_str(self):
    self.assertEqual(str(self.term_a), '6.7j [3+ 1 4+]')

  def test_str_number_site(self):
    self.assertEqual(str(number_operator(self.n_qubits, 1)), '1.0 [1+ 1]')

  def test_str_fermion_identity(self):
    self.assertEqual(str(fermion_identity(self.n_qubits)), '1.0 []')

  def test_hermitian_conjugated(self):
    self.term_a.hermitian_conjugate()
    self.assertEqual(self.term_a.operators[0], (4, 0))
    self.assertEqual(self.term_a.operators[1], (1, 1))
    self.assertEqual(self.term_a.operators[2], (3, 0))
    self.assertEqual(-6.7j, self.term_a.coefficient)

  def test_hermitian_conjugate_fermion_identity(self):
    result = fermion_identity(self.n_qubits)
    result.hermitian_conjugate()
    self.assertEqual(fermion_identity(self.n_qubits), result)

  def test_hermitian_conjugate_number_site(self):
    term = number_operator(self.n_qubits, 1)
    term.hermitian_conjugate()
    self.assertEqual(term, number_operator(self.n_qubits, 1))

  def test_hermitian_conjugated(self):
    hermitian_conjugate = self.term_a.hermitian_conjugated()
    self.assertEqual(hermitian_conjugate.operators[0], (4, 0))
    self.assertEqual(hermitian_conjugate.operators[1], (1, 1))
    self.assertEqual(hermitian_conjugate.operators[2], (3, 0))
    self.assertEqual(-6.7j, hermitian_conjugate.coefficient)

  def test_hermitian_conjugated_fermion_identity(self):
    result = fermion_identity(self.n_qubits).hermitian_conjugated()
    self.assertEqual(fermion_identity(self.n_qubits), result)

  def test_hermitian_conjugated_number_site(self):
    term = number_operator(self.n_qubits, 1)
    self.assertEqual(term, term.hermitian_conjugated())

  def test_is_normal_ordered(self):
    self.assertFalse(self.term_a.is_normal_ordered())
    self.assertFalse(self.term_b.is_normal_ordered())
    self.assertTrue(self.normal_ordered_a.is_normal_ordered())
    self.assertTrue(self.normal_ordered_b1.is_normal_ordered())
    self.assertTrue(self.normal_ordered_b2.is_normal_ordered())

  def test_normal_ordered_single_term(self):
    self.assertEqual(self.operator_a_normal, self.term_a.normal_ordered())

  def test_normal_ordered_two_term(self):
    normal_ordered_b = self.term_b.normal_ordered()
    self.assertEqual(2, len(normal_ordered_b.terms))
    self.assertEqual(normal_ordered_b, self.operator_b_normal)
    normal_ordered_b *= -1.
    normal_ordered_b += self.normal_ordered_b1
    normal_ordered_b += self.normal_ordered_b2
    self.assertEqual(len(normal_ordered_b.terms), 0)

  def test_normal_ordered_number(self):
    number_term2 = FermionTerm(self.n_qubits, 1, [(2, 1), (2, 0)])
    number_op2 = FermionOperator(self.n_qubits, number_term2)
    self.assertEqual(number_op2, number_term2.normal_ordered())

  def test_normal_ordered_number_reversed(self):
    n_term_rev2 = FermionTerm(self.n_qubits, 1, [(2, 0), (2, 1)])
    number_term2 = FermionTerm(self.n_qubits, 1, [(2, 1), (2, 0)])
    number_op2 = FermionOperator(self.n_qubits, number_term2)
    self.assertEqual(fermion_identity(self.n_qubits) - number_op2,
                     n_term_rev2.normal_ordered())

  def test_normal_ordered_offsite(self):
    term = FermionTerm(self.n_qubits, 1, [(3, 1), (2, 0)])
    op = FermionOperator(self.n_qubits, term)
    self.assertEqual(op, term.normal_ordered())

  def test_normal_ordered_offsite_reversed(self):
    term = FermionTerm(self.n_qubits, 1, [(3, 0), (2, 1)])
    expected = FermionTerm(self.n_qubits, -1, [(2, 1), (3, 0)])
    op = FermionOperator(self.n_qubits, expected)
    self.assertEqual(op, term.normal_ordered())

  def test_normal_ordered_double_create(self):
    term = FermionTerm(self.n_qubits, 1, [(2, 0), (3, 1), (3, 1)])
    expected = FermionTerm(self.n_qubits, 0.0)
    op = FermionOperator(self.n_qubits, expected)
    self.assertEqual(op, term.normal_ordered())

  def test_normal_ordered_multi(self):
    term = FermionTerm(self.n_qubits, 1, [(2, 0), (1, 1), (2, 1)])
    ordered_212 = FermionTerm(self.n_qubits, -1, [(2, 1), (1, 1), (2, 0)])
    ordered_1 = FermionTerm(self.n_qubits, -1, [(1, 1)])
    ordered_op = FermionOperator(self.n_qubits, [ordered_1, ordered_212])
    self.assertEqual(ordered_op, term.normal_ordered())

  def test_normal_ordered_triple(self):
    term_132 = FermionTerm(self.n_qubits, 1, [(1, 1), (3, 0), (2, 0)])
    op_132 = FermionOperator(self.n_qubits, term_132)

    term_123 = FermionTerm(self.n_qubits, 1, [(1, 1), (2, 0), (3, 0)])
    op_123 = FermionOperator(self.n_qubits, term_123)

    term_321 = FermionTerm(self.n_qubits, 1, [(3, 0), (2, 0), (1, 1)])
    op_321 = FermionOperator(self.n_qubits, term_321)

    self.assertEqual(term_123.normal_ordered(), -op_132)
    self.assertEqual(term_132.normal_ordered(), op_132)
    self.assertEqual(term_321.normal_ordered(), op_132)

  def test_jordan_wigner_transform_raise3(self):
    raising = FermionTerm(self.n_qubits, 1.,
                          [(3, 1)]).jordan_wigner_transform()
    self.assertEqual(len(raising), 2)

    correct_operators_x = [(0, 'Z'), (1, 'Z'), (2, 'Z'), (3, 'X')]
    correct_operators_y = [(0, 'Z'), (1, 'Z'), (2, 'Z'), (3, 'Y')]

    self.assertEqual(raising[correct_operators_x], 0.5)
    self.assertEqual(raising[correct_operators_y], -0.5j)
    self.assertEqual(raising.n_qubits, self.n_qubits)

  def test_jordan_wigner_transform_raise1(self):
    raising = FermionTerm(self.n_qubits, 1.,
                          [(1, 1)]).jordan_wigner_transform()
    self.assertEqual(len(raising), 2)

    correct_operators_x = [(0, 'Z'), (1, 'X')]
    correct_operators_y = [(0, 'Z'), (1, 'Y')]

    self.assertEqual(raising[correct_operators_x], 0.5)
    self.assertEqual(raising[correct_operators_y], -0.5j)
    self.assertEqual(raising.n_qubits, self.n_qubits)

  def test_jordan_wigner_transform_lower3(self):
    lowering = FermionTerm(self.n_qubits, 1.,
                           [(3, 0)]).jordan_wigner_transform()
    self.assertEqual(len(lowering), 2)

    correct_operators_x = [(0, 'Z'), (1, 'Z'), (2, 'Z'), (3, 'X')]
    correct_operators_y = [(0, 'Z'), (1, 'Z'), (2, 'Z'), (3, 'Y')]
    qtermx = QubitTerm(self.n_qubits, 0.5, correct_operators_x)
    qtermy = QubitTerm(self.n_qubits, 0.5j, correct_operators_y)

    self.assertEqual(lowering[correct_operators_x], 0.5)
    self.assertEqual(lowering[correct_operators_y], 0.5j)
    self.assertEqual(lowering, QubitOperator(self.n_qubits,
                                             [qtermx, qtermy]))
    self.assertEqual(lowering.n_qubits, self.n_qubits)

  def test_jordan_wigner_transform_lower2(self):
    lowering = FermionTerm(self.n_qubits, 1.,
                           [(2, 0)]).jordan_wigner_transform()
    self.assertEqual(len(lowering), 2)

    correct_operators_x = [(0, 'Z'), (1, 'Z'), (2, 'X')]
    correct_operators_y = [(0, 'Z'), (1, 'Z'), (2, 'Y')]

    self.assertEqual(lowering[correct_operators_x], 0.5)
    self.assertEqual(lowering[correct_operators_y], 0.5j)
    self.assertEqual(lowering.n_qubits, self.n_qubits)

  def test_jordan_wigner_transform_lower1(self):
    lowering = FermionTerm(self.n_qubits, 1.,
                           [(1, 0)]).jordan_wigner_transform()
    self.assertEqual(len(lowering), 2)

    correct_operators_x = [(0, 'Z'), (1, 'X')]
    correct_operators_y = [(0, 'Z'), (1, 'Y')]

    self.assertEqual(lowering[correct_operators_x], 0.5)
    self.assertEqual(lowering[correct_operators_y], 0.5j)
    self.assertEqual(lowering.n_qubits, self.n_qubits)

  def test_jordan_wigner_transform_lower0(self):
    lowering = FermionTerm(self.n_qubits, 1.,
                           [(0, 0)]).jordan_wigner_transform()
    self.assertEqual(len(lowering), 2)

    correct_operators_x = [(0, 'X')]
    correct_operators_y = [(0, 'Y')]

    self.assertEqual(lowering[correct_operators_x], 0.5)
    self.assertEqual(lowering[correct_operators_y], 0.5j)
    self.assertEqual(lowering.n_qubits, self.n_qubits)

  def test_jordan_wigner_transform_raise3lower0(self):
    # recall that creation gets -1j on Y and annihilation gets +1j on Y.
    term = FermionTerm(self.n_qubits, 1.,
                       [(3, 1), (0, 0)]).jordan_wigner_transform()
    self.assertEqual(term[((0, 'X'), (1, 'Z'), (2, 'Z'), (3, 'Y'))],
                     0.25 * 1 * -1j)
    self.assertEqual(term[((0, 'Y'), (1, 'Z'), (2, 'Z'), (3, 'Y'))],
                     0.25 * 1j * -1j)
    self.assertEqual(term[((0, 'Y'), (1, 'Z'), (2, 'Z'), (3, 'X'))],
                     0.25 * 1j * 1)
    self.assertEqual(term[((0, 'X'), (1, 'Z'), (2, 'Z'), (3, 'X'))],
                     0.25 * 1 * 1)
    self.assertEqual(self.n_qubits, term.n_qubits)

  def test_jordan_wigner_transform_number(self):
    n = number_operator(self.n_qubits, 3)
    n_jw = n.jordan_wigner_transform()
    self.assertEqual(n_jw[[(3, 'Z')]], -0.5)
    self.assertEqual(n_jw[[]], 0.5)
    self.assertEqual(len(n_jw), 2)
    self.assertEqual(self.n_qubits, n_jw.n_qubits)

  def test_bravyi_kitaev_transform(self):
    # Check that the QubitOperators are two-term.
    lowering = FermionTerm(self.n_qubits, 1.,
                           [(3, 0)]).bravyi_kitaev_transform()
    raising = FermionTerm(self.n_qubits, 1.,
                          [(3, 1)]).bravyi_kitaev_transform()
    self.assertEqual(len(raising), 2)
    self.assertEqual(len(lowering), 2)

    #  Test the locality invariant for N=2^d qubits
    # (c_j majorana is always log2N+1 local on qubits)
    n_qubits = 16
    invariant = numpy.log2(n_qubits) + 1
    for index in range(n_qubits):
      operator = FermionTerm(n_qubits, 1.,
                             [(index, 0)]).bravyi_kitaev_transform()
      qubit_terms = operator.terms.items()  # Get the majorana terms.

      for item in qubit_terms:
        term = item[1]

        #  Identify the c majorana terms by real
        #  coefficients and check their length.
        if not isinstance(term.coefficient, complex):
          self.assertEqual(len(term), invariant)

    #  Hardcoded coefficient test on 16 qubits
    lowering = FermionTerm(n_qubits, 1., [(9, 0)]).bravyi_kitaev_transform()
    raising = FermionTerm(n_qubits, 1., [(9, 1)]).bravyi_kitaev_transform()

    correct_operators_c = [(7, 'Z'), (8, 'Z'), (9, 'X'), (11, 'X'), (15, 'X')]
    correct_operators_d = [(7, 'Z'), (9, 'Y'), (11, 'X'), (15, 'X')]

    self.assertEqual(lowering[correct_operators_c], 0.5)
    self.assertEqual(lowering[correct_operators_d], 0.5j)
    self.assertEqual(raising[correct_operators_d], -0.5j)
    self.assertEqual(raising[correct_operators_c], 0.5)

  def test_add_terms(self):
    sum_terms = self.term_a + self.term_b
    diff_terms = self.term_a - self.term_b
    self.assertEqual(2. * self.term_a + self.term_b - self.term_b,
                     sum_terms + diff_terms)
    self.assertIsInstance(sum_terms, FermionOperator)
    self.assertIsInstance(diff_terms, FermionOperator)

  def test_jw_ccr_offsite_even_ca(self):
    c2 = FermionTerm(5, 1, [(2, 1)])
    a4 = FermionTerm(5, 1, [(4, 0)])
    self.assertEqual((c2 * a4).normal_ordered(), (-a4 * c2).normal_ordered())

    self.assertEqual((c2 * a4).jordan_wigner_transform(),
                     (-a4 * c2).jordan_wigner_transform())

  def test_jw_ccr_offsite_odd_ca(self):
    c1 = FermionTerm(5, 1, [(1, 1)])
    a4 = FermionTerm(5, 1, [(4, 0)])
    self.assertEqual((c1 * a4).normal_ordered(), (-a4 * c1).normal_ordered())

    self.assertEqual((c1 * a4).jordan_wigner_transform(),
                     (-a4 * c1).jordan_wigner_transform())

  def test_jw_ccr_offsite_even_cc(self):
    c2 = FermionTerm(5, 1, [(2, 1)])
    c4 = FermionTerm(5, 1, [(4, 1)])
    self.assertEqual((c2 * c4).normal_ordered(), (-c4 * c2).normal_ordered())

    self.assertEqual((c2 * c4).jordan_wigner_transform(),
                     (-c4 * c2).jordan_wigner_transform())

  def test_jw_ccr_offsite_odd_cc(self):
    c1 = FermionTerm(5, 1, [(1, 1)])
    c4 = FermionTerm(5, 1, [(4, 1)])
    self.assertEqual((c1 * c4).normal_ordered(), (-c4 * c1).normal_ordered())

    self.assertEqual((c1 * c4).jordan_wigner_transform(),
                     (-c4 * c1).jordan_wigner_transform())

  def test_jw_ccr_offsite_even_aa(self):
    a2 = FermionTerm(5, 1, [(2, 0)])
    a4 = FermionTerm(5, 1, [(4, 0)])
    self.assertEqual((a2 * a4).normal_ordered(), (-a4 * a2).normal_ordered())

    self.assertEqual((a2 * a4).jordan_wigner_transform(),
                     (-a4 * a2).jordan_wigner_transform())

  def test_jw_ccr_offsite_odd_aa(self):
    a1 = FermionTerm(5, 1, [(1, 0)])
    a4 = FermionTerm(5, 1, [(4, 0)])
    self.assertEqual((a1 * a4).normal_ordered(), (-a4 * a1).normal_ordered())

    self.assertEqual((a1 * a4).jordan_wigner_transform(),
                     (-a4 * a1).jordan_wigner_transform())

  def test_jw_ccr_onsite(self):
    c1 = FermionTerm(5, 1, [(1, 1)])
    a1 = c1.hermitian_conjugated()
    self.assertEqual((c1 * a1).normal_ordered(),
                     fermion_identity(5) - (a1 * c1).normal_ordered())
    self.assertEqual((c1 * a1).jordan_wigner_transform(),
                     (qubit_identity(5) -
                      (a1 * c1).jordan_wigner_transform()))

  def test_is_molecular_term_fermion_identity(self):
    term = FermionTerm(self.n_qubits, 1.0)
    self.assertTrue(term.is_molecular_term())

  def test_is_molecular_term_number(self):
    term = number_operator(self.n_qubits, 3)
    self.assertTrue(term.is_molecular_term())

  def test_is_molecular_term_updown(self):
    term = FermionTerm(self.n_qubits, 1.0, [(2, 1), (4, 0)])
    self.assertTrue(term.is_molecular_term())

  def test_is_molecular_term_downup(self):
    term = FermionTerm(self.n_qubits, 1.0, [(2, 0), (4, 1)])
    self.assertTrue(term.is_molecular_term())

  def test_is_molecular_term_downup_badspin(self):
    term = FermionTerm(self.n_qubits, 1.0, [(2, 0), (3, 1)])
    self.assertFalse(term.is_molecular_term())

  def test_is_molecular_term_three(self):
    term = FermionTerm(self.n_qubits, 1.0, [(0, 1), (2, 1), (4, 0)])
    self.assertFalse(term.is_molecular_term())

  def test_is_molecular_term_four(self):
    term = FermionTerm(self.n_qubits, 1.0, [(0, 1), (2, 0), (1, 1), (3, 0)])
    self.assertTrue(term.is_molecular_term())


class FermionOperatorsTest(unittest.TestCase):

  def setUp(self):
    self.n_qubits = 5

    self.coefficient_a = 6.7j
    self.coefficient_b = -88.
    self.coefficient_c = 3.

    self.operators_a = [(3, 1), (1, 0), (4, 1)]
    self.operators_b = [(2, 0), (4, 0), (2, 1)]
    self.operators_c = [(2, 1), (3, 0), (0, 0), (3, 1)]

    self.term_a = FermionTerm(self.n_qubits, self.coefficient_a,
                              self.operators_a)
    self.term_b = FermionTerm(self.n_qubits, self.coefficient_b,
                              self.operators_b)
    self.term_c = FermionTerm(self.n_qubits, self.coefficient_c,
                              self.operators_c)

    self.operator = FermionOperator(self.n_qubits, [self.term_a, self.term_b])
    self.operator_a = FermionOperator(self.n_qubits, self.term_a)
    self.operator_bc = FermionOperator(self.n_qubits,
                                       [self.term_b, self.term_c])
    self.operator_abc = FermionOperator(
        self.n_qubits, [self.term_a, self.term_b, self.term_c])
    self.operator_c = FermionOperator(self.n_qubits, self.term_c)
    self.normal_ordered_a = FermionTerm(self.n_qubits, self.coefficient_a,
                                        [(4, 1), (3, 1), (1, 0)])
    self.normal_ordered_b1 = FermionTerm(self.n_qubits, -self.coefficient_b,
                                         [(4, 0)])
    self.normal_ordered_b2 = FermionTerm(self.n_qubits, -self.coefficient_b,
                                         [(2, 1), (4, 0), (2, 0)])
    self.normal_ordered_operator = FermionOperator(
        self.n_qubits, [self.normal_ordered_a, self.normal_ordered_b1,
                        self.normal_ordered_b2])

  def test_init_list(self):
    self.assertEqual(self.n_qubits, self.operator_a.n_qubits)

    self.assertEqual(self.term_a, self.operator_a.terms.values()[0])
    self.assertEqual(self.coefficient_b,
                     self.operator_abc[self.operators_b])
    self.assertEqual(0.0, self.operator_abc[(3, 0), (1, 0), (4, 1)])
    self.assertEqual(len(self.operator_abc), 3)

  def test_init_dict(self):
    d = {}
    d[((3, 1), (1, 0), (4, 1))] = self.term_a
    d[((2, 1), (3, 0), (0, 0), (3, 1))] = self.term_c
    op_ac = FermionOperator(self.n_qubits, d)
    self.assertEqual(len(op_ac), 2)
    self.assertEqual(self.n_qubits, op_ac.n_qubits)
    self.assertEqual(self.coefficient_a,
                     op_ac[tuple(self.operators_a)])
    self.assertEqual(self.coefficient_c,
                     op_ac[tuple(self.operators_c)])
    self.assertEqual(0.0,
                     op_ac[tuple(self.operators_b)])

  def test_init_fermionterm(self):
    self.assertEqual(self.operator_a,
                     FermionOperator(self.n_qubits, [self.term_a]))
    self.assertEqual(len(self.operator_a), 1)
    self.assertEqual(self.coefficient_a,
                     self.operator_a[tuple(self.operators_a)])
    self.assertEqual(0.0, self.operator_a[tuple(self.operators_b)])
    self.assertEqual(0.0, self.operator_a[tuple(self.operators_c)])

  def test_init_badterm(self):
    with self.assertRaises(TypeError):
      local_operators.LocalOperator(self.n_qubits, 1)

  def test_init_list_protection(self):
    coeff1 = 2.j-3
    operators1 = ((0, 1), (1, 0), (2, 0))
    terms = [FermionTerm(self.n_qubits, coeff1, operators1)]

    operator1 = FermionOperator(self.n_qubits, terms)
    terms.append((3, 1))

    expected_term = FermionTerm(self.n_qubits, coeff1, operators1)
    expected_op = FermionOperator(self.n_qubits, expected_term)
    self.assertEqual(operator1, expected_op)

  def test_init_dict_protection(self):
    d = {}
    d[((3, 1), (1, 0), (4, 1))] = self.term_a
    d[((2, 1), (3, 0), (0, 0), (3, 1))] = self.term_c

    op_ac = local_operators.LocalOperator(self.n_qubits, d)
    self.assertEqual(len(op_ac), 2)
    self.assertEqual(self.n_qubits, op_ac.n_qubits)

    # add a new element to the old dictionary
    d[tuple(self.operators_b)] = self.term_b

    self.assertEqual(self.coefficient_a,
                     op_ac[tuple(self.operators_a)])
    self.assertEqual(self.coefficient_c,
                     op_ac[tuple(self.operators_c)])
    self.assertEqual(0.0, op_ac[tuple(self.operators_b)])

  def test_change_nqubits_error(self):
    with self.assertRaises(local_operators.LocalOperatorError):
      self.operator_a.n_qubits = 2

  def test_eq(self):
    self.assertTrue(self.operator_a == self.operator_a)
    self.assertFalse(self.operator_a == self.operator_bc)

  def test_neq(self):
    self.assertTrue(self.operator_a != self.operator_bc)
    self.assertFalse(self.operator_a != self.operator_a)

  def test_neq_different_nqubits(self):
    with self.assertRaises(local_operators.LocalOperatorError):
      self.operator_abc != FermionOperator(1, [])

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
    zero = FermionOperator(self.n_qubits)
    self.assertEqual(zero, new_term)

  def test_add_fermionterm(self):
    self.assertEqual(self.operator_a + self.term_a,
                     self.term_a + self.operator_a)

  def test_sub_fermionterm_cancel(self):
    self.assertEqual(self.operator_a - self.term_a,
                     self.term_a - self.operator_a)
    expected = local_operators.LocalOperator(self.n_qubits)
    self.assertEqual(self.operator_a - self.term_a, expected)

  def test_neg(self):
    term = FermionTerm(self.n_qubits, -self.coefficient_a,
                       self.operators_a)
    expected = FermionOperator(self.n_qubits, term)
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

  def test_mul_by_zero_fermionterm(self):
    zero_term1 = FermionTerm(self.n_qubits, 0.0, [(0, 1)])
    zero_op1 = FermionOperator(self.n_qubits, zero_term1)
    zero_term2 = FermionTerm(self.n_qubits, 0.0)
    zero_op2 = FermionOperator(self.n_qubits, zero_term2)
    self.assertEqual(self.operator_abc * zero_term1, zero_op1)
    self.assertEqual(self.operator_abc * zero_term1, zero_op2)
    self.assertEqual(self.operator_abc * zero_term2, zero_op1)

  def test_mul_by_zero_op(self):
    zero_term = FermionTerm(self.n_qubits, 0.0)
    zero_op = FermionOperator(self.n_qubits, zero_term)
    self.assertEqual(self.operator_abc * zero_op, zero_op)

  def test_mul_by_identity_term(self):
    identity_term = FermionTerm(self.n_qubits, 1.0)
    self.assertEqual(self.operator_abc * identity_term, self.operator_abc)

  def test_mul_by_identity_op(self):
    identity_term = FermionTerm(self.n_qubits, 1.0)
    identity_op = FermionOperator(self.n_qubits, identity_term)
    self.assertEqual(self.operator_abc * identity_op, self.operator_abc)

  def test_mul_npfloat64(self):
    self.assertEqual(self.operator * numpy.float64(2.303),
                     self.operator * 2.303)
    self.assertEqual(numpy.float64(2.303) * self.operator,
                     self.operator * 2.303)

  def test_mul_npfloat128(self):
    self.assertEqual(self.operator * numpy.float128(2.303),
                     self.operator * 2.303)
    self.assertEqual(numpy.float128(2.303) * self.operator,
                     self.operator * 2.303)

  def test_mul_scalar_commute(self):
    self.assertEqual(3.2 * self.operator, self.operator * 3.2)

  def test_imul_fermionterm(self):
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
    self.assertIn(self.operators_a, self.operator_abc)
    self.assertIn(self.operators_b, self.operator_abc)

  def test_contains_false(self):
    self.assertNotIn(self.operators_a, self.operator_bc)

  def test_pow_sq(self):
    self.assertEqual(self.operator_abc ** 2,
                     self.operator_abc * self.operator_abc)

  def test_pow_zero(self):
    identity_term = FermionTerm(self.n_qubits, 1.0)
    identity_op = local_operators.LocalOperator(self.n_qubits, identity_term)
    self.assertEqual(self.operator_abc ** 0, identity_op)

  def test_str(self):
    self.assertEqual(str(self.operator_abc),
                     "6.7j [3+ 1 4+]\n-88.0 [2 4 2+]\n3.0 [2+ 3 0 3+]\n")

  def test_str_zero(self):
    self.assertEqual('0', str(FermionOperator(3)))

  def test_contains(self):
    self.assertNotIn(((3, 0)), self.operator_abc)
    self.assertIn(self.operators_a, self.operator_abc)
    self.assertIn(self.operators_b, self.operator_abc)
    self.assertIn(self.operators_c, self.operator_abc)

  def test_get(self):
    self.assertEqual(self.coefficient_a, self.operator_abc[self.operators_a])
    self.assertEqual(self.coefficient_b, self.operator_abc[self.operators_b])
    self.assertEqual(self.coefficient_c, self.operator_abc[self.operators_c])
    self.assertEqual(0.0, self.operator_abc[(3, 0)])

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
      self.assertIn(term, actual)
    for term in actual:
      self.assertIn(term, expected)
      self.assertEqual(self.operator_abc[term], term.coefficient)

  def test_normal_order(self):
    self.operator.normal_order()
    self.assertEqual(self.operator, self.normal_ordered_operator)

  def test_normal_order_sign(self):
    term_132 = FermionTerm(self.n_qubits, 1, [(1, 1), (3, 0), (2, 0)])
    op_132 = FermionOperator(self.n_qubits, term_132)

    term_123 = FermionTerm(self.n_qubits, 1, [(1, 1), (2, 0), (3, 0)])
    op_123 = FermionOperator(self.n_qubits, term_123)
    self.assertEqual(op_123.normal_ordered(), -op_132.normal_ordered())

  def test_jordan_wigner_transform(self):
    n = number_operator(self.n_qubits)
    n_jw = n.jordan_wigner_transform()
    self.assertEqual(self.n_qubits + 1, len(n_jw))
    self.assertEqual(self.n_qubits / 2., n_jw[[]])
    for qubit in range(self.n_qubits):
      operators = [(qubit, 'Z')]
      self.assertEqual(n_jw[operators], -0.5)

  def test_get_molecular_operator(self):
    molecular_operator = self.operator_c.get_molecular_operator()
    fermion_operator = molecular_operator.get_fermion_operator()
    fermion_operator.normal_order()
    self.operator_c.normal_order()
    self.assertEqual(self.operator_c, fermion_operator)

  def test_bk_jw_integration(self):

    # Initialize a random fermionic operator.
    n_qubits = 5
    fermion_operator = FermionTerm(
        n_qubits, -4.3, [(3, 1), (2, 1), (1, 0), (0, 0)])
    fermion_operator += FermionTerm(
        n_qubits, 8.17, [(3, 1), (1, 0)])
    fermion_operator += 3.2 * fermion_identity(n_qubits)
    fermion_operator **= 3

    # Map to qubits and compare matrix versions.
    jw_qubit_operator = fermion_operator.jordan_wigner_transform()
    bk_qubit_operator = fermion_operator.bravyi_kitaev_transform()
    jw_sparse = jw_qubit_operator.get_sparse_operator()
    bk_sparse = bk_qubit_operator.get_sparse_operator()

    # Diagonalize and make sure the spectra are the same.
    jw_spectrum = jw_sparse.get_eigenspectrum()
    bk_spectrum = bk_sparse.get_eigenspectrum()
    self.assertAlmostEqual(0., numpy.amax(
        numpy.absolute(jw_spectrum - bk_spectrum)))


if __name__ == '__main__':
  unittest.main()
