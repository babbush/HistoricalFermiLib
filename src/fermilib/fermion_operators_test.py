"""Tests for fermion_operators.py."""
from __future__ import absolute_import

import unittest

import numpy

from fermilib import local_operators
from fermilib import local_terms
from fermilib.config import *
from fermilib.fermion_operators import (fermion_identity, number_operator,
                                        one_body_term, two_body_term,
                                        FermionTerm, FermionOperator,
                                        FermionTermError, FermionOperatorError)
from fermilib.qubit_operators import qubit_identity, QubitTerm, QubitOperator


class HoppingOperatorsTest(unittest.TestCase):

    def setUp(self):
        self.n_qubits = 5

    def test_one_body_term(self):
        n14 = one_body_term(1, 4)
        t14 = FermionTerm([(1, 1), (4, 0)])
        self.assertEqual(n14, t14)


class NumberOperatorsTest(unittest.TestCase):

    def setUp(self):
        self.n_qubits = 5

    def test_number_operator(self):
        n3 = number_operator(self.n_qubits, 3)
        self.assertEqual(n3, FermionTerm([(3, 1), (3, 0)]))

    def test_number_operator_total(self):
        total_n = number_operator(self.n_qubits)
        self.assertEqual(len(total_n.terms), self.n_qubits)
        for qubit in range(self.n_qubits):
            operators = [(qubit, 1), (qubit, 0)]
            self.assertEqual(total_n[operators], 1.)


class FermionTermsTest(unittest.TestCase):

    def setUp(self):
        self.n_qubits = 5
        self.coefficient_a = 6.7j
        self.coefficient_b = -88.
        self.operators_a = [(3, 1), (1, 0), (4, 1)]
        self.operators_b = [(2, 0), (4, 0), (2, 1)]
        self.term_a = FermionTerm(self.operators_a, self.coefficient_a)
        self.term_b = FermionTerm(self.operators_b, self.coefficient_b)
        self.normal_ordered_a = FermionTerm([(4, 1), (3, 1), (1, 0)],
                                            self.coefficient_a)
        self.normal_ordered_b1 = FermionTerm([(4, 0)], -self.coefficient_b)
        self.normal_ordered_b2 = FermionTerm([(2, 1), (4, 0), (2, 0)],
                                             -self.coefficient_b)

        self.operator_a = FermionOperator(self.term_a)
        self.operator_b = FermionOperator(self.term_b)
        self.operator_ab = FermionOperator([self.term_a, self.term_b])

        self.operator_a_normal = FermionOperator(self.normal_ordered_a)
        self.operator_b_normal = FermionOperator([self.normal_ordered_b1,
                                                  self.normal_ordered_b2])

    def test_init(self):
        self.assertEqual(self.term_a.coefficient, self.coefficient_a)
        self.assertEqual(self.term_a.operators, self.operators_a)

        self.assertEqual(self.term_b.coefficient, self.coefficient_b)
        self.assertEqual(self.term_b.operators, self.operators_b)

    def test_init_badaction(self):
        with self.assertRaises(ValueError):
            term = FermionTerm([(1, 2)])

    def test_init_str(self):
        str_input_a = FermionTerm('3^ 1 4^') * 6.7j
        str_input_b = -88 * FermionTerm('2 4 2^')
        self.assertEqual(self.term_a, str_input_a)
        self.assertEqual(self.term_b, str_input_b)

    def test_init_str_badaction(self):
        with self.assertRaises(ValueError):
            FermionTerm('1!')

    def test_init_str_identity(self):
        self.assertEqual(fermion_identity(), FermionTerm(''))

    def test_add_fermionterm(self):
        self.assertEqual(self.term_a + self.term_b, self.operator_ab)

    def test_sub_fermionterm(self):
        expected = FermionOperator([self.term_a, -self.term_b])
        self.assertEqual(self.term_a - self.term_b, expected)

    def test_sub_cancel(self):
        expected = FermionOperator()
        self.assertEqual(self.term_b - self.term_b, expected)

    def test_sub_fermionop(self):
        expected = FermionOperator(-self.term_b)
        self.assertEqual(self.term_a - self.operator_ab, expected)

    def test_eq(self):
        self.assertTrue(self.term_a == self.term_a)
        self.assertFalse(self.term_a == self.term_b)

    def test_eq_within_tol_same_ops(self):
        self.term1 = FermionTerm([(1, 1)], coefficient=1.0)
        self.term2 = FermionTerm([(1, 1)], coefficient=(1 + 9e-13))
        self.assertEqual(self.term1, self.term2)

    def test_eq_within_tol_diff_ops(self):
        self.term1 = FermionTerm([(1, 1)], coefficient=9e-13)
        self.term2 = FermionTerm([(1, 0)], coefficient=7e-13)
        self.assertEqual(self.term1, self.term2)

    def test_neq(self):
        self.assertTrue(self.term_a != self.term_b)
        self.assertFalse(self.term_a != self.term_a)

    def test_neq_outside_tol(self):
        self.term1 = FermionTerm([(1, 1)], coefficient=1.0)
        self.term2 = FermionTerm([(1, 1)],
                                 coefficient=(1 + 2 * EQ_TOLERANCE))
        self.assertNotEqual(self.term1, self.term2)
        self.assertFalse(self.term1 == self.term2)

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
        term1 = FermionTerm(operators=[(1, 1)])
        with self.assertRaises(local_terms.LocalTermError):
            term1[2] = [(2, 1)]

    def test_get_not_in(self):
        with self.assertRaises(local_terms.LocalTermError):
            self.term_a[11]

    def test_del_not_in(self):
        term1 = FermionTerm(operators=[(1, 1)])
        with self.assertRaises(local_terms.LocalTermError):
            del term1[10]

    def test_slicing_del(self):
        term1 = FermionTerm([(i, 1) for i in range(10)])
        del term1[3:6]
        self.assertEqual(term1.operators,
                         ([(i, 1) for i in range(3)] +
                          [(i, 1) for i in range(6, 10)]))

    def test_add_fermionterms(self):
        self.assertEqual(self.term_a + self.term_b,
                         FermionOperator([self.term_a, self.term_b]))

    def test_add_localterms_reverse(self):
        self.assertEqual(self.term_b + self.term_a,
                         FermionOperator([self.term_a, self.term_b]))

    def test_add_localterms_error(self):
        with self.assertRaises(TypeError):
            self.term_a + 1

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
                         FermionOperator([self.term_a, -self.term_b]))

    def test_sub_cancel(self):
        self.assertEqual(self.term_a - self.term_a, FermionOperator())

    def test_neg(self):
        expected = FermionTerm(self.operators_a, -self.coefficient_a)
        self.assertEqual(-self.term_a, expected)

    def test_rmul(self):
        term = self.term_a * -3.
        expected = FermionTerm(self.operators_a, self.coefficient_a * -3.)
        self.assertEqual(term, expected)

    def test_lmul_rmul(self):
        term = 7. * self.term_a
        self.assertEqual(3. * term, term * 3.)
        self.assertEqual(7. * self.term_a.coefficient, term.coefficient)
        self.assertEqual(self.term_a.operators, term.operators)

    def test_imul_scalar(self):
        term = self.term_a * 1
        term *= -3. + 2j
        expected = FermionTerm(
            self.operators_a, self.coefficient_a * (-3. + 2j))
        self.assertEqual(term, expected)

    def test_imul_localterm(self):
        expected_coeff = self.term_a.coefficient * self.term_b.coefficient
        expected_ops = self.term_a.operators + self.term_b.operators
        expected = FermionTerm(expected_ops, expected_coeff)

        self.term_a *= self.term_b
        self.assertEqual(self.term_a, expected)

    def test_mul_by_scalarzero(self):
        term1 = self.term_a * 0
        expected = FermionTerm(self.term_a.operators, 0.0)
        self.assertEqual(term1, expected)

    def test_mul_by_fermiontermzero(self):
        term0 = FermionTerm([], 0)
        term0d = self.term_a * term0
        self.assertEqual(term0d, term0)

    def test_mul_by_self(self):
        new_term = self.term_a * self.term_a
        self.assertEqual(self.term_a.coefficient ** 2.,
                         new_term.coefficient)
        self.assertEqual(2 * self.term_a.operators, new_term.operators)

    def test_lmul_identity(self):
        self.assertEqual(self.term_b, fermion_identity() * self.term_b)

    def test_rmul_identity(self):
        self.assertEqual(self.term_b, self.term_b * fermion_identity())

    def test_mul_by_multiple_of_identity(self):
        self.assertEqual(3.0 * self.term_a,
                         (3.0 * fermion_identity()) * self.term_a)

    def test_mul_triple(self):
        new_term = self.term_a * self.term_a * self.term_a
        self.assertEqual(self.term_a.coefficient ** 3.,
                         new_term.coefficient)
        self.assertEqual(3 * self.term_a.operators, new_term.operators)

    def test_mul_scalar(self):
        expected = local_terms.LocalTerm(self.operators_a,
                                         self.coefficient_a * (-3. + 2j))
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
        expected = FermionTerm(self.operators_a + self.operators_a,
                               self.coefficient_a ** 2)
        self.assertEqual(squared, self.term_a * self.term_a)
        self.assertEqual(squared, expected)

    def test_pow_zero(self):
        zerod = self.term_a ** 0
        expected = FermionTerm()
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
        expected = FermionTerm(self.operators_a * 10,
                               self.term_a.coefficient ** 10)
        self.assertAlmostEqual(expected.coefficient, high.coefficient)
        self.assertEqual(high.operators, expected.operators)

    def test_abs_complex(self):
        term = FermionTerm([(0, 1), (1, 0)], 2. + 3j)
        self.assertEqual(abs(term).coefficient, abs(term.coefficient))

    def test_len(self):
        term = FermionTerm([(0, 0), (1, 1)], 2. + 3j)
        self.assertEqual(len(term), 2)
        self.assertEqual(len(self.term_a), 3)
        self.assertEqual(len(self.term_b), 3)

    def test_str(self):
        self.assertEqual(str(self.term_a), '6.7j [3^ 1 4^]')

    def test_str_number_site(self):
        self.assertEqual(str(number_operator(self.n_qubits, 1)), '1.0 [1^ 1]')

    def test_str_fermion_identity(self):
        self.assertEqual(str(fermion_identity()), '1.0 []')

    def test_str_negcomplexidentity(self):
        self.assertEqual(str(FermionTerm([], -3.7j)), '-3.7j []')

    def test_hermitian_conjugated(self):
        self.term_a.hermitian_conjugate()
        self.assertEqual(self.term_a.operators[0], (4, 0))
        self.assertEqual(self.term_a.operators[1], (1, 1))
        self.assertEqual(self.term_a.operators[2], (3, 0))
        self.assertEqual(-6.7j, self.term_a.coefficient)

    def test_hermitian_conjugate_fermion_identity(self):
        result = fermion_identity()
        result.hermitian_conjugate()
        self.assertEqual(fermion_identity(), result)

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
        result = fermion_identity().hermitian_conjugated()
        self.assertEqual(fermion_identity(), result)

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
        number_term2 = FermionTerm([(2, 1), (2, 0)])
        number_op2 = FermionOperator(number_term2)
        self.assertEqual(number_op2, number_term2.normal_ordered())

    def test_normal_ordered_number_reversed(self):
        n_term_rev2 = FermionTerm([(2, 0), (2, 1)])
        number_term2 = FermionTerm([(2, 1), (2, 0)])
        number_op2 = FermionOperator(number_term2)
        self.assertEqual(fermion_identity() - number_op2,
                         n_term_rev2.normal_ordered())

    def test_normal_ordered_offsite(self):
        term = FermionTerm([(3, 1), (2, 0)])
        op = FermionOperator(term)
        self.assertEqual(op, term.normal_ordered())

    def test_normal_ordered_offsite_reversed(self):
        term = FermionTerm([(3, 0), (2, 1)])
        expected = -FermionTerm([(2, 1), (3, 0)])
        op = FermionOperator(expected)
        self.assertEqual(op, term.normal_ordered())

    def test_normal_ordered_double_create(self):
        term = FermionTerm([(2, 0), (3, 1), (3, 1)])
        expected = FermionTerm([], 0.0)
        op = FermionOperator(expected)
        self.assertEqual(op, term.normal_ordered())

    def test_normal_ordered_multi(self):
        term = FermionTerm([(2, 0), (1, 1), (2, 1)])
        ordered_212 = -FermionTerm([(2, 1), (1, 1), (2, 0)])
        ordered_1 = -FermionTerm([(1, 1)])
        ordered_op = FermionOperator([ordered_1, ordered_212])
        self.assertEqual(ordered_op, term.normal_ordered())

    def test_normal_ordered_triple(self):
        term_132 = FermionTerm([(1, 1), (3, 0), (2, 0)])
        op_132 = FermionOperator(term_132)

        term_123 = FermionTerm([(1, 1), (2, 0), (3, 0)])
        op_123 = FermionOperator(term_123)

        term_321 = FermionTerm([(3, 0), (2, 0), (1, 1)])
        op_321 = FermionOperator(term_321)

        self.assertEqual(term_123.normal_ordered(), -op_132)
        self.assertEqual(term_132.normal_ordered(), op_132)
        self.assertEqual(term_321.normal_ordered(), op_132)

    def test_add_terms(self):
        sum_terms = self.term_a + self.term_b
        diff_terms = self.term_a - self.term_b
        self.assertEqual(2. * self.term_a + self.term_b - self.term_b,
                         sum_terms + diff_terms)
        self.assertIsInstance(sum_terms, FermionOperator)
        self.assertIsInstance(diff_terms, FermionOperator)

    def test_is_molecular_term_fermion_identity(self):
        term = FermionTerm()
        self.assertTrue(term.is_molecular_term())

    def test_is_molecular_term_number(self):
        term = number_operator(self.n_qubits, 3)
        self.assertTrue(term.is_molecular_term())

    def test_is_molecular_term_updown(self):
        term = FermionTerm([(2, 1), (4, 0)])
        self.assertTrue(term.is_molecular_term())

    def test_is_molecular_term_downup(self):
        term = FermionTerm([(2, 0), (4, 1)])
        self.assertTrue(term.is_molecular_term())

    def test_is_molecular_term_downup_badspin(self):
        term = FermionTerm([(2, 0), (3, 1)])
        self.assertFalse(term.is_molecular_term())

    def test_is_molecular_term_three(self):
        term = FermionTerm([(0, 1), (2, 1), (4, 0)])
        self.assertFalse(term.is_molecular_term())

    def test_is_molecular_term_four(self):
        term = FermionTerm([(0, 1), (2, 0), (1, 1), (3, 0)])
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

        self.term_a = FermionTerm(self.operators_a, self.coefficient_a)
        self.term_b = FermionTerm(self.operators_b, self.coefficient_b)
        self.term_c = FermionTerm(self.operators_c, self.coefficient_c)

        self.operator = FermionOperator([self.term_a, self.term_b])
        self.operator_a = FermionOperator(self.term_a)
        self.operator_bc = FermionOperator([self.term_b, self.term_c])
        self.operator_abc = FermionOperator(
            [self.term_a, self.term_b, self.term_c])
        self.operator_c = FermionOperator(self.term_c)
        self.normal_ordered_a = FermionTerm([(4, 1), (3, 1), (1, 0)],
                                            self.coefficient_a)
        self.normal_ordered_b1 = FermionTerm([(4, 0)], -self.coefficient_b)
        self.normal_ordered_b2 = FermionTerm([(2, 1), (4, 0), (2, 0)],
                                             -self.coefficient_b)
        self.normal_ordered_operator = FermionOperator(
            [self.normal_ordered_a, self.normal_ordered_b1,
                self.normal_ordered_b2])

    def test_init_list(self):
        self.assertEqual(self.term_a, self.operator_a.terms.values()[0])
        self.assertEqual(self.coefficient_b,
                         self.operator_abc[self.operators_b])
        self.assertEqual(0.0, self.operator_abc[(3, 0), (1, 0), (4, 1)])
        self.assertEqual(len(self.operator_abc), 3)

    def test_init_dict(self):
        d = {}
        d[((3, 1), (1, 0), (4, 1))] = self.term_a
        d[((2, 1), (3, 0), (0, 0), (3, 1))] = self.term_c
        op_ac = FermionOperator(d)
        self.assertEqual(len(op_ac), 2)
        self.assertEqual(self.coefficient_a,
                         op_ac[tuple(self.operators_a)])
        self.assertEqual(self.coefficient_c,
                         op_ac[tuple(self.operators_c)])
        self.assertEqual(0.0,
                         op_ac[tuple(self.operators_b)])

    def test_init_fermionterm(self):
        self.assertEqual(self.operator_a, FermionOperator([self.term_a]))
        self.assertEqual(len(self.operator_a), 1)
        self.assertEqual(self.coefficient_a,
                         self.operator_a[tuple(self.operators_a)])
        self.assertEqual(0.0, self.operator_a[tuple(self.operators_b)])
        self.assertEqual(0.0, self.operator_a[tuple(self.operators_c)])

    def test_init_list_protection(self):
        coeff1 = 2.j - 3
        operators1 = ((0, 1), (1, 0), (2, 0))
        terms = [FermionTerm(operators1, coeff1,)]

        operator1 = FermionOperator(terms)
        terms.append((3, 1))

        expected_term = FermionTerm(operators1, coeff1)
        expected_op = FermionOperator(expected_term)
        self.assertEqual(operator1, expected_op)

    def test_init_dict_protection(self):
        d = {}
        d[((3, 1), (1, 0), (4, 1))] = self.term_a
        d[((2, 1), (3, 0), (0, 0), (3, 1))] = self.term_c

        op_ac = local_operators.LocalOperator(d)
        self.assertEqual(len(op_ac), 2)

        # add a new element to the old dictionary
        d[tuple(self.operators_b)] = self.term_b

        self.assertEqual(self.coefficient_a,
                         op_ac[tuple(self.operators_a)])
        self.assertEqual(self.coefficient_c,
                         op_ac[tuple(self.operators_c)])
        self.assertEqual(0.0, op_ac[tuple(self.operators_b)])

    def test_init_badterm(self):
        with self.assertRaises(TypeError):
            FermionOperator(1)

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
        zero = FermionOperator()
        self.assertEqual(zero, new_term)

    def test_add_fermionterm(self):
        self.assertEqual(self.operator_a + self.term_a,
                         self.term_a + self.operator_a)

    def test_sub_fermionterm_cancel(self):
        self.assertEqual(self.operator_a - self.term_a,
                         self.term_a - self.operator_a)
        expected = local_operators.LocalOperator()
        self.assertEqual(self.operator_a - self.term_a, expected)

    def test_neg(self):
        term = FermionTerm(self.operators_a, -self.coefficient_a)
        expected = FermionOperator(term)
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
        zero_term1 = FermionTerm([(0, 1)], 0.0)
        zero_op1 = FermionOperator(zero_term1)
        zero_term2 = FermionTerm([], 0.0)
        zero_op2 = FermionOperator(zero_term2)
        self.assertEqual(self.operator_abc * zero_term1, zero_op1)
        self.assertEqual(self.operator_abc * zero_term1, zero_op2)
        self.assertEqual(self.operator_abc * zero_term2, zero_op1)

    def test_mul_by_zero_op(self):
        zero_term = FermionTerm([], 0.0)
        zero_op = FermionOperator(zero_term)
        self.assertEqual(self.operator_abc * zero_op, zero_op)

    def test_mul_by_identity_term(self):
        identity_term = FermionTerm()
        self.assertEqual(self.operator_abc * identity_term, self.operator_abc)

    def test_mul_by_identity_op(self):
        identity_term = FermionTerm()
        identity_op = FermionOperator(identity_term)
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
        self.assertEqual(
            self.operator_abc[(self.term_a * self.term_a).operators],
            self.coefficient_a ** 2)
        self.assertEqual(
            self.operator_abc[(self.term_a * self.term_b).operators],
            0.0)
        self.assertEqual(
            self.operator_abc[(self.term_b * self.term_a).operators],
            self.coefficient_a * self.coefficient_b)
        self.assertEqual(
            self.operator_abc[(self.term_c * self.term_a).operators],
            self.coefficient_a * self.coefficient_c)
        self.assertEqual(self.operator_abc[self.operators_a], 0.0)
        self.assertEqual(self.operator_abc[self.operators_b], 0.0)

    def test_imul_scalar(self):
        self.operator_a *= 3
        self.assertEqual(
            self.operator_a[self.operators_a], 3 * self.coefficient_a)

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
        identity_term = FermionTerm()
        identity_op = local_operators.LocalOperator(identity_term)
        self.assertEqual(self.operator_abc ** 0, identity_op)

    def test_str(self):
        self.assertEqual(str(self.operator_abc),
                         '6.7j [3^ 1 4^]\n-88.0 [2 4 2^]\n3.0 [2^ 3 0 3^]\n')

    def test_str_zero(self):
        self.assertEqual('0', str(FermionOperator()))

    def test_contains(self):
        self.assertNotIn(((3, 0)), self.operator_abc)
        self.assertIn(self.operators_a, self.operator_abc)
        self.assertIn(self.operators_b, self.operator_abc)
        self.assertIn(self.operators_c, self.operator_abc)

    def test_get(self):
        self.assertEqual(self.coefficient_a,
                         self.operator_abc[self.operators_a])
        self.assertEqual(self.coefficient_b,
                         self.operator_abc[self.operators_b])
        self.assertEqual(self.coefficient_c,
                         self.operator_abc[self.operators_c])
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
        term_132 = FermionTerm([(1, 1), (3, 0), (2, 0)])
        op_132 = FermionOperator(term_132)

        term_123 = FermionTerm([(1, 1), (2, 0), (3, 0)])
        op_123 = FermionOperator(term_123)
        self.assertEqual(op_123.normal_ordered(), -op_132.normal_ordered())


if __name__ == '__main__':
    unittest.main()
