"""Tests for local_operators.py"""
import local_operators
import unittest
import copy


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


class LocalOperatorsTest(unittest.TestCase):

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
    self.assertEqual(
        len(self.local_operator_a.terms), self.local_operator_a.count_terms())
    self.assertEqual(self.local_operator_a.count_terms(), 1)
    self.assertEqual(self.local_operator_a.terms, self.local_operator_a.terms)
    self.assertTrue(self.local_operator_a == self.local_operator_a)
    self.assertFalse(self.local_operator_a == self.local_operator_bc)
    self.assertTrue(self.local_operator_a != self.local_operator_bc)
    self.assertFalse(self.local_operator_a != self.local_operator_a)

  def test_addition(self):
    self.local_operator_a.add_operator(self.local_operator_bc)
    self.assertTrue(self.local_operator_a == self.local_operator_abc)

    term_a_copy = copy.deepcopy(self.term_a)
    term_b_copy = copy.deepcopy(self.term_b)
    term_c_copy = copy.deepcopy(self.term_c)
    term_a_copy.coefficient *= -1.
    term_b_copy.coefficient *= -1.
    term_c_copy.coefficient *= -1.
    self.local_operator_a.add_terms_list([term_b_copy, term_c_copy])
    self.local_operator_a.add_term(term_a_copy)
    self.assertEqual(self.local_operator_a.terms, {})

  def test_multiplication(self):
    self.local_operator_abc.multiply_by_scalar(2.)
    new_coefficients = set(self.local_operator_abc.list_coefficients())
    correct_coefficients = set([2. * self.coefficient_a,
                                2. * self.coefficient_b,
                                2. * self.coefficient_c])
    self.assertEqual(new_coefficients, correct_coefficients)

    local_operator_a_copy = copy.deepcopy(self.local_operator_a)
    self.local_operator_a.multiply_by_term(self.term_a)
    local_operator_a_copy.multiply_by_operator(local_operator_a_copy)
    self.assertTrue(local_operator_a_copy == self.local_operator_a)

  def test_look_up_coefficient(self):
    self.assertAlmostEqual(
        self.local_operator_abc.look_up_coefficient(
            self.term_a.operators), self.coefficient_a)

  def test_remove_term(self):
    self.local_operator_abc.remove_term(self.term_a.operators)
    self.assertTrue(self.local_operator_abc, self.local_operator_a)


if __name__ == '__main__':
  unittest.main()
