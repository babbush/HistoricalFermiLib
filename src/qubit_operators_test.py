"""Tests for qubit_operators.py"""
import sparse_operators
from qubit_operators import (QubitTerm, QubitOperator, qubit_identity,
                             QubitTermError, QubitOperatorError)
import unittest
import copy


class QubitTermsTest(unittest.TestCase):

  def setUp(self):
    self._n_qubits = 12
    self.coefficient = 0.5
    self.operators = [(1, 'X'), (3, 'Y'), (8, 'Z')]
    self.term = QubitTerm(self._n_qubits, self.coefficient, self.operators)
    self.identity = QubitTerm(self._n_qubits)

  def test_init_bad_tensor_factors(self):
    with self.assertRaises(QubitTermError):
      n_qubits = 3
      operators = [(i, 'Z') for i in xrange(4)]
      QubitTerm(n_qubits, 1, operators)

  def test_init_bad_action(self):
    with self.assertRaises(QubitTermError):
      QubitTerm(1, 1, [(0, 'Q')])

  def test_init_sort_equiv(self):
    y0x1z2 = QubitTerm(3, 1j, [(0, 'Y'), (1, 'X'), (2, 'Z')])
    x1z2y0 = QubitTerm(3, 1j, [(1, 'X'), (2, 'Z'), (0, 'Y')])
    z2x1y0 = QubitTerm(3, 1j, [(2, 'Z'), (1, 'X'), (0, 'Y')])
    self.assertEqual(y0x1z2, x1z2y0)
    self.assertEqual(y0x1z2, z2x1y0)
    self.assertEqual(x1z2y0, z2x1y0)

  def test_correct_input(self):
    self.assertEqual(self.term.coefficient, 0.5)
    self.assertEqual(self.term._n_qubits, 12)
    self.assertEqual(len(self.term), 3)

  def test_eq_same(self):
    self.assertEqual(self.term, self.term)
    self.assertFalse(self.term != self.term)

  def test_correct_multiply(self):
    term = QubitTerm(self._n_qubits, 3.j, [(0, 'Y'), (3, 'X'),
                                           (8, 'Z'), (11, 'X')])
    product = copy.deepcopy(term)
    product *= self.term
    correct_coefficient = 1.j * term.coefficient * self.coefficient
    correct_operators = [(0, 'Y'), (1, 'X'), (3, 'Z'), (11, 'X')]
    correct_product = QubitTerm(self._n_qubits, correct_coefficient,
                                correct_operators)
    self.assertEqual(correct_product, product)

  def test_sparse_matrix_Y(self):
    term = QubitTerm(1, 1, [(0, 'Y')])
    matrix = term.get_sparse_matrix()
    self.assertEqual(list(matrix.data), [1j, -1j])
    self.assertEqual(list(matrix.indices), [1, 0])

  def test_sparse_matrix_ZX(self):
    n_qubits = 2
    coefficient = 2.
    operators = [(0, 'Z'), (1, 'X')]
    term = QubitTerm(n_qubits, coefficient, operators)
    matrix = term.get_sparse_matrix()
    self.assertEqual(list(matrix.data), [2., 2., -2., -2.])
    self.assertEqual(list(matrix.indices), [1, 0, 3, 2])
    self.assertTrue(sparse_operators.is_hermitian(matrix))

  def test_sparse_matrix_ZIZ(self):
    n_qubits = 3
    operators = [(0, 'Z'), (2, 'Z')]
    term = QubitTerm(n_qubits, 1, operators)
    matrix = term.get_sparse_matrix()
    self.assertEqual(list(matrix.data), [1, -1, 1, -1, -1, 1, -1, 1])
    self.assertEqual(list(matrix.indices), range(8))

  def test_reverse_jordan_wigner(self):
    pauli_x = QubitTerm(self._n_qubits, 1., [(2, 'X')])
    pauli_y = QubitTerm(self._n_qubits, 1., [(2, 'Y')])
    pauli_z = QubitTerm(self._n_qubits, 1., [(2, 'Z')])

    transformed_x = pauli_x.reverse_jordan_wigner()
    retransformed_x = transformed_x.jordan_wigner_transform()
    self.assertEqual(1, len(retransformed_x))
    self.assertEqual(pauli_x, retransformed_x.list_terms()[0])

    transformed_y = pauli_y.reverse_jordan_wigner()
    retransformed_y = transformed_y.jordan_wigner_transform()
    self.assertEqual(1, len(retransformed_y))
    self.assertEqual(pauli_y, retransformed_y.list_terms()[0])

    transformed_z = pauli_z.reverse_jordan_wigner()
    retransformed_z = transformed_z.jordan_wigner_transform()
    self.assertEqual(1, len(retransformed_z))
    self.assertEqual(pauli_z, retransformed_z.list_terms()[0])

    transformed_i = self.identity.reverse_jordan_wigner()
    retransformed_i = transformed_i.jordan_wigner_transform()
    self.assertEqual(1, len(retransformed_i))
    self.assertEqual(self.identity, (retransformed_i.list_terms()[0]))

    transformed_term = self.term.reverse_jordan_wigner()
    retransformed_term = transformed_term.jordan_wigner_transform()
    self.assertEqual(1, len(retransformed_term))
    self.assertEqual(self.term, retransformed_term.list_terms()[0])

  def test_sum_terms(self):
    self.coefficient = 0.5
    self.operators = [(1, 'X'), (3, 'Y'), (8, 'Z')]
    new_term = 2. * self.term + self.term - 2. * self.term
    self.assertEqual(QubitOperator(self._n_qubits, self.term), new_term)
    self.assertIsInstance(self.term + self.term, QubitOperator)
    self.assertIsInstance(self.term - self.term, QubitOperator)


class QubitOperatorsTest(unittest.TestCase):

  def setUp(self):
    self._n_qubits = 12
    self.identity = QubitTerm(self._n_qubits)
    self.term_a = QubitTerm(
        self._n_qubits, 0.5, [(1, 'X'), (3, 'Y'), (8, 'Z')])
    self.term_b = QubitTerm(
        self._n_qubits, 1.2, [(1, 'Z'), (3, 'X'), (8, 'Z')])
    self.term_c = QubitTerm(
        self._n_qubits, 1.4, [(1, 'Z'), (3, 'Y'), (9, 'Z')])
    self.qubit_operator = QubitOperator(self._n_qubits)
    self.assertEqual(self.qubit_operator.terms, {})
    self.qubit_operator += self.term_a
    self.qubit_operator += self.term_b

  def test_reverse_jordan_wigner(self):
    transformed_operator = self.qubit_operator.reverse_jordan_wigner()
    retransformed_operator = transformed_operator.jordan_wigner_transform()
    self.assertTrue(self.qubit_operator == retransformed_operator)

  def test_expectation(self):
    expectation = self.qubit_operator.expectation(self.qubit_operator)
    coefficients = self.qubit_operator.list_coefficients()
    expected_expectation = sum([x * x for x in coefficients])
    self.assertAlmostEqual(expectation, expected_expectation)

  def test_sparse_matrix_combo(self):
    matrix = (QubitTerm(2, -0.1j, [(0, 'Y'), (1, 'X')]) +
              QubitTerm(2, 3+2j, [(0, 'X'), (1, 'Z')])).get_sparse_matrix()
    self.assertEqual(list(matrix.data),
                     [3+2j, 0.1, 0.1, -3-2j, 3+2j, -0.1, -0.1, -3-2j])
    self.assertEqual(list(matrix.indices), [2, 3, 2, 3, 0, 1, 0, 1])

if __name__ == "__main__":
  unittest.main()
