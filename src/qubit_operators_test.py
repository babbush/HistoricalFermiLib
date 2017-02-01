"""Tests for qubit_operators.py"""
import sparse_operators
from qubit_operators import (QubitTerm, QubitOperator, qubit_identity,
                             QubitTermError, QubitOperatorError)
import local_terms
import unittest
import copy


class QubitTermsTest(unittest.TestCase):

  def setUp(self):
    self.n_qubits = 12
    self.coefficient = 0.5
    self.operators = [(1, 'X'), (3, 'Y'), (8, 'Z')]
    self.term = QubitTerm(self.n_qubits, self.coefficient, self.operators)
    self.identity = QubitTerm(self.n_qubits)

  def test_init(self):
    self.assertEqual(self.term.coefficient, 0.5)
    self.assertEqual(self.term.n_qubits, 12)
    self.assertEqual(len(self.term), 3)

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

  def test_set_nqubits_protect(self):
    with self.assertRaises(local_terms.LocalTermError):
      self.term.n_qubits = 5

  def test_qubit_identity(self):
    term = qubit_identity(3)
    self.assertEqual(term.n_qubits, 3)
    self.assertEqual(term.operators, [])
    self.assertEqual(term.coefficient, 1)

  def test_eq_neq_self(self):
    self.assertEqual(self.term, self.term)
    self.assertFalse(self.term != self.term)

  def test_eq_tol(self):
    term1 = QubitTerm(self.n_qubits, self.coefficient, self.operators)
    term2 = QubitTerm(self.n_qubits, self.coefficient + 7e-13, self.operators)
    self.assertEqual(term1, term2)

  def test_correct_multiply(self):
    term = QubitTerm(self.n_qubits, 3.j, [(0, 'Y'), (3, 'X'),
                                          (8, 'Z'), (11, 'X')])
    product = copy.deepcopy(term)
    product *= self.term
    correct_coefficient = 1.j * term.coefficient * self.coefficient
    correct_operators = [(0, 'Y'), (1, 'X'), (3, 'Z'), (11, 'X')]
    correct_product = QubitTerm(self.n_qubits, correct_coefficient,
                                correct_operators)
    self.assertEqual(correct_product, product)

  def test_sparse_matrix_Y(self):
    term = QubitTerm(1, 1, [(0, 'Y')])
    matrix = term.get_sparse_matrix()
    self.assertEqual(list(matrix.data), [1j, -1j])
    self.assertEqual(list(matrix.indices), [1, 0])
    self.assertTrue(sparse_operators.is_hermitian(matrix))

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
    self.assertTrue(sparse_operators.is_hermitian(matrix))

  def test_reverse_jordan_wigner_x(self):
    pauli_x = QubitTerm(self.n_qubits, 1., [(2, 'X')])
    transformed_x = pauli_x.reverse_jordan_wigner()
    retransformed_x = transformed_x.jordan_wigner_transform()
    self.assertEqual(1, len(retransformed_x))
    self.assertEqual(QubitOperator(self.n_qubits, pauli_x), retransformed_x)

  def test_reverse_jordan_wigner_y(self):
    pauli_y = QubitTerm(self.n_qubits, 1., [(2, 'Y')])
    transformed_y = pauli_y.reverse_jordan_wigner()
    retransformed_y = transformed_y.jordan_wigner_transform()
    self.assertEqual(1, len(retransformed_y))
    self.assertEqual(QubitOperator(self.n_qubits, pauli_y), retransformed_y)

  def test_reverse_jordan_wigner_z(self):
    pauli_z = QubitTerm(self.n_qubits, 1., [(2, 'Z')])
    transformed_z = pauli_z.reverse_jordan_wigner()
    retransformed_z = transformed_z.jordan_wigner_transform()
    self.assertEqual(1, len(retransformed_z))
    self.assertEqual(QubitOperator(self.n_qubits, pauli_z), retransformed_z)

  def test_reverse_jordan_wigner_identity(self):
    transformed_i = self.identity.reverse_jordan_wigner()
    retransformed_i = transformed_i.jordan_wigner_transform()
    self.assertEqual(1, len(retransformed_i))
    self.assertEqual(QubitOperator(self.n_qubits, self.identity),
                     retransformed_i)

  def test_reverse_jordan_wigner_yzxz(self):
    yzxz = QubitTerm(4, 1., [(0, 'Y'), (1, 'Z'), (2, 'X'), (3, 'Z')])
    transformed_yzxz = yzxz.reverse_jordan_wigner()
    retransformed_yzxz = transformed_yzxz.jordan_wigner_transform()
    self.assertEqual(1, len(retransformed_yzxz))
    self.assertEqual(QubitOperator(4, yzxz), retransformed_yzxz)

  def test_reverse_jordan_wigner_term(self):
    transformed_term = self.term.reverse_jordan_wigner()
    retransformed_term = transformed_term.jordan_wigner_transform()
    self.assertEqual(1, len(retransformed_term))
    self.assertEqual(QubitOperator(self.n_qubits, self.term),
                     retransformed_term)

  def test_add_term(self):
    term_a = QubitTerm(3, 1, [(1, 'Y')])
    term_b = QubitTerm(3, -1j, [(0, 'Z')])
    op_ab = QubitOperator(3, [term_a, term_b])
    self.assertEqual(term_a + term_b, op_ab)

  def test_add_term_negate(self):
    term_a = QubitTerm(3, 1, [(1, 'Y'), (0, 'X')])
    term_b = QubitTerm(3, -2, [(1, 'Y'), (0, 'X')])
    op_ab = QubitOperator(3, [-term_a])
    self.assertEqual(term_a + term_b, op_ab)

  def test_add_mul_term_cancel(self):
    term_a = QubitTerm(3, 1, [(1, 'Y'), (0, 'X')])
    term_b = QubitTerm(3, -2, [(1, 'Y'), (0, 'X')])
    op_ab = QubitOperator(3)
    self.assertEqual(2 * term_a + term_b, op_ab)

  def test_add_sub_self(self):
    self.coefficient = 0.5
    self.operators = [(1, 'X'), (3, 'Y'), (8, 'Z')]
    new_term = 2. * self.term + self.term - 2. * self.term
    self.assertEqual(QubitOperator(self.n_qubits, self.term), new_term)

  def test_add_convert_to_op(self):
    self.assertIsInstance(self.term + self.term, QubitOperator)

  def test_sub_convert_to_op(self):
    self.assertIsInstance(self.term - self.term, QubitOperator)

  def test_lmul_constant(self):
    term_a = QubitTerm(3, -1j, [(1, 'Y'), (0, 'X')])
    term_3a = 3 * term_a
    self.assertEqual(term_3a.operators, [(0, 'X'), (1, 'Y')])
    self.assertEqual(term_3a.coefficient, -3j)
    self.assertEqual(term_3a.n_qubits, 3)

  def test_rmul_constant(self):
    term_a = QubitTerm(3, -1j, [(1, 'Y'), (0, 'X')])
    term_ma = term_a * -0.1j
    self.assertEqual(term_ma.operators, [(0, 'X'), (1, 'Y')])
    self.assertEqual(term_ma.coefficient, -0.1)
    self.assertEqual(term_ma.n_qubits, 3)

  def test_imul_constant(self):
    term_a = QubitTerm(3, -1j, [(1, 'Y'), (0, 'X')])
    term_a *= 3 + 2j
    self.assertEqual(term_a.operators, [(0, 'X'), (1, 'Y')])
    self.assertEqual(term_a.coefficient, 2 - 3j)
    self.assertEqual(term_a.n_qubits, 3)

  def test_mul_term(self):
    term_a = QubitTerm(3, -1j, [(1, 'Y'), (0, 'X')])
    term_b = QubitTerm(3, 2.5, [(0, 'Z'), (1, 'Y')])
    term_amulb = QubitTerm(3, -2.5, [(0, 'Y')])
    self.assertEqual(term_a * term_b, term_amulb)
    self.assertEqual(term_b * term_a, -term_amulb)

  def test_imul_term(self):
    term_a = QubitTerm(3, -1j, [(1, 'Y'), (0, 'X')])
    term_b = QubitTerm(3, -1.5, [(1, 'Y'), (0, 'X'), (2, 'Z')])
    term_a *= term_b
    self.assertEqual(term_b.coefficient, -1.5)
    self.assertEqual(term_b.operators, [(0, 'X'), (1, 'Y'), (2, 'Z')])
    self.assertEqual(term_a.coefficient, 1.5j)
    self.assertEqual(term_a.operators, [(2, 'Z')])

  def test_imul_term_bidir(self):
    term_a = QubitTerm(3, -1j, [(1, 'Y'), (0, 'X')])
    term_b = QubitTerm(3, -1.5, [(1, 'Y'), (0, 'X'), (2, 'Z')])
    term_a *= term_b
    term_b *= term_a
    self.assertEqual(term_b.coefficient, -2.25j)
    self.assertEqual(term_b.operators, [(0, 'X'), (1, 'Y')])
    self.assertEqual(term_a.coefficient, 1.5j)
    self.assertEqual(term_a.operators, [(2, 'Z')])

  def test_str_X(self):
    term = QubitTerm(1, 1, [(0, 'X')])
    self.assertEqual(str(term), '1 X0')

  def test_str_YX(self):
    self.assertEqual(str(QubitTerm(8, 2, [(4, 'Y'), (7, 'X')])),
                     '2 Y4 X7')

  def test_str_init_sort(self):
    self.assertEqual(str(QubitTerm(8, 2, [(4, 'Y'), (7, 'X')])),
                     str(QubitTerm(8, 2, [(7, 'X'), (4, 'Y')])))

  def test_str_identity(self):
    self.assertEqual(str(QubitTerm(1, 1)), '1 I')

  def test_str_negcomplexidentity(self):
    self.assertEqual(str(QubitTerm(3, -3.7j)), '-3.7j I')


class QubitOperatorsTest(unittest.TestCase):

  def setUp(self):
    self.n_qubits = 12
    self.identity = QubitTerm(self.n_qubits)
    self.term_a = QubitTerm(
        self.n_qubits, 0.5, [(1, 'X'), (3, 'Y'), (8, 'Z')])
    self.term_b = QubitTerm(
        self.n_qubits, 1.2, [(1, 'Z'), (3, 'X'), (8, 'Z')])
    self.term_c = QubitTerm(
        self.n_qubits, 1.4, [(1, 'Z'), (3, 'Y'), (9, 'Z')])
    self.qubit_operator = QubitOperator(self.n_qubits)
    self.assertEqual(self.qubit_operator.terms, {})
    self.qubit_operator += self.term_a
    self.qubit_operator += self.term_b

  def test_reverse_jordan_wigner(self):
    transformed_operator = self.qubit_operator.reverse_jordan_wigner()
    retransformed_operator = transformed_operator.jordan_wigner_transform()
    self.assertEqual(self.qubit_operator, retransformed_operator)

  def test_expectation(self):
    expectation = self.qubit_operator.expectation(self.qubit_operator)
    coefficients = self.qubit_operator.list_coefficients()
    expected_expectation = sum([x * x for x in coefficients])
    self.assertEqual(expectation, expected_expectation)

  def test_sparse_matrix_combo(self):
    matrix = (QubitTerm(2, -0.1j, [(0, 'Y'), (1, 'X')]) +
              QubitTerm(2, 3+2j, [(0, 'X'), (1, 'Z')])).get_sparse_matrix()
    self.assertEqual(list(matrix.data),
                     [3+2j, 0.1, 0.1, -3-2j, 3+2j, -0.1, -0.1, -3-2j])
    self.assertEqual(list(matrix.indices), [2, 3, 2, 3, 0, 1, 0, 1])

if __name__ == "__main__":
  unittest.main()
