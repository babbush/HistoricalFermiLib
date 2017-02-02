"""Tests for qubit_operators.py"""
import sparse_operators
from qubit_operators import (QubitTerm, QubitOperator, qubit_identity,
                             QubitTermError, QubitOperatorError)
import fermion_operators as fo
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

  def test_imul_different_nqubits(self):
    term_a = QubitTerm(2)
    with self.assertRaises(QubitTermError):
      term_a *= QubitTerm(1)

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

    expected_terms = [fo.fermion_identity(self.n_qubits),
                      fo.FermionTerm(self.n_qubits, -2, [(2, 1), (2, 0)])]
    expected = fo.FermionOperator(self.n_qubits, expected_terms)
    self.assertEqual(transformed_z, expected)

    retransformed_z = transformed_z.jordan_wigner_transform()
    self.assertEqual(1, len(retransformed_z))
    self.assertEqual(QubitOperator(self.n_qubits, pauli_z), retransformed_z)

  def test_reverse_jordan_wigner_identity(self):
    transformed_i = self.identity.reverse_jordan_wigner()
    expected_i_term = fo.fermion_identity(self.identity.n_qubits)
    expected_i = fo.FermionOperator(self.identity.n_qubits, [expected_i_term])
    self.assertEqual(transformed_i, expected_i)

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

  def test_reverse_jordan_wigner_xx(self):
    xx = QubitTerm(6, 2., [(3, 'X'), (4, 'X')])
    transformed_xx = xx.reverse_jordan_wigner()
    retransformed_xx = transformed_xx.jordan_wigner_transform()

    expected1 = (fo.FermionTerm(6, 2., [(3, 1)]) -
                 fo.FermionTerm(6, 2., [(3, 0)]))
    expected2 = (fo.FermionTerm(6, 1., [(4, 1)]) +
                 fo.FermionTerm(6, 1., [(4, 0)]))
    expected = expected1 * expected2

    self.assertEqual(QubitOperator(6, [xx]), retransformed_xx)
    self.assertEqual(transformed_xx.normal_ordered(),
                     expected.normal_ordered())

  def test_reverse_jordan_wigner_yy(self):
    yy = QubitTerm(6, 2., [(2, 'Y'), (3, 'Y')])
    transformed_yy = yy.reverse_jordan_wigner()
    retransformed_yy = transformed_yy.jordan_wigner_transform()

    expected1 = -(fo.FermionTerm(6, 2., [(2, 1)]) +
                  fo.FermionTerm(6, 2., [(2, 0)]))
    expected2 = (fo.FermionTerm(6, 1., [(3, 1)]) -
                 fo.FermionTerm(6, 1., [(3, 0)]))
    expected = expected1 * expected2

    self.assertEqual(QubitOperator(6, [yy]), retransformed_yy)
    self.assertEqual(transformed_yy.normal_ordered(),
                     expected.normal_ordered())

  def test_reverse_jordan_wigner_xy(self):
    xy = QubitTerm(6, -2.j, [(4, 'X'), (5, 'Y')])
    transformed_xy = xy.reverse_jordan_wigner()
    retransformed_xy = transformed_xy.jordan_wigner_transform()

    expected1 = -2j * (fo.FermionTerm(6, 1j, [(4, 1)]) -
                       fo.FermionTerm(6, 1j, [(4, 0)]))
    expected2 = (fo.FermionTerm(6, 1., [(5, 1)]) -
                 fo.FermionTerm(6, 1., [(5, 0)]))
    expected = expected1 * expected2

    self.assertEqual(QubitOperator(6, [xy]), retransformed_xy)
    self.assertEqual(transformed_xy.normal_ordered(),
                     expected.normal_ordered())

  def test_reverse_jordan_wigner_yx(self):
    yx = QubitTerm(6, -0.5, [(0, 'Y'), (1, 'X')])
    transformed_yx = yx.reverse_jordan_wigner()
    retransformed_yx = transformed_yx.jordan_wigner_transform()

    expected1 = 1j * (fo.FermionTerm(6, 1., [(0, 1)]) +
                      fo.FermionTerm(6, 1., [(0, 0)]))
    expected2 = -0.5 * (fo.FermionTerm(6, 1., [(1, 1)]) +
                        fo.FermionTerm(6, 1., [(1, 0)]))
    expected = expected1 * expected2

    self.assertEqual(QubitOperator(6, [yx]), retransformed_yx)
    self.assertEqual(transformed_yx.normal_ordered(),
                     expected.normal_ordered())


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
    self.qubit_operator = QubitOperator(self.n_qubits,
                                        [self.term_a, self.term_b])

  def test_init(self):
    self.assertEqual(self.qubit_operator.terms,
                     {((1, 'X'), (3, 'Y'), (8, 'Z')): self.term_a,
                      ((1, 'Z'), (3, 'X'), (8, 'Z')): self.term_b})
    self.assertEqual(len(self.qubit_operator), 2)

  def test_set_in(self):
    self.qubit_operator[((1, 'X'), (3, 'Y'), (8, 'Z'))] = 0.1
    self.assertEqual(self.qubit_operator.terms,
                     {((1, 'X'), (3, 'Y'), (8, 'Z')): self.term_a * 0.2,
                      ((1, 'Z'), (3, 'X'), (8, 'Z')): self.term_b})

  def test_set_not_in(self):
    self.qubit_operator[((1, 'Z'), (3, 'Y'), (9, 'Z'))] = 4.2
    self.assertEqual(self.qubit_operator.terms,
                     {((1, 'X'), (3, 'Y'), (8, 'Z')): self.term_a,
                      ((1, 'Z'), (3, 'X'), (8, 'Z')): self.term_b,
                      ((1, 'Z'), (3, 'Y'), (9, 'Z')): self.term_c * 3})

  def test_set_not_in_zero(self):
    zero = QubitOperator(3)
    zero[((1, 'X'),)] = 1
    self.assertEqual(zero, QubitOperator(3, [QubitTerm(3, 1, [(1, 'X')])]))

  def test_set_protect_bad_operator(self):
    with self.assertRaises(QubitTermError):
      self.qubit_operator[((1, 'Q'),)] = 1

  def test_set_protect_bad_tensor_factor(self):
    with self.assertRaises(QubitTermError):
      self.qubit_operator[((19, 'X'),)] = 1

  def test_str_zero(self):
    self.assertEqual('0', str(QubitOperator(3)))

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
    self.assertEqual(list(matrix.data), [3+2j, 0.1, 0.1, -3-2j,
                                         3+2j, -0.1, -0.1, -3-2j])
    self.assertEqual(list(matrix.indices), [2, 3, 2, 3, 0, 1, 0, 1])

  def test_get_molecular_rdm(self):
    pass

if __name__ == "__main__":
  unittest.main()
