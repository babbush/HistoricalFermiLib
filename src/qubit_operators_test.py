"""Tests for qubit_operators.py"""
import sparse_operators
from qubit_operators import (QubitTerm, QubitOperator, qubit_identity,
                             QubitTermError, QubitOperatorError)
import fermion_operators as fo
import local_terms
import local_operators
import unittest
import copy
import numpy


class QubitTermsTest(unittest.TestCase):

  def setUp(self):
    self.n_qubits = 12
    self.coefficient = 0.5
    self.operators = [(1, 'X'), (3, 'Y'), (8, 'Z')]
    self.term = QubitTerm(self.n_qubits, self.coefficient, self.operators)
    self.identity = QubitTerm(self.n_qubits)
    self.coefficient_a = 6.7j
    self.coefficient_b = -88.
    self.operators_a = [(3, 'Z'), (1, 'Y'), (4, 'Y')]
    self.operators_b = [(2, 'X'), (3, 'Y')]
    self.term_a = QubitTerm(self.n_qubits, self.coefficient_a,
                            self.operators_a)
    self.term_b = QubitTerm(self.n_qubits, self.coefficient_b,
                            self.operators_b)

    self.operator_a = QubitOperator(self.n_qubits, self.term_a)
    self.operator_b = QubitOperator(self.n_qubits, self.term_b)
    self.operator_ab = QubitOperator(self.n_qubits,
                                     [self.term_a, self.term_b])

  def test_init(self):
    self.assertEqual(self.term.coefficient, 0.5)
    self.assertEqual(self.term.n_qubits, 12)
    self.assertEqual(len(self.term), 3)
    self.assertEqual(self.term.operators, self.operators)
    self.assertEqual(len(self.term_b), 2)

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

  def test_eq_self(self):
    self.assertTrue(self.term == self.term)
    self.assertFalse(self.term == self.term_b)

  def test_neq_self(self):
    self.assertTrue(self.term_a != self.term_b)
    self.assertFalse(self.term_a != self.term_a)

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

  def test_slicing(self):
    for i in range(len(self.term_a)):
      if i == 0:
        self.assertEqual(self.term_a[i], (1, 'Y'))
      elif i == 1:
        self.assertEqual(self.term_a[i], (3, 'Z'))
      else:
        self.assertEqual(self.term_a[i], (4, 'Y'))

  def test_set_not_in(self):
    term1 = QubitTerm(5, 1, [(1, 'Y')])
    with self.assertRaises(local_terms.LocalTermError):
      term1[2] = 2

  def test_get_not_in(self):
    with self.assertRaises(local_terms.LocalTermError):
      self.term_a[11]

  def test_del_not_in(self):
    term1 = QubitTerm(10, coefficient=1,
                      operators=[(i, 'Y') for i in range(10)])
    with self.assertRaises(local_terms.LocalTermError):
      del term1[10]

  def test_slicing_del(self):
    term1 = QubitTerm(11, coefficient=1,
                      operators=[(i, 'Y') for i in range(10)])
    del term1[3:6]
    self.assertEqual(term1.operators,
                     ([(i, 'Y') for i in range(3)] +
                      [(i, 'Y') for i in range(6, 10)]))

  def test_add_term(self):
    term_a = QubitTerm(3, 1, [(1, 'Y')])
    term_b = QubitTerm(3, -1j, [(0, 'Z')])
    op_ab = QubitOperator(3, [term_a, term_b])
    self.assertEqual(term_a + term_b, op_ab)
    self.assertEqual(self.term_a + self.term_b, self.operator_ab)

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
    self.assertEqual(self.term + self.term,
                     QubitOperator(self.n_qubits, [self.term * 2]))

  def test_add_localterms_error(self):
    with self.assertRaises(TypeError):
      self.term_a + 1

  def test_add_different_nqubits_error(self):
    self.term1 = QubitTerm(5, 2j, [(1, 'X')])
    self.term2 = QubitTerm(2, -1, [(0, 'Y')])
    with self.assertRaises(local_terms.LocalTermError):
      self.term1 + self.term2

  def test_sub_cancel(self):
    self.assertEqual(self.term - self.term, QubitOperator(self.n_qubits))

  def test_neg(self):
    expected = QubitTerm(self.n_qubits, -self.coefficient_a,
                         self.operators_a)
    self.assertEqual(-self.term_a, expected)

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

  def test_mul_by_scalarzero(self):
    term1 = self.term_a * 0
    expected = QubitTerm(self.n_qubits, 0, self.term_a.operators)
    self.assertEqual(term1, expected)

  def test_mul_by_localtermzero(self):
    term0 = QubitTerm(self.n_qubits, 0, [])
    term0d = self.term_a * term0
    self.assertEqual(term0d, term0)

  def test_mul_by_self(self):
    new_term = self.term_a * self.term_a
    self.assertEqual(self.term_a.coefficient ** 2.,
                     new_term.coefficient)
    self.assertEqual([], new_term.operators)

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
    self.assertEqual(self.term_a.operators, new_term.operators)

  def test_mul_npfloat64(self):
    self.assertEqual(self.term * numpy.float64(2.303),
                     self.term * 2.303)
    self.assertEqual(numpy.float64(2.303) * self.term,
                     self.term * 2.303)

  def test_mul_npfloat128(self):
    self.assertEqual(self.term * numpy.float128(2.303),
                     self.term * 2.303)
    self.assertEqual(numpy.float128(2.303) * self.term,
                     self.term * 2.303)

  def test_mul_scalar_commute(self):
    self.assertEqual(3.2j * self.term, self.term * 3.2j)

  def test_div(self):
    new_term = self.term_a / 3
    self.assertEqual(new_term.coefficient, self.term_a.coefficient / 3)
    self.assertEqual(new_term.operators, self.term_a.operators)

  def test_idiv(self):
    self.term_a /= 2
    self.assertEqual(self.term_a.coefficient, self.coefficient_a / 2)
    self.assertEqual(self.term_a.operators,
                     sorted(self.operators_a,
                            key=lambda operator: operator[0]))

  def test_pow_square(self):
    squared = self.term_a ** 2
    expected = QubitTerm(self.n_qubits, self.coefficient_a ** 2)
    self.assertEqual(squared, self.term_a * self.term_a)
    self.assertEqual(squared, expected)

  def test_pow_zero(self):
    zerod = self.term_a ** 0
    expected = QubitTerm(self.n_qubits, 1.0)
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
    high = self.term_a ** 11
    expected = QubitTerm(self.n_qubits, self.term_a.coefficient ** 11,
                         self.operators_a)
    self.assertEqual(high, expected)

  def test_abs(self):
    abs_term_a = abs(self.term_a)
    self.assertEqual(abs(self.term_a.coefficient),
                     abs_term_a.coefficient)

  def test_abs_complex(self):
    term1 = QubitTerm(3, 2. + 3j, [])
    self.assertEqual(abs(term1).coefficient, abs(term1.coefficient))

  def test_len(self):
    self.assertEqual(len(self.term_a), 3)
    self.assertEqual(len(self.term_b), 2)

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
    self.coefficient_a = 0.5
    self.coefficient_b = 1.2
    self.coefficient_c = 1.4j
    self.operators_a = ((1, 'X'), (3, 'Y'), (8, 'Z'))
    self.operators_b = ((1, 'Z'), (3, 'X'), (8, 'Z'))
    self.operators_c = ((1, 'Z'), (3, 'Y'), (9, 'Z'))
    self.term_a = QubitTerm(
        self.n_qubits, 0.5, [(1, 'X'), (3, 'Y'), (8, 'Z')])
    self.term_b = QubitTerm(
        self.n_qubits, 1.2, [(1, 'Z'), (3, 'X'), (8, 'Z')])
    self.term_c = QubitTerm(
        self.n_qubits, 1.4j, [(1, 'Z'), (3, 'Y'), (9, 'Z')])
    self.qubit_operator = QubitOperator(self.n_qubits,
                                        [self.term_a, self.term_b])
    self.operator_a = QubitOperator(self.n_qubits, self.term_a)
    self.operator_b = QubitOperator(self.n_qubits, self.term_b)
    self.operator_bc = QubitOperator(self.n_qubits, [self.term_b, self.term_c])
    self.operator_abc = QubitOperator(self.n_qubits,
                                      [self.term_a, self.term_b, self.term_c])

  def test_init(self):
    self.assertEqual(self.qubit_operator.terms,
                     {self.operators_a: self.term_a,
                      self.operators_b: self.term_b})
    self.assertEqual(len(self.qubit_operator), 2)
    self.assertEqual(self.coefficient_a,
                     self.qubit_operator[tuple(self.operators_a)])
    self.assertEqual(0, self.qubit_operator[tuple(self.operators_c)])
    self.assertEqual(self.coefficient_b,
                     self.qubit_operator[tuple(self.operators_b)])

  def test_init_list(self):
    self.assertEqual(self.n_qubits, self.operator_a.n_qubits)
    self.assertEqual(self.coefficient_a,
                     self.operator_a[tuple(self.operators_a)])
    self.assertEqual(self.term_a, self.operator_a.terms.values()[0])
    self.assertEqual(self.coefficient_b,
                     self.operator_abc[self.operators_b])
    self.assertEqual(0.0, self.operator_abc[(0, 'X')])
    self.assertEqual(len(self.operator_a), 1)
    self.assertEqual(len(self.operator_abc), 3)

  def test_init_dict(self):
    d = {}
    d[tuple(self.operators_a)] = self.term_a
    d[tuple(self.operators_c)] = self.term_c
    op_ac = local_operators.LocalOperator(self.n_qubits, d)
    self.assertEqual(len(op_ac), 2)
    self.assertEqual(self.n_qubits, op_ac.n_qubits)
    self.assertEqual(self.coefficient_a,
                     op_ac[tuple(self.operators_a)])
    self.assertEqual(self.coefficient_c,
                     op_ac[tuple(self.operators_c)])
    self.assertEqual(0.0, op_ac[tuple(self.operators_b)])

  def test_init_qubitterm(self):
    self.assertEqual(self.operator_a,
                     QubitOperator(self.n_qubits, self.term_a))

  def test_init_badterm(self):
    with self.assertRaises(TypeError):
      QubitOperator(self.n_qubits, 1)

  def test_init_list_protection(self):
    coeff1 = 2.j - 3
    operators1 = [(0, 'X'), (5, 'Z')]
    term1 = QubitTerm(self.n_qubits, coeff1, operators1)

    operator1 = QubitOperator(self.n_qubits, [term1])
    operators1.append((6, 'X'))

    expected_term = QubitTerm(self.n_qubits, coeff1, operators1[:-1])
    expected_op = QubitOperator(self.n_qubits, expected_term)
    self.assertEqual(operator1, expected_op)

  def test_init_dict_protection(self):
    d = {}
    d[tuple(self.operators_a)] = self.term_a
    d[tuple(self.operators_c)] = self.term_c
    op_ac = QubitOperator(self.n_qubits, d)
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
      self.operator_abc != QubitOperator(1, [])

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
    expected = QubitOperator(self.n_qubits)
    self.assertEqual(expected, new_term)

  def test_add_qubitterm(self):
    self.assertEqual(self.operator_a + self.term_a,
                     self.term_a + self.operator_a)

  def test_sub_qubitterm_cancel(self):
    self.assertEqual(self.operator_a - self.term_a,
                     self.term_a - self.operator_a)
    expected = QubitOperator(self.n_qubits)
    self.assertEqual(self.operator_a - self.term_a, expected)

  def test_neg(self):
    term = QubitTerm(self.n_qubits, -self.coefficient_a, self.operators_a)
    expected = QubitOperator(self.n_qubits, term)
    self.assertEqual(-self.operator_a, expected)

  def test_mul(self):
    new_operator = self.operator_abc * self.operator_abc

    ops_ab = [(1, 'Y'), (3, 'Z')]
    ops_ac = [(1, 'Y'), (8, 'Z'), (9, 'Z')]
    ops_bc = [(3, 'Z'), (8, 'Z'), (9, 'Z')]

    term_i = QubitTerm(self.n_qubits,
                       (self.coefficient_a ** 2 + self.coefficient_b ** 2 +
                        self.coefficient_c ** 2))
    term_ab = QubitTerm(self.n_qubits,
                        2j * 1j * self.coefficient_a * self.coefficient_b,
                        ops_ab)

    expected = QubitOperator(self.n_qubits, [term_i, term_ab])
    self.assertEqual(new_operator, expected)

  def test_mul_by_zero_qubitterm(self):
    zero_term = QubitTerm(self.n_qubits, 0.0, [(1, 'X')])
    zero_op = QubitOperator(self.n_qubits, zero_term)
    self.assertEqual(self.operator_abc * zero_term, zero_op)

  def test_mul_by_zero_op(self):
    zero_term = QubitTerm(self.n_qubits, 0.0, [(1, 'X')])
    zero_op = QubitOperator(self.n_qubits, zero_term)
    self.assertEqual(self.operator_abc * zero_op, zero_op)

  def test_mul_by_identity_term(self):
    identity_term = QubitTerm(self.n_qubits, 1.0)
    self.assertEqual(self.operator_abc * identity_term, self.operator_abc)

  def test_mul_by_identity_op(self):
    identity_term = QubitTerm(self.n_qubits, 1.0)
    identity_op = QubitOperator(self.n_qubits, identity_term)
    self.assertEqual(self.operator_abc * identity_op, self.operator_abc)

  def test_mul_npfloat64(self):
    self.assertEqual(self.qubit_operator * numpy.float64(2.303),
                     self.qubit_operator * 2.303)
    self.assertEqual(numpy.float64(2.303) * self.qubit_operator,
                     self.qubit_operator * 2.303)

  def test_mul_npfloat128(self):
    self.assertEqual(self.qubit_operator * numpy.float128(2.303),
                     self.qubit_operator * 2.303)
    self.assertEqual(numpy.float128(2.303) * self.qubit_operator,
                     self.qubit_operator * 2.303)

  def test_mul_scalar_commute(self):
    self.assertEqual(3.2j * self.qubit_operator, self.qubit_operator * 3.2j)

  def test_imul_qubitterm(self):
    self.operator_abc *= self.term_a
    self.assertEqual(self.operator_abc[[]], self.coefficient_a ** 2)
    self.assertEqual(self.operator_abc[((1, 'Y'), (3, 'Z'))],
                     1j * 1j * self.coefficient_a * self.coefficient_b)
    self.assertEqual(self.operator_abc[((1, 'Y'), (8, 'Z'), (9, 'Z'))],
                     1j * self.coefficient_a * self.coefficient_c)
    self.assertEqual(self.operator_abc[self.operators_a], 0.0)
    self.assertEqual(self.operator_abc[self.operators_b], 0.0)

  def test_imul_scalar(self):
    self.operator_a *= 3
    self.assertEqual(self.operator_a[self.operators_a], 3 * self.coefficient_a)

  def test_imul_op(self):
    self.operator_abc *= self.operator_abc
    ops_ab = [(1, 'Y'), (3, 'Z')]
    term_i = QubitTerm(self.n_qubits,
                       (self.coefficient_a ** 2 + self.coefficient_b ** 2 +
                        self.coefficient_c ** 2))
    term_ab = QubitTerm(self.n_qubits,
                        2j * 1j * self.coefficient_a * self.coefficient_b,
                        ops_ab)

    expected = QubitOperator(self.n_qubits, [term_i, term_ab])
    self.assertEqual(self.operator_abc, expected)

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
    op = QubitOperator(2, [qubit_identity(2) * 3,
                           QubitTerm(2, 1.0, [(1, 'X')]),
                           QubitTerm(2, 2.0, [(0, 'X'), (1, 'Y')])])
    expect_sq = QubitOperator(2, [14 * qubit_identity(2),
                                  QubitTerm(2, 6.0, [(1, 'X')]),
                                  QubitTerm(2, 12.0, [(0, 'X'), (1, 'Y')])])
    self.assertEqual(op ** 2, expect_sq)
    self.assertEqual(self.operator_abc ** 2,
                     self.operator_abc * self.operator_abc)

  def test_pow_sq_selfinverse(self):
    op = QubitOperator(2, QubitTerm(2, -1.5j, [(1, 'X')]))
    self.assertEqual(op ** 2, QubitOperator(2, qubit_identity(2) * -2.25))

  def test_pow_operator_commute(self):
    term = QubitTerm(2, -1.5j, [(1, 'X')])
    op = QubitOperator(2, term)

    self.assertEqual(QubitOperator(2, term ** 2), op ** 2)

  def test_pow_zero(self):
    identity_op = QubitOperator(4, qubit_identity(4))
    op = QubitOperator(4, [qubit_identity(4) * 3,
                           QubitTerm(4, 1.0, [(1, 'X')]),
                           QubitTerm(4, 2.0, [(0, 'X'), (1, 'Y')])])
    self.assertEqual(op ** 0, identity_op)

  def test_set_in(self):
    self.qubit_operator[((1, 'X'), (3, 'Y'), (8, 'Z'))] = 0.1
    self.assertEqual(self.qubit_operator.terms,
                     {((1, 'X'), (3, 'Y'), (8, 'Z')): self.term_a * 0.2,
                      ((1, 'Z'), (3, 'X'), (8, 'Z')): self.term_b})

  def test_set_not_in(self):
    self.qubit_operator[((1, 'Z'), (3, 'Y'), (9, 'Z'))] = 4.2j
    self.assertEqual(self.qubit_operator.terms,
                     {((1, 'X'), (3, 'Y'), (8, 'Z')): self.term_a,
                      ((1, 'Z'), (3, 'X'), (8, 'Z')): self.term_b,
                      ((1, 'Z'), (3, 'Y'), (9, 'Z')): self.term_c * 3})

  def test_set_not_in_zero(self):
    zero = QubitOperator(3)
    zero[((1, 'X'),)] = 1
    self.assertEqual(zero, QubitOperator(3, [QubitTerm(3, 1, [(1, 'X')])]))

  def test_setting_identity(self):
    zero = QubitOperator(3)
    zero[()] = 3.7
    self.assertEqual(zero, QubitOperator(3, 3.7 * qubit_identity(3)))

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

  def test_sparse_matrix_zero_1qubit(self):
    matrix = QubitOperator(1).get_sparse_matrix()
    self.assertEqual(len(list(matrix.data)), 0)
    self.assertEqual(matrix.shape, (2, 2))

  def test_sparse_matrix_zero_5qubit(self):
    matrix = QubitOperator(5).get_sparse_matrix()
    self.assertEqual(len(list(matrix.data)), 0)
    self.assertEqual(matrix.shape, (32, 32))

  def test_sparse_matrix_identity_1qubit(self):
    matrix = QubitOperator(1, qubit_identity(1)).get_sparse_matrix()
    self.assertEqual(list(matrix.data), [1] * 2)
    self.assertEqual(matrix.shape, (2, 2))

  def test_sparse_matrix_identity_5qubit(self):
    matrix = QubitOperator(5, qubit_identity(5)).get_sparse_matrix()
    self.assertEqual(list(matrix.data), [1] * 32)
    self.assertEqual(matrix.shape, (32, 32))

  def test_sparse_matrix_linearity(self):
    identity = QubitOperator(4, qubit_identity(4))
    zzzz = QubitOperator(4, QubitTerm(4, 1.0, [(i, 'Z') for i in range(4)]))

    matrix1 = (identity + zzzz).get_sparse_matrix()
    matrix2 = identity.get_sparse_matrix() + zzzz.get_sparse_matrix()

    self.assertEqual(list(matrix1.data), [2] * 8)
    self.assertEqual(list(matrix1.indices), [0, 3, 5, 6, 9, 10, 12, 15])
    self.assertEqual(list(matrix2.data), [2] * 8)
    self.assertEqual(list(matrix2.indices), [0, 3, 5, 6, 9, 10, 12, 15])

  def test_reverse_jw_linearity(self):
    term1 = QubitTerm(4, -0.5, [(0, 'X'), (1, 'Y')])
    term2 = QubitTerm(4, -1j, [(0, 'Y'), (1, 'X'), (2, 'Y'), (3, 'Y')])

    op12 = term1.reverse_jordan_wigner() - term2.reverse_jordan_wigner()
    self.assertEqual(op12, (term1 - term2).reverse_jordan_wigner())

  @unittest.skip("Should this work? I don't know.")
  def test_get_molecular_rdm(self):
    term1 = QubitTerm(4, -0.5, [(0, 'X')])
    op1 = QubitOperator(4, term1)
    mol1 = op1.get_molecular_rdm()

    self.assertEqual(op1, mol1.jordan_wigner_transform())

if __name__ == "__main__":
  unittest.main()
