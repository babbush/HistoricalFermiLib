"""Tests for fermion_operators.py"""
from fermion_operators import (fermion_identity, number_operator,
                               FermionTerm, FermionOperator,
                               FermionTermError, FermionOperatorError,
                               JordanWignerError)
import qubit_operators as qo
import unittest


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

  def test_init_bad(self):
    with self.assertRaises(ValueError):
      term = FermionTerm(4, 1, [(1, 2)])

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

  def test_iadd(self):
    self.term_a += self.term_b
    self.assertEqual(self.term_a, self.operator_ab)

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
    qtermx = qo.QubitTerm(self.n_qubits, 0.5, correct_operators_x)
    qtermy = qo.QubitTerm(self.n_qubits, 0.5j, correct_operators_y)

    self.assertEqual(lowering[correct_operators_x], 0.5)
    self.assertEqual(lowering[correct_operators_y], 0.5j)
    self.assertEqual(lowering, qo.QubitOperator(self.n_qubits,
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
                     (qo.qubit_identity(5) -
                      (a1 * c1).jordan_wigner_transform()))

  def test_add_terms(self):
    sum_terms = self.term_a + self.term_b
    diff_terms = self.term_a - self.term_b
    self.assertEqual(2. * self.term_a + self.term_b - self.term_b,
                     sum_terms + diff_terms)
    self.assertIsInstance(sum_terms, FermionOperator)
    self.assertIsInstance(diff_terms, FermionOperator)

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

  def test_str_zero(self):
    self.assertEqual('0', str(FermionOperator(3)))


if __name__ == '__main__':
  unittest.main()
