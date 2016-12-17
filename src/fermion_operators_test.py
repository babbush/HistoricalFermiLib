"""Tests for fermion_operators.py"""
import unittest
import fermion_operators


class StandAloneFunctionsTest(unittest.TestCase):

  def setUp(self):
    self.n_qubits = 5

  def test_jordan_wigner_ladder(self):
    lowering = fermion_operators.jordan_wigner_ladder(self.n_qubits, 2, 0)
    raising = fermion_operators.jordan_wigner_ladder(self.n_qubits, 2, 1)
    self.assertEqual(len(raising.terms), 2)
    self.assertEqual(len(lowering.terms), 2)
    correct_operators_x = [(0, 'Z'), (1, 'Z'), (2, 'X')]
    correct_operators_y = [(0, 'Z'), (1, 'Z'), (2, 'Y')]
    self.assertAlmostEqual(
        raising.look_up_coefficient(correct_operators_x), 0.5)
    self.assertAlmostEqual(
        raising.look_up_coefficient(correct_operators_y), -0.5j)
    self.assertAlmostEqual(
        lowering.look_up_coefficient(correct_operators_x), 0.5)
    self.assertAlmostEqual(
        lowering.look_up_coefficient(correct_operators_y), 0.5j)

  def test_number_operator(self):
    number_operator = fermion_operators.number_operator(self.n_qubits, 3)
    self.assertTrue(number_operator ==
                    fermion_operators.FermionTerm(
                        self.n_qubits, 1., [(3, 1), (3, 0)]))
    total_number_operator = fermion_operators.number_operator(self.n_qubits)
    self.assertEqual(len(total_number_operator.terms), self.n_qubits)
    for qubit in range(self.n_qubits):
      operators = [(qubit, 1), (qubit, 0)]
      self.assertAlmostEqual(total_number_operator(operators), 1.)


class FermionTermsTest(unittest.TestCase):

  def setUp(self):
    self.n_qubits = 5
    self.coefficient_a = 6.7j
    self.coefficient_b = -88.
    self.operators_a = [(3, 1), (1, 0), (4, 1)]
    self.operators_b = [(2, 0), (4, 0), (2, 1)]
    self.term_a = fermion_operators.FermionTerm(
        self.n_qubits, self.coefficient_a, self.operators_a)
    self.term_b = fermion_operators.FermionTerm(
        self.n_qubits, self.coefficient_b, self.operators_b)
    self.normal_ordered_a = fermion_operators.FermionTerm(
        self.n_qubits, self.coefficient_a, [(4, 1), (3, 1), (1, 0)])
    self.normal_ordered_b1 = fermion_operators.FermionTerm(
        self.n_qubits, -self.coefficient_b, [(4, 0)])
    self.normal_ordered_b2 = fermion_operators.FermionTerm(
        self.n_qubits, -self.coefficient_b, [(2, 1), (4, 0), (2, 0)])

  def test_str(self):
    self.assertEqual(self.term_a.__str__(), '6.7j (3+ 1 4+)')

  def test_get_hermitian_conjugate(self):
    hermitian_conjugate = self.term_a.get_hermitian_conjugate()
    self.assertEqual(hermitian_conjugate.operators[0], (4, 0))
    self.assertEqual(hermitian_conjugate.operators[1], (1, 1))
    self.assertEqual(hermitian_conjugate.operators[2], (3, 0))
    self.assertAlmostEqual(-6.7j, hermitian_conjugate.coefficient)

  def test_is_normal_ordered(self):
    self.assertFalse(self.term_a.is_normal_ordered())
    self.assertFalse(self.term_b.is_normal_ordered())
    self.assertTrue(self.normal_ordered_a.is_normal_ordered())
    self.assertTrue(self.normal_ordered_b1.is_normal_ordered())
    self.assertTrue(self.normal_ordered_b2.is_normal_ordered())

  def test_return_normal_order(self):
    self.assertTrue(self.normal_ordered_a ==
                    self.term_a.return_normal_order().list_terms()[0])

    normal_ordered_b = self.term_b.return_normal_order()
    self.assertEqual(2, len(normal_ordered_b.terms))
    normal_ordered_b.multiply_by_scalar(-1.)
    normal_ordered_b.add_terms_list([self.normal_ordered_b1,
                                     self.normal_ordered_b2])
    self.assertFalse(len(normal_ordered_b.terms))

  def test_jordan_wigner_transform(self):
    number_operator = fermion_operators.number_operator(self.n_qubits, 3)
    number_operator_jw = number_operator.jordan_wigner_transform()
    number_operator_jw.print_operator()
    self.assertAlmostEqual(number_operator_jw([(3, 'Z')]), 0.5)
    self.assertAlmostEqual(number_operator_jw([]), -0.5)
    self.assertEqual(number_operator_jw.count_terms(), 2)


class FermionOperatorsTest(unittest.TestCase):

  def setUp(self):
    self.n_qubits = 5
    self.coefficient_a = 6.7j
    self.coefficient_b = -88.
    self.operators_a = [(3, 1), (1, 0), (4, 1)]
    self.operators_b = [(2, 0), (4, 0), (2, 1)]
    self.term_a = fermion_operators.FermionTerm(
        self.n_qubits, self.coefficient_a, self.operators_a)
    self.term_b = fermion_operators.FermionTerm(
        self.n_qubits, self.coefficient_b, self.operators_b)
    self.operator = fermion_operators.FermionOperator(
        self.n_qubits, [self.term_a, self.term_b])
    self.normal_ordered_a = fermion_operators.FermionTerm(
        self.n_qubits, self.coefficient_a, [(4, 1), (3, 1), (1, 0)])
    self.normal_ordered_b1 = fermion_operators.FermionTerm(
        self.n_qubits, -self.coefficient_b, [(4, 0)])
    self.normal_ordered_b2 = fermion_operators.FermionTerm(
        self.n_qubits, -self.coefficient_b, [(2, 1), (4, 0), (2, 0)])
    self.normal_ordered_operator = fermion_operators.FermionOperator(
        self.n_qubits, [self.normal_ordered_a,
                        self.normal_ordered_b1,
                        self.normal_ordered_b2])

  def test_normal_order(self):
    self.operator.normal_order()
    self.assertTrue(self.operator == self.normal_ordered_operator)

  def test_jordan_wigner_transform(self):
    number_operator = fermion_operators.number_operator(self.n_qubits)
    number_operator_jw = number_operator.jordan_wigner_transform()
    self.assertEqual(self.n_qubits + 1, number_operator_jw.count_terms())
    self.assertAlmostEqual(self.n_qubits / 2., number_operator_jw([]))
    for qubit in range(self.n_qubits):
      operators = [(qubit, 'Z')]
      self.assertAlmostEqual(total_number_operator(operators), 0.5)


if __name__ == '__main__':
  unittest.main()
