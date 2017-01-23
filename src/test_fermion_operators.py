"""Tests for fermion_operators.py"""
import fermion_operators
import unittest


class NumberOperatorsTest(unittest.TestCase):

  def setUp(self):
    self.n_qubits = 5

  def test_number_operator(self):
    number_operator = fermion_operators.number_operator(self.n_qubits, 3)
    self.assertEqual(number_operator,
                     fermion_operators.FermionTerm(self.n_qubits,
                                                   1., [(3, 1), (3, 0)]))

  def test_number_operator_total(self):
    total_number_operator = fermion_operators.number_operator(self.n_qubits)
    self.assertEqual(len(total_number_operator.terms), self.n_qubits)
    for qubit in xrange(self.n_qubits):
      operators = [(qubit, 1), (qubit, 0)]
      self.assertEqual(total_number_operator[operators], 1.)

  def test_number_operator_above_range(self):
    with self.assertRaises(ValueError):
      number_operator = fermion_operators.number_operator(self.n_qubits, 5)

  def test_number_operator_below_range(self):
    with self.assertRaises(ValueError):
      number_operator = fermion_operators.number_operator(self.n_qubits, -1)


class FermionTermsTest(unittest.TestCase):

  def setUp(self):
    self.n_qubits = 5
    self.coefficient_a = 6.7j
    self.coefficient_b = -88.
    self.operators_a = [(3, 1), (1, 0), (4, 1)]
    self.operators_b = [(2, 0), (4, 0), (2, 1)]
    self.term_a = fermion_operators.FermionTerm(self.n_qubits,
                                                self.coefficient_a,
                                                self.operators_a)
    self.term_b = fermion_operators.FermionTerm(self.n_qubits,
                                                self.coefficient_b,
                                                self.operators_b)
    self.normal_ordered_a = fermion_operators.FermionTerm(
        self.n_qubits, self.coefficient_a, [(4, 1), (3, 1), (1, 0)])
    self.normal_ordered_b1 = fermion_operators.FermionTerm(
        self.n_qubits, -self.coefficient_b, [(4, 0)])
    self.normal_ordered_b2 = fermion_operators.FermionTerm(
        self.n_qubits, -self.coefficient_b, [(2, 1), (4, 0), (2, 0)])
<<<<<<< HEAD
=======

    self.operator_a = fermion_operators.FermionOperator(
        self.n_qubits, [self.term_a])
    self.operator_b = fermion_operators.FermionOperator(
        self.n_qubits, [self.term_b])
    self.operator_ab = fermion_operators.FermionOperator(
        self.n_qubits, [self.term_a, self.term_b])

  def test_add_fermionterm(self):
    self.assertEqual(self.term_a + self.term_b, self.operator_ab)

  def test_sub_fermionterm(self):
    neg_term_b = -1 * self.term_b
    expected = fermion_operators.FermionOperator(self.n_qubits,
                                                 [self.term_a, neg_term_b])
    self.assertEqual(self.term_a - self.term_b, expected)

  def test_sub_cancel(self):
    expected = fermion_operators.FermionOperator(self.n_qubits)
    self.assertEqual(self.term_b - self.term_b, expected)

  def test_sub_fermionop(self):
    neg_term_b = -1 * self.term_b
    expected = fermion_operators.FermionOperator(self.n_qubits, [neg_term_b])
    self.assertEqual(self.term_a - self.operator_ab, expected)

  def test_iadd(self):
    self.term_a += self.term_b
    self.assertEqual(self.term_a, self.operator_ab)

  def test_str(self):
    self.assertEqual(str(self.term_a), '6.7j (3+ 1 4+)')

  def test_str_number_site(self):
    self.assertEqual(str(fermion_operators.number_operator(self.n_qubits, 1)),
                     '1.0 (1+ 1)')

  def test_str_identity(self):
    self.assertEqual(str(fermion_operators.identity(self.n_qubits)), '1.0 ()')

  def test_hermitian_conjugate(self):
    hermitian_conjugate = self.term_a.hermitian_conjugate()
    self.assertEqual(hermitian_conjugate.operators[0], (4, 0))
    self.assertEqual(hermitian_conjugate.operators[1], (1, 1))
    self.assertEqual(hermitian_conjugate.operators[2], (3, 0))
    self.assertEqual(-6.7j, hermitian_conjugate.coefficient)

  def test_hermitian_conjugate_identity(self):
    result = fermion_operators.identity(self.n_qubits).hermitian_conjugate()
    self.assertEqual(fermion_operators.identity(self.n_qubits), result)

  def test_hermitian_conjugate_number_site(self):
    term = fermion_operators.number_operator(self.n_qubits, 1)
    self.assertEqual(term, term.hermitian_conjugate())

  def test_is_normal_ordered(self):
    self.assertFalse(self.term_a.is_normal_ordered())
    self.assertFalse(self.term_b.is_normal_ordered())
    self.assertTrue(self.normal_ordered_a.is_normal_ordered())
    self.assertTrue(self.normal_ordered_b1.is_normal_ordered())
    self.assertTrue(self.normal_ordered_b2.is_normal_ordered())

  def test_normal_order(self):
    self.assertTrue(self.normal_ordered_a ==
                    self.term_a.normal_order().list_terms()[0])

    normal_ordered_b = self.term_b.normal_order()
    self.assertEqual(2, len(normal_ordered_b.terms))
    normal_ordered_b *= -1.
    normal_ordered_b += self.normal_ordered_b1
    normal_ordered_b += self.normal_ordered_b2
    self.assertEqual(len(normal_ordered_b.terms), 0)

  def test_jordan_wigner_transform(self):
    lowering = fermion_operators.FermionTerm(
        self.n_qubits, 1., [(3, 0)]).jordan_wigner_transform()
    raising = fermion_operators.FermionTerm(
        self.n_qubits, 1., [(3, 1)]).jordan_wigner_transform()
    self.assertEqual(len(raising), 2)
    self.assertEqual(len(lowering), 2)

    correct_operators_x = [(0, 'Z'), (1, 'Z'), (2, 'Z'), (3, 'X')]
    correct_operators_y = [(0, 'Z'), (1, 'Z'), (2, 'Z'), (3, 'Y')]
    self.assertEqual(raising[correct_operators_x], 0.5)
    self.assertEqual(raising[correct_operators_y], -0.5j)
    self.assertEqual(lowering[correct_operators_x], 0.5)
    self.assertEqual(lowering[correct_operators_y], 0.5j)

    number_operator = fermion_operators.number_operator(self.n_qubits, 3)
    number_operator_jw = number_operator.jordan_wigner_transform()
    self.assertEqual(number_operator_jw[[(3, 'Z')]], -0.5)
    self.assertEqual(number_operator_jw[[]], 0.5)
    self.assertEqual(len(number_operator_jw), 2)

  def test_add_terms(self):
    sum_terms = self.term_a + self.term_b
    diff_terms = self.term_a - self.term_b
    self.assertEqual(2. * self.term_a + self.term_b - self.term_b,
                     sum_terms + diff_terms)
    self.assertIsInstance(sum_terms, fermion_operators.FermionOperator)
    self.assertIsInstance(diff_terms, fermion_operators.FermionOperator)


class FermionOperatorsTest(unittest.TestCase):

  def setUp(self):
    self.n_qubits = 5

    self.coefficient_a = 6.7j
    self.coefficient_b = -88.
    self.coefficient_c = 3.

    self.operators_a = [(3, 1), (1, 0), (4, 1)]
    self.operators_b = [(2, 0), (4, 0), (2, 1)]
    self.operators_c = [(2, 1), (3, 0), (0, 0), (3, 1)]

    self.term_a = fermion_operators.FermionTerm(
        self.n_qubits, self.coefficient_a, self.operators_a)
    self.term_b = fermion_operators.FermionTerm(
        self.n_qubits, self.coefficient_b, self.operators_b)
    self.term_c = fermion_operators.FermionTerm(
        self.n_qubits, self.coefficient_c, self.operators_c)

    self.operator = fermion_operators.FermionOperator(
        self.n_qubits, [self.term_a, self.term_b])
    self.operator_c = fermion_operators.FermionOperator(
        self.n_qubits, [self.term_c])
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
    self.assertEqual(self.operator, self.normal_ordered_operator)

  def test_jordan_wigner_transform(self):
    number_operator = fermion_operators.number_operator(self.n_qubits)
    number_operator_jw = number_operator.jordan_wigner_transform()
    self.assertEqual(self.n_qubits + 1, len(number_operator_jw))
    self.assertEqual(self.n_qubits / 2., number_operator_jw[[]])
    for qubit in range(self.n_qubits):
      operators = [(qubit, 'Z')]
      self.assertEqual(number_operator_jw[operators], -0.5)

  def test_get_molecular_operator(self):
    molecular_operator = self.operator_c.get_molecular_operator()
    fermion_operator = molecular_operator.get_fermion_operator()
    fermion_operator.normal_order()
    self.operator_c.normal_order()
    self.assertEqual(self.operator_c, fermion_operator)

if __name__ == '__main__':
  n_qubits = 5
  coefficient_a = 6.7j
  coefficient_b = -88.
  operators_a = [(3, 1), (1, 0), (4, 1)]
  operators_b = [(2, 0), (4, 0), (2, 1)]
  term_a = fermion_operators.FermionTerm(n_qubits, coefficient_a, operators_a)
  term_b = fermion_operators.FermionTerm(n_qubits, coefficient_b, operators_b)

  operator_ab = fermion_operators.FermionOperator(n_qubits, [term_a, term_b])

  print term_a._n_qubits
  print term_a.coefficient
  print term_a.operators

  print issubclass(type(term_a), fermion_operators.FermionOperator)
  print issubclass(type(term_a), fermion_operators.FermionTerm)
  print issubclass(type(operator_ab), fermion_operators.FermionOperator)
  print issubclass(type(operator_ab), fermion_operators.FermionTerm)

  print type(operator_ab.n_qubits)

  unittest.main()
