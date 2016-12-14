"""Tests for fermionic_data.py"""
import unittest
import fermionic_data


class FermionicDataTest(unittest.TestCase):

  def setUp(self):
    n_sites = 10
    self.n_qubits = n_sites
    coefficient_a = 6.7j
    coefficient_b = -88.
    operators_a = [(3, 1), (1, 0), (7, 1)]
    operators_b = [(2, 0), (7, 0), (2, 1)]
    self.fermionic_term_a = fermionic_data.FermionicTerm(n_sites,
                                                         coefficient_a,
                                                         operators_a)
    self.fermionic_term_b = fermionic_data.FermionicTerm(n_sites,
                                                         coefficient_b,
                                                         operators_b)
    self.fermionic_operator = fermionic_data.FermionicOperator(
        n_sites, [self.fermionic_term_a, self.fermionic_term_b])

  def test_class_init_term(self):
    self.assertEqual(len(self.fermionic_term_a.operators), 3)
    self.assertEqual(self.fermionic_term_a.operators[0][0], 3)
    self.assertEqual(self.fermionic_term_a.operators[0][1], 1)
    self.assertEqual(self.fermionic_term_a.operators[1][0], 1)
    self.assertEqual(self.fermionic_term_a.operators[1][1], 0)
    self.assertEqual(self.fermionic_term_a.operators[2][0], 7)
    self.assertEqual(self.fermionic_term_a.operators[2][1], 1)
    self.assertAlmostEqual(6.7j, self.fermionic_term_a.coefficient)

  def test_get_hermitian_conjugate(self):
    hermitian_conjugate = self.fermionic_term_a.get_hermitian_conjugate()
    self.assertEqual(hermitian_conjugate.operators[0], (7, 0))
    self.assertEqual(hermitian_conjugate.operators[1], (1, 1))
    self.assertEqual(hermitian_conjugate.operators[2], (3, 0))
    self.assertAlmostEqual(-6.7j, hermitian_conjugate.coefficient)

  def test_return_normal_order(self):

    # Term A is already normal ordered.
    normal_ordered_a = self.fermionic_term_a.return_normal_order().terms[-1]
    self.assertTrue(normal_ordered_a.is_identical_term(
        self.fermionic_term_a))

    # Make sure Term B is normal ordered correctly.
    normal_ordered_b = self.fermionic_term_b.return_normal_order()
    self.assertEqual(normal_ordered_b.terms[0].coefficient,
                     -1. * self.fermionic_term_b.coefficient)
    self.assertEqual(normal_ordered_b.terms[1].coefficient,
                     -1. * self.fermionic_term_b.coefficient)
    self.assertEqual(normal_ordered_b.terms[0].operators[0], (7, 0))
    self.assertEqual(normal_ordered_b.terms[1].operators[0], (2, 1))
    self.assertEqual(normal_ordered_b.terms[1].operators[1], (7, 0))
    self.assertEqual(normal_ordered_b.terms[1].operators[2], (2, 0))

  def test_global_normal_order(self):
    self.fermionic_operator.normal_order()
    normal_term_a = self.fermionic_operator.terms[0]
    normal_term_b = self.fermionic_operator.terms[1]
    normal_term_c = self.fermionic_operator.terms[2]
    self.assertTrue(normal_term_a.is_identical_term(self.fermionic_term_a))
    self.assertEqual(normal_term_b.coefficient,
                     -1. * self.fermionic_term_b.coefficient)
    self.assertEqual(normal_term_c.coefficient,
                     -1. * self.fermionic_term_b.coefficient)
    self.assertEqual(normal_term_b.operators[0], (7, 0))
    self.assertEqual(normal_term_c.operators[0], (2, 1))
    self.assertEqual(normal_term_c.operators[1], (7, 0))
    self.assertEqual(normal_term_c.operators[2], (2, 0))

  def test_jordan_wigner_ladder(self):
    raising = fermionic_data.jordan_wigner_ladder(self.n_qubits, 3, 1)
    self.assertAlmostEqual(raising.terms[0].coefficient, 0.5)
    self.assertAlmostEqual(raising.terms[1].coefficient, -0.5j)
    self.assertEqual(raising.terms[0].operators[-1], (3, 'X'))
    self.assertEqual(raising.terms[1].operators[-1], (3, 'Y'))
    for operator_index in range(3):
      self.assertEqual(
          raising.terms[0].operators[operator_index], (operator_index, 'Z'))
      self.assertEqual(
          raising.terms[1].operators[operator_index], (operator_index, 'Z'))

  def test_number_operator(self):
    number_operator = fermionic_data.number_operator(self.n_qubits, 3)
    self.assertTrue(number_operator.is_identical_term(
        fermionic_data.FermionicTerm(self.n_qubits, 1., [(3, 1), (3, 0)])))
    total_number_operator = fermionic_data.number_operator(self.n_qubits)
    for site, term in enumerate(total_number_operator.terms):
      self.assertEqual(term.operators[0], (site, 1))
      self.assertEqual(term.operators[1], (site, 0))

  def test_jordan_wigner_transform(self):

    # Test single term transformation.
    number_operator = fermionic_data.number_operator(self.n_qubits, 3)
    number_operator_pauli = number_operator.jordan_wigner_transform()
    self.assertFalse(number_operator_pauli.terms[0].operators)
    self.assertAlmostEqual(number_operator_pauli.terms[0].coefficient, 0.5)
    self.assertEqual(number_operator_pauli.terms[1].operators[0], (3, 'Z'))
    self.assertAlmostEqual(number_operator_pauli.terms[1].coefficient, -0.5)

    # Test multi term transformation.
    number_operator = fermionic_data.number_operator(self.n_qubits)
    number_operator_pauli = number_operator.jordan_wigner_transform()
    self.assertAlmostEqual(
        number_operator_pauli.terms[0].coefficient, 0.5 * self.n_qubits)
    self.assertFalse(number_operator_pauli.terms[0].operators)
    for site, term in enumerate(number_operator_pauli.terms[1::]):
      self.assertAlmostEqual(term.coefficient, -0.5)
      self.assertEqual(term.operators[0], (site, 'Z'))

  def test_remove_term(self):
    self.fermionic_operator.remove_term([(3, 1), (1, 0), (7, 1)])
    n_sites = 10
    coefficient = -88.
    operators = [(2, 0), (7, 0), (2, 1)]
    fermionic_term = fermionic_data.FermionicTerm(n_sites,
                                                  coefficient,
                                                  operators)
    fermionic_operator = fermionic_data.FermionicOperator(
        n_sites, [fermionic_term])
    self.assertTrue(self.fermionic_operator.is_identical_operator(
        fermionic_operator))


if __name__ == '__main__':
  unittest.main()
