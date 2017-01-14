"""Tests for local_terms.py"""
import local_terms
import unittest


class LocalTermsTest(unittest.TestCase):

  def setUp(self):
    self.n_qubits = 5
    self.coefficient_a = -2.
    self.coefficient_b = 8.
    self.operators_a = [0, 1, 2, 3, 4]
    self.operators_b = [1, 2]
    self.term_a = local_terms.LocalTerm(
        self.n_qubits, self.coefficient_a, self.operators_a)
    self.term_b = local_terms.LocalTerm(
        self.n_qubits, self.coefficient_b, self.operators_b)

  def test_init(self):
    self.assertEqual(self.term_a.operators, self.operators_a)
    self.assertAlmostEqual(self.term_a.coefficient, self.coefficient_a)
    
  def test_init_list_protection(self):
    arr = []
    self.term1 = local_terms.LocalTerm(1, coefficient=1, operators=[])
    arr.append(3)
    self.assertEqual(self.term1.operators, [])
    
  def test_len(self):
    self.assertEqual(len(self.term_a), 5)
    
  def test_change_nqubits_error(self):
    with self.assertRaises(local_terms.LocalTermError):
      self.term_a.n_qubits = 2    

  def test_eq(self):
    self.assertTrue(self.term_a == self.term_a)
    self.assertFalse(self.term_a == self.term_b)
    
  def test_eq_within_tol(self):
    self.term1 = local_terms.LocalTerm(1, coefficient=1)
    self.term2 = local_terms.LocalTerm(1, coefficient=(1+9e-13))
    self.assertEqual(self.term1, self.term2)
    
  def test_eq_different_nqubits_error(self):
    self.term1 = local_terms.LocalTerm(1, coefficient=1)
    self.term2 = local_terms.LocalTerm(2, coefficient=1)
    with self.assertRaises(local_terms.LocalTermError):
      self.term1 == self.term2  
    
  def test_neq(self):
    self.assertTrue(self.term_a != self.term_b)
    self.assertFalse(self.term_a != self.term_a)

  def test_slicing(self):
    for i in xrange(len(self.term_a)):
      self.assertEqual(self.term_a[i], i)
      
  def test_slicing_set(self):
    for i in xrange(len(self.term_a)):
      self.term_a[i] += 1
    for i in xrange(len(self.term_a)):
      self.assertEqual(self.term_a[i], i + 1)
      
  def test_slicing_del(self):
    self.term1 = local_terms.LocalTerm(5, coefficient=1, operators=range(10))
    del self.term1[3:6]
    self.assertEqual(self.term1.operators, [0, 1, 2, 6, 7, 8, 9])
    
  # Main addition tests are in local_operators_test.py: term addition
  # is written using that.
  def test_add_localterms_error(self):
    with self.assertRaises(TypeError):
      self.term_a + self.term_b
  
  def test_add_different_nqubits_error(self):
    self.term1 = local_terms.LocalTerm(5, 1)
    self.term2 = local_terms.LocalTerm(2, 1)
    with self.assertRaises(TypeError):
      self.term1 + self.term2
      
  # TODO Ian: imul, rmul, mul, pow, str tests

  def test_multiplication(self):
    new_term = 7. * self.term_a
    self.assertTrue(3. * new_term == new_term * 3.)
    self.assertAlmostEqual(7. * self.term_a.coefficient, new_term.coefficient)
    self.assertEqual(self.term_a.operators, new_term.operators)

    self.term_a *= 7.
    self.assertTrue(self.term_a == new_term)

    new_term = self.term_a * self.term_a
    self.assertAlmostEqual(self.term_a.coefficient ** 2.,
                           new_term.coefficient)
    self.assertEqual(2 * self.term_a.operators, new_term.operators)

    new_term = self.term_a * self.term_a * self.term_a
    self.assertAlmostEqual(self.term_a.coefficient ** 3.,
                           new_term.coefficient)
    self.assertEqual(3 * self.term_a.operators, new_term.operators)
    
  def test_pow(self):
    squared = self.term_a ** 2
    self.assertEqual(squared, self.term_a * self.term_a)  

  def test_iter(self):
    for i, operator in enumerate(self.term_a):
      self.assertEqual(self.term_a.operators[i], operator)

  def test_abs(self):
    abs_term_a = abs(self.term_a)
    self.assertAlmostEqual(abs(self.term_a.coefficient),
                           abs_term_a.coefficient)


if __name__ == '__main__':
  unittest.main()
