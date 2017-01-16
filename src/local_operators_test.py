"""Tests for local_operators.py"""
import local_operators
import local_terms
import unittest


class LocalOperatorsTest(unittest.TestCase):

  def setUp(self):
    self.n_qubits = 5
    
    self.coefficient_a = 6.7j
    self.coefficient_b = -88.
    self.coefficient_c = 2.
    
    self.operators_a = [1, 2, 3, 4]
    self.operators_b = [1, 2]
    self.operators_c = [0, 3, 4]
    
    self.term_a = local_terms.LocalTerm(
      self.n_qubits, self.coefficient_a, self.operators_a)
    self.term_b = local_terms.LocalTerm(
      self.n_qubits, self.coefficient_b, self.operators_b)
    self.term_c = local_terms.LocalTerm(
      self.n_qubits, self.coefficient_c, self.operators_c)
    
    self.operator_a = local_operators.LocalOperator(
      self.n_qubits, [self.term_a])
    self.operator_bc = local_operators.LocalOperator(
      self.n_qubits, [self.term_b, self.term_c])
    self.operator_abc = local_operators.LocalOperator(
      self.n_qubits, [self.term_a, self.term_b, self.term_c])

  def test_init_list(self):
    self.assertEqual(self.n_qubits, self.operator_a.n_qubits)
    self.assertEqual(self.coefficient_a, 
                     self.operator_a[tuple(self.operators_a)])
    self.assertEqual(self.term_a, self.operator_a.terms.values()[0])
    self.assertEqual(self.coefficient_b, 
                     self.operator_abc[self.operators_b])
    self.assertEqual(0.0, self.operator_abc[(1, 2, 9)])
    self.assertEqual(len(self.operator_a), 1)
    self.assertEqual(len(self.operator_abc), 3)
    
  def test_init_dict(self):
    d = {}
    d[(1, 2, 3, 4)] = self.term_a
    d[(0, 3, 4)] = self.term_c
    op_ac = local_operators.LocalOperator(self.n_qubits, d)
    self.assertEqual(len(op_ac), 2)
    self.assertEqual(self.n_qubits, op_ac.n_qubits)
    self.assertEqual(self.coefficient_a, 
                     op_ac[tuple(self.operators_a)])
    self.assertEqual(self.coefficient_c, 
                     op_ac[tuple(self.operators_c)])
    self.assertEqual(0.0, 
                     op_ac[tuple(self.operators_b)])    
    
  def test_init_badterm(self):
    with self.assertRaises(TypeError):
      local_operators.LocalOperator(self.n_qubits, 1)
  
  def test_init_list_protection(self):
    self.coeff1 = 2.j-3
    self.operators1 = [6, 7, 8, 11]
    self.term1 = local_terms.LocalTerm(self.n_qubits, self.coeff1,
                                       self.operators1)
    
    self.operator1 = local_operators.LocalOperator(self.n_qubits, [self.term1])
    self.term1 *= 2
    
    self.assertEqual(self.operator1, local_operators.LocalOperator(
      self.n_qubits, [local_terms.LocalTerm(self.n_qubits, self.coeff1, 
                                            self.operators1)]),
      "Got {}.".format(self.operator1))
    
  def test_init_dict_protection(self):
    d = {}
    d[(1, 2, 3, 4)] = self.term_a
    d[(0, 3, 4)] = self.term_c
    op_ac = local_operators.LocalOperator(self.n_qubits, d)
    self.assertEqual(len(op_ac), 2)
    self.assertEqual(self.n_qubits, op_ac.n_qubits)
    
    # add a new element to the old dictionary
    d[tuple(self.operators_b)] = self.term_b
    
    self.assertEqual(self.coefficient_a, 
                     op_ac[tuple(self.operators_a)])
    self.assertEqual(self.coefficient_c, 
                     op_ac[tuple(self.operators_c)])
    self.assertEqual(0.0, 
                     op_ac[tuple(self.operators_b)]) 
      
  def test_change_nqubits_error(self):
    with self.assertRaises(local_operators.LocalOperatorError):
      self.operator_a.n_qubits = 2   

  def test_eq(self):
    self.assertTrue(self.operator_a == self.operator_a)
    self.assertFalse(self.operator_a == self.operator_bc)
    
  def test_neq(self):
    self.assertTrue(self.operator_a != self.operator_bc)
    self.assertFalse(self.operator_a != self.operator_a)
    
  def test_addition(self):
    new_term = self.operator_a + self.operator_bc
    self.assertEqual(new_term, self.operator_abc)
    
  def test_isub(self):
    self.operator_abc -= self.operator_a
    self.assertEqual(self.operator_abc, self.operator_bc)

  def test_iadd(self):
    self.operator_bc += self.operator_a
    self.assertEqual(self.operator_bc, self.operator_abc)

  def test_add3(self):
    new_term = self.operator_abc + self.operator_abc + self.operator_abc
    for term in new_term:
      self.assertEqual(term.coefficient,
                             3. * self.operator_abc[term.operators])

  def test_sub_cancel(self):
    new_term = self.operator_abc - self.operator_abc
    self.assertEqual(local_operators.LocalOperator(self.n_qubits, []),
                     new_term)

  def test_add_localterm(self):
    self.assertEqual(self.operator_a + self.term_a,
                     self.term_a + self.operator_a)

  def test_sub_localterm_cancel(self):
    self.assertEqual(self.operator_a - self.term_a,
                    self.term_a - self.operator_a)
    self.assertEqual(self.operator_a - self.term_a,
                     local_operators.LocalOperator(self.n_qubits, []))
    
  def test_mul(self):
    new_operator = self.operator_abc * self.operator_abc
    new_term = self.term_a * self.term_a
    self.assertEqual(new_term.coefficient,
                     new_operator[new_term.operators])

  def test_mul_scalar_commute(self):
    new_operator = self.operator_abc * self.operator_abc
    self.assertEqual(3.2 * new_operator, new_operator * 3.2)

  def test_imul_op(self):
    new_term = self.term_a * self.term_a
    self.operator_abc *= self.operator_abc
    self.assertEqual((self.term_a * self.term_a).coefficient,
                     self.operator_abc[new_term.operators])

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
      
  def test_str(self):
    self.assertEqual(str(self.operator_abc), 
                     "-88.0 [1, 2]\n2.0 [0, 3, 4]\n6.7j [1, 2, 3, 4]\n")

if __name__ == '__main__':
  unittest.main()
