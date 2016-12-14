"""Tests for pauli_data.py"""
import sparse_operators
import pauli_data
import unittest
import copy


class PauliString(unittest.TestCase):

  def setUp(self):

    # Test term acts on 12 qubits: 0.5 * X1 Y3 Z8.
    self.n_qubits = 12
    coefficient = 0.5
    operators = [(1, 'X'), (3, 'Y'), (8, 'Z')]
    self.pauli_string = pauli_data.PauliString(
        self.n_qubits, coefficient, operators)
    self.identity = pauli_data.PauliString(self.n_qubits)

  def test_correct_input(self):

    # Make sure things initialize correctly.
    self.assertAlmostEqual(self.pauli_string.coefficient, 0.5)
    self.assertEqual(self.pauli_string.n_qubits, 12)
    self.assertEqual(len(self.pauli_string.operators), 3)
    self.assertTrue(self.pauli_string.is_identical_string(self.pauli_string))

  def test_correct_identity(self):

    # Test the special case of initializing the identity operator.
    self.identity.n_qubits = 10
    self.assertEqual(self.identity.n_qubits, 10)
    self.assertAlmostEqual(self.identity.coefficient, 1.0)
    self.assertEqual(self.identity.operators, [])
    with self.assertRaises(pauli_data.ErrorPauliString):
      pauli_data.PauliString(10.1, 0.5, [])
    with self.assertRaises(pauli_data.ErrorPauliString):
      self.pauli_string.is_identical_string(self.identity)
    self.identity.n_qubits = 12
    self.assertFalse(
        self.pauli_string.is_identical_string(self.identity))

  def test_correct_multiply(self):

    # Test to make sure terms multiply correct.
    term = pauli_data.PauliString(
        12, 3.j, [(0, 'Y'), (3, 'X'), (8, 'Z'), (11, 'X')])
    product = pauli_data.multiply_pauli_strings(term, self.pauli_string)
    correct_coefficient = 1.5j * 1.j
    correct_operators = [(0, 'Y'), (1, 'X'), (3, 'Z'), (11, 'X')]
    correct_product = pauli_data.PauliString(12, correct_coefficient,
                                             correct_operators)
    self.assertTrue(correct_product.is_identical_string(product))
    self.assertAlmostEqual(correct_coefficient, product.coefficient)
    term.multiply_by_string(self.pauli_string)
    self.assertTrue(product.is_identical_string(term))

  def test_sparse_matrix(self):
    n_qubits = 2
    coefficient = 2.
    operators = [(0, 'Z'), (1, 'X')]
    pauli_string = pauli_data.PauliString(n_qubits, coefficient, operators)
    matrix = pauli_string.to_sparse_matrix()
    self.assertAlmostEqual(list(matrix.data), [2., 2., -2., -2.])
    self.assertAlmostEqual(list(matrix.indices), [1, 0, 3, 2])
    self.assertTrue(sparse_operators.is_hermitian(matrix))

  def test_reverse_jordan_wigner_pauli(self):

    # Test transformation of Z operator.
    transformed_operator = pauli_data.reverse_jordan_wigner_pauli(
        self.n_qubits, 2, 'Z')
    un_transformed_operator = transformed_operator.jordan_wigner_transform()
    self.assertEqual(len(un_transformed_operator.terms), 1)
    origin_term = un_transformed_operator.terms[0]
    self.assertEqual(origin_term.n_qubits, self.n_qubits)
    self.assertAlmostEqual(origin_term.coefficient, 1.)
    self.assertEqual(len(origin_term.operators), 1)
    self.assertEqual(origin_term.operators[0], (2, 'Z'))

    # Test transformation of X operator.
    transformed_operator = pauli_data.reverse_jordan_wigner_pauli(
        self.n_qubits, 2, 'X')
    un_transformed_operator = transformed_operator.jordan_wigner_transform()
    self.assertEqual(len(un_transformed_operator.terms), 1)
    origin_term = un_transformed_operator.terms[0]
    self.assertEqual(origin_term.n_qubits, self.n_qubits)
    self.assertAlmostEqual(origin_term.coefficient, 1.)
    self.assertEqual(len(origin_term.operators), 1)
    self.assertEqual(origin_term.operators[0], (2, 'X'))

    # Test transformation of Y operator.
    transformed_operator = pauli_data.reverse_jordan_wigner_pauli(
        self.n_qubits, 2, 'Y')
    un_transformed_operator = transformed_operator.jordan_wigner_transform()
    self.assertEqual(len(un_transformed_operator.terms), 1)
    origin_term = un_transformed_operator.terms[0]
    self.assertEqual(origin_term.n_qubits, self.n_qubits)
    self.assertAlmostEqual(origin_term.coefficient, 1.)
    self.assertEqual(len(origin_term.operators), 1)
    self.assertEqual(origin_term.operators[0], (2, 'Y'))

  def test_reverse_jordan_wigner(self):
    transformed_operator = self.pauli_string.reverse_jordan_wigner()
    un_transformed_operator = transformed_operator.jordan_wigner_transform()
    self.assertEqual(len(un_transformed_operator.terms), 1)
    origin_term = un_transformed_operator.terms[0]
    self.assertEqual(origin_term.n_qubits, self.n_qubits)
    self.assertAlmostEqual(
        origin_term.coefficient, self.pauli_string.coefficient)
    self.assertEqual(origin_term.operators, self.pauli_string.operators)


class QubitOperator(unittest.TestCase):

  def setUp(self):
    self.n_qubits = 12
    self.identity = pauli_data.PauliString(self.n_qubits)
    self.pauli_string_a = pauli_data.PauliString(
        self.n_qubits, 0.5, [(1, 'X'), (3, 'Y'), (8, 'Z')])
    self.pauli_string_b = pauli_data.PauliString(
        self.n_qubits, 1.2, [(1, 'Z'), (3, 'X'), (8, 'Z')])
    self.pauli_string_c = pauli_data.PauliString(
        self.n_qubits, 1.4, [(1, 'Z'), (3, 'Y'), (9, 'Z')])
    self.qubit_operator = pauli_data.QubitOperator(self.n_qubits)
    self.assertEqual(self.qubit_operator.terms, [])
    self.qubit_operator.add_term(self.pauli_string_a)
    self.qubit_operator.add_term(self.pauli_string_b)

  def test_add_term(self):
    self.qubit_operator = pauli_data.QubitOperator(self.n_qubits)
    self.qubit_operator.add_term(self.pauli_string_a)
    self.qubit_operator.add_term(self.pauli_string_b)
    self.assertAlmostEqual([0.5, 1.2], self.qubit_operator.get_coefficients())
    self.qubit_operator.add_term(self.pauli_string_a)
    self.assertAlmostEqual([1., 1.2], self.qubit_operator.get_coefficients())
    self.qubit_operator.add_term(self.pauli_string_c)
    self.assertAlmostEqual(
        [1., 1.2, 1.4], self.qubit_operator.get_coefficients())

  def test_multiply_by_term(self):
    self.qubit_operator = pauli_data.QubitOperator(self.n_qubits)
    self.qubit_operator.add_term(self.identity)
    self.qubit_operator.add_term(copy.copy(self.pauli_string_a))
    self.qubit_operator.multiply_by_term(self.pauli_string_c)
    self.assertAlmostEqual(1.4, self.qubit_operator.get_coefficients()[0])
    self.assertAlmostEqual(-0.7j, self.qubit_operator.get_coefficients()[1])
    self.assertTrue(self.qubit_operator.terms[0].is_identical_string(
        self.pauli_string_c))
    self.assertTrue(self.qubit_operator.terms[1].is_identical_string(
        pauli_data.multiply_pauli_strings(self.pauli_string_a,
                                          self.pauli_string_c)))

  def test_add_operator(self):
    qubit_operator_a = pauli_data.QubitOperator(
        self.n_qubits, [self.pauli_string_a])
    qubit_operator_b = pauli_data.QubitOperator(
        self.n_qubits, [self.pauli_string_b])
    qubit_operator_a.add_operator(qubit_operator_b)
    self.assertTrue(qubit_operator_a.terms[0].is_identical_string(
        self.qubit_operator.terms[0]))
    self.assertTrue(qubit_operator_a.terms[1].is_identical_string(
        self.qubit_operator.terms[1]))

  def test_multiply_by_operator(self):
    qubit_operator = pauli_data.QubitOperator(
        self.n_qubits, [self.pauli_string_a, self.identity])
    qubit_operator.multiply_by_operator(copy.deepcopy(qubit_operator))
    self.assertTrue(qubit_operator.terms[0].is_identical_string(
        self.identity))
    self.assertTrue(qubit_operator.terms[1].is_identical_string(
        self.pauli_string_a))

  def test_reverse_jordan_wigner(self):
    transformed_operator = self.qubit_operator.reverse_jordan_wigner()
    un_transformed_operator = transformed_operator.jordan_wigner_transform()
    self.assertTrue(un_transformed_operator.is_identical_operator(
        self.qubit_operator))

  def test_remove_term(self):
    self.qubit_operator.remove_term([(1, 'Z'), (3, 'X'), (8, 'Z')])
    pauli_string = pauli_data.PauliString(
        n_qubits, 0.5, [(1, 'X'), (3, 'Y'), (8, 'Z')])
    qubit_operator = pauli_data.QubitOperator(self.n_qubits)
    qubit_operator.add_term(self.pauli_string)
    self.assertTrue(self.qubit_operator.is_identical_operator(
        qubit_operator))


if __name__ == "__main__":
  unittest.main()
