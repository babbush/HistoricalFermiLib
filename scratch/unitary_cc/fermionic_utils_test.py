"""Tests for fermionic_utils."""

import unittest

import fermionic_utils
import hamiltonian_utils


class FermionicOperatorTest(unittest.TestCase):

  def test_initialization(self):
    operator = fermionic_utils.FermionicOperator(-1.0, [1, -2, 3])
    self.assertEqual(len(operator.fermionic_operator), 3)
    self.assertEqual(operator.fermionic_operator[0], 1)
    self.assertEqual(operator.fermionic_operator[1], -2)
    self.assertEqual(operator.fermionic_operator[2], 3)
    self.assertAlmostEqual(-1.0, operator.coefficient)

  def test_get_hermitian_conjugate(self):
    operator = fermionic_utils.FermionicOperator(1 + 2j, [1, -2, 3])
    hermitian_operator = operator.get_hermitian_conjugate()
    self.assertEqual(hermitian_operator.fermionic_operator[0], -3)
    self.assertEqual(hermitian_operator.fermionic_operator[1], 2)
    self.assertEqual(hermitian_operator.fermionic_operator[2], -1)
    self.assertAlmostEqual(1 - 2j, hermitian_operator.coefficient)


class JordanWignerTransform(unittest.TestCase):

  def test_jordan_wigner(self):
    operator = fermionic_utils.FermionicOperator(-1.0, [4, -2])
    # Correct result:
    string1 = hamiltonian_utils.PauliString(4, -0.25j, [3], [1], [2])
    string2 = hamiltonian_utils.PauliString(4, -0.25, [], [1, 3], [2])
    string3 = hamiltonian_utils.PauliString(4, -0.25, [1, 3], [], [2])
    string4 = hamiltonian_utils.PauliString(4, 0.25j, [1], [3], [2])
    # Compare
    result = fermionic_utils.jordan_wigner_transform(operator, 4)
    self.assertEqual(len(result), 4)
    compare = [False, False, False, False]
    for ii in range(4):
      for string in [string1, string2, string3, string4]:
        if string.is_identical_pauli(result[ii]):
          compare[ii] = True
          # Compare coefficient of identical tensor factors
          self.assertAlmostEqual(string.coefficient, result[ii].coefficient)
    # Check that all tensor factors are there
    self.assertTrue(all(compare))

  def test_not_enough_qubits(self):
    operator = fermionic_utils.FermionicOperator(1.0, [2, -3])
    with self.assertRaises(fermionic_utils.ErrorJordanWigner):
      fermionic_utils.jordan_wigner_transform(operator, 2)


if __name__ == '__main__':
  unittest.main()
