"""Tests  _reverse_jordan_wigner.py."""
from __future__ import absolute_import
import unittest

from fermilib.ops import FermionOperator, normal_ordered
from fermilib.transforms import jordan_wigner, reverse_jordan_wigner
from projectqtemp.ops._qubit_operator import QubitOperator, QubitOperatorError


class ReverseJWTest(unittest.TestCase):

    def setUp(self):
        self.coefficient = 0.5
        self.operators = ((1, 'X'), (3, 'Y'), (8, 'Z'))
        self.term = QubitOperator(self.operators, self.coefficient)
        self.identity = QubitOperator(())
        self.coefficient_a = 6.7j
        self.coefficient_b = -88.
        self.operators_a = ((3, 'Z'), (1, 'Y'), (4, 'Y'))
        self.operators_b = ((2, 'X'), (3, 'Y'))
        self.operator_a = QubitOperator(self.operators_a, self.coefficient_a)
        self.operator_b = QubitOperator(self.operators_b, self.coefficient_b)
        self.operator_ab = self.operator_a + self.operator_b
        self.qubit_operator = QubitOperator(
                ((1, 'X'), (3, 'Y'), (8, 'Z')), 0.5)
        self.qubit_operator += QubitOperator(
                ((1, 'Z'), (3, 'X'), (8, 'Z')), 1.2)

    def test_identity_jwterm(self):
        self.assertTrue(FermionOperator(()).isclose(
            reverse_jordan_wigner(QubitOperator(()))))

    def test_x(self):
        pauli_x = QubitOperator(((2, 'X'),))
        transmed_x = reverse_jordan_wigner(pauli_x)
        print transmed_x
        retransmed_x = jordan_wigner(transmed_x)
        self.assertTrue(pauli_x.isclose(retransmed_x))


if __name__ == '__main__':
    unittest.main()
