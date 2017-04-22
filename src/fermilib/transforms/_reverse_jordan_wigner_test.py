"""Tests for _reverse_jordan_wigner.py."""
from __future__ import absolute_import
import unittest

from fermilib import fermion_operators as fo
from fermilib.qubit_operators import QubitTerm, QubitOperator
from ._reverse_jordan_wigner import reverse_jordan_wigner
from ._jordan_wigner import jordan_wigner


class ReverseJWTermTest(unittest.TestCase):

    def setUp(self):
        self.coefficient = 0.5
        self.operators = [(1, 'X'), (3, 'Y'), (8, 'Z')]
        self.term = QubitTerm(self.operators, self.coefficient)
        self.identity = QubitTerm()
        self.coefficient_a = 6.7j
        self.coefficient_b = -88.
        self.operators_a = [(3, 'Z'), (1, 'Y'), (4, 'Y')]
        self.operators_b = [(2, 'X'), (3, 'Y')]
        self.term_a = QubitTerm(self.operators_a, self.coefficient_a)
        self.term_b = QubitTerm(self.operators_b, self.coefficient_b)

        self.operator_a = QubitOperator(self.term_a)
        self.operator_b = QubitOperator(self.term_b)
        self.operator_ab = QubitOperator([self.term_a, self.term_b])

    def test_x(self):
        pauli_x = QubitTerm([(2, 'X')])
        transformed_x = reverse_jordan_wigner(pauli_x)
        retransformed_x = jordan_wigner(transformed_x)
        self.assertEqual(1, len(retransformed_x))
        self.assertEqual(QubitOperator(pauli_x), retransformed_x)

    def test_y(self):
        pauli_y = QubitTerm([(2, 'Y')])
        transformed_y = reverse_jordan_wigner(pauli_y)
        retransformed_y = jordan_wigner(transformed_y)
        self.assertEqual(1, len(retransformed_y))
        self.assertEqual(QubitOperator(pauli_y), retransformed_y)

    def test_z(self):
        pauli_z = QubitTerm([(2, 'Z')])
        transformed_z = reverse_jordan_wigner(pauli_z)

        expected_terms = [fo.fermion_identity(),
                          fo.FermionTerm([(2, 1), (2, 0)], -2.)]
        expected = fo.FermionOperator(expected_terms)
        self.assertEqual(transformed_z, expected)

        retransformed_z = jordan_wigner(transformed_z)
        self.assertEqual(1, len(retransformed_z))
        self.assertEqual(QubitOperator(pauli_z), retransformed_z)

    def test_identity(self):
        n_qubits = 5
        transformed_i = reverse_jordan_wigner(self.identity, n_qubits)
        expected_i_term = fo.fermion_identity()
        expected_i = fo.FermionOperator([expected_i_term])
        self.assertEqual(transformed_i, expected_i)

        retransformed_i = jordan_wigner(transformed_i)
        self.assertEqual(1, len(retransformed_i))
        self.assertEqual(QubitOperator(self.identity), retransformed_i)

    def test_yzxz(self):
        yzxz = QubitTerm([(0, 'Y'), (1, 'Z'), (2, 'X'), (3, 'Z')])
        transformed_yzxz = reverse_jordan_wigner(yzxz)
        retransformed_yzxz = jordan_wigner(transformed_yzxz)
        self.assertEqual(1, len(retransformed_yzxz))
        self.assertEqual(QubitOperator(yzxz), retransformed_yzxz)

    def test_term(self):
        transformed_term = reverse_jordan_wigner(self.term)
        retransformed_term = jordan_wigner(transformed_term)
        self.assertEqual(1, len(retransformed_term))
        self.assertEqual(QubitOperator(self.term),
                         retransformed_term)

    def test_xx(self):
        xx = QubitTerm([(3, 'X'), (4, 'X')], 2.)
        transformed_xx = reverse_jordan_wigner(xx)
        retransformed_xx = jordan_wigner(transformed_xx)

        expected1 = (fo.FermionTerm([(3, 1)], 2.) -
                     fo.FermionTerm([(3, 0)], 2.))
        expected2 = (fo.FermionTerm([(4, 1)], 1.) +
                     fo.FermionTerm([(4, 0)], 1.))
        expected = expected1 * expected2

        self.assertEqual(QubitOperator([xx]), retransformed_xx)
        self.assertEqual(transformed_xx.normal_ordered(),
                         expected.normal_ordered())

    def test_yy(self):
        yy = QubitTerm([(2, 'Y'), (3, 'Y')], 2.)
        transformed_yy = reverse_jordan_wigner(yy)
        retransformed_yy = jordan_wigner(transformed_yy)

        expected1 = -(fo.FermionTerm([(2, 1)], 2.) +
                      fo.FermionTerm([(2, 0)], 2.))
        expected2 = (fo.FermionTerm([(3, 1)]) -
                     fo.FermionTerm([(3, 0)]))
        expected = expected1 * expected2

        self.assertEqual(QubitOperator([yy]), retransformed_yy)
        self.assertEqual(transformed_yy.normal_ordered(),
                         expected.normal_ordered())

    def test_xy(self):
        xy = QubitTerm([(4, 'X'), (5, 'Y')], -2.j)
        transformed_xy = reverse_jordan_wigner(xy)
        retransformed_xy = jordan_wigner(transformed_xy)

        expected1 = -2j * (fo.FermionTerm([(4, 1)], 1j) -
                           fo.FermionTerm([(4, 0)], 1j))
        expected2 = (fo.FermionTerm([(5, 1)]) -
                     fo.FermionTerm([(5, 0)]))
        expected = expected1 * expected2

        self.assertEqual(QubitOperator([xy]), retransformed_xy)
        self.assertEqual(transformed_xy.normal_ordered(),
                         expected.normal_ordered())

    def test_yx(self):
        yx = QubitTerm([(0, 'Y'), (1, 'X')], -0.5)
        transformed_yx = reverse_jordan_wigner(yx)
        retransformed_yx = jordan_wigner(transformed_yx)

        expected1 = 1j * (fo.FermionTerm([(0, 1)]) +
                          fo.FermionTerm([(0, 0)]))
        expected2 = -0.5 * (fo.FermionTerm([(1, 1)]) +
                            fo.FermionTerm([(1, 0)]))
        expected = expected1 * expected2

        self.assertEqual(QubitOperator([yx]), retransformed_yx)
        self.assertEqual(transformed_yx.normal_ordered(),
                         expected.normal_ordered())


class ReverseJWOperatorTest(unittest.TestCase):

    def setUp(self):
        self.identity = QubitTerm()
        self.coefficient_a = 0.5
        self.coefficient_b = 1.2
        self.operators_a = ((1, 'X'), (3, 'Y'), (8, 'Z'))
        self.operators_b = ((1, 'Z'), (3, 'X'), (8, 'Z'))
        self.term_a = QubitTerm([(1, 'X'), (3, 'Y'), (8, 'Z')], 0.5)
        self.term_b = QubitTerm([(1, 'Z'), (3, 'X'), (8, 'Z')], 1.2)

        self.qubit_operator = QubitOperator([self.term_a, self.term_b])

    def test_reverse_jordan_wigner(self):
        transformed_operator = reverse_jordan_wigner(self.qubit_operator)
        retransformed_operator = jordan_wigner(transformed_operator)
        self.assertEqual(self.qubit_operator, retransformed_operator)

    def test_reverse_jw_linearity(self):
        term1 = QubitTerm([(0, 'X'), (1, 'Y')], -0.5)
        term2 = QubitTerm([(0, 'Y'), (1, 'X'), (2, 'Y'), (3, 'Y')], -1j)

        op12 = reverse_jordan_wigner(term1) - reverse_jordan_wigner(term2)
        self.assertEqual(op12, reverse_jordan_wigner(term1 - term2))
