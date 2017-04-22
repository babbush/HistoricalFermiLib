"""Tests for _jordan_wigner.py."""
from __future__ import absolute_import
import unittest

from fermilib import fermion_operators as fo
from fermilib.fermion_operators import FermionTerm, number_operator
from fermilib.qubit_operators import qubit_identity, QubitTerm, QubitOperator

from ._jordan_wigner import jordan_wigner


class JordanWignerTransformTest(unittest.TestCase):
    def setUp(self):
        self.n_qubits = 5

    def test_transform_raise3(self):
        raising = jordan_wigner(FermionTerm([(3, 1)]))
        self.assertEqual(len(raising), 2)

        correct_operators_x = [(0, 'Z'), (1, 'Z'), (2, 'Z'), (3, 'X')]
        correct_operators_y = [(0, 'Z'), (1, 'Z'), (2, 'Z'), (3, 'Y')]

        self.assertEqual(raising[correct_operators_x], 0.5)
        self.assertEqual(raising[correct_operators_y], -0.5j)

    def test_transform_raise1(self):
        raising = jordan_wigner(FermionTerm([(1, 1)]))
        self.assertEqual(len(raising), 2)

        correct_operators_x = [(0, 'Z'), (1, 'X')]
        correct_operators_y = [(0, 'Z'), (1, 'Y')]

        self.assertEqual(raising[correct_operators_x], 0.5)
        self.assertEqual(raising[correct_operators_y], -0.5j)

    def test_transform_lower3(self):
        lowering = jordan_wigner(FermionTerm([(3, 0)]))
        self.assertEqual(len(lowering), 2)

        correct_operators_x = [(0, 'Z'), (1, 'Z'), (2, 'Z'), (3, 'X')]
        correct_operators_y = [(0, 'Z'), (1, 'Z'), (2, 'Z'), (3, 'Y')]
        qtermx = QubitTerm(correct_operators_x, 0.5)
        qtermy = QubitTerm(correct_operators_y, 0.5j)

        self.assertEqual(lowering[correct_operators_x], 0.5)
        self.assertEqual(lowering[correct_operators_y], 0.5j)
        self.assertEqual(lowering, QubitOperator([qtermx, qtermy]))

    def test_transform_lower2(self):
        lowering = jordan_wigner(FermionTerm([(2, 0)]))
        self.assertEqual(len(lowering), 2)

        correct_operators_x = [(0, 'Z'), (1, 'Z'), (2, 'X')]
        correct_operators_y = [(0, 'Z'), (1, 'Z'), (2, 'Y')]

        self.assertEqual(lowering[correct_operators_x], 0.5)
        self.assertEqual(lowering[correct_operators_y], 0.5j)

    def test_transform_lower1(self):
        lowering = jordan_wigner(FermionTerm([(1, 0)]))
        self.assertEqual(len(lowering), 2)

        correct_operators_x = [(0, 'Z'), (1, 'X')]
        correct_operators_y = [(0, 'Z'), (1, 'Y')]

        self.assertEqual(lowering[correct_operators_x], 0.5)
        self.assertEqual(lowering[correct_operators_y], 0.5j)

    def test_transform_lower0(self):
        lowering = jordan_wigner(FermionTerm([(0, 0)]))
        self.assertEqual(len(lowering), 2)

        correct_operators_x = [(0, 'X')]
        correct_operators_y = [(0, 'Y')]

        self.assertEqual(lowering[correct_operators_x], 0.5)
        self.assertEqual(lowering[correct_operators_y], 0.5j)

    def test_transform_raise3lower0(self):
        # recall that creation gets -1j on Y and annihilation gets +1j on Y.
        term = jordan_wigner(FermionTerm([(3, 1), (0, 0)]))
        self.assertEqual(term[((0, 'X'), (1, 'Z'), (2, 'Z'), (3, 'Y'))],
                         0.25 * 1 * -1j)
        self.assertEqual(term[((0, 'Y'), (1, 'Z'), (2, 'Z'), (3, 'Y'))],
                         0.25 * 1j * -1j)
        self.assertEqual(term[((0, 'Y'), (1, 'Z'), (2, 'Z'), (3, 'X'))],
                         0.25 * 1j * 1)
        self.assertEqual(term[((0, 'X'), (1, 'Z'), (2, 'Z'), (3, 'X'))],
                         0.25 * 1 * 1)

    def test_transform_number(self):
        n = number_operator(self.n_qubits, 3)
        n_jw = jordan_wigner(n)
        self.assertEqual(n_jw[[(3, 'Z')]], -0.5)
        self.assertEqual(n_jw[[]], 0.5)
        self.assertEqual(len(n_jw), 2)

    def test_ccr_offsite_even_ca(self):
        c2 = FermionTerm([(2, 1)])
        a4 = FermionTerm([(4, 0)])
        self.assertEqual((c2 * a4).normal_ordered(),
                         (-a4 * c2).normal_ordered())
        self.assertEqual(jordan_wigner(c2 * a4), jordan_wigner(-a4 * c2))

    def test_ccr_offsite_odd_ca(self):
        c1 = FermionTerm([(1, 1)])
        a4 = FermionTerm([(4, 0)])
        self.assertEqual((c1 * a4).normal_ordered(),
                         (-a4 * c1).normal_ordered())

        self.assertEqual(jordan_wigner(c1 * a4), jordan_wigner(-a4 * c1))

    def test_ccr_offsite_even_cc(self):
        c2 = FermionTerm([(2, 1)])
        c4 = FermionTerm([(4, 1)])
        self.assertEqual((c2 * c4).normal_ordered(),
                         (-c4 * c2).normal_ordered())

        self.assertEqual(jordan_wigner(c2 * c4), jordan_wigner(-c4 * c2))

    def test_ccr_offsite_odd_cc(self):
        c1 = FermionTerm([(1, 1)])
        c4 = FermionTerm([(4, 1)])
        self.assertEqual((c1 * c4).normal_ordered(),
                         (-c4 * c1).normal_ordered())

        self.assertEqual(jordan_wigner(c1 * c4), jordan_wigner(-c4 * c1))

    def test_ccr_offsite_even_aa(self):
        a2 = FermionTerm([(2, 0)])
        a4 = FermionTerm([(4, 0)])
        self.assertEqual((a2 * a4).normal_ordered(),
                         (-a4 * a2).normal_ordered())

        self.assertEqual(jordan_wigner(a2 * a4), jordan_wigner(-a4 * a2))

    def test_ccr_offsite_odd_aa(self):
        a1 = FermionTerm([(1, 0)])
        a4 = FermionTerm([(4, 0)])
        self.assertEqual((a1 * a4).normal_ordered(),
                         (-a4 * a1).normal_ordered())

        self.assertEqual(jordan_wigner(a1 * a4), jordan_wigner(-a4 * a1))

    def test_ccr_onsite(self):
        c1 = FermionTerm([(1, 1)])
        a1 = c1.hermitian_conjugated()
        self.assertEqual((c1 * a1).normal_ordered(),
                         fo.fermion_identity() - (a1 * c1).normal_ordered())
        self.assertEqual(jordan_wigner(c1 * a1),
                         qubit_identity() - jordan_wigner(a1 * c1))

    def test_jordan_wigner_transform_op(self):
        n = number_operator(self.n_qubits)
        n_jw = jordan_wigner(n)
        self.assertEqual(self.n_qubits + 1, len(n_jw))
        self.assertEqual(self.n_qubits / 2., n_jw[[]])
        for qubit in range(self.n_qubits):
            operators = [(qubit, 'Z')]
            self.assertEqual(n_jw[operators], -0.5)
