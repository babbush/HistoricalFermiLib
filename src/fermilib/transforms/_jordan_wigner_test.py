"""Tests for _jordan_wigner.py."""
from __future__ import absolute_import
import numpy
import unittest

from fermilib.ops import (FermionOperator,
                          InteractionOperator,
                          number_operator)
from fermilib.transforms import jordan_wigner

from projectqtemp.ops import qubit_identity, QubitOperator

from transforms._jordan_wigner import jordan_wigner
from transforms._jordan_wigner_interaction_op import (jordan_wigner_one_body,
                                                      jordan_wigner_two_body)


class JordanWignerTransformTest(unittest.TestCase):
    def setUp(self):
        self.n_qubits = 5

    def test_bad_input(self):
        with self.assertRaises(TypeError):
            jordan_wigner(3)

    def test_transform_raise3(self):
        raising = jordan_wigner(FermionOperator(((3, 1),)))
        self.assertEqual(len(raising.terms), 3)  # identity is also there

        correct_operators_x = ((0, 'Z'), (1, 'Z'), (2, 'Z'), (3, 'X'))
        correct_operators_y = ((0, 'Z'), (1, 'Z'), (2, 'Z'), (3, 'Y'))
        qtermx = QubitOperator(correct_operators_x, 0.5)
        qtermy = QubitOperator(correct_operators_y, -0.5j)

        self.assertEqual(raising.terms[correct_operators_x], 0.5)
        self.assertEqual(raising.terms[correct_operators_y], -0.5j)
        self.assertTrue(raising.isclose(qtermx + qtermy))

    def test_transform_raise1(self):
        raising = jordan_wigner(FermionOperator(((1, 1),)))

        correct_operators_x = ((0, 'Z'), (1, 'X'))
        correct_operators_y = ((0, 'Z'), (1, 'Y'))
        qtermx = QubitOperator(correct_operators_x, 0.5)
        qtermy = QubitOperator(correct_operators_y, -0.5j)

        self.assertEqual(raising.terms[correct_operators_x], 0.5)
        self.assertEqual(raising.terms[correct_operators_y], -0.5j)
        self.assertTrue(raising.isclose(qtermx + qtermy))

    def test_transform_lower3(self):
        lowering = jordan_wigner(FermionOperator(((3, 0),)))

        correct_operators_x = ((0, 'Z'), (1, 'Z'), (2, 'Z'), (3, 'X'))
        correct_operators_y = ((0, 'Z'), (1, 'Z'), (2, 'Z'), (3, 'Y'))
        qtermx = QubitOperator(correct_operators_x, 0.5)
        qtermy = QubitOperator(correct_operators_y, 0.5j)

        self.assertEqual(lowering.terms[correct_operators_x], 0.5)
        self.assertEqual(lowering.terms[correct_operators_y], 0.5j)
        self.assertTrue(lowering.isclose(qtermx + qtermy))

    def test_transform_lower2(self):
        lowering = jordan_wigner(FermionOperator(((2, 0),)))

        correct_operators_x = ((0, 'Z'), (1, 'Z'), (2, 'X'))
        correct_operators_y = ((0, 'Z'), (1, 'Z'), (2, 'Y'))
        qtermx = QubitOperator(correct_operators_x, 0.5)
        qtermy = QubitOperator(correct_operators_y, 0.5j)

        self.assertEqual(lowering.terms[correct_operators_x], 0.5)
        self.assertEqual(lowering.terms[correct_operators_y], 0.5j)
        self.assertTrue(lowering.isclose(qtermx + qtermy))

    def test_transform_lower1(self):
        lowering = jordan_wigner(FermionOperator(((1, 0),)))

        correct_operators_x = ((0, 'Z'), (1, 'X'))
        correct_operators_y = ((0, 'Z'), (1, 'Y'))
        qtermx = QubitOperator(correct_operators_x, 0.5)
        qtermy = QubitOperator(correct_operators_y, 0.5j)

        self.assertEqual(lowering.terms[correct_operators_x], 0.5)
        self.assertEqual(lowering.terms[correct_operators_y], 0.5j)
        self.assertTrue(lowering.isclose(qtermx + qtermy))

    def test_transform_lower0(self):
        lowering = jordan_wigner(FermionOperator(((0, 0),)))

        correct_operators_x = ((0, 'X'),)
        correct_operators_y = ((0, 'Y'),)
        qtermx = QubitOperator(correct_operators_x, 0.5)
        qtermy = QubitOperator(correct_operators_y, 0.5j)

        self.assertEqual(lowering.terms[correct_operators_x], 0.5)
        self.assertEqual(lowering.terms[correct_operators_y], 0.5j)
        self.assertTrue(lowering.isclose(qtermx + qtermy))

    def test_transform_raise3lower0(self):
        # recall that creation gets -1j on Y and annihilation gets +1j on Y.
        term = jordan_wigner(FermionOperator(((3, 1), (0, 0))))
        self.assertEqual(term.terms[((0, 'X'), (1, 'Z'), (2, 'Z'), (3, 'Y'))],
                         0.25 * 1 * -1j)
        self.assertEqual(term.terms[((0, 'Y'), (1, 'Z'), (2, 'Z'), (3, 'Y'))],
                         0.25 * 1j * -1j)
        self.assertEqual(term.terms[((0, 'Y'), (1, 'Z'), (2, 'Z'), (3, 'X'))],
                         0.25 * 1j * 1)
        self.assertEqual(term.terms[((0, 'X'), (1, 'Z'), (2, 'Z'), (3, 'X'))],
                         0.25 * 1 * 1)

    def test_transform_number(self):
        n = number_operator(self.n_qubits, 3)
        n_jw = jordan_wigner(n)
        self.assertEqual(n_jw.terms[((3, 'Z'),)], -0.5)
        self.assertEqual(n_jw.terms[()], 0.5)
        self.assertEqual(len(n_jw.terms), 2)

    def test_ccr_offsite_even_ca(self):
        c2 = FermionOperator(((2, 1),))
        a4 = FermionOperator(((4, 0),))

        self.assertTrue((c2 * a4).normal_ordered().isclose(
            (-a4 * c2).normal_ordered()))
        self.assertTrue(jordan_wigner(c2 * a4).isclose(
            jordan_wigner(-a4 * c2)))

    def test_ccr_offsite_odd_ca(self):
        c1 = FermionOperator(((1, 1),))
        a4 = FermionOperator(((4, 0),))
        self.assertTrue((c1 * a4).normal_ordered().isclose(
            (-a4 * c1).normal_ordered()))

        self.assertTrue(jordan_wigner(c1 * a4).isclose(
            jordan_wigner(-a4 * c1)))

    def test_ccr_offsite_even_cc(self):
        c2 = FermionOperator(((2, 1),))
        c4 = FermionOperator(((4, 1),))
        self.assertTrue((c2 * c4).normal_ordered().isclose(
            (-c4 * c2).normal_ordered()))

        self.assertTrue(jordan_wigner(c2 * c4).isclose(
            jordan_wigner(-c4 * c2)))

    def test_ccr_offsite_odd_cc(self):
        c1 = FermionOperator(((1, 1),))
        c4 = FermionOperator(((4, 1),))
        self.assertTrue((c1 * c4).normal_ordered().isclose(
            (-c4 * c1).normal_ordered()))

        self.assertTrue(jordan_wigner(c1 * c4).isclose(
            jordan_wigner(-c4 * c1)))

    def test_ccr_offsite_even_aa(self):
        a2 = FermionOperator(((2, 0),))
        a4 = FermionOperator(((4, 0),))
        self.assertTrue((a2 * a4).normal_ordered().isclose(
            (-a4 * a2).normal_ordered()))

        self.assertTrue(jordan_wigner(a2 * a4).isclose(
            jordan_wigner(-a4 * a2)))

    def test_ccr_offsite_odd_aa(self):
        a1 = FermionOperator(((1, 0),))
        a4 = FermionOperator(((4, 0),))
        self.assertTrue((a1 * a4).normal_ordered().isclose(
            (-a4 * a1).normal_ordered()))

        self.assertTrue(jordan_wigner(a1 * a4).isclose(
            jordan_wigner(-a4 * a1)))

    def test_ccr_onsite(self):
        c1 = FermionOperator(((1, 1),))
        a1 = fo.hermitian_conjugated(c1)
        self.assertTrue((c1 * a1).normal_ordered().isclose(
            fo.fermion_identity() - (a1 * c1).normal_ordered()))
        self.assertTrue(jordan_wigner(c1 * a1).isclose(
            qubit_identity() - jordan_wigner(a1 * c1)))

    def test_jordan_wigner_transform_op(self):
        n = number_operator(self.n_qubits)
        n_jw = jordan_wigner(n)
        self.assertEqual(self.n_qubits + 1, len(n_jw.terms))
        self.assertEqual(self.n_qubits / 2., n_jw.terms[()])
        for qubit in range(self.n_qubits):
            operators = ((qubit, 'Z'),)
            self.assertEqual(n_jw.terms[operators], -0.5)


class InteractionOperatorsJWTest(unittest.TestCase):

    def setUp(self):
        self.n_qubits = 5
        self.constant = 0.
        self.one_body = numpy.zeros((self.n_qubits, self.n_qubits), float)
        self.two_body = numpy.zeros((self.n_qubits, self.n_qubits,
                                     self.n_qubits, self.n_qubits), float)
        self.interaction_operator = InteractionOperator(self.constant,
                                                        self.one_body,
                                                        self.two_body)

    def test_jordan_wigner_one_body(self):
        # Make sure it agrees with jordan_wigner(FermionTerm).
        for p in range(self.n_qubits):
            for q in range(self.n_qubits):
                # Get test qubit operator.
                test_operator = jordan_wigner_one_body(p, q)

                # Get correct qubit operator.
                fermion_term = FermionOperator(((p, 1), (q, 0)))
                correct_op = jordan_wigner(fermion_term)
                hermitian_conjugate = fo.hermitian_conjugated(fermion_term)
                if not fermion_term.isclose(hermitian_conjugate):
                    correct_op += jordan_wigner(hermitian_conjugate)

                self.assertTrue(test_operator.isclose(correct_op))

    def test_jordan_wigner_two_body(self):
        # Make sure it agrees with jordan_wigner(FermionTerm).
        for p in range(self.n_qubits):
            for q in range(self.n_qubits):
                for r in range(self.n_qubits):
                    for s in range(self.n_qubits):
                        # Get test qubit operator.
                        test_operator = jordan_wigner_two_body(p, q, r, s)

                        # Get correct qubit operator.
                        fermion_term = FermionOperator(((p, 1), (q, 1),
                                                        (r, 0), (s, 0)))
                        correct_op = jordan_wigner(fermion_term)
                        hermitian_conjugate = fo.hermitian_conjugated(
                            fermion_term)
                        if not fermion_term.isclose(hermitian_conjugate):
                            correct_op += jordan_wigner(hermitian_conjugate)

                        self.assertTrue(test_operator.isclose(correct_op))


if __name__ == '__main__':
    unittest.main()
