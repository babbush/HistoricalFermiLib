"""Tests for transformations.py."""
from __future__ import absolute_import

import copy
import unittest

import numpy

from fermilib import fermion_operators as fo
from fermilib.fermion_operators import (FermionTerm, FermionOperator,
                                        number_operator, one_body_term,
                                        two_body_term)
from fermilib.interaction_operators import InteractionOperator
from fermilib.qubit_operators import QubitTerm, QubitOperator, qubit_identity
from transforms import (bravyi_kitaev, eigenspectrum, get_fermion_operator,
                        get_interaction_operator, jordan_wigner,
                        jordan_wigner_one_body, jordan_wigner_two_body,
                        reverse_jordan_wigner, reverse_jordan_wigner_term)


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


class BravyiKitaevTransformTest(unittest.TestCase):

    def test_bravyi_kitaev_transform(self):
        # Check that the QubitOperators are two-term.
        lowering = bravyi_kitaev(FermionTerm([(3, 0)]))
        raising = bravyi_kitaev(FermionTerm([(3, 1)]))
        self.assertEqual(len(raising), 2)
        self.assertEqual(len(lowering), 2)

        #  Test the locality invariant for N=2^d qubits
        # (c_j majorana is always log2N+1 local on qubits)
        n_qubits = 16
        invariant = numpy.log2(n_qubits) + 1
        for index in range(n_qubits):
            operator = bravyi_kitaev(FermionTerm([(index, 0)]), n_qubits)
            qubit_terms = operator.terms.items()  # Get the majorana terms.

            for item in qubit_terms:
                term = item[1]

                #  Identify the c majorana terms by real
                #  coefficients and check their length.
                if not isinstance(term.coefficient, complex):
                    self.assertEqual(len(term), invariant)

        #  Hardcoded coefficient test on 16 qubits
        lowering = bravyi_kitaev(FermionTerm([(9, 0)]), n_qubits)
        raising = bravyi_kitaev(FermionTerm([(9, 1)]), n_qubits)

        correct_operators_c = [
            (7, 'Z'), (8, 'Z'), (9, 'X'), (11, 'X'), (15, 'X')]
        correct_operators_d = [(7, 'Z'), (9, 'Y'), (11, 'X'), (15, 'X')]

        self.assertEqual(lowering[correct_operators_c], 0.5)
        self.assertEqual(lowering[correct_operators_d], 0.5j)
        self.assertEqual(raising[correct_operators_d], -0.5j)
        self.assertEqual(raising[correct_operators_c], 0.5)

    def test_bk_jw_number_operator(self):
        # Check if number operator has the same spectrum in both
        # representations
        n = fo.number_operator(1, 0)
        jw_n = jordan_wigner(n)
        bk_n = bravyi_kitaev(n)

        # Diagonalize and make sure the spectra are the same.
        jw_spectrum = eigenspectrum(jw_n)
        bk_spectrum = eigenspectrum(bk_n)

        self.assertAlmostEqual(0., numpy.amax(
            numpy.absolute(jw_spectrum - bk_spectrum)))

    def test_bk_jw_number_operators(self):
        # Check if number operator has the same spectrum in both
        # representations
        n_qubits = 2
        n1 = fo.number_operator(n_qubits, 0)
        n2 = fo.number_operator(n_qubits, 1)
        n = n1 + n2

        jw_n = jordan_wigner(n)
        bk_n = bravyi_kitaev(n)

        # Diagonalize and make sure the spectra are the same.
        jw_spectrum = eigenspectrum(jw_n)
        bk_spectrum = eigenspectrum(bk_n)

        self.assertAlmostEqual(0., numpy.amax(
            numpy.absolute(jw_spectrum - bk_spectrum)))

    def test_bk_jw_number_operator_scaled(self):
        # Check if number operator has the same spectrum in both
        # representations
        n_qubits = 1
        n = number_operator(n_qubits, 0, coefficient=2)  # eigenspectrum (0,2)
        jw_n = jordan_wigner(n)
        bk_n = bravyi_kitaev(n)

        # Diagonalize and make sure the spectra are the same.
        jw_spectrum = eigenspectrum(jw_n)
        bk_spectrum = eigenspectrum(bk_n)

        self.assertAlmostEqual(0., numpy.amax(
                               numpy.absolute(jw_spectrum - bk_spectrum)))

    def test_bk_jw_hopping_operator(self):
        # Check if the spectrum fits for a single hoppping operator
        n_qubits = 5
        ho = one_body_term(1, 4) + one_body_term(4, 1)
        jw_ho = jordan_wigner(ho)
        bk_ho = bravyi_kitaev(ho)

        # Diagonalize and make sure the spectra are the same.
        jw_spectrum = eigenspectrum(jw_ho)
        bk_spectrum = eigenspectrum(bk_ho)

        self.assertAlmostEqual(0., numpy.amax(
                               numpy.absolute(jw_spectrum - bk_spectrum)))

    def test_bk_jw_majoranas(self):
        n_qubits = 7

        a = FermionTerm([(1, 0)])
        a_dag = FermionTerm([(1, 1)])

        c = a + a_dag
        d = 1j * (a_dag - a)

        c_spins = [jordan_wigner(c), bravyi_kitaev(c)]
        d_spins = [jordan_wigner(d), bravyi_kitaev(d)]

        c_sparse = [c_spins[0].get_sparse_operator(),
                    c_spins[1].get_sparse_operator()]
        d_sparse = [d_spins[1].get_sparse_operator(),
                    d_spins[1].get_sparse_operator()]

        c_spectrum = [eigenspectrum(c_spins[0]),
                      eigenspectrum(c_spins[1])]
        d_spectrum = [eigenspectrum(d_spins[0]),
                      eigenspectrum(d_spins[1])]

        # ^ Majoranas have the same spectra. Fine
        self.assertAlmostEqual(0., numpy.amax(numpy.absolute(d_spectrum[0] -
                                                             d_spectrum[1])))

    def test_bk_jw_integration(self):
        # Initialize a random fermionic operator.
        n_qubits = 4

        # Minimal failing example:
        fo = FermionTerm([(3, 1)])

        jw = jordan_wigner(fo)
        bk = bravyi_kitaev(fo)

        jw_spectrum = eigenspectrum(jw)
        bk_spectrum = eigenspectrum(bk)

        self.assertAlmostEqual(0., numpy.amax(numpy.absolute(jw_spectrum -
                                                             bk_spectrum)))

    def test_bk_jw_integration_original(self):
        # Initialize a random fermionic operator.
        n_qubits = 5
        fermion_operator = FermionTerm([(3, 1), (2, 1), (1, 0), (0, 0)], -4.3)
        fermion_operator += FermionTerm([(3, 1), (1, 0)], 8.17)
        fermion_operator += 3.2 * fo.fermion_identity()
        fermion_operator **= 3

        # Map to qubits and compare matrix versions.
        jw_qubit_operator = jordan_wigner(fermion_operator)
        bk_qubit_operator = bravyi_kitaev(fermion_operator)

        # Diagonalize and make sure the spectra are the same.
        jw_spectrum = eigenspectrum(jw_qubit_operator)
        bk_spectrum = eigenspectrum(bk_qubit_operator)
        self.assertAlmostEqual(0., numpy.amax(numpy.absolute(jw_spectrum -
                                                             bk_spectrum)))


class GetInteractionOperatorTest(unittest.TestCase):

    def test_get_molecular_operator(self):
        coefficient = 3.
        operators = [(2, 1), (3, 0), (0, 0), (3, 1)]
        term = FermionTerm(operators, coefficient)
        op = FermionOperator(term)

        molecular_operator = get_interaction_operator(op)
        fermion_operator = get_fermion_operator(molecular_operator)
        fermion_operator.normal_order()
        op.normal_order()
        self.assertEqual(op, fermion_operator)


class InteractionOperatorsTest(unittest.TestCase):

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
                fermion_term = FermionTerm([(p, 1), (q, 0)], 1.)
                correct_op = jordan_wigner(fermion_term)
                hermitian_conjugate = fermion_term.hermitian_conjugated()
                if fermion_term != hermitian_conjugate:
                    correct_op += jordan_wigner(hermitian_conjugate)

                self.assertEqual(test_operator, correct_op)

    def test_jordan_wigner_two_body(self):
        # Make sure it agrees with jordan_wigner(FermionTerm).
        for p in range(self.n_qubits):
            for q in range(self.n_qubits):
                for r in range(self.n_qubits):
                    for s in range(self.n_qubits):

                        # Get test qubit operator.
                        test_operator = jordan_wigner_two_body(p, q, r, s)

                        # Get correct qubit operator.
                        fermion_term = FermionTerm([(p, 1), (q, 1),
                                                    (r, 0), (s, 0)], 1.)
                        correct_op = jordan_wigner(fermion_term)
                        hermitian_conjugate = (
                            fermion_term.hermitian_conjugated())
                        if fermion_term != hermitian_conjugate:
                            correct_op += jordan_wigner(hermitian_conjugate)

                        self.assertEqual(test_operator, correct_op)


if __name__ == '__main__':
    unittest.main()
