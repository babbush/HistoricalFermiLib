"""Tests for _bravyi_kitaev.py."""
from __future__ import absolute_import
import unittest
import numpy

from projectqtemp.ops import _fermion_operator as fo
from projectqtemp.ops._fermion_operator import (FermionOperator, one_body_term,
                                                number_operator)

from transforms._bravyi_kitaev import bravyi_kitaev
from transforms._jordan_wigner import jordan_wigner
from transforms._conversion import eigenspectrum, get_sparse_operator

class BravyiKitaevTransformTest(unittest.TestCase):

    # add an identity test - I suspect that will fail

    def test_bravyi_kitaev_transform(self):
        # Check that the QubitOperators are two-term.
        lowering = bravyi_kitaev(FermionOperator(((3, 0),)))
        raising = bravyi_kitaev(FermionOperator(((3, 1),)))
        self.assertEqual(len(raising), 2)
        self.assertEqual(len(lowering), 2)

        #  Test the locality invariant for N=2^d qubits
        # (c_j majorana is always log2N+1 local on qubits)
        n_qubits = 16
        invariant = numpy.log2(n_qubits) + 1
        for index in range(n_qubits):
            operator = bravyi_kitaev(FermionOperator(((index, 0),)), n_qubits)
            qubit_terms = operator.terms.items()  # Get the majorana terms.

            for item in qubit_terms:
                term = item[1]

                #  Identify the c majorana terms by real
                #  coefficients and check their length.
                if not isinstance(term.coefficient, complex):
                    self.assertEqual(len(term), invariant)

        #  Hardcoded coefficient test on 16 qubits
        lowering = bravyi_kitaev(FermionOperator(((9, 0),)), n_qubits)
        raising = bravyi_kitaev(FermionOperator(((9, 1),)), n_qubits)

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

        a = FermionOperator(((1, 0),))
        a_dag = FermionOperator(((1, 1),))

        c = a + a_dag
        d = 1j * (a_dag - a)

        c_spins = [jordan_wigner(c), bravyi_kitaev(c)]
        d_spins = [jordan_wigner(d), bravyi_kitaev(d)]

        c_sparse = [get_sparse_operator(c_spins[0]),
                    get_sparse_operator(c_spins[1])]
        d_sparse = [get_sparse_operator(d_spins[0]),
                    get_sparse_operator(d_spins[1])]

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
        fo = FermionOperator(((3, 1),))

        jw = jordan_wigner(fo)
        bk = bravyi_kitaev(fo)

        jw_spectrum = eigenspectrum(jw)
        bk_spectrum = eigenspectrum(bk)

        self.assertAlmostEqual(0., numpy.amax(numpy.absolute(jw_spectrum -
                                                             bk_spectrum)))

    def test_bk_jw_integration_original(self):
        # Initialize a random fermionic operator.
        n_qubits = 5
        fermion_operator = FermionOperator(((3, 1), (2, 1), (1, 0), (0, 0)),
                                           -4.3)
        fermion_operator += FermionOperator(((3, 1), (1, 0)), 8.17)
        fermion_operator += 3.2 * fo.fermion_identity()
        # fermion_operator **= 3

        # Map to qubits and compare matrix versions.
        jw_qubit_operator = jordan_wigner(fermion_operator)
        bk_qubit_operator = bravyi_kitaev(fermion_operator)

        # Diagonalize and make sure the spectra are the same.
        jw_spectrum = eigenspectrum(jw_qubit_operator)
        bk_spectrum = eigenspectrum(bk_qubit_operator)
        self.assertAlmostEqual(0., numpy.amax(numpy.absolute(jw_spectrum -
                                                             bk_spectrum)))

if __name__ == '__main__':
    unittest.main()