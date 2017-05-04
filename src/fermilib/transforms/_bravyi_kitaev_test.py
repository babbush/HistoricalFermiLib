"""Tests for _bravyi_kitaev.py."""
from __future__ import absolute_import
import unittest
import numpy

from fermilib.ops import FermionOperator, number_operator

from fermilib.transforms import (bravyi_kitaev,
                                 get_eigenspectrum,
                                 get_sparse_operator,
                                 jordan_wigner)

from projectqtemp.ops import QubitOperator


class BravyiKitaevTransformTest(unittest.TestCase):

    def test_bravyi_kitaev_transform(self):
        # Check that the QubitOperators are two-term.
        lowering = bravyi_kitaev(FermionOperator(((3, 0),)))
        raising = bravyi_kitaev(FermionOperator(((3, 1),)))
        self.assertEqual(len(raising.terms), 2)
        self.assertEqual(len(lowering.terms), 2)

        #  Test the locality invariant for N=2^d qubits
        # (c_j majorana is always log2N+1 local on qubits)
        n_qubits = 16
        invariant = numpy.log2(n_qubits) + 1
        for index in range(n_qubits):
            operator = bravyi_kitaev(FermionOperator(((index, 0),)), n_qubits)
            qubit_terms = operator.terms.items()  # Get the majorana terms.

            for item in qubit_terms:
                coeff = item[1]

                #  Identify the c majorana terms by real
                #  coefficients and check their length.
                if not isinstance(coeff, complex):
                    self.assertEqual(len(item[0]), invariant)

        #  Hardcoded coefficient test on 16 qubits
        lowering = bravyi_kitaev(FermionOperator(((9, 0),)), n_qubits)
        raising = bravyi_kitaev(FermionOperator(((9, 1),)), n_qubits)

        correct_operators_c = ((7, 'Z'), (8, 'Z'), (9, 'X'),
                               (11, 'X'), (15, 'X'))
        correct_operators_d = ((7, 'Z'), (9, 'Y'), (11, 'X'), (15, 'X'))

        self.assertEqual(lowering.terms[correct_operators_c], 0.5)
        self.assertEqual(lowering.terms[correct_operators_d], 0.5j)
        self.assertEqual(raising.terms[correct_operators_d], -0.5j)
        self.assertEqual(raising.terms[correct_operators_c], 0.5)

    def test_bk_identity(self):
        self.assertTrue(bravyi_kitaev(FermionOperator(())).isclose(
                        QubitOperator(())))

    def test_bk_jw_number_operator(self):
        # Check if number operator has the same spectrum in both
        # BK and JW representations
        n = number_operator(1, 0)
        jw_n = jordan_wigner(n)
        bk_n = bravyi_kitaev(n)

        # Diagonalize and make sure the spectra are the same.
        jw_spectrum = get_eigenspectrum(jw_n)
        bk_spectrum = get_eigenspectrum(bk_n)

        self.assertAlmostEqual(0., numpy.amax(
            numpy.absolute(jw_spectrum - bk_spectrum)))

    def test_bk_jw_number_operators(self):
        # Check if a number operator has the same spectrum in both
        # JW and BK representations
        n_qubits = 2
        n1 = number_operator(n_qubits, 0)
        n2 = number_operator(n_qubits, 1)
        n = n1 + n2

        jw_n = jordan_wigner(n)
        bk_n = bravyi_kitaev(n)

        # Diagonalize and make sure the spectra are the same.
        jw_spectrum = get_eigenspectrum(jw_n)
        bk_spectrum = get_eigenspectrum(bk_n)

        self.assertAlmostEqual(0., numpy.amax(
            numpy.absolute(jw_spectrum - bk_spectrum)))

    def test_bk_jw_number_operator_scaled(self):
        # Check if number operator has the same spectrum in both
        # JW and BK representations
        n_qubits = 1
        n = number_operator(n_qubits, 0, coefficient=2)  # eigenspectrum (0,2)
        jw_n = jordan_wigner(n)
        bk_n = bravyi_kitaev(n)

        # Diagonalize and make sure the spectra are the same.
        jw_spectrum = get_eigenspectrum(jw_n)
        bk_spectrum = get_eigenspectrum(bk_n)

        self.assertAlmostEqual(0., numpy.amax(
                               numpy.absolute(jw_spectrum - bk_spectrum)))

    def test_bk_jw_hopping_operator(self):
        # Check if the spectrum fits for a single hoppping operator
        n_qubits = 5
        ho = FermionOperator(((1, 1), (4, 0))) + FermionOperator(
            ((4, 1), (1, 0)))
        jw_ho = jordan_wigner(ho)
        bk_ho = bravyi_kitaev(ho)

        # Diagonalize and make sure the spectra are the same.
        jw_spectrum = get_eigenspectrum(jw_ho)
        bk_spectrum = get_eigenspectrum(bk_ho)

        self.assertAlmostEqual(0., numpy.amax(
                               numpy.absolute(jw_spectrum - bk_spectrum)))

    def test_bk_jw_majoranas(self):
        # Check if the Majorana operators have the same spectrum
        # irrespectively of the transform.
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

        c_spectrum = [get_eigenspectrum(c_spins[0]),
                      get_eigenspectrum(c_spins[1])]
        d_spectrum = [get_eigenspectrum(d_spins[0]),
                      get_eigenspectrum(d_spins[1])]

        self.assertAlmostEqual(0., numpy.amax(numpy.absolute(d_spectrum[0] -
                                                             d_spectrum[1])))

    def test_bk_jw_integration(self):
        # This is a legacy test, which was a minimal failing example when
        # optimization for hermitian operators was used.
        n_qubits = 4

        # Minimal failing example:
        fo = FermionOperator(((3, 1),))

        jw = jordan_wigner(fo)
        bk = bravyi_kitaev(fo)

        jw_spectrum = get_eigenspectrum(jw)
        bk_spectrum = get_eigenspectrum(bk)

        self.assertAlmostEqual(0., numpy.amax(numpy.absolute(jw_spectrum -
                                                             bk_spectrum)))

    def test_bk_jw_integration_original(self):
        # This is a legacy test, which was an example proposed by Ryan,
        # failing when optimization for hermitian operators was used.
        n_qubits = 5
        fermion_operator = FermionOperator(((3, 1), (2, 1), (1, 0), (0, 0)),
                                           -4.3)
        fermion_operator += FermionOperator(((3, 1), (1, 0)), 8.17)
        fermion_operator += 3.2 * FermionOperator()

        # Map to qubits and compare matrix versions.
        jw_qubit_operator = jordan_wigner(fermion_operator)
        bk_qubit_operator = bravyi_kitaev(fermion_operator)

        # Diagonalize and make sure the spectra are the same.
        jw_spectrum = get_eigenspectrum(jw_qubit_operator)
        bk_spectrum = get_eigenspectrum(bk_qubit_operator)
        self.assertAlmostEqual(0., numpy.amax(numpy.absolute(jw_spectrum -
                                                             bk_spectrum)))


if __name__ == '__main__':
    unittest.main()
