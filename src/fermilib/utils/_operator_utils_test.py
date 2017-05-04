"""Tests for operator_utils."""
from __future__ import absolute_import

import numpy
import unittest

from fermilib.ops import *
from fermilib.transforms import jordan_wigner, get_interaction_operator
from fermilib.utils import eigenspectrum, is_identity, count_qubits


class OperatorUtilsTest(unittest.TestCase):

    def setUp(self):
        self.n_qubits = 5
        self.fermion_term = FermionOperator('1^ 2^ 3 4', -3.17)
        self.fermion_operator = self.fermion_term + hermitian_conjugated(
                self.fermion_term)
        self.qubit_operator = jordan_wigner(self.fermion_operator)
        self.interaction_operator = get_interaction_operator(
                self.fermion_operator)

    def test_n_qubits(self):
        self.assertEqual(self.n_qubits,
                         count_qubits(self.fermion_term))
        self.assertEqual(self.n_qubits,
                         count_qubits(self.fermion_operator))
        self.assertEqual(self.n_qubits,
                         count_qubits(self.qubit_operator))
        self.assertEqual(self.n_qubits,
                         count_qubits(self.interaction_operator))

    def test_eigenspectrum(self):
        fermion_eigenspectrum = eigenspectrum(self.fermion_operator)
        qubit_eigenspectrum = eigenspectrum(self.qubit_operator)
        interaction_eigenspectrum = eigenspectrum(self.interaction_operator)
        for i in range(2 ** self.n_qubits):
            self.assertAlmostEqual(fermion_eigenspectrum[i],
                                   qubit_eigenspectrum[i])
            self.assertAlmostEqual(fermion_eigenspectrum[i],
                                   interaction_eigenspectrum[i])


if __name__ == '__main__':
    unittest.main()


    #def test_isidentity_identity():
    #    assert FermionOperator().is_identity()
    #
    #
    #def test_isidentity_mulidentity():
    #    op = FermionOperator() * 2
    #    assert op.is_identity()
    #
    #
    #def test_isidentity_zero():
    #    op = 0 * FermionOperator() + FermionOperator('2^', 0.0)
    #    assert op.is_identity()
    #
    #
    #def test_isidentity_zeroX():
    #    op = -2 * FermionOperator() + FermionOperator('2', 0.0)
    #    assert op.is_identity()
    #
    #
    #def test_isidentity_IX():
    #    op = -2 * FermionOperator() + FermionOperator('0', 0.03j)
    #    assert not op.is_identity()


    #def test_nqubits_0():
    #    op = FermionOperator()
    #    assert op.n_qubits() == 0
    #
    #
    #def test_nqubits_1():
    #    op = FermionOperator('0', 3)
    #    assert op.n_qubits() == 1
    #
    #
    #def test_nqubits_doubledigit():
    #    op = FermionOperator('27 5^ 11^')
    #    assert op.n_qubits() == 28
    #
    #
    #def test_nqubits_multiterm():
    #    op = (FermionOperator() + FermionOperator('1 2 3') +
    #          FermionOperator())
    #    assert op.n_qubits() == 4
