"""Tests for interaction_rdms.py."""
from __future__ import absolute_import

import unittest

from fermilib.config import *
from fermilib.ops import MolecularData
from fermilib.transforms import jordan_wigner


class InteractionRDMTest(unittest.TestCase):

    def setUp(self):
        geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 0.7414))]
        basis = 'sto-3g'
        multiplicity = 1
        filename = THIS_DIRECTORY + '/tests/testdata/H2_sto-3g_singlet'
        self.molecule = MolecularData(
            geometry, basis, multiplicity, filename=filename)
        self.molecule.load()
        self.cisd_energy = self.molecule.cisd_energy
        self.rdm = self.molecule.get_molecular_rdm()
        self.hamiltonian = self.molecule.get_molecular_hamiltonian()

    def test_get_qubit_expectations(self):
        qubit_operator = jordan_wigner(self.hamiltonian)
        qubit_expectations = self.rdm.get_qubit_expectations(qubit_operator)

        test_energy = qubit_operator.terms[()]
        for qubit_term in qubit_expectations.terms:
            term_coefficient = qubit_operator.terms[qubit_term]
            test_energy += (term_coefficient *
                            qubit_expectations.terms[qubit_term])

        self.assertLess(abs(test_energy - self.cisd_energy), EQ_TOLERANCE)

    def test_get_molecular_operator_expectation(self):
        expectation = self.rdm.expectation(self.hamiltonian)
        self.assertAlmostEqual(expectation, self.cisd_energy, places=7)


# Test.
if __name__ == '__main__':
    unittest.main()
