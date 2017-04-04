"""Tests for interaction_rdms.py"""
import numpy
import unittest

from config import *
from molecular_data import MolecularData
from interaction_rdms import InteractionRDM
from run_psi4 import run_psi4


class InteractionRDMTest(unittest.TestCase):

  def setUp(self):
    geometry = [('H', (0, 0, 0)), ('H', (0, 0, 0.2))]
    molecule = MolecularData(geometry, 'sto-3g', 1, autosave=False)
    molecule = run_psi4(molecule,
                        run_scf=True,
                        run_mp2=False,
                        run_cisd=True,
                        run_ccsd=False,
                        run_fci=False,
                        delete_input=True,
                        delete_output=True)
    self.cisd_energy = molecule.cisd_energy
    self.rdm = molecule.get_molecular_rdm()
    self.hamiltonian = molecule.get_molecular_hamiltonian()

  def test_get_qubit_expectations(self):
    qubit_operator = self.hamiltonian.jordan_wigner_transform()
    qubit_expectations = self.rdm.get_qubit_expectations(qubit_operator)
    test_energy = qubit_operator[[]]
    for qubit_term in qubit_expectations:
      term_coefficient = qubit_operator[qubit_term]
      test_energy += term_coefficient * qubit_term.coefficient
    self.assertLess(abs(test_energy - self.cisd_energy), EQ_TOLERANCE)

  def test_get_molecular_operator_expectation(self):
    expectation = self.rdm.expectation(self.hamiltonian)
    self.assertAlmostEqual(expectation, self.cisd_energy, places=7)


# Test.
if __name__ == "__main__":
  unittest.main()
