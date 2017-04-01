"""Tests for molecular_rdm.py"""
import numpy
import unittest

from molecular_data import MolecularData
from molecular_rdm import MolecularRDM
from run_psi4 import run_psi4


class MolecularRDMTest(unittest.TestCase):

  def setUp(self):
    geometry = [('H', (0, 0, 0)), ('H', (0, 0, 0.2))]
    molecule = MolecularData(geometry, 'sto-3g', 1, autosave=False)
    molecule = run_psi4(molecule,
                        run_scf=False,
                        run_mp2=False,
                        run_cisd=True,
                        run_ccsd=False,
                        run_fci=False,
                        delete_input=True,
                        delete_output=True)
    self.cisd_energy = molecule.cisd_energy
    self.rdm = molecule.get_molecular_rdm()
    self.hamiltonion = molecule.get_molecular_hamiltonian()

  def test_get_molecular_operator_expectation(self):
    expectation = self.rdm.get_molecular_operator_expectation(self.hamiltonion)
    self.assertEqual(expectation, self.cisd_energy)


# Test.
if __name__ == "__main__":
  unittest.main()
