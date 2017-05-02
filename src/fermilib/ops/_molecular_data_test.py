"""Tests for molecular_data."""
from __future__ import absolute_import

import unittest

import numpy.random
import scipy.linalg

from fermilib.config import *
from fermilib.ops._molecular_data import MolecularData, name_molecule


class MolecularDataTest(unittest.TestCase):

    def setUp(self):
        self.geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 0.7414))]
        self.basis = 'sto-3g'
        self.multiplicity = 1
        filename = THIS_DIRECTORY + '/tests/testdata/H2_sto-3g_singlet'
        self.molecule = MolecularData(
            self.geometry, self.basis, self.multiplicity, filename=filename)
        self.molecule.load()

    def test_name_molecule(self):
        charge = 0
        correct_name = 'H2_sto-3g_singlet'
        computed_name = name_molecule(self.geometry,
                                      self.basis,
                                      self.multiplicity,
                                      charge,
                                      description=None)
        self.assertEqual(correct_name, computed_name)
        self.assertEqual(correct_name, self.molecule.name)

    def test_save_load(self):

        # Set number of atoms to be one higher than it should be.
        n_atoms = self.molecule.n_atoms
        self.molecule.n_atoms += 1
        self.assertEqual(self.molecule.n_atoms, n_atoms + 1)

        # Refresh the molecule and make sure the number of atoms is restored.
        self.molecule.load()
        self.assertEqual(self.molecule.n_atoms, n_atoms)

    def test_energies(self):

        # Check energies.
        self.assertAlmostEqual(self.molecule.hf_energy, -1.1167, places=4)
        self.assertAlmostEqual(self.molecule.mp2_energy, -1.1299, places=4)
        self.assertAlmostEqual(self.molecule.cisd_energy, -1.1373, places=4)
        self.assertAlmostEqual(self.molecule.ccsd_energy, -1.1373, places=4)
        self.assertAlmostEqual(self.molecule.ccsd_energy, -1.1373, places=4)

    def test_rdm_and_rotation(self):

        # Compute total energy from RDM.
        molecular_hamiltonian = self.molecule.get_molecular_hamiltonian()
        molecular_rdm = self.molecule.get_molecular_rdm()
        total_energy = molecular_rdm.expectation(molecular_hamiltonian)
        self.assertAlmostEqual(total_energy, self.molecule.cisd_energy)

        # Build random rotation with correction dimension.
        num_spatial_orbitals = self.molecule.n_orbitals
        rotation_generator = numpy.random.randn(
            num_spatial_orbitals, num_spatial_orbitals)
        rotation_matrix = scipy.linalg.expm(
            rotation_generator - rotation_generator.T)

        # Compute total energy from RDM under some basis set rotation.
        molecular_rdm.rotate_basis(rotation_matrix)
        molecular_hamiltonian.rotate_basis(rotation_matrix)
        total_energy = molecular_rdm.expectation(molecular_hamiltonian)
        self.assertAlmostEqual(total_energy, self.molecule.cisd_energy)


if __name__ == '__main__':
    unittest.main()
