"""Tests for molecular_data and run_psi4."""
from __future__ import absolute_import

import unittest

import numpy.random
import scipy.linalg

from fermilib import molecular_data
from fermilib import run_psi4


class MolecularData(unittest.TestCase):

  def setUp(self):

    # Set up molecule.
    self.n_atoms = 2
    self.geometry = [('H', (0., 0., 0.7414 * x)) for x in range(self.n_atoms)]
    self.basis = 'sto-3g'
    self.multiplicity = 1
    self.charge = 0
    self.description = 'eq'
    self.molecule = molecular_data.MolecularData(self.geometry,
                                                 self.basis,
                                                 self.multiplicity,
                                                 self.charge,
                                                 self.description)

    # Run calculations.
    run_scf = 1
    run_mp2 = 1
    run_cisd = 1
    run_ccsd = 1
    run_fci = 1
    delete_input = 1
    delete_output = 1
    verbose = 0
    self.molecule = run_psi4.run_psi4(self.molecule,
                                      run_scf=run_scf,
                                      run_mp2=run_mp2,
                                      run_cisd=run_cisd,
                                      run_ccsd=run_ccsd,
                                      run_fci=run_fci,
                                      verbose=verbose,
                                      delete_input=delete_input,
                                      delete_output=delete_output)

  def test_name_molecule(self):
    correct_name = 'H2_sto-3g_singlet_eq'
    computed_name = molecular_data.name_molecule(self.geometry,
                                                 self.basis,
                                                 self.multiplicity,
                                                 self.charge,
                                                 self.description)
    self.assertEqual(correct_name, computed_name)
    self.assertEqual(correct_name, self.molecule.name)

  def test_save_refresh(self):

    # Set number of atoms to be one higher than it should be.
    n_atoms = self.molecule.n_atoms
    self.molecule.n_atoms += 1
    self.assertEqual(self.molecule.n_atoms, n_atoms + 1)

    # Refresh the molecule and make sure the number of atoms is restored.
    self.molecule.refresh()
    self.assertEqual(self.molecule.n_atoms, n_atoms)

    # Now change the number of atoms again and save it this time.
    self.molecule.n_atoms += 1
    self.molecule.save()

    # Load the molecule.
    self.molecule = molecular_data.MolecularData(self.geometry,
                                                 self.basis,
                                                 self.multiplicity,
                                                 self.charge,
                                                 self.description)
    self.assertTrue(self.molecule.n_atoms, n_atoms + 1)

    # Finally, correct the number of atoms and restore.
    self.molecule.n_atoms -= 1
    self.molecule.save()

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

    # Compute total energy from RDM under some arbitrary basis set rotation.
    molecular_rdm.rotate_basis(rotation_matrix)
    molecular_hamiltonian.rotate_basis(rotation_matrix)
    total_energy = molecular_rdm.expectation(molecular_hamiltonian)
    self.assertAlmostEqual(total_energy, self.molecule.cisd_energy)


if __name__ == '__main__':
  unittest.main()
