"""Tests many modules to compute energy of LiH."""
from __future__ import absolute_import

import unittest
import os

import numpy
import scipy.sparse
from fermilib.ops import *
from fermilib.transforms import *
from fermilib.utils import uccsd_operator


def get_test_filename(name):
  _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
  return os.path.join(_THIS_DIR, name)


class LiHIntegrationTest(unittest.TestCase):

    def setUp(self):

        # Set up molecule.
        self.geometry = [('Li', (0., 0., 0.)), ('H', (0., 0., 1.45))]
        self.basis = 'sto-3g'
        self.multiplicity = 1
        self.molecule = MolecularData(self.geometry, self.basis,
                                      self.multiplicity)
        self.molecule.refresh(filename=get_test_filename(
            "testdata/H1-Li1_sto-3g_singlet"))

        # Get molecular Hamiltonian.
        self.molecular_hamiltonian = self.molecule.get_molecular_hamiltonian(
            filename=get_test_filename("testdata/H1-Li1_sto-3g_singlet"))
        #self.molecular_hamiltonian_no_core = self.molecule.\
            #    get_molecular_hamiltonian(active_space_start=1)

        # Get FCI RDM.
        self.fci_rdm = self.molecule.get_molecular_rdm(use_fci=1,
            filename=get_test_filename("testdata/H1-Li1_sto-3g_singlet"))

        # Get explicit coefficients.
        self.nuclear_repulsion = self.molecular_hamiltonian.constant
        self.one_body = self.molecular_hamiltonian.one_body_tensor
        self.two_body = self.molecular_hamiltonian.two_body_tensor

        # Get fermion Hamiltonian.
        self.fermion_hamiltonian = normal_ordered(get_fermion_operator(
            self.molecular_hamiltonian))

        # Get qubit Hamiltonian.
        self.qubit_hamiltonian = jordan_wigner(self.fermion_hamiltonian)

        # Get explicit coefficients.
        self.nuclear_repulsion = self.molecular_hamiltonian.constant
        self.one_body = self.molecular_hamiltonian.one_body_tensor
        self.two_body = self.molecular_hamiltonian.two_body_tensor

        # Get matrix form.
        self.hamiltonian_matrix = get_sparse_operator(
            self.molecular_hamiltonian)
        #self.hamiltonian_matrix_no_core = get_sparse_operator(
        #    self.molecular_hamiltonian_no_core)


    def test_reverse_jordan_wigner(self):
        fermion_hamiltonian = reverse_jordan_wigner(self.qubit_hamiltonian)
        fermion_hamiltonian = normal_ordered(fermion_hamiltonian)
        self.assertTrue(self.fermion_hamiltonian.isclose(fermion_hamiltonian))

    def test_interaction_operator_mapping(self):
        fermion_hamiltonian = get_fermion_operator(self.molecular_hamiltonian)
        fermion_hamiltonian = normal_ordered(fermion_hamiltonian)
        self.assertTrue(self.fermion_hamiltonian.isclose(fermion_hamiltonian))

    def test_rdm_energy(self):
        fci_rdm_energy = self.nuclear_repulsion
        fci_rdm_energy += numpy.sum(self.fci_rdm.one_body_tensor *
                                    self.one_body)
        fci_rdm_energy += numpy.sum(self.fci_rdm.two_body_tensor *
                                    self.two_body)
        self.assertAlmostEqual(fci_rdm_energy, self.molecule.fci_energy)

        # Confirm expectation on qubit Hamiltonian using reverse JW matches.
        qubit_rdm = self.fci_rdm.get_qubit_expectations(self.qubit_hamiltonian)
        qubit_energy = self.qubit_hamiltonian.terms[()]
        for term, coefficient in qubit_rdm.terms.items():
          qubit_energy += coefficient * self.qubit_hamiltonian.terms[term]
        self.assertAlmostEqual(qubit_energy, self.molecule.fci_energy)

        # Confirm fermionic RDMs can be built from measured qubit RDMs.
        new_fermi_rdm = get_interaction_rdm(qubit_rdm)
        fermi_rdm_energy = new_fermi_rdm.expectation(
            self.molecular_hamiltonian)
        self.assertAlmostEqual(fci_rdm_energy, self.molecule.fci_energy)

    def test_sparse(self):
        energy, wavefunction = self.hamiltonian_matrix.get_ground_state()
        self.assertAlmostEqual(energy, self.molecule.fci_energy)
        expected_energy = self.hamiltonian_matrix.expectation(wavefunction)
        self.assertAlmostEqual(expected_energy, energy)

        # Make sure you can reproduce Hartree-Fock energy.
        hf_state = jw_hartree_fock_state(
            self.molecule.n_electrons, self.qubit_hamiltonian.n_qubits())
        hf_density = get_density_matrix([hf_state], [1.])
        expected_hf_density_energy = self.hamiltonian_matrix.expectation(
            hf_density)
        expected_hf_energy = self.hamiltonian_matrix.expectation(hf_state)
        self.assertAlmostEqual(expected_hf_energy, self.molecule.hf_energy)
        self.assertAlmostEqual(expected_hf_density_energy,
                               self.molecule.hf_energy)

        # Check that frozen core result matches frozen core FCI from psi4.
        # Recore frozen core result from external calculation.
        #self.frozen_core_fci_energy = -7.8807607374168
        #no_core_fci_energy = scipy. \
        #    linalg.eigh(self.hamiltonian_matrix_no_core.matrix.todense())[0][0]
        #self.assertAlmostEqual(no_core_fci_energy,
        #                       self.frozen_core_fci_energy)


if __name__ == '__main__':
    unittest.main()
