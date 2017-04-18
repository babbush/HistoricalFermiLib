"""Tests many modules to compute energy of hydrogen."""
from __future__ import absolute_import

import unittest

import numpy
import scipy.sparse

from fermilib import molecular_data
from fermilib import run_psi4
from fermilib import sparse_operators
from fermilib import unitary_cc
from fermilib.transformations import (jordan_wigner_transform,
                                      reverse_jordan_wigner,
                                      get_interaction_rdm,
                                      get_interaction_operator)


class HydrogenIntegrationTest(unittest.TestCase):

    def setUp(self):

        # Set up molecule.
        self.geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 0.7414))]
        self.basis = 'sto-3g'
        self.multiplicity = 1
        self.molecule = molecular_data.MolecularData(
            self.geometry, self.basis, self.multiplicity)

        # Run calculations.
        run_scf = 1
        run_ccsd = 1
        run_fci = 1
        verbose = 0
        delete_input = 1
        delete_output = 0
        self.molecule = run_psi4.run_psi4(self.molecule,
                                          run_scf=run_scf,
                                          run_ccsd=run_ccsd,
                                          run_fci=run_fci,
                                          verbose=verbose,
                                          delete_input=delete_input,
                                          delete_output=delete_output)

        # Get molecular Hamiltonian.
        self.molecular_hamiltonian = self.molecule.get_molecular_hamiltonian()

        # Get FCI RDM.
        self.fci_rdm = self.molecule.get_molecular_rdm(use_fci=run_fci)

        # Get explicit coefficients.
        self.nuclear_repulsion = self.molecular_hamiltonian.constant
        self.one_body = self.molecular_hamiltonian.one_body_tensor
        self.two_body = self.molecular_hamiltonian.two_body_tensor

        # Get fermion Hamiltonian.
        self.fermion_hamiltonian = (
            self.molecular_hamiltonian.get_fermion_operator())
        self.fermion_hamiltonian.normal_order()

        # Get qubit Hamiltonian.
        self.qubit_hamiltonian = jordan_wigner_transform(
            self.fermion_hamiltonian)

        # Get matrix form.
        self.hamiltonian_matrix = (
            self.molecular_hamiltonian.get_sparse_operator())

        # Initialize coefficients given in Seeley and Love paper, arXiv:
        # 1208.5986.
        self.g0 = 0.71375
        self.g1 = -1.2525
        self.g2 = -0.47593
        self.g3 = 0.67449 / 2.
        self.g4 = 0.69740 / 2.
        self.g5 = 0.66347 / 2.
        self.g6 = 0.18129 / 2.

        # Below are qubit term coefficients, also from Seeley and Love.
        self.f0 = -0.8126
        self.f1 = 0.1712
        self.f2 = -0.2228
        self.f3 = 0.1686
        self.f4 = 0.1205
        self.f5 = 0.1659
        self.f6 = 0.1743
        self.f7 = 0.04532

    def test_molecular_hydrogen(self):

        # Check that all the transforms work.
        qubit_hamiltonian = jordan_wigner_transform(self.fermion_hamiltonian)
        self.assertTrue(self.qubit_hamiltonian == qubit_hamiltonian)

        # Check reverse transform.
        fermion_hamiltonian = reverse_jordan_wigner(qubit_hamiltonian)
        fermion_hamiltonian.normal_order()
        self.assertTrue(self.fermion_hamiltonian == fermion_hamiltonian)

        # Make sure the mapping of FermionOperator to MolecularOperator works.
        molecular_hamiltonian = get_interaction_operator(
            self.fermion_hamiltonian)
        fermion_hamiltonian = molecular_hamiltonian.get_fermion_operator()
        self.assertTrue(self.fermion_hamiltonian == fermion_hamiltonian)

        # Make sure mapping of MolecularOperator to QubitOperator works.
        qubit_hamiltonian = jordan_wigner_transform(self.molecular_hamiltonian)
        self.assertEqual(self.qubit_hamiltonian, qubit_hamiltonian)

        # Check that FCI prior has the correct energy.
        fci_rdm_energy = self.nuclear_repulsion
        fci_rdm_energy += numpy.sum(self.fci_rdm.one_body_tensor *
                                    molecular_hamiltonian.one_body_tensor)
        fci_rdm_energy += numpy.sum(self.fci_rdm.two_body_tensor *
                                    molecular_hamiltonian.two_body_tensor)
        self.assertAlmostEqual(fci_rdm_energy, self.molecule.fci_energy)

        # Check the integrals.
        self.assertAlmostEqual(self.nuclear_repulsion, self.g0, places=4)

        self.assertAlmostEqual(self.one_body[0, 0], self.g1, places=4)
        self.assertAlmostEqual(self.one_body[1, 1], self.g1, places=4)

        self.assertAlmostEqual(self.one_body[2, 2], self.g2, places=4)
        self.assertAlmostEqual(self.one_body[3, 3], self.g2, places=4)

        self.assertAlmostEqual(self.two_body[0, 1, 1, 0], self.g3, places=4)
        self.assertAlmostEqual(self.two_body[1, 0, 0, 1], self.g3, places=4)

        self.assertAlmostEqual(self.two_body[2, 3, 3, 2], self.g4, places=4)
        self.assertAlmostEqual(self.two_body[3, 2, 2, 3], self.g4, places=4)

        self.assertAlmostEqual(self.two_body[0, 2, 2, 0], self.g5, places=4)
        self.assertAlmostEqual(self.two_body[0, 3, 3, 0], self.g5, places=4)
        self.assertAlmostEqual(self.two_body[1, 2, 2, 1], self.g5, places=4)
        self.assertAlmostEqual(self.two_body[1, 3, 3, 1], self.g5, places=4)
        self.assertAlmostEqual(self.two_body[2, 0, 0, 2], self.g5, places=4)
        self.assertAlmostEqual(self.two_body[3, 0, 0, 3], self.g5, places=4)
        self.assertAlmostEqual(self.two_body[2, 1, 1, 2], self.g5, places=4)
        self.assertAlmostEqual(self.two_body[3, 1, 1, 3], self.g5, places=4)

        self.assertAlmostEqual(self.two_body[0, 2, 0, 2], self.g6, places=4)
        self.assertAlmostEqual(self.two_body[1, 3, 1, 3], self.g6, places=4)
        self.assertAlmostEqual(self.two_body[2, 1, 3, 0], self.g6, places=4)
        self.assertAlmostEqual(self.two_body[2, 3, 1, 0], self.g6, places=4)
        self.assertAlmostEqual(self.two_body[0, 3, 1, 2], self.g6, places=4)
        self.assertAlmostEqual(self.two_body[0, 1, 3, 2], self.g6, places=4)

        # Make sure its actually normal ordered.
        for term in self.fermion_hamiltonian:
            self.assertTrue(term.is_normal_ordered())

        # Test the local Hamiltonian terms.
        self.assertAlmostEqual(
            self.qubit_hamiltonian[[(0, 'Z')]], self.f1, places=4)
        self.assertAlmostEqual(
            self.qubit_hamiltonian[[(1, 'Z')]], self.f1, places=4)

        self.assertAlmostEqual(
            self.qubit_hamiltonian[[(2, 'Z')]], self.f2, places=4)
        self.assertAlmostEqual(
            self.qubit_hamiltonian[[(3, 'Z')]], self.f2, places=4)

        self.assertAlmostEqual(
            self.qubit_hamiltonian[[(0, 'Z'), (1, 'Z')]], self.f3, places=4)

        self.assertAlmostEqual(
            self.qubit_hamiltonian[[(0, 'Z'), (2, 'Z')]], self.f4, places=4)
        self.assertAlmostEqual(
            self.qubit_hamiltonian[[(1, 'Z'), (3, 'Z')]], self.f4, places=4)

        self.assertAlmostEqual(
            self.qubit_hamiltonian[[(1, 'Z'), (2, 'Z')]], self.f5, places=4)
        self.assertAlmostEqual(
            self.qubit_hamiltonian[[(0, 'Z'), (3, 'Z')]], self.f5, places=4)

        self.assertAlmostEqual(
            self.qubit_hamiltonian[[(2, 'Z'), (3, 'Z')]], self.f6, places=4)

        self.assertAlmostEqual(
            self.qubit_hamiltonian[[(0, 'Y'), (1, 'Y'),
                                    (2, 'X'), (3, 'X')]], -self.f7, places=4)
        self.assertAlmostEqual(
            self.qubit_hamiltonian[[(0, 'X'), (1, 'X'),
                                    (2, 'Y'), (3, 'Y')]], -self.f7, places=4)

        self.assertAlmostEqual(
            self.qubit_hamiltonian[[(0, 'X'), (1, 'Y'),
                                    (2, 'Y'), (3, 'X')]], self.f7, places=4)
        self.assertAlmostEqual(
            self.qubit_hamiltonian[[(0, 'Y'), (1, 'X'),
                                    (2, 'X'), (3, 'Y')]], self.f7, places=4)

        # Test the matrix representation.
        energy, wavefunction = self.hamiltonian_matrix.get_ground_state()
        self.assertAlmostEqual(energy, self.molecule.fci_energy)
        expected_energy = self.hamiltonian_matrix.expectation(wavefunction)
        self.assertAlmostEqual(expected_energy, energy)

        # Make sure you can reproduce Hartree-Fock energy.
        hf_state = sparse_operators.jw_hartree_fock_state(
            self.molecule.n_electrons, self.qubit_hamiltonian.n_qubits())
        hf_density = sparse_operators.get_density_matrix([hf_state], [1.])
        expected_hf_density_energy = self.hamiltonian_matrix.expectation(
            hf_density)
        expected_hf_energy = self.hamiltonian_matrix.expectation(hf_state)
        self.assertAlmostEqual(expected_hf_energy, self.molecule.hf_energy)
        self.assertAlmostEqual(expected_hf_density_energy,
                               self.molecule.hf_energy)

        # Confirm expectation on qubit Hamiltonian using reverse JW matches.
        qubit_rdm = self.fci_rdm.get_qubit_expectations(self.qubit_hamiltonian)
        qubit_energy = qubit_rdm.expectation(self.qubit_hamiltonian)
        self.assertAlmostEqual(qubit_energy, self.molecule.fci_energy)

        # Confirm fermionic RDMs can be built from measured qubit RDMs
        new_fermi_rdm = get_interaction_rdm(qubit_rdm)
        new_fermi_rdm.expectation(self.molecular_hamiltonian)
        self.assertAlmostEqual(fci_rdm_energy, self.molecule.fci_energy)

        # Test UCCSD for reasonable accuracy against FCI using loaded t
        # amplitudes
        uccsd_operator = unitary_cc.\
            uccsd_operator(self.molecule.ccsd_amplitudes.one_body_tensor,
                           self.molecule.ccsd_amplitudes.two_body_tensor)

        uccsd_sparse = sparse_operators.\
            jordan_wigner_operator_sparse(uccsd_operator,
                                          self.qubit_hamiltonian.n_qubits())
        uccsd_state = scipy.sparse.linalg.expm_multiply(uccsd_sparse.matrix,
                                                        hf_state)
        expected_uccsd_energy = self.hamiltonian_matrix.expectation(
            uccsd_state)
        self.assertAlmostEqual(expected_uccsd_energy, self.molecule.fci_energy,
                               places=4)

        # print("UCCSD Energy: {}".format(expected_uccsd_energy))

        # Test CCSD for precise match against FCI using loaded t amplitudes
        ccsd_operator = unitary_cc.\
            uccsd_operator(self.molecule.ccsd_amplitudes.one_body_tensor,
                           self.molecule.ccsd_amplitudes.two_body_tensor,
                           anti_hermitian=False)

        ccsd_sparse_r = sparse_operators.\
            jordan_wigner_operator_sparse(ccsd_operator,
                                          self.qubit_hamiltonian.n_qubits())
        ccsd_sparse_l = sparse_operators.\
            jordan_wigner_operator_sparse(
                -ccsd_operator.hermitian_conjugated(),
                self.qubit_hamiltonian.n_qubits())
        ccsd_state_r = scipy.sparse.linalg.expm_multiply(ccsd_sparse_r.matrix,
                                                         hf_state)
        ccsd_state_l = scipy.sparse.linalg.expm_multiply(ccsd_sparse_l.matrix,
                                                         hf_state)
        expected_ccsd_energy = ccsd_state_l.getH().dot(
            self.hamiltonian_matrix.matrix.dot(ccsd_state_r))[0, 0]
        self.assertAlmostEqual(expected_ccsd_energy, self.molecule.fci_energy)

        # print("CCSD Energy: {}".format(expected_ccsd_energy))


if __name__ == '__main__':
    unittest.main()
