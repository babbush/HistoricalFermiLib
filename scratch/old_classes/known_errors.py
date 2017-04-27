"""The purpose of this file is to document and demonstrate known bugs."""
from __future__ import absolute_import

import unittest
import numpy

from fermilib.ops import MolecularData
from psi4tmp import run_psi4


class OddNumberElectronBug(unittest.TestCase):
    """Demonstrate incorrect qubit expectation values for odd numbers of e.

    The transformation from fermionic RDMs to qubit operators seems to
    provide erroneous values for odd numbers of electrons. However, the
    energy seems to be correctly computed with these unphysical Pauli
    expectation values. This tests it for H3 and demonstrates the error
    in the simplest case.

    """

    def setUp(self):

        # Set up molecule.
        self.geometry = [('H', (0., 0., x * 0.7414)) for x in range(3)]
        self.basis = 'sto-3g'
        self.multiplicity = 2
        self.molecule = MolecularData(
            self.geometry, self.basis, self.multiplicity)

        # Run calculations.
        run_scf = 1
        run_fci = 1
        verbose = 0
        delete_input = 1
        delete_output = 1
        self.molecule = run_psi4(self.molecule,
                                 run_scf=run_scf,
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
        self.qubit_hamiltonian = (
            self.fermion_hamiltonian.jordan_wigner_transform())

        # Get matrix form.
        self.hamiltonian_matrix = (
            self.molecular_hamiltonian.get_sparse_operator())

    def test_qubit_to_molecular_rdm(self):
        qubit_rdm = self.fci_rdm.get_qubit_expectations(self.qubit_hamiltonian)
        molecular_rdm = qubit_rdm.get_interaction_rdm()
        self.assertTrue(molecular_rdm == self.fci_rdm)

    def test_molecular_hydrogen(self):

        # Check that all the transforms work.
        qubit_hamiltonian = self.fermion_hamiltonian.jordan_wigner_transform()
        self.assertTrue(self.qubit_hamiltonian == qubit_hamiltonian)

        # Check reverse transform.
        fermion_hamiltonian = qubit_hamiltonian.reverse_jordan_wigner()
        fermion_hamiltonian.normal_order()
        self.assertTrue(self.fermion_hamiltonian == fermion_hamiltonian)

        # Make sure the mapping of FermionOperator to MolecularOperator works.
        molecular_hamiltonian = (
            self.fermion_hamiltonian.get_interaction_operator())
        fermion_hamiltonian = molecular_hamiltonian.get_fermion_operator()
        self.assertTrue(self.fermion_hamiltonian == fermion_hamiltonian)

        # Check that FCI prior has the correct energy.
        fci_rdm_energy = self.fci_rdm.expectation(self.molecular_hamiltonian)
        self.assertAlmostEqual(fci_rdm_energy, self.molecule.fci_energy)

        # Make sure its actually normal ordered.
        for term in self.fermion_hamiltonian:
            self.assertTrue(term.is_normal_ordered())

        # Confirm expectation on qubit Hamiltonian using reverse JW matches
        qubit_rdm = self.fci_rdm.get_qubit_expectations(self.qubit_hamiltonian)
        qubit_energy = self.qubit_hamiltonian.expectation(qubit_rdm)
        self.assertAlmostEqual(qubit_energy, self.molecule.fci_energy)

        # Confirm fermionic RDMs can be built from measured qubit RDMs
        new_fermi_rdm = qubit_rdm.get_interaction_rdm()
        fci_rdm_energy = new_fermi_rdm.expectation(self.molecular_hamiltonian)
        self.assertAlmostEqual(fci_rdm_energy, self.molecule.fci_energy)

        # Check that qubit expectations have physical values
        for term in qubit_rdm:
            if (numpy.abs(term.coefficient) > 1.):
                print('Mismatch on term: {}'.format(term))
                print('Hamiltonian Coefficient of Term: {}'.format(
                      self.qubit_hamiltonian[term]))
                print('Reverse JW of term: {}'.format(
                      term.reverse_jordan_wigner().normal_order()))
                self.assertTrue(False)


# Run test.
if __name__ == '__main__':
    unittest.main()
