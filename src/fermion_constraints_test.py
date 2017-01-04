"""Tests for fermion_constraints.py"""
import numpy
import unittest
import run_psi4
import molecular_data
import sparse_operators
import molecular_operators
import fermion_constraints


class FermionConstraintsTest(unittest.TestCase):

  def setUp(self):

    # Set up molecule.
    self.geometry = [('H', (0., 0., 0.)),
                     ('H', (0., 0., 0.7414)),
                     ('H', (0., 0., 0.7414 * 2.)),
                     ('H', (0., 0., 0.7414 * 3))]
    self.basis = 'sto-3g'
    self.multiplicity = 1
    self.molecule = molecular_data.MolecularData(
        self.geometry, self.basis, self.multiplicity)
    self.n_electrons = 4

    # Run calculations.
    run_scf = 1
    run_fci = 1
    verbose = 0
    delete_input = 1
    delete_output = 1
    self.molecule = run_psi4.run_psi4(self.molecule,
                                      run_scf=run_scf,
                                      run_fci=run_fci,
                                      verbose=verbose,
                                      delete_input=delete_input,
                                      delete_output=delete_output)

    # Get molecular Hamiltonian.
    self.molecular_hamiltonian = self.molecule.get_molecular_hamiltonian()

    # Get FCI RDM.
    self.fci_rdm = self.molecule.get_molecular_rdm(use_fci=run_fci)

    # Get fermion Hamiltonian.
    self.fermion_hamiltonian = (
        self.molecular_hamiltonian.get_fermion_operator())
    self.fermion_hamiltonian.normal_order()

  def test_one_body_constraints(self):
    # One-Body Constraints
    constraints_generator = fermion_constraints.\
        FermionConstraints(self.fermion_hamiltonian.n_qubits,
                           self.n_electrons).\
        constraints_one_body()

    for constraint in constraints_generator:
      qubit_rdm = self.fci_rdm.get_qubit_expectations(constraint[0])
      constraint_value = constraint[0].expectation(qubit_rdm)
      self.assertAlmostEqual(constraint_value, constraint[1])

  def test_two_body_constraints(self):
    # Two-Body Constraints
    constraints_generator = fermion_constraints. \
        FermionConstraints(self.fermion_hamiltonian.n_qubits,
                           self.n_electrons). \
        constraints_two_body()

    for constraint in constraints_generator:
      qubit_rdm = self.fci_rdm.get_qubit_expectations(constraint[0])
      constraint_value = constraint[0].expectation(qubit_rdm)
      self.assertAlmostEqual(constraint_value, constraint[1])

if __name__ == '__main__':
  unittest.main()
