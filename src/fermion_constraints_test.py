"""Tests for fermion_constraints.py"""
import numpy
import numpy.random
import scipy
import unittest
import run_psi4
import molecular_data
import sparse_operators
import molecular_operators
import fermion_constraints
import qubit_operators


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

    # Get qubit Hamiltonian.
    self.qubit_hamiltonian = (
        self.fermion_hamiltonian.jordan_wigner_transform())

    self.n_qubits = self.qubit_hamiltonian.n_qubits

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

  def test_positivity_constraints(self):
    # Positivity of one- and two-rdm
    constraint_fermi = fermion_constraints.\
        FermionConstraints(self.fermion_hamiltonian.n_qubits,
                           self.n_electrons)
    projected_rdm = constraint_fermi.apply_positivity(self.fci_rdm)

    # Check energy on projected RDM
    molecular_hamiltonian = self.fermion_hamiltonian.get_molecular_operator()

    # Original
    fci_rdm_energy = molecular_hamiltonian.constant
    fci_rdm_energy += numpy.sum(self.fci_rdm.one_body_coefficients *
                                molecular_hamiltonian.one_body_coefficients)
    fci_rdm_energy += numpy.sum(self.fci_rdm.two_body_coefficients *
                                molecular_hamiltonian.two_body_coefficients)
    self.assertAlmostEqual(fci_rdm_energy, self.molecule.fci_energy)

    # Projected
    fci_rdm_energy = molecular_hamiltonian.constant
    fci_rdm_energy += numpy.sum(self.fci_rdm.one_body_coefficients *
                                molecular_hamiltonian.one_body_coefficients)
    fci_rdm_energy += numpy.sum(self.fci_rdm.two_body_coefficients *
                                molecular_hamiltonian.two_body_coefficients)
    self.assertAlmostEqual(fci_rdm_energy, self.molecule.fci_energy)

    # Now take a ``noisy'' set of Pauli measurements, project, check properties
    qubit_rdm = self.fci_rdm.get_qubit_expectations(self.qubit_hamiltonian)
    noisy_qubit_rdm = qubit_operators.QubitOperator(self.n_qubits)
    for term in qubit_rdm:
      noisy_qubit_rdm += qubit_operators.\
          QubitTerm(self.n_qubits,
                    term.coefficient + numpy.random.randn(),
                    term.operators)
    noisy_fermi_rdm = noisy_qubit_rdm.get_fermion_expectations()
    projected_noisy_rdm = constraint_fermi.apply_positivity(noisy_fermi_rdm)

    # Check Trace
    self.assertAlmostEqual(numpy.trace(projected_noisy_rdm.
                                       one_body_coefficients),
                           self.n_electrons)
    self.assertAlmostEqual(
        numpy.trace(projected_noisy_rdm.
                    two_body_coefficients.reshape((self.n_qubits**2,
                                                   self.n_qubits**2))),
        -self.n_electrons * (self.n_electrons - 1))

    # Check positive semi-definiteness
    self.assertTrue(
        numpy.all(
            scipy.linalg.eigh(
                projected_noisy_rdm.one_body_coefficients)[0] >= -1e-12))

    self.assertTrue(
        numpy.all(
            scipy.linalg.eigh(
                projected_noisy_rdm.
                two_body_coefficients.reshape((self.n_qubits ** 2,
                                               self.n_qubits ** 2)))[0] <=
            1e-12))

if __name__ == '__main__':
  unittest.main()
