"""The purpose of this file is to document and demonstrate known bugs."""
import unittest
import molecular_data
import run_psi4
import local_operators
import local_terms
import sparse_operators
import numpy
import copy


class NumpyScalarBug(unittest.TestCase):
  """THIS BUG HAS BEEN FIXED --- IAN
  
  Multiplying certain classes by numpy scalars returns invalid objects.

  When a FermionOpeator, QubitOperator or LocalOperator is multiplied by
  a scalar it is supposed to return that operator times the scalar.
  Something unexpected happens when the scalar is from a numpy data type,
  for instance when the scalar is a numpy.float64 instead of a "float".
  I have tried to change the __rmul__ methods in local_operators and
  qubit_operators to handle numpy scalar types by recognizing them with
  numpy.isscalar() but to no avail. Note that the bug is only with the
  __rmul__ and not with __mul__/__lmul__. Note that the bug may also
  need to be fixed in the QubitTerm which overwrites the __imul__ class
  from its parent, LocalTerm."""

  def setUp(self):

    # Get the numpy scalar and also a normal one for comparison.
    self.good_scalar = -2.9
    self.bad_scalar = numpy.float64(self.good_scalar)

    # Get a LocalTerm to test things out.
    self.n_qubits = 8
    self.coefficient = 1.
    self.operators = [3, 4, 5]
    self.local_term = local_terms.LocalTerm(
        self.n_qubits, self.coefficient, self.operators)

    # Make a LocalOperator to test things out.
    new_term = copy.deepcopy(self.local_term)
    self.local_operator = local_operators.LocalOperator(
        self.n_qubits, [new_term, self.local_term])

  def test_right_local_term_multiply(self):
    correct = self.good_scalar * self.local_term
    buggy = self.bad_scalar * self.local_term
    print 'The correct result for a single term follows:\n{}'.format(correct)
    print 'The buggy results for a single term follows:\n{}'.format(buggy)
    self.assertEqual(correct, buggy)

  def test_right_local_operator_multiply(self):
    correct = self.good_scalar * self.local_operator
    buggy = self.bad_scalar * self.local_operator
    print 'The correct result for an operator follows:\n{}'.format(correct)
    print 'The buggy results for an operator follows:\n{}'.format(buggy)
    self.assertEqual(correct, buggy)


class OddNumberElectronBug(unittest.TestCase):
  """Demonstrate incorrect qubit expectation values for odd numbers of e

  The transformation from fermionic RDMs to qubit operators seems to
  provide erroneous values for odd numbers of electrons. However,
  the energy seems to be correctly computed with these unphysical Pauli
  expectation values. This tests it for H3 and demonstrates the error in
  the simplest case."""

  def setUp(self):

    # Set up molecule.
    self.geometry = [('H', (0., 0., x * 0.7414)) for x in range(3)]
    self.basis = 'sto-3g'
    self.multiplicity = 2
    self.molecule = molecular_data.MolecularData(
        self.geometry, self.basis, self.multiplicity)

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

    # Get explicit coefficients.
    self.nuclear_repulsion = self.molecular_hamiltonian.constant
    self.one_body = self.molecular_hamiltonian.one_body_coefficients
    self.two_body = self.molecular_hamiltonian.two_body_coefficients

    # Get fermion Hamiltonian.
    self.fermion_hamiltonian = (
        self.molecular_hamiltonian.get_fermion_operator())
    self.fermion_hamiltonian.normal_order()

    # Get qubit Hamiltonian.
    self.qubit_hamiltonian = (
        self.fermion_hamiltonian.jordan_wigner_transform())

    # Get matrix form.
    self.hamiltonian_matrix = self.molecular_hamiltonian.get_sparse_matrix()

  def test_qubit_to_molecular_rdm(self):
    qubit_rdm = self.fci_rdm.get_qubit_expectations(self.qubit_hamiltonian)
    molecular_rdm = qubit_rdm.get_molecular_rdm()
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
    molecular_hamiltonian = self.fermion_hamiltonian.get_molecular_operator()
    fermion_hamiltonian = molecular_hamiltonian.get_fermion_operator()
    self.assertTrue(self.fermion_hamiltonian == fermion_hamiltonian)

    # Check that FCI prior has the correct energy.
    fci_rdm_energy = self.nuclear_repulsion
    fci_rdm_energy += numpy.sum(self.fci_rdm.one_body_coefficients *
                                molecular_hamiltonian.one_body_coefficients)
    fci_rdm_energy += numpy.sum(self.fci_rdm.two_body_coefficients *
                                molecular_hamiltonian.two_body_coefficients)
    self.assertAlmostEqual(fci_rdm_energy, self.molecule.fci_energy)

    # Make sure its actually normal ordered.
    for term in self.fermion_hamiltonian:
      self.assertTrue(term.is_normal_ordered())

    # Test the matrix representation.
    wavefunction, energy = sparse_operators.get_ground_state(
        self.hamiltonian_matrix)
    self.assertAlmostEqual(energy, self.molecule.fci_energy)
    expected_energy = sparse_operators.expectation(
        self.hamiltonian_matrix, wavefunction)
    self.assertAlmostEqual(expected_energy, energy)

    # Make sure you can reproduce Hartree-Fock energy.
    hf_state = sparse_operators.hartree_fock_state(
        self.molecule.n_electrons, self.qubit_hamiltonian.n_qubits)
    hf_density = sparse_operators.get_density_matrix([hf_state], [1.])
    expected_hf_density_energy = sparse_operators.expectation(
        self.hamiltonian_matrix, hf_density)
    expected_hf_energy = sparse_operators.expectation(
        self.hamiltonian_matrix, hf_state)
    self.assertAlmostEqual(expected_hf_energy, self.molecule.hf_energy)
    self.assertAlmostEqual(expected_hf_density_energy,
                           self.molecule.hf_energy)

    # Confirm expectation on qubit Hamiltonian using reverse JW matches
    qubit_rdm = self.fci_rdm.get_qubit_expectations(self.qubit_hamiltonian)
    qubit_energy = self.qubit_hamiltonian.expectation(qubit_rdm)
    self.assertAlmostEqual(qubit_energy, self.molecule.fci_energy)

    # Confirm fermionic RDMs can be built from measured qubit RDMs
    new_fermi_rdm = qubit_rdm.get_molecular_rdm()
    fci_rdm_energy = self.nuclear_repulsion
    fci_rdm_energy += numpy.sum(new_fermi_rdm.one_body_coefficients *
                                molecular_hamiltonian.one_body_coefficients)
    fci_rdm_energy += numpy.sum(new_fermi_rdm.two_body_coefficients *
                                molecular_hamiltonian.two_body_coefficients)
    self.assertAlmostEqual(fci_rdm_energy, self.molecule.fci_energy)

    # Check that qubit expectations have physical values
    for term in qubit_rdm:
      if (numpy.abs(term.coefficient) > 1.):
        print("Mismatch on term: {}".format(term))
        print("Hamiltonian Coefficient of Term: {}".format(
              self.qubit_hamiltonian[term]))
        print("Reverse JW of term: {}".format(term.reverse_jordan_wigner(

        ).normal_order()))
        self.assertTrue(False)

# Run test.
if __name__ == '__main__':
  unittest.main()
