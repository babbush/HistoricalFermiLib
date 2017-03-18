import unittest
import numpy
import scipy
import scipy.linalg
import scipy.sparse
import scipy.optimize
import fermion_operators
import jellium
import jellium_ucc


class JelliumUCCTest(unittest.TestCase):

  def setUp(self):
    self.n_fermions = 2
    self.jellium_hamiltonian = jellium.jellium_model(2, 2, 0.1)
    self.n_qubits = self.jellium_hamiltonian.n_qubits
    print("Number of qubits in simulation: {}".format(self.n_qubits))

  def test_optimize_uccsd(self):
    """Optimize UCCSD ansatz with 2 electrons and check against FCI"""

    # Build a number operator for partitioning space into different occupations
    number_operator = fermion_operators.FermionOperator(self.n_qubits)
    for i in range(self.n_qubits):
      number_operator += fermion_operators.FermionTerm(self.n_qubits, 1.0,
                                                       [(i, 1), (i, 0)])

    # Build matrix representations of number operator, jellium, and uccsd
    number_qubit = number_operator.\
        jordan_wigner_transform().\
        get_sparse_matrix()

    jellium_qubit = self.jellium_hamiltonian.\
        jordan_wigner_transform().\
        get_sparse_matrix()

    evals, evecs = scipy.sparse.linalg.eigsh(jellium_qubit, 30,
                                             which="SA", maxiter=1e3)

    # Find minimum energy for state with 2 electrons
    minimum_2_fermion_energy = 1.e10
    for i, eval in enumerate(evals):
      evec = evecs[:, i][:, numpy.newaxis]
      expected_number = numpy.dot(numpy.conj(evec.T),
                                  number_qubit.dot(evec))
      if ((numpy.abs(expected_number - 2.0) < 1e-7) and
         (eval < minimum_2_fermion_energy)):
        minimum_2_fermion_energy = eval
    print("Minimum 2-fermion energy: {}".format(minimum_2_fermion_energy))

    # Create simple initial state with 2 fermions
    initial_state = jellium_ucc.simple_initial_state(self.n_qubits, 2)

    def energy_objective(amplitudes):
      return jellium_ucc.\
          jellium_uccsd_energy(self.n_qubits, 2, initial_state,
                               jellium_qubit, amplitudes)

    initial_amplitudes = \
        numpy.zeros(jellium_ucc.jellium_uccsd_amplitude_count(self.n_qubits,
                                                              self.n_fermions))

    # Optimize UCCSD energy with 2 electrons
    result = scipy.optimize.minimize(energy_objective, initial_amplitudes,
                                     method='CG',
                                     options={'disp': True})
    print("Optimal Energy Found with UCCSD: {}".format(result.fun))

    self.assertAlmostEqual(minimum_2_fermion_energy, result.fun)

# Run test.
if __name__ == '__main__':
  unittest.main()
