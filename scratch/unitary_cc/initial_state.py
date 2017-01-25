"""Testing if initial estimate of coupled cluster amplitudes works.
"""

import scipy.optimize
import scipy.sparse
import scipy.sparse.linalg


import emulation_utils as eu
import hamiltonian_utils as hu
import ucc_ansatz


def main():
  # input:
  hamiltonian_file = "data/H6_sto-3g_JW_12.ham"
  cc_amplitude_file = "data/H6_sto-3g.ccamp"
  num_qubits = 12
  num_electrons = 6
  closed_shell = True

  # Process Hamiltonian:
  hamiltonian = hu.read_hamiltonian_file(hamiltonian_file, num_qubits)
  h_matrix = eu.get_qubit_hamiltonian_matrix(hamiltonian)
  # Exact:
  e0_energy, e0_wavefunction = eu.calculate_exact_ground_state(h_matrix)
  # Hartree Fock:
  hf_wavefunction = eu.get_hf_wavefunction(num_electrons, num_qubits)
  hf_energy = eu.get_expectation_value(h_matrix, hf_wavefunction)
  overlap_of_hf = eu.calculate_overlap(hf_wavefunction, e0_wavefunction)
  # Coupled cluster energy from amplitudes:
  cc_energy = eu.calculate_cc_energy(cc_amplitude_file, num_qubits,
                                     num_electrons, closed_shell, h_matrix)
  # UCC Ansatz:
  ansatz = ucc_ansatz.UccAnsatz(num_qubits, num_electrons)
  ansatz.set_parameters_to_cc_amplitudes(cc_amplitude_file, closed_shell)
  ucc_wavefunction = eu.apply_ucc_ansatz(ansatz, hf_wavefunction)
  ucc_energy = eu.get_expectation_value(h_matrix, ucc_wavefunction)
  overlap_of_ucc = eu.calculate_overlap(ucc_wavefunction, e0_wavefunction)

  # Output:
  print "Exact energy: {0}".format(e0_energy)
  print "Hartree Fock energy: {0}".format(hf_energy)
  print "Overlap of Hartree Fock with true ground state: {0}".format(
      overlap_of_hf)
  print "CC ansatz energy: {0}".format(cc_energy)
  print "UCC ansatz energy: {0}".format(ucc_energy)
  print "Overlap of UCC ansatz with true ground state: {0}".format(
      overlap_of_ucc)

  # UCC optimization
  def objective(x, ansatz, hf_wavefunction, h_matrix):
    ansatz.set_parameters(x)
    ucc_wavefunction = eu.apply_ucc_ansatz(ansatz, hf_wavefunction)
    return eu.get_expectation_value(h_matrix, ucc_wavefunction)

  optimized_params = scipy.optimize.fmin_cobyla(
      objective,
      ansatz.get_parameters(), [],
      args=(ansatz, hf_wavefunction, h_matrix),
      rhobeg=1e-1,
      maxfun=1000,
      disp=0)
  ansatz.set_parameters(optimized_params)
  vqe_wavefunction = eu.apply_ucc_ansatz(ansatz, hf_wavefunction)
  vqe_energy = eu.get_expectation_value(h_matrix, vqe_wavefunction)
  overlap_of_vqe = eu.calculate_overlap(vqe_wavefunction, e0_wavefunction)
  print "After optimisation loop"
  print "Optimal parameters: {0}".format(optimized_params)
  print "VQE energy: {0}".format(vqe_energy)
  print "Overlap of VQE ansatz with true ground state: {0}".format(
      overlap_of_vqe)


if __name__ == "__main__":
  main()
