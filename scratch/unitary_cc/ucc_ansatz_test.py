"""Tests for ucc_ansatz."""

import os
import unittest

import fermionic_utils
import hamiltonian_utils
import ucc_ansatz

_BASE_PATH = "testdata/"


def get_test_filename(name):
  return os.path.join(_BASE_PATH, name)


class UccAnsatzTest(unittest.TestCase):

  def test_initialization_even_num_electrons(self):
    ansatz = ucc_ansatz.UccAnsatz(6, 2)
    # Correct fermion terms
    correct = [[3, -1], [5, -1], [4, -2], [6, -2], [4, 3, -2, -1],
               [6, 3, -2, -1], [5, 4, -2, -1], [6, 5, -2, -1]]
    self.assertEqual(len(ansatz.ucc_terms), len(correct))
    term_found = len(correct) * [False]
    for term in ansatz.ucc_terms:
      self.assertEqual(term[0], 0)
      for ii in range(len(correct)):
        if correct[ii] == term[1].fermionic_operator:
          term_found[ii] = True
    self.assertTrue(all(term_found))

  def test_initialization_odd_num_electrons(self):
    ansatz = ucc_ansatz.UccAnsatz(10, 5)
    # Correct fermion terms
    correct = [[7, -1], [9, -1], [6, -2], [8, -2], [10, -2], [7, -3], [9, -3],
               [6, -4], [8, -4], [10, -4], [7, -5], [9, -5], [7, 6, -2, -1],
               [9, 6, -2, -1], [8, 7, -2, -1], [10, 7, -2, -1], [9, 8, -2, -1],
               [10, 9, -2, -1], [9, 7, -3, -1], [7, 6, -4, -1], [9, 6, -4, -1],
               [8, 7, -4, -1], [10, 7, -4, -1], [9, 8, -4, -1], [10, 9, -4, -1],
               [9, 7, -5, -1], [7, 6, -3, -2], [9, 6, -3, -2], [8, 7, -3, -2],
               [10, 7, -3, -2], [9, 8, -3, -2], [10, 9, -3, -2], [8, 6, -4, -2],
               [10, 6, -4, -2], [10, 8, -4, -2], [7, 6, -5, -2], [9, 6, -5, -2],
               [8, 7, -5, -2], [10, 7, -5, -2], [9, 8, -5, -2], [10, 9, -5, -2],
               [7, 6, -4, -3], [9, 6, -4, -3], [8, 7, -4, -3], [10, 7, -4, -3],
               [9, 8, -4, -3], [10, 9, -4, -3], [9, 7, -5, -3], [7, 6, -5, -4],
               [9, 6, -5, -4], [8, 7, -5, -4], [10, 7, -5, -4], [9, 8, -5, -4],
               [10, 9, -5, -4]]
    self.assertEqual(len(ansatz.ucc_terms), len(correct))
    term_found = len(correct) * [False]
    for term in ansatz.ucc_terms:
      self.assertEqual(term[0], 0)
      for ii in range(len(correct)):
        if correct[ii] == term[1].fermionic_operator:
          term_found[ii] = True
    self.assertTrue(all(term_found))

  def test_transform_fermion_to_spin(self):
    num_qubits = 4
    operator = fermionic_utils.FermionicOperator(-1.0, [4, -2])
    # Correct result:
    string1 = hamiltonian_utils.PauliString(4, -0.25j, [3], [1], [2])
    string2 = hamiltonian_utils.PauliString(4, -0.25, [], [1, 3], [2])
    string3 = hamiltonian_utils.PauliString(4, -0.25, [1, 3], [], [2])
    string4 = hamiltonian_utils.PauliString(4, 0.25j, [1], [3], [2])
    # Compare
    ansatz = ucc_ansatz.UccAnsatz(2, 1)
    result = ansatz._transform_fermion_to_spin(operator, num_qubits)
    self.assertEqual(len(result), 4)
    compare = [False, False, False, False]
    for ii in range(4):
      for string in [string1, string2, string3, string4]:
        if string.is_identical_pauli(result[ii]):
          compare[ii] = True
          # Compare coefficient of identical tensor factors
          self.assertAlmostEqual(string.coefficient, result[ii].coefficient)
    # Check that all tensor factors are there
    self.assertTrue(all(compare))

  def test_get_ucc_spin_hamiltonian(self):
    num_qubits = 4
    operator = fermionic_utils.FermionicOperator(-1.0, [4, -2])
    # Correct result -1.0 * (a_4^dagger a_2 - a_2^dagger a_4) / (-1.0j):
    string1 = hamiltonian_utils.PauliString(4, 0.5, [3], [1], [2])
    string2 = hamiltonian_utils.PauliString(4, -0.5, [1], [3], [2])
    # Compare
    ansatz = ucc_ansatz.UccAnsatz(num_qubits, 1)
    hamiltonian = ansatz._get_ucc_spin_hamiltonian(operator)
    self.assertEqual(len(hamiltonian.terms), 2)
    compare = [False, False]
    for ii in range(2):
      for string in [string1, string2]:
        if string.is_identical_pauli(hamiltonian.terms[ii]):
          compare[ii] = True
          # Compare coefficient of identical tensor factors
          self.assertAlmostEqual(string.coefficient,
                                 hamiltonian.terms[ii].coefficient)
    # Check that all tensor factors are there
    self.assertTrue(all(compare))

  def test_get_and_set_parameters(self):
    ansatz = ucc_ansatz.UccAnsatz(10, 5)
    correct_parameters = len(ansatz.ucc_terms) * [0]
    initial_parameters = ansatz.get_parameters()
    self.assertEqual(initial_parameters, correct_parameters)
    new_parameters = len(ansatz.ucc_terms) * [0.5]
    ansatz.set_parameters(new_parameters)
    self.assertAlmostEqual(ansatz.get_parameters(), new_parameters)
    with self.assertRaises(ucc_ansatz.ErrorCoupledCluster):
      ansatz.set_parameters([0])

  def test_set_parameters_with_dict(self):
    ansatz = ucc_ansatz.UccAnsatz(4, 2)
    ansatz.set_parameters(len(ansatz.ucc_terms) * [1])
    parameter_dict = {}
    parameter_dict[(4, 3, -2, -1)] = 2
    # Set all parameters to 0 except (4,3,-2,-1) to 2
    ansatz.set_parameters_with_dict(parameter_dict)
    for term in ansatz.ucc_terms:
      if tuple(term[1].fermionic_operator) != (4, 3, -2, -1):
        self.assertEqual(term[0], 0)
      else:
        self.assertEqual(term[0], 2)


class CCAmplitudeTest(unittest.TestCase):

  def test_read_psi4_file(self):
    # Even number of electrons
    filename_1 = get_test_filename("HRing4_sto-3g.ccamp")
    result = ucc_ansatz.read_cc_file(filename_1)
    self.assertEqual(len(result), 5)
    self.assertEqual(len(result[0]), 2)
    self.assertEqual(len(result[1]), 0)
    self.assertEqual(len(result[2]), 0)
    self.assertEqual(len(result[3]), 0)
    self.assertEqual(len(result[4]), 8)
    self.assertAlmostEqual(result[0][1], [0, 1, -2.75e-08])
    self.assertAlmostEqual(result[4][7], [1, 1, 1, 1, -0.0008085547])
    # Odd number of electrons
    filename_2 = get_test_filename("HRing5_sto-3g.ccamp")
    result_2 = ucc_ansatz.read_cc_file(filename_2)
    self.assertEqual(len(result_2), 5)
    self.assertEqual(len(result_2[0]), 6)
    self.assertEqual(len(result_2[1]), 6)
    self.assertEqual(len(result_2[2]), 3)
    self.assertEqual(len(result_2[3]), 3)
    self.assertEqual(len(result_2[4]), 36)
    self.assertAlmostEqual(result_2[2][0], [2, 1, 1, 0, -0.0338409982])

  def test_convert_cc_numbering_to_spin_orbitals_open_shell(self):
    # Odd number of electrons (5 electrons, 10 spin orbitals)
    # Input:
    num_electrons = 5
    cc_occupied_spin_up = [0, 1, 2]
    cc_occupied_spin_down = [0, 1]
    cc_virtual_spin_up = [0, 1]
    cc_virtual_spin_down = [0, 1, 2]
    # Output:
    occupied_spin_up = []
    occupied_spin_down = []
    virtual_spin_up = []
    virtual_spin_down = []
    # Correct result:
    correct_occupied_spin_up = [1, 3, 5]
    correct_occupied_spin_down = [2, 4]
    correct_virtual_spin_up = [7, 9]
    correct_virtual_spin_down = [6, 8, 10]
    # Test:
    for spatial in cc_occupied_spin_up:
      occupied_spin_up.append(
          ucc_ansatz.convert_cc_numbering_to_spin_orbital(spatial, True, True,
                                                          num_electrons))
    for spatial in cc_occupied_spin_down:
      occupied_spin_down.append(
          ucc_ansatz.convert_cc_numbering_to_spin_orbital(spatial, False, True,
                                                          num_electrons))
    for spatial in cc_virtual_spin_up:
      virtual_spin_up.append(
          ucc_ansatz.convert_cc_numbering_to_spin_orbital(spatial, True, False,
                                                          num_electrons))
    for spatial in cc_virtual_spin_down:
      virtual_spin_down.append(
          ucc_ansatz.convert_cc_numbering_to_spin_orbital(spatial, False, False,
                                                          num_electrons))
    self.assertEqual(occupied_spin_up, correct_occupied_spin_up)
    self.assertEqual(occupied_spin_down, correct_occupied_spin_down)
    self.assertEqual(virtual_spin_up, correct_virtual_spin_up)
    self.assertEqual(virtual_spin_down, correct_virtual_spin_down)

  def test_convert_cc_numbering_to_spin_orbitals_closed_shell(self):
    # Even number of electrons (6 electrons, 12 spin orbitals)
    # Input:
    num_electrons = 6
    cc_occupied_spin_up = [0, 1, 2]
    cc_occupied_spin_down = [0, 1, 2]
    cc_virtual_spin_up = [0, 1, 2]
    cc_virtual_spin_down = [0, 1, 2]
    # Output:
    occupied_spin_up = []
    occupied_spin_down = []
    virtual_spin_up = []
    virtual_spin_down = []
    # Correct result:
    correct_occupied_spin_up = [1, 3, 5]
    correct_occupied_spin_down = [2, 4, 6]
    correct_virtual_spin_up = [7, 9, 11]
    correct_virtual_spin_down = [8, 10, 12]
    # Test:
    for spatial in cc_occupied_spin_up:
      occupied_spin_up.append(
          ucc_ansatz.convert_cc_numbering_to_spin_orbital(spatial, True, True,
                                                          num_electrons))
    for spatial in cc_occupied_spin_down:
      occupied_spin_down.append(
          ucc_ansatz.convert_cc_numbering_to_spin_orbital(spatial, False, True,
                                                          num_electrons))
    for spatial in cc_virtual_spin_up:
      virtual_spin_up.append(
          ucc_ansatz.convert_cc_numbering_to_spin_orbital(spatial, True, False,
                                                          num_electrons))
    for spatial in cc_virtual_spin_down:
      virtual_spin_down.append(
          ucc_ansatz.convert_cc_numbering_to_spin_orbital(spatial, False, False,
                                                          num_electrons))
    self.assertEqual(occupied_spin_up, correct_occupied_spin_up)
    self.assertEqual(occupied_spin_down, correct_occupied_spin_down)
    self.assertEqual(virtual_spin_up, correct_virtual_spin_up)
    self.assertEqual(virtual_spin_down, correct_virtual_spin_down)

  def test_convert_cc_amplitudes_to_ucc_closed_shell(self):
    filename = get_test_filename("H4_sto-3g.ccamp")
    cc_amplitudes = ucc_ansatz.read_cc_file(filename)
    ucc_amplitude_dict = ucc_ansatz.convert_cc_amplitudes_to_ucc(cc_amplitudes,
                                                                 4, True)
    # Correct result obtained by manual calculation:
    t00 = -0.0062919007
    t11 = 0.0030948932
    t1100 = -0.1299315906
    t0000 = -0.049109294
    t0101 = -0.0468249907
    t1010 = -0.0468249907
    t0011 = -0.029138475
    t1111 = -0.0272126304
    t0110 = -0.0240235256
    t1001 = -0.0240235256
    c2 = -1.  # Because they use a different sign convention for the doubles
    manual_ucc_dict = {}
    manual_ucc_dict[(5, -1)] = t00
    manual_ucc_dict[(7, -3)] = t11
    manual_ucc_dict[(6, -2)] = t00
    manual_ucc_dict[(8, -4)] = t11
    manual_ucc_dict[(6, 5, -4, -3)] = c2 * t1100
    manual_ucc_dict[(6, 5, -2, -1)] = c2 * t0000
    manual_ucc_dict[(8, 5, -4, -1)] = c2 * t0101
    manual_ucc_dict[(7, 6, -3, -2)] = c2 * t1010
    manual_ucc_dict[(8, 7, -2, -1)] = c2 * t0011
    manual_ucc_dict[(8, 7, -4, -3)] = c2 * t1111
    manual_ucc_dict[(7, 6, -4, -1)] = c2 * -1. * t0110
    manual_ucc_dict[(8, 5, -3, -2)] = c2 * -1. * t1001
    manual_ucc_dict[(7, 5, -3, -1)] = (
        c2 * t0101 + c2 * t1010 - c2 * t0110 - c2 * t1001) / 2.
    manual_ucc_dict[(8, 6, -4, -2)] = (
        c2 * t0101 + c2 * t1010 - c2 * t0110 - c2 * t1001) / 2.
    # Test equal:
    for key in manual_ucc_dict:
      self.assertAlmostEqual(manual_ucc_dict[key], ucc_amplitude_dict[key])
    self.assertEqual(
        len(manual_ucc_dict.keys()), len(ucc_amplitude_dict.keys()))


if __name__ == "__main__":
  unittest.main()
