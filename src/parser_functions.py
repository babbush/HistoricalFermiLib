"""Helper functions for parsing data files of different types"""

import molecular_operators
import numpy


def parse_psi4_ccsd_amplitudes(number_orbitals,
                               n_alpha_electrons, n_beta_electrons,
                               psi_filename):
  """Parse coupled cluster singles and doubles amplitudes from psi4 file

  Args:
    number_orbitals(int): Number of total spin orbitals in the system
    n_alpha_electrons(int): Number of alpha electrons in the system
    n_beta_electrons(int): Number of beta electrons in the system
    psi_filename(str): Filename of psi4 output file

  Returns:
    molecule(MolecularOperator): Molecular Operator instance holding ccsd
      amplitudes
  """
  output_buffer = [line for line in open(psi_filename)]

  T1IA_index = None
  T1ia_index = None
  T2IJAB_index = None
  T2ijab_index = None
  T2IjAb_index = None

  # Find Start Indices
  for i, line in enumerate(output_buffer):
    if ("Largest TIA Amplitudes:" in line):
      T1IA_index = i

    elif ("Largest Tia Amplitudes:" in line):
      T1ia_index = i

    elif ("Largest TIJAB Amplitudes:" in line):
      T2IJAB_index = i

    elif ("Largest Tijab Amplitudes:" in line):
      T2ijab_index = i

    elif ("Largest TIjAb Amplitudes:" in line):
      T2IjAb_index = i

  T1IA_Amps = []
  T1ia_Amps = []

  T2IJAB_Amps = []
  T2ijab_Amps = []
  T2IjAb_Amps = []

  # Read T1's
  if (T1IA_index is not None):
    for line in output_buffer[T1IA_index + 1:]:
      ivals = line.split()
      if not ivals:
        break
      T1IA_Amps.append([int(ivals[0]), int(ivals[1]), float(ivals[2])])

  if (T1ia_index is not None):
    for line in output_buffer[T1ia_index + 1:]:
      ivals = line.split()
      if not ivals:
        break
      T1ia_Amps.append([int(ivals[0]), int(ivals[1]), float(ivals[2])])

  # Read T2's
  if (T2IJAB_index is not None):
    for line in output_buffer[T2IJAB_index + 1:]:
      ivals = line.split()
      if not ivals:
        break
      T2IJAB_Amps.append([int(ivals[0]), int(ivals[1]),
                          int(ivals[2]), int(ivals[3]),
                          float(ivals[4])])

  if (T2ijab_index is not None):
    for line in output_buffer[T2ijab_index + 1:]:
      ivals = line.split()
      if not ivals:
        break
      T2ijab_Amps.append([int(ivals[0]), int(ivals[1]),
                          int(ivals[2]), int(ivals[3]),
                          float(ivals[4])])

  if (T2IjAb_index is not None):
    for line in output_buffer[T2IjAb_index + 1:]:
      ivals = line.split()
      if not ivals:
        break
      T2IjAb_Amps.append([int(ivals[0]), int(ivals[1]),
                          int(ivals[2]), int(ivals[3]),
                          float(ivals[4])])

  # Determine if calculation is restricted / closed shell or otherwise
  restricted = T1ia_index is None and T2ijab_index is None

  # Store amplitudes with spin-orbital indexing, including appropriate symmetry
  single_amplitudes = numpy.zeros((number_orbitals, ) * 2)
  double_amplitudes = numpy.zeros((number_orbitals, ) * 4)

  # Define local helper routines for clear indexing of orbitals
  def alpha_occupied(i):
    return 2 * i

  def alpha_unoccupied(i):
    return 2 * (i + n_alpha_electrons)

  def beta_occupied(i):
    return 2 * i + 1

  def beta_unoccupied(i):
    return 2 * (i + n_beta_electrons) + 1

  # Store singles
  for entry in T1IA_Amps:
    i, a, value = entry
    single_amplitudes[alpha_occupied(i),
                      alpha_unoccupied(a)] = value
    if (restricted):
      single_amplitudes[beta_occupied(i),
                        beta_unoccupied(a)] = value

  for entry in T1ia_Amps:
    i, a, value = entry
    single_amplitudes[beta_occupied(i),
                      beta_unoccupied(a)] = value

  # Store doubles, include factor of 1/2 for convention
  for entry in T2IJAB_Amps:
    i, j, a, b, value = entry
    double_amplitudes[alpha_occupied(i),
                      alpha_unoccupied(a),
                      alpha_occupied(j),
                      alpha_unoccupied(b)] = -value / 2.
    if (restricted):
      double_amplitudes[beta_occupied(i),
                        beta_unoccupied(a),
                        beta_occupied(j),
                        beta_unoccupied(b)] = -value / 2.

  for entry in T2ijab_Amps:
    i, j, a, b, value = entry
    double_amplitudes[beta_occupied(i),
                      beta_unoccupied(a),
                      beta_occupied(j),
                      beta_unoccupied(b)] = -value / 2.

  for entry in T2IjAb_Amps:
    i, j, a, b, value = entry
    double_amplitudes[alpha_occupied(i),
                      alpha_unoccupied(a),
                      beta_occupied(j),
                      beta_unoccupied(b)] = -value / 2.

    if (restricted):
      double_amplitudes[beta_occupied(i),
                        beta_unoccupied(a),
                        alpha_occupied(j),
                        alpha_unoccupied(b)] = -value / 2.

  # Package into molecular operator
  molecule = molecular_operators.MolecularOperator(0.0,
                                                   single_amplitudes,
                                                   double_amplitudes)
  return molecule
