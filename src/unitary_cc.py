"""Module to create and manipulate unitary coupled cluster operators"""

import itertools
import fermion_operators


def uccsd_operator(single_amplitudes, double_amplitudes):
  """Create a fermionic operator that is the generator of uccsd

  Args:
    single_amplitudes(ndarray): [NxN] array storing single excitation
      amplitudes corresponding to t[i,j] * (a_i^\dagger a_j + H.C.)
    double_amplitudes(ndarray): [NxNxNxN] array storing double excitation
      amplitudes corresponding to
      t[i,j,k,l] * (a_i^\dagger a_j a_k^\dagger a_l + H.C.)

  Returns:
    uccsd_operator(FermionOperator): Anti-hermitian fermion operator that is
      the generator for the uccsd wavefunction.

  """
  n_orbitals = single_amplitudes.shape[0]
  assert(n_orbitals == double_amplitudes.shape[0])
  uccsd_operator = fermion_operators.FermionOperator()

  # Add single excitations
  for i, j in itertools.product(range(n_orbitals), repeat=2):
    if single_amplitudes[i, j] == 0.:
      continue
    uccsd_operator += fermion_operators. \
        FermionTerm([(i, 1), (j, 0)], single_amplitudes[i, j])
    uccsd_operator += fermion_operators. \
        FermionTerm([(j, 1), (i, 0)], -single_amplitudes[i, j])

    # Add double excitations
  for i, j, k, l in itertools.product(range(n_orbitals), repeat=4):
    if double_amplitudes[i, j, k, l] == 0.:
      continue
    uccsd_operator += fermion_operators. \
        FermionTerm([(i, 1), (j, 0), (k, 1), (l, 0)],
                    double_amplitudes[i, j, k, l])
    uccsd_operator += fermion_operators. \
        FermionTerm([(l, 1), (k, 0), (j, 1), (i, 0)],
                    -double_amplitudes[i, j, k, l])

  return uccsd_operator


def ccsd_operator(single_amplitudes, double_amplitudes):
  """Create a fermionic operator that is the generator of ccsd

  Non-unitary coupled cluster with single and double excitations.  This is
  mainly used to test that the coefficients are being correctly
  loaded from external packages.

  Args:
    single_amplitudes(ndarray): [NxN] array storing single excitation
      amplitudes corresponding to t[i,j] * (a_i^\dagger a_j)
    double_amplitudes(ndarray): [NxNxNxN] array storing double excitation
      amplitudes corresponding to
      t[i,j,k,l] * (a_i^\dagger a_j a_k^\dagger a_l)

  Returns:
    ccsd_operator(FermionOperator): Anti-hermitian fermion operator that is
      the generator for the uccsd wavefunction.

  """
  n_orbitals = single_amplitudes.shape[0]
  assert(n_orbitals == double_amplitudes.shape[0])
  ccsd_operator = fermion_operators.FermionOperator()

  # Add single excitations
  for i, j in itertools.product(range(n_orbitals), repeat=2):
    if single_amplitudes[i, j] == 0.:
      continue
    ccsd_operator += fermion_operators. \
        FermionTerm([(i, 1), (j, 0)], single_amplitudes[i, j])

  # Add double excitations
  for i, j, k, l in itertools.product(range(n_orbitals), repeat=4):
    if double_amplitudes[i, j, k, l] == 0.:
      continue
    ccsd_operator += fermion_operators. \
        FermionTerm([(i, 1), (j, 0), (k, 1), (l, 0)],
                    double_amplitudes[i, j, k, l])

  return ccsd_operator
