"""Class and functions to store molecular density operators."""
from molecular_coefficients import MolecularCoefficients
import fermion_operators
import qubit_operators
import itertools
import numpy
import copy
from functools import reduce


class MolecularRDMError(Exception):
  pass


# TODO(Wei Sun): Consider make this function a method of MolecularRDM?
def unpack_spatial_rdm(one_rdm_a,
                       one_rdm_b,
                       two_rdm_aa,
                       two_rdm_ab,
                       two_rdm_bb):
    """Covert from spin compact spatial format to spin-orbital format for RDM.

    Note: the compact 2-RDM is stored as follows where A/B are spin up/down:
    RDM[pqrs] = <| a_{p, A}^\dagger a_{r, A}^\dagger a_{q, A} a_{s, A} |>
      for 'AA'/'BB' spins.
    RDM[pqrs] = <| a_{p, A}^\dagger a_{r, B}^\dagger a_{q, B} a_{s, A} |>
      for 'AB' spins.

    Args:
      one_rdm_a: 2-index numpy array storing alpha spin
        sector of 1-electron reduced density matrix.
      one_rdm_b: 2-index numpy array storing beta spin
        sector of 1-electron reduced density matrix.
      two_rdm_aa: 4-index numpy array storing alpha-alpha spin
        sector of 2-electron reduced density matrix.
      two_rdm_ab: 4-index numpy array storing alpha-beta spin
        sector of 2-electron reduced density matrix.
      two_rdm_bb: 4-index numpy array storing beta-beta spin
        sector of 2-electron reduced density matrix.

    Returns:
      one_rdm: 2-index numpy array storing 1-electron density matrix
        in full spin-orbital space.
      two_rdm: 4-index numpy array storing 2-electron density matrix
        in full spin-orbital space.
    """
    # Initialize RDMs.
    n_orbitals = one_rdm_a.shape[0]
    n_qubits = 2 * n_orbitals
    one_rdm = numpy.zeros((n_qubits, n_qubits))
    two_rdm = numpy.zeros((n_qubits, n_qubits,
                           n_qubits, n_qubits))

    # Unpack compact representation.
    for p in range(n_orbitals):
      for q in range(n_orbitals):

        # Populate 1-RDM.
        one_rdm[2*p, 2*q] = one_rdm_a[p, q]
        one_rdm[2*p + 1, 2*q + 1] = one_rdm_b[p, q]

        # Continue looping to unpack 2-RDM.
        for r in range(n_orbitals):
          for s in range(n_orbitals):

            # Handle case of same spin.
            two_rdm[2*p, 2*q, 2*r, 2*s] = (
                two_rdm_aa[p, r, q, s])
            two_rdm[2*p + 1, 2*q + 1, 2*r + 1, 2*s + 1] = (
                two_rdm_bb[p, r, q, s])

            # Handle case of mixed spin.
            two_rdm[2*p, 2*q + 1, 2*r, 2*s + 1] = (
                two_rdm_ab[p, r, q, s])
            two_rdm[2*p, 2*q + 1, 2*r + 1, 2*s] = (
                -1. * two_rdm_ab[p, s, q, r])
            two_rdm[2*p + 1, 2*q, 2*r + 1, 2*s] = (
                two_rdm_ab[q, s, p, r])
            two_rdm[2*p + 1, 2*q, 2*r, 2*s + 1] = (
                -1. * two_rdm_ab[q, r, p, s])

    # Map to physicist notation and return.
    two_rdm = numpy.einsum('pqsr', two_rdm)
    return one_rdm, two_rdm


# TODO(Wei Sun): Consider make this function a method of MolecularRDM?
def restrict_to_active_space(one_body_integrals, two_body_integrals,
                             active_space_start, active_space_stop):
  """Restrict the molecule at a spatial orbital level to the active space
  defined by active_space=[start,stop]. Note that one_body_integrals and
  two_body_integrals must be defined in an orthonormal basis set,
  which is typically the case when defining an active space.

    Args:
      one_body_integrals: (N,N) numpy array containing the one-electron
        spatial integrals for a molecule.
      two_body_integrals: (N,N,N,N) numpy array containing the two-electron
        spatial integrals.
      active_space_start(int): spatial orbital index defining active
        space start.

    Returns:
      core_constant: Adjustment to constant shift in Hamiltonian from
        integrating out core orbitals
      one_body_integrals_new: New one-electron integrals over active space.
      two_body_integrals_new: New two-electron integrals over active space.
  """
  # Determine core constant
  core_constant = 0.0
  for i in range(active_space_start):
    core_constant += 2 * one_body_integrals[i, i]
    for j in range(active_space_start):
      core_constant += (2 * two_body_integrals[i, j, j, i] -
                        two_body_integrals[i, j, i, j])

  # Modified one electron integrals
  one_body_integrals_new = numpy.copy(one_body_integrals)
  for u in range(active_space_start, active_space_stop):
    for v in range(active_space_start, active_space_stop):
      for i in range(active_space_start):
        one_body_integrals_new[u, v] += (2 * two_body_integrals[i, u, v, i] -
                                         two_body_integrals[i, u, i, v])

  # Restrict integral ranges and change M appropriately
  return (core_constant,
          one_body_integrals_new[active_space_start: active_space_stop,
                                 active_space_start: active_space_stop],
          two_body_integrals[active_space_start: active_space_stop,
                             active_space_start: active_space_stop,
                             active_space_start: active_space_stop,
                             active_space_start: active_space_stop])


class MolecularRDM(MolecularCoefficients):
  """Class for storing molecular RDM, including 1-RDM and 2RDM.

  Attributes:
    n_qubits: An int giving the number of qubits.
    one_body_coefficients: The coefficients of the one-body terms (h[p, q]).
        This is an n_qubits x n_qubits numpy array of floats.
    two_body_coefficients: The coefficients of the two-body terms
        (h[p, q, r, s]). This is an n_qubits x n_qubits x n_qubits x
        n_qubits numpy array of floats.
    constant: A constant term in the operator given as a float.
        For instance, the nuclear repulsion energy.
  """

  def __init__(self, constant, one_body_coefficients, two_body_coefficients):
    """Initialize the MolecularRDM class.

    Args:
      constant: A constant term in the operator given as a float.
          For instance, the nuclear repulsion energy.
      one_body_coefficients: The coefficients of the one-body terms (h[p, q]).
          This is an n_qubits x n_qubits numpy array of floats.
      two_body_coefficients: The coefficients of the two-body terms
          (h[p, q, r, s]). This is an n_qubits x n_qubits x n_qubits x
          n_qubits numpy array of floats.
    """
    super(MolecularRDM, self).__init__(constant, one_body_coefficients,
                                       two_body_coefficients)

  def get_qubit_term_expectation(self, qubit_term):
    """Return expectation value of a QubitTerm with a molecular RDM (self).

    Args:
      qubit_term: QubitTerm instance to be evaluated on this MolecularRDM.

    Returns:
      expectation: A float giving the expectation value.

    Raises:
      MolecularRDMError: Observable not contained in 1-RDM or 2-RDM.
    """
    expectation = 0.
    reversed_fermion_operators = qubit_term.reverse_jordan_wigner(
        self.n_qubits)
    reversed_fermion_operators.normal_order()

    for fermion_term in reversed_fermion_operators:
      # Handle molecular terms.
      if fermion_term.is_molecular_term():
        indices = [operator[0] for operator in fermion_term]
        rdm_element = self[indices]
        expectation += rdm_element * fermion_term.coefficient
      # Handle non-molecular terms.
      elif len(fermion_term.operators) > 4:
        raise MolecularRDMError('Observable not contained '
                                'in 1-RDM or 2-RDM.')

    return expectation / qubit_term.coefficient

  def get_qubit_expectations(self, qubit_operator):
    """Return expectations of qubit op as coefficients of new qubit op.

    Args:
      qubit_operator: QubitOperator instance to be evaluated on this
          MolecularRDM.

    Returns:
      qubit_operator_expectations: QubitOperator with coefficients
          corresponding to expectation values of those operators.

    Raises:
      MolecularRDMError: Observable not contained in 1-RDM or 2-RDM.
    """
    qubit_operator_expectations = copy.deepcopy(qubit_operator)
    for qubit_term in qubit_operator_expectations:
      qubit_term.coefficient = self.get_qubit_term_expectation(qubit_term)
    return qubit_operator_expectations
