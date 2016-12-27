"""Class and functions to store molecular Hamiltonians / density operators."""
import fermion_operators
import qubit_operators
import numpy
import copy


class ErrorMolecularOperator(Exception):
  pass


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
    one_rdm = numpy.zeros((2 * n_orbitals, 2 * n_orbitals))
    two_rdm = numpy.zeros((2 * n_orbitals, 2 * n_orbitals,
                           2 * n_orbitals, 2 * n_orbitals))

    # Unpack compact representation.
    for p in range(n_orbitals):
      for q in range(n_orbitals):

        # Populate 1-RDM.
        one_rdm[2 * p, 2 * q] = one_rdm_a[p, q]
        one_rdm[2 * p + 1, 2 * q + 1] = one_rdm_b[p, q]

        # Continue looping to unpack 2-RDM.
        for r in range(n_orbitals):
          for s in range(n_orbitals):

            # Handle case of same spin.
            two_rdm[2 * p, 2 * q, 2 * r, 2 * s] = (
                two_rdm_aa[p, r, q, s])
            two_rdm[2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1] = (
                two_rdm_bb[p, r, q, s])

            # Handle case of mixed spin.
            two_rdm[2 * p, 2 * q + 1, 2 * r, 2 * s + 1] = (
                two_rdm_ab[p, r, q, s])
            two_rdm[2 * p, 2 * q + 1, 2 * r + 1, 2 * s] = (
                -1. * two_rdm_ab[p, s, q, r])
            two_rdm[2 * p + 1, 2 * q, 2 * r + 1, 2 * s] = (
                two_rdm_ab[q, s, p, r])
            two_rdm[2 * p + 1, 2 * q, 2 * r, 2 * s + 1] = (
                -1. * two_rdm_ab[q, r, p, s])

    # Map to physicist notation and return.
    two_rdm = numpy.einsum('pqsr', two_rdm)
    return one_rdm, two_rdm


def one_body_basis_change(one_body_operator,
                          rotation_matrix):
  """Change the basis of 1-body fermionic operators, e.g. the 1-RDM.

  M' = R^T.M.R where R is the rotation matrix, M is the fermion operator
  and M' is the transformed fermion operator.

  Args:
    one_body_operator: A square numpy array or matrix containing information
      about a 1-body operator such as the 1-body integrals or 1-RDM.
    rotation_matrix: A square numpy array or matrix having dimensions of
      n_orbitals by n_orbitals. Assumed to be real and invertible.

  Returns:
    transformed_one_body_operator: one_body_operator in the rotated basis.
  """
  # If operator acts on spin degrees of freedom, enlarge rotation matrix.
  n_orbitals = rotation_matrix.shape[0]
  if one_body_operator.shape[0] == 2 * n_orbitals:
    rotation_matrix = numpy.kron(rotation_matrix, numpy.eye(2))

  # Effect transformation and return.
  transformed_one_body_operator = numpy.einsum('qp, qr, rs',
                                               rotation_matrix,
                                               one_body_operator,
                                               rotation_matrix)
  return transformed_one_body_operator


def two_body_basis_change(two_body_operator,
                          rotation_matrix):
  """Change the basis of 2-body fermionic operators, e.g. the 2-RDM.

  Procedure we use is an N^5 transformation which can be expressed as
  (pq|rs) = \sum_a R^p_a (\sum_b R^q_b (\sum_c R^r_c (\sum_d R^s_d (ab|cd)))).

  Args:
    two_body_operator: a square rank 4 tensor in a numpy array containing
      information about a 2-body fermionic operator.
    rotation_matrix: A square numpy array or matrix having dimensions of
      n_orbitals by n_orbitals. Assumed to be real and invertible.

  Returns:
    transformed_two_body_operator: two_body_operator matrix in rotated basis.
  """
  # If operator acts on spin degrees of freedom, enlarge rotation matrix.
  n_orbitals = rotation_matrix.shape[0]
  if two_body_operator.shape[0] == 2 * n_orbitals:
    rotation_matrix = numpy.kron(rotation_matrix, numpy.eye(2))

  # Effect transformation and return.
  # TODO Jarrod: Make work without the two lines that perform permutations.
  two_body_operator = numpy.einsum('prsq', two_body_operator)
  first_sum = numpy.einsum('ds, abcd', rotation_matrix, two_body_operator)
  second_sum = numpy.einsum('cr, abcs', rotation_matrix, first_sum)
  third_sum = numpy.einsum('bq, abrs', rotation_matrix, second_sum)
  transformed_two_body_operator = numpy.einsum('ap, aqrs',
                                               rotation_matrix, third_sum)
  transformed_two_body_operator = numpy.einsum('psqr',
                                               transformed_two_body_operator)
  return transformed_two_body_operator


def restrict_to_active_space(one_body_integrals,
                             two_body_integrals,
                             active_space_start,
                             active_space_stop):
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
      core_constant += (2 * two_body_integrals[i, j, i, j] -
                        two_body_integrals[i, j, j, i])

  # Modified one electron integrals
  one_body_integrals_new = numpy.copy(one_body_integrals)
  for u in range(active_space_start, active_space_stop):
    for v in range(active_space_start, active_space_stop):
      for i in range(active_space_start):
        one_body_integrals_new[u, v] += (2 * two_body_integrals[i, u, i, v] -
                                         two_body_integrals[i, u, v, i])

  # Restrict integral ranges and change M appropriately
  return (core_constant,
          one_body_integrals_new[active_space_start: active_space_stop,
                                 active_space_start:active_space_stop],
          two_body_integrals[active_space_start:active_space_stop,
                             active_space_start:active_space_stop,
                             active_space_start:active_space_stop,
                             active_space_start:active_space_stop])


class MolecularOperator(object):

  """Class for storing 'molecular operators' which are defined to be
  fermionic operators consisting of one-body and two-body terms which
  conserve particle number and spin. The most common examples of data
  that will use this structure are molecular Hamiltonians and molecular
  2-RDM density operators. In principle, everything stored in this class
  could also be represented as the more general FermionOperator class.
  However, this class is able to exploit specific properties of molecular
  operators in order to enable more efficient manipulation of the data.
  Note that the operators stored in this class take the form:
      constant + \sum_{p, q} h_[p, q] a^\dagger_p a_q +
      \sum_{p, q, r, s} h_[p, q, r, s] a^\dagger_p a^\dagger_q a_r a_s.

  Attributes:
    n_orbitals: An int giving the number of orbitals.
    one_body_coefficients: The coefficients of the one-body terms (h[p, q]).
        This is an n_orbital x n_orbital numpy array of floats.
    two_body_coefficients: The coefficients of the two-body terms
        (h[p, q, r, s]). This is an n_orbital x n_orbital x n_orbital x
        n_orbital numpy array of floats.
    constant: A constant term in the operator given as a float.
        For instance, the nuclear repulsion energy.
  """

  def __init__(self,
               constant,
               one_body_coefficients,
               two_body_coefficients):
    """Initialize the MolecularOperator class.

    Args:
      constant: A constant term in the operator given as a float.
          For instance, the nuclear repulsion energy.
      one_body_coefficients: The coefficients of the one-body terms (h[p, q]).
          This is an n_orbital x n_orbital numpy array of floats.
      two_body_coefficients: The coefficients of the two-body terms
          (h[p, q, r, s]). This is an n_orbital x n_orbital x n_orbital x
          n_orbital numpy array of floats.
    """
    self.n_orbitals = one_body_coefficients.shape[0]
    self.constant = constant
    self.one_body_coefficients = one_body_coefficients
    self.two_body_coefficients = two_body_coefficients

  def rotate_basis(self, rotation_matrix):
    """Rotate the orbital basis of the MolecularOperator.

    Args:
      rotation_matrix: A square numpy array or matrix having dimensions of
        n_orbitals by n_orbitals. Assumed to be real and invertible.
    """
    self.one_body_coefficients = one_body_basis_change(
        self.one_body_coefficients, rotation_matrix)
    self.two_body_coefficients = two_body_basis_change(
        self.two_body_coefficients, rotation_matrix)

  def get_fermion_operator(self):
    """Output MolecularOperator as an instance of FermionOperator class.

    Returns:
      fermion_operator: An instance of the FermionOperator class.
    """
    # Add identity term.
    identity = fermion_operators.FermionTerm(self.n_orbitals, self.constant)
    terms = [identity]

    # Loop through terms.
    for p in range(self.n_orbitals):
      for q in range(self.n_orbitals):

        # Add one-body terms.
        coefficient = self.one_body_coefficients[p, q]
        terms += [fermion_operators.FermionTerm(
                  self.n_orbitals, coefficient, [(p, 1), (q, 0)])]

        # Keep looping.
        for r in range(self.n_orbitals):
          for s in range(self.n_orbitals):

            # Add two-body terms.
            coefficient = self.two_body_coefficients[p, q, r, s]
            terms += [fermion_operators.FermionTerm(
                      self.n_orbitals, coefficient,
                      [(p, 1), (q, 1), (r, 0), (s, 0)])]

    # Make operator and return.
    fermion_operator = fermion_operators.FermionOperator(
        self.n_orbitals, terms)
    return fermion_operator

  def jordan_wigner_transform(self):
    """Output MolecularOperator as QubitOperator class under JW transform.

    Returns:
      qubit_operator: An instance of the QubitOperator class.
    """
    # TODO: hard code the transformation without going through fermionic
    # class in order to improve performance.
    fermion_operator = self.get_fermion_operator()
    qubit_operator = fermion_operator.jordan_wigner_transform()
    return qubit_operator

  def get_sparse_matrix(self):
    # TODO: hard code the transformation without going through pauli
    # class in order to improve performance.
    qubit_operator = self.jordan_wigner_transform()
    sparse_operator = qubit_operator.get_sparse_matrix()
    return sparse_operator

  def get_qubit_expectations(self, qubit_operator):
    """Take a qubit operator and return expectation values as coefficients
      of a new operator

    Args:
      qubit_operator: QubitOperator instance to be evaluated on this
        molecular_operators reduced density matrices

    Returns:
      qubit_operator_expectations: qubit operator with coefficients
        corresponding to expectation values of those operators
      """
    qubit_operator_expectations = copy.deepcopy(qubit_operator)
    for qubit_term in qubit_operator_expectations:
      tmp_qubit_term = qubit_operators.QubitTerm(self.n_orbitals,
                                                 1.0,
                                                 qubit_term.operators)
      tmp_qubit_operator = qubit_operators.QubitOperator(self.n_orbitals,
                                                         [tmp_qubit_term])
      expectation_value = tmp_qubit_operator.expectation_fermion(self)
      qubit_term.coefficient = expectation_value
    return qubit_operator_expectations

  def get_jordan_wigner_rdm(self):
    """Transform an RDM into an RDM over qubit operators.

    Returns:
      qubit_rdm: The RDM represented as a qubit operator.
    """
    # Map density operator to qubits.
    one_body_terms = [fermion_operators.FermionTerm(self.n_orbitals,
                                                    1.0,
                                                    [(i, 1), (j, 0)])
                      for i in range(self.n_orbitals)
                      for j in range(self.n_orbitals)]
    two_body_terms = [fermion_operators.FermionTerm(self.n_orbitals, 1.0,
                                                    [(i, 1), (j, 1),
                                                     (k, 0), (l, 0)])
                      for i in range(self.n_orbitals)
                      for j in range(self.n_orbitals)
                      for k in range(self.n_orbitals)
                      for l in range(self.n_orbitals)]
    fermion_rdm = fermion_operators.FermionOperator(self.n_orbitals,
                                                    one_body_terms +
                                                    two_body_terms)
    fermion_rdm.normal_order()
    qubit_rdm = fermion_rdm.jordan_wigner_transform()

    # Compute QubitTerm variances.
    for qubit_term in qubit_rdm:
      qubit_term.coefficient = 1.0
      qubit_operator = qubit_operators.QubitOperator(
          self.n_orbitals, [qubit_term])
      expectation_value = qubit_operator.expectation_fermion(self)
      qubit_term.coefficient = expectation_value

    # Return.
    return qubit_rdm
