"""Class and functions to store molecular Hamiltonians / density operators."""
import fermion_operators
import qubit_operators
import itertools
import numpy
import copy


class MolecularOperatorError(Exception):
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


def one_body_basis_change(one_body_operator, rotation_matrix):
  """Change the basis of 1-body fermionic operators, e.g. the 1-RDM.

  M' = R^T.M.R where R is the rotation matrix, M is the fermion operator
  and M' is the transformed fermion operator.

  Args:
    one_body_operator: A square numpy array or matrix containing information
      about a 1-body operator such as the 1-body integrals or 1-RDM.
    rotation_matrix: A square numpy array or matrix having dimensions of
      n_qubits by n_qubits. Assumed to be real and invertible.

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


def two_body_basis_change(two_body_operator, rotation_matrix):
  """Change the basis of 2-body fermionic operators, e.g. the 2-RDM.

  Procedure we use is an N^5 transformation which can be expressed as
  (pq|rs) = \sum_a R^p_a (\sum_b R^q_b (\sum_c R^r_c (\sum_d R^s_d (ab|cd)))).

  Args:
    two_body_operator: a square rank 4 tensor in a numpy array containing
      information about a 2-body fermionic operator.
    rotation_matrix: A square numpy array or matrix having dimensions of
      n_qubits by n_qubits. Assumed to be real and invertible.

  Returns:
    transformed_two_body_operator: two_body_operator matrix in rotated basis.
  """
  # If operator acts on spin degrees of freedom, enlarge rotation matrix.
  n_orbitals = rotation_matrix.shape[0]
  if two_body_operator.shape[0] == 2 * n_orbitals:
    rotation_matrix = numpy.kron(rotation_matrix, numpy.eye(2))

  # Effect transformation and return.
  # TODO: Make work without the two lines that perform permutations.
  two_body_operator = numpy.einsum('prsq', two_body_operator)
  first_sum = numpy.einsum('ds, abcd', rotation_matrix, two_body_operator)
  second_sum = numpy.einsum('cr, abcs', rotation_matrix, first_sum)
  third_sum = numpy.einsum('bq, abrs', rotation_matrix, second_sum)
  transformed_two_body_operator = numpy.einsum('ap, aqrs',
                                               rotation_matrix, third_sum)
  transformed_two_body_operator = numpy.einsum('psqr',
                                               transformed_two_body_operator)
  return transformed_two_body_operator


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
    """Initialize the MolecularOperator class.

    Args:
      constant: A constant term in the operator given as a float.
          For instance, the nuclear repulsion energy.
      one_body_coefficients: The coefficients of the one-body terms (h[p, q]).
          This is an n_qubits x n_qubits numpy array of floats.
      two_body_coefficients: The coefficients of the two-body terms
          (h[p, q, r, s]). This is an n_qubits x n_qubits x n_qubits x
          n_qubits numpy array of floats.
    """
    # Make sure nonzero elements are only for normal ordered terms.
    self.n_qubits = one_body_coefficients.shape[0]
    self.constant = constant
    self.one_body_coefficients = one_body_coefficients
    self.two_body_coefficients = two_body_coefficients

  def __getitem__(self, args):
    if len(args) == 4:
      p, q, r, s = args
      return self.two_body_coefficients[p, q, r, s]
    elif len(args) == 2:
      p, q = args
      return self.one_body_coefficients[p, q]
    elif not len(args):
      return self.constant

  def __setitem__(self, args, value):
    if len(args) == 4:
      p, q, r, s = args
      self.two_body_coefficients[p, q, r, s] = value
    elif len(args) == 2:
      p, q = args
      self.one_body_coefficients[p, q] = value
    elif not len(args):
      self.constant = value
    else:
      raise ValueError('args must be of length 0, 2, or 4.')

  def __eq__(self, molecular_operator):
    tol = 1e-12
    diff = max(abs(self.constant - molecular_operator.constant),
               numpy.amax(
                   numpy.absolute(self.one_body_coefficients -
                                  molecular_operator.one_body_coefficients)),
               numpy.amax(
                   numpy.absolute(self.two_body_coefficients -
                                  molecular_operator.two_body_coefficients)))
    return diff < tol

  def __neq__(self, molecular_operator):
    return not (self == molecular_operator)

  def __str__(self):
    """Print out the elements of the MolecularOperator in readable fashion."""

    # Start with the constant.
    string = '[] {}\n\n'.format(self.constant)

    # Loop over one-body terms.
    for p in range(self.n_qubits):
      for q in range(self.n_qubits):
        coefficient = self.one_body_coefficients[p, q]
        if coefficient:
          string += '[{} {}] {}\n'.format(p, q, coefficient)

    # Loop over two-body terms.
    for p in range(self.n_qubits):
      for q in range(self.n_qubits):
        for r in range(self.n_qubits):
          for s in range(self.n_qubits):
            coefficient = self.two_body_coefficients[p, q, r, s]
            if coefficient:
              string += '\n[{} {} {} {}] {}'.format(p, q, r, s, coefficient)

    # Return.
    return string if string else '0'

  def __repr__(self):
    return str(self)

  def rotate_basis(self, rotation_matrix):
    """Rotate the orbital basis of the MolecularOperator.

    Args:
      rotation_matrix: A square numpy array or matrix having dimensions of
        n_qubits by n_qubits. Assumed to be real and invertible.
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
    # Initialize with identity term.
    identity = fermion_operators.FermionTerm(self.n_qubits, [], self.constant)
    fermion_operator = fermion_operators.FermionOperator(
        self.n_qubits, [identity])

    for p in range(self.n_qubits):
      for q in range(self.n_qubits):
        # Add one-body terms.
        coefficient = self[p, q]
        fermion_operator += fermion_operators.FermionTerm(
            self.n_qubits, [(p, 1), (q, 0)], coefficient)

        for r in range(self.n_qubits):
          for s in range(self.n_qubits):
            # Add two-body terms.
            coefficient = self[p, q, r, s]
            fermion_operator += fermion_operators.FermionTerm(
                self.n_qubits, [(p, 1), (q, 1), (r, 0), (s, 0)], coefficient)

    return fermion_operator

  @staticmethod
  def jordan_wigner_one_body(n_qubits, p, q):
    """Map the term a^\dagger_p a_q + a^\dagger_q a_p to a qubit operator.

    Note that the diagonal terms are divided by a factor of 2 because they
    are equal to their own Hermitian conjugate."""
    # Handle off-diagonal terms.
    qubit_operator = qubit_operators.QubitOperator(n_qubits)
    if p != q:
      a, b = sorted([p, q])
      parity_string = [(z, 'Z') for z in range(a + 1, b)]
      for operator in ['X', 'Y']:
        operators = [(a, operator)] + parity_string + [(b, operator)]
        qubit_operator += qubit_operators.QubitTerm(n_qubits, operators, .5)

    # Handle diagonal terms.
    else:
      qubit_operator += qubit_operators.QubitTerm(n_qubits, [], .5)
      qubit_operator += qubit_operators.QubitTerm(n_qubits, [(p, 'Z')], -.5)

    return qubit_operator

  @staticmethod
  def jordan_wigner_two_body(n_qubits, p, q, r, s):
    """Map the term a^\dagger_p a^\dagger_q a_r a_s + h.c. to qubit operator.

    Note that the diagonal terms are divided by a factor of two because they
    are equal to their own Hermitian conjugate."""
    # Initialize qubit operator.
    qubit_operator = qubit_operators.QubitOperator(n_qubits)

    # Return zero terms.
    if (p == q) or (r == s):
      return qubit_operator

    # Handle case of four unique indices.
    elif len(set([p, q, r, s])) == 4:

      # Loop through different operators which act on each tensor factor.
      for operator_p, operator_q, operator_r in itertools.product(['X', 'Y'],
                                                                  repeat=3):
        if [operator_p, operator_q, operator_r].count('X') % 2:
          operator_s = 'X'
        else:
          operator_s = 'Y'

        # Sort operators.
        [(a, operator_a), (b, operator_b),
         (c, operator_c), (d, operator_d)] = sorted(
             [(p, operator_p), (q, operator_q),
              (r, operator_r), (s, operator_s)], key=lambda pair: pair[0])

        # Computer operator strings.
        operators = [(a, operator_a)]
        operators += [(z, 'Z') for z in range(a + 1, b)]
        operators += [(b, operator_b)]
        operators += [(c, operator_c)]
        operators += [(z, 'Z') for z in range(c + 1, d)]
        operators += [(d, operator_d)]

        # Get coefficients.
        coefficient = .125
        parity_condition = bool(operator_p != operator_q or
                                operator_p == operator_r)
        if (p > q) ^ (r > s):
          if not parity_condition:
            coefficient *= -1.
        elif parity_condition:
          coefficient *= -1.

        # Add term.
        qubit_operator += qubit_operators.QubitTerm(
            n_qubits, operators, coefficient)

    # Handle case of three unique indices.
    elif len(set([p, q, r, s])) == 3:

      # Identify equal tensor factors.
      if p == r:
        a, b = sorted([q, s])
        c = p
      elif p == s:
        a, b = sorted([q, r])
        c = p
      elif q == r:
        a, b = sorted([p, s])
        c = q
      elif q == s:
        a, b = sorted([p, r])
        c = q

      # Get operators.
      parity_string = [(z, 'Z') for z in range(a + 1, b)]
      pauli_z = qubit_operators.QubitTerm(n_qubits, [(c, 'Z')], 1.)
      for operator in ['X', 'Y']:
        operators = [(a, operator)] + parity_string + [(b, operator)]

        # Get coefficient.
        if (p == s) or (q == r):
          coefficient = .25
        else:
          coefficient = -.25

        # Add term.
        hopping_term = qubit_operators.QubitTerm(
            n_qubits, operators, coefficient)
        qubit_operator -= pauli_z * hopping_term
        qubit_operator += hopping_term

    # Handle case of two unique indices.
    elif len(set([p, q, r, s])) == 2:

      # Get coefficient.
      if (p, q, r, s) == (s, r, q, p):
        coefficient = .25
      else:
        coefficient = .5
      if p == s:
        coefficient *= -1.

      # Add terms.
      qubit_operator -= qubit_operators.QubitTerm(
          n_qubits, [], coefficient)
      qubit_operator += qubit_operators.QubitTerm(
          n_qubits, [(p, 'Z')], coefficient)
      qubit_operator += qubit_operators.QubitTerm(
          n_qubits, [(q, 'Z')], coefficient)
      qubit_operator -= qubit_operators.QubitTerm(
          n_qubits, [(min(q, p), 'Z'), (max(q, p), 'Z')], coefficient)

    return qubit_operator

  def jordan_wigner_transform(self):
    """Output MolecularOperator as QubitOperator class under JW transform.

    One could accomplish this very easily by first mapping to fermions and
    then mapping to qubits. We skip the middle step for the sake of speed.

    Returns:
      qubit_operator: An instance of the QubitOperator class.
    """
    # Initialize qubit operator.
    qubit_operator = qubit_operators.QubitOperator(self.n_qubits)

    # Add constant.
    qubit_operator += qubit_operators.QubitTerm(self.n_qubits, [],
                                                self.constant)

    # Loop through all indices.
    for p in range(self.n_qubits):
      for q in range(self.n_qubits):

        # Handle one-body terms.
        coefficient = float(self[p, q])
        if coefficient and p >= q:
          qubit_operator += coefficient * self.jordan_wigner_one_body(
              self.n_qubits, p, q)

        # Keep looping for the two-body terms.
        for r in range(self.n_qubits):
          for s in range(self.n_qubits):
            coefficient = float(self[p, q, r, s])

            # Skip zero terms.
            if (not coefficient) or (p == q) or (r == s):
              continue

            # Identify and skip one of the complex conjugates.
            if [p, q, r, s] != [s, r, q, p]:
              if len(set([p, q, r, s])) == 4:
                if min(r, s) < min(p, q):
                  continue
              else:
                if q < p:
                  continue

            # Handle the two-body terms.
            transformed_term = self.jordan_wigner_two_body(
                self.n_qubits, p, q, r, s)
            transformed_term *= coefficient
            qubit_operator += transformed_term

    return qubit_operator

  def get_qubit_term_expectation(self, qubit_term):
    """Return expectation value of a QubitTerm with a molecular RDM (self).

    Args:
      qubit_term: QubitTerm instance to be evaluated on this
          MolecularOperator representing a reduced density matrix.

    Returns:
      expectation: A float giving the expectation value.

    Raises:
      MolecularOperatorError: Observable not contained in 1-RDM or 2-RDM.
    """
    expectation = 0.
    reversed_fermion_operators = qubit_term.reverse_jordan_wigner()
    reversed_fermion_operators.normal_order()

    for fermion_term in reversed_fermion_operators:
      # Handle molecular terms.
      if fermion_term.is_molecular_term():
        indices = [operator[0] for operator in fermion_term]
        rdm_element = self[indices]
        expectation += rdm_element * fermion_term.coefficient
      # Handle non-molecular terms.
      elif len(fermion_term.operators) > 4:
        raise MolecularOperatorError('Observable not contained '
                                     'in 1-RDM or 2-RDM.')

    return expectation / qubit_term.coefficient

  def get_qubit_expectations(self, qubit_operator):
    """Return expectations of qubit op as coefficients of new qubit op.

    Note that this method is designed to be called on RDM MolecularOperators.

    Args:
      qubit_operator: QubitOperator instance to be evaluated on this
          MolecularOperator reduced density matrices.

    Returns:
      qubit_operator_expectations: QubitOperator with coefficients
          corresponding to expectation values of those operators.

    Raises:
      MolecularOperatorError: Observable not contained in 1-RDM or 2-RDM.
    """
    qubit_operator_expectations = copy.deepcopy(qubit_operator)
    for qubit_term in qubit_operator_expectations:
      qubit_term.coefficient = self.get_qubit_term_expectation(qubit_term)
    return qubit_operator_expectations

  def get_sparse_operator(self):
    fermion_operator = self.get_fermion_operator()
    sparse_operator = fermion_operator.jordan_wigner_sparse()
    return sparse_operator
