"""Class and functions to store molecular Hamiltonians."""
import fermion_operators
import qubit_operators
import itertools
import numpy
import copy
from molecular_coefficients import MolecularCoefficients


class MolecularOperatorError(Exception):
  pass


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


class MolecularOperator(MolecularCoefficients):
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
    super(MolecularOperator, self).__init__(constant, one_body_coefficients,
                                            two_body_coefficients)

  def __symmetry_iter_helper(self, symmetry):
    """Iterate all terms that are not in the same symmetry group.
    Four point symmetry:
      1. pq = qp.
      2. pqrs = srqp = qpsr = rspq.
    Eight point symmetry:
      1. pq = qp.
      2. pqrs = rqps = psrq = srqp = qpsr = rspq = spqr = qrsp.

    Args:
      symmetry: The symmetry, 4 or 8, to represent four point or eight point.
    """
    if symmetry != 4 and symmetry != 8:
      raise ValueError('The symmetry must be one of 4, 8.')

    if self.constant:  # Constant.
      yield []

    for p in range(self.n_qubits):  # One-body terms.
      for q in range(p + 1):
        if self.one_body_coefficients[p, q]:
          yield [p, q]

    record_map = {}
    for p in range(self.n_qubits):  # Two-body terms.
      for q in range(self.n_qubits):
        for r in range(self.n_qubits):
          for s in range(self.n_qubits):
            if self.two_body_coefficients[p, q, r, s] and \
               (p, q, r, s) not in record_map:
              yield [p, q, r, s]
              record_map[(p, q, r, s)] = []
              record_map[(s, r, q, p)] = []
              record_map[(q, p, s, r)] = []
              record_map[(r, s, p, q)] = []
              if symmetry == 8:
                record_map[(p, s, r, q)] = []
                record_map[(s, p, q, r)] = []
                record_map[(q, r, s, p)] = []
                record_map[(r, q, p, s)] = []

  def four_point_iter(self):
    for key in self.__symmetry_iter_helper(4):
      yield key

  def eight_point_iter(self):
    for key in self.__symmetry_iter_helper(8):
      yield key

  def get_fermion_operator(self):
    """Output MolecularOperator as an instance of FermionOperator class.

    Returns:
      fermion_operator: An instance of the FermionOperator class.
    """
    # Initialize with identity term.
    identity = fermion_operators.FermionTerm([], self.constant)
    fermion_operator = fermion_operators.FermionOperator([identity])

    for p in range(self.n_qubits):
      for q in range(self.n_qubits):
        # Add one-body terms.
        coefficient = self[p, q]
        fermion_operator += fermion_operators.FermionTerm(
            [(p, 1), (q, 0)], coefficient)

        for r in range(self.n_qubits):
          for s in range(self.n_qubits):
            # Add two-body terms.
            coefficient = self[p, q, r, s]
            fermion_operator += fermion_operators.FermionTerm(
                [(p, 1), (q, 1), (r, 0), (s, 0)], coefficient)

    return fermion_operator

  @staticmethod
  def jordan_wigner_one_body(p, q):
    """Map the term a^\dagger_p a_q + a^\dagger_q a_p to a qubit operator.

    Note that the diagonal terms are divided by a factor of 2 because they
    are equal to their own Hermitian conjugate."""
    # Handle off-diagonal terms.
    qubit_operator = qubit_operators.QubitOperator()
    if p != q:
      a, b = sorted([p, q])
      parity_string = [(z, 'Z') for z in range(a + 1, b)]
      for operator in ['X', 'Y']:
        operators = [(a, operator)] + parity_string + [(b, operator)]
        qubit_operator += qubit_operators.QubitTerm(operators, .5)

    # Handle diagonal terms.
    else:
      qubit_operator += qubit_operators.QubitTerm([], .5)
      qubit_operator += qubit_operators.QubitTerm([(p, 'Z')], -.5)

    return qubit_operator

  @staticmethod
  def jordan_wigner_two_body(p, q, r, s):
    """Map the term a^\dagger_p a^\dagger_q a_r a_s + h.c. to qubit operator.

    Note that the diagonal terms are divided by a factor of two because they
    are equal to their own Hermitian conjugate."""
    # Initialize qubit operator.
    qubit_operator = qubit_operators.QubitOperator()

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
        qubit_operator += qubit_operators.QubitTerm(operators, coefficient)

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
      pauli_z = qubit_operators.QubitTerm([(c, 'Z')], 1.)
      for operator in ['X', 'Y']:
        operators = [(a, operator)] + parity_string + [(b, operator)]

        # Get coefficient.
        if (p == s) or (q == r):
          coefficient = .25
        else:
          coefficient = -.25

        # Add term.
        hopping_term = qubit_operators.QubitTerm(operators, coefficient)
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
      qubit_operator -= qubit_operators.QubitTerm([], coefficient)
      qubit_operator += qubit_operators.QubitTerm([(p, 'Z')], coefficient)
      qubit_operator += qubit_operators.QubitTerm([(q, 'Z')], coefficient)
      qubit_operator -= qubit_operators.QubitTerm(
          [(min(q, p), 'Z'), (max(q, p), 'Z')], coefficient)

    return qubit_operator

  def jordan_wigner_transform(self):
    """Output MolecularOperator as QubitOperator class under JW transform.

    One could accomplish this very easily by first mapping to fermions and
    then mapping to qubits. We skip the middle step for the sake of speed.

    Returns:
      qubit_operator: An instance of the QubitOperator class.
    """
    # Initialize qubit operator.
    qubit_operator = qubit_operators.QubitOperator()

    # Add constant.
    qubit_operator += qubit_operators.QubitTerm([], self.constant)

    # Loop through all indices.
    for p in range(self.n_qubits):
      for q in range(self.n_qubits):

        # Handle one-body terms.
        coefficient = float(self[p, q])
        if coefficient and p >= q:
          qubit_operator += coefficient * self.jordan_wigner_one_body(p, q)

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
            transformed_term = self.jordan_wigner_two_body(p, q, r, s)
            transformed_term *= coefficient
            qubit_operator += transformed_term

    return qubit_operator

  def get_sparse_operator(self):
    fermion_operator = self.get_fermion_operator()
    sparse_operator = fermion_operator.jordan_wigner_sparse()
    return sparse_operator
