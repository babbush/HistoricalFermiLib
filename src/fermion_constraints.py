"""Constraints on fermionic reduced density matrices"""
import fermion_operators
import qubit_operators
import copy
import numpy


class FermionConstraints(object):
  """Generates constraints on fermion reduced density matrices
    in terms of Pauli operators.

    The specific constraints implemented are the known 2-positivity constraints
      on the one- and two-fermion reduced density matrices. Constraints are
      generated in the form of tuples of (QubitOperator, float),
      such that the expected value of the QubitOperator on a physical state
      should be equal to the float that is provided.  Generators are used
      to avoid issues with memory in the two-body constraints.

  Attributes:
    n_qubits(int): number of qubits the operators act on
    n_fermions(int): number of fermions in the system
  """

  def __init__(self, n_qubits, n_fermions):
    """ Initialize constraint class

    Args:
      n_qubits(int): Number of qubits in the system
      n_fermions(int): Number of fermions in the system
      """
    self.n_qubits = n_qubits
    self.n_fermions = n_fermions

  @staticmethod
  def extract_imaginary_operator(operator):
    """Returns the imaginary component of qubit operator

    Args:
      operator(QubitOperator): Full operator to get imaginary component from

    Returns:
      new_operator(QubitOperator): Imaginary component of operator
    """
    new_operator = qubit_operators.QubitOperator(operator.n_qubits)
    for term in operator:
      if (numpy.abs(numpy.imag(term.coefficient)) > 1e-12):
        new_operator += qubit_operators.\
            QubitTerm(operator.n_qubits,
                      numpy.imag(term.coefficient),
                      term.operators)
    return new_operator

  @staticmethod
  def extract_real_operator(operator):
    """Returns the real component of qubit operator

    Args:
      operator(QubitOperator): Full operator to get real component from

    Returns:
      new_operator(QubitOperator): Real component of operator
    """
    new_operator = qubit_operators.QubitOperator(operator.n_qubits)
    for term in operator:
      if (numpy.abs(numpy.real(term.coefficient)) > 1e-12):
        new_operator += qubit_operators.\
            QubitTerm(operator.n_qubits,
                      numpy.real(term.coefficient),
                      term.operators)
    return new_operator

  @staticmethod
  def constraint_trivial(operator):
    """Check if an operator contains any non-trivial terms (i.e. X, Y, Z)

    Args:
      operator(QubitOperator): Operator to check for non-trivial terms

    Returns:
      False if any term is non-trivial, True otherwise
    """
    for term in operator:
      if term.operators:
        return False
    return True

  def constraints_one_body(self):
    """A generator for one-body positivity constraints

      Yields:
        Constraint tuples of the form (QubitOperator, float), where
        the expectation value of the QubitOperator should be equal to the float
    """

    # One-RDM Trace Condition
    #print("1-RDM Trace Condition")
    constraint_operator = qubit_operators.QubitOperator(self.n_qubits)
    for i in range(self.n_qubits):
      transformed_term = fermion_operators.FermionTerm(self.n_qubits,
                                                       1.0,
                                                       [(i, 1), (i, 0)]).\
          jordan_wigner_transform()
      constraint_operator += transformed_term

    real_constraint = self.extract_real_operator(constraint_operator)
    if not self.constraint_trivial(real_constraint):
        yield (real_constraint, self.n_fermions)

    imaginary_constraint = self.extract_imaginary_operator(constraint_operator)
    if not self.constraint_trivial(imaginary_constraint):
        yield (imaginary_constraint, 0.0)

    # Diagonal Particle-Hole
    #print("1-RDM Diagonal Particle-Hole")
    for i in range(self.n_qubits):
      constraint_operator = qubit_operators.QubitOperator(self.n_qubits)
      transformed_term = fermion_operators.FermionTerm(self.n_qubits,
                                                       1.0,
                                                       [(i, 1), (i, 0)]).\
          jordan_wigner_transform()
      constraint_operator += transformed_term
      transformed_term = fermion_operators.FermionTerm(self.n_qubits,
                                                       1.0,
                                                       [(i, 0), (i, 1)]).\
          jordan_wigner_transform()
      constraint_operator += transformed_term

      real_constraint = self.extract_real_operator(constraint_operator)
      if not self.constraint_trivial(real_constraint):
        yield (real_constraint, 1.0)

      imaginary_constraint = self.\
          extract_imaginary_operator(constraint_operator)
      if not self.constraint_trivial(imaginary_constraint):
        yield (imaginary_constraint, 0.0)

    # Off-Diagonal Particle-Hole
    #print("1-RDM Off-Diagonal Particle-Hole")
    for i in range(self.n_qubits):
      for j in range(i + 1, self.n_qubits):
        constraint_operator = qubit_operators.QubitOperator(self.n_qubits)
        transformed_term = fermion_operators.FermionTerm(self.n_qubits,
                                                         1.0,
                                                         [(i, 1), (j, 0)]).\
            jordan_wigner_transform()
        constraint_operator += transformed_term
        transformed_term = fermion_operators.FermionTerm(self.n_qubits,
                                                         1.0,
                                                         [(i, 0), (j, 1)]).\
            jordan_wigner_transform()
        constraint_operator += transformed_term

        real_constraint = self.extract_real_operator(constraint_operator)
        if not self.constraint_trivial(real_constraint):
          yield (real_constraint, 0.0)

        imaginary_constraint = self.extract_imaginary_operator(
            constraint_operator)
        if not self.constraint_trivial(imaginary_constraint):
          yield (imaginary_constraint, 0.0)

    # One-Body Hermiticity
    #print("1-RDM Hermiticity")
    for i in range(self.n_qubits):
      for j in range(i + 1, self.n_qubits):
        constraint_operator = qubit_operators.QubitOperator(self.n_qubits)
        transformed_term = fermion_operators.FermionTerm(self.n_qubits,
                                                         1.0,
                                                         [(i, 1), (j, 0)]). \
            jordan_wigner_transform()
        constraint_operator += transformed_term
        transformed_term = fermion_operators.FermionTerm(self.n_qubits,
                                                         -1.0,
                                                         [(j, 1), (i, 0)]). \
            jordan_wigner_transform()
        constraint_operator += transformed_term

        real_constraint = self.extract_real_operator(constraint_operator)
        if not self.constraint_trivial(real_constraint):
          yield (real_constraint, 0.0)

        imaginary_constraint = self.extract_imaginary_operator(
            constraint_operator)
        if not self.constraint_trivial(imaginary_constraint):
          yield (imaginary_constraint, 0.0)

  def constraints_two_body(self):
    """A generator for two-body positivity constraints

      Yields:
        Constraint tuples of the form (QubitOperator, float), where
        the expectation value of the QubitOperator should be equal to the float
    """
    # 2 Body Trace Condition
    constraint_operator = qubit_operators.QubitOperator(self.n_qubits)
    for i in range(self.n_qubits):
      for j in range(self.n_qubits):
        transformed_term = fermion_operators.FermionTerm(self.n_qubits,
                                                         1.0,
                                                         [(i, 1), (j, 1),
                                                          (j, 0), (i, 0)]). \
            jordan_wigner_transform()

    real_constraint = self.extract_real_operator(constraint_operator)
    if not self.constraint_trivial(real_constraint):
      yield (real_constraint, self.n_fermions * (self.n_fermions - 1))

    imaginary_constraint = self.extract_imaginary_operator(
        constraint_operator)
    if not self.constraint_trivial(imaginary_constraint):
      yield (imaginary_constraint, 0.0)

    # 2 Body Hermiticity Conditions
    for ij in range(self.n_qubits**2):
      i, j = (ij / self.n_qubits), (ij % self.n_qubits)
      for kl in range(ij + 1, self.n_qubits**2):
        k, l = (kl / self.n_qubits), (kl % self.n_qubits)
        constraint_operator = qubit_operators.QubitOperator(self.n_qubits)
        transformed_term = fermion_operators.FermionTerm(self.n_qubits,
                                                         1.0,
                                                         [(i, 1), (j, 1),
                                                          (l, 0), (k, 0)]). \
            jordan_wigner_transform()
        constraint_operator += transformed_term
        transformed_term = fermion_operators.FermionTerm(self.n_qubits,
                                                         -1.0,
                                                         [(k, 1), (l, 1),
                                                          (j, 0), (i, 0)]). \
            jordan_wigner_transform()
        constraint_operator += transformed_term
        if not self.constraint_trivial(real_constraint):
          yield (real_constraint, 0.0)

        imaginary_constraint = self.extract_imaginary_operator(
            constraint_operator)
        if not self.constraint_trivial(imaginary_constraint):
          yield (imaginary_constraint, 0.0)

    # 2 Body Anti-Symmetry
    for ij in range(self.n_qubits**2):
      i, j = (ij / self.n_qubits), (ij % self.n_qubits)
      for kl in range(ij + 1, self.n_qubits**2):
        k, l = (kl / self.n_qubits), (kl % self.n_qubits)
        constraint_operator = qubit_operators.QubitOperator(self.n_qubits)
        transformed_term = fermion_operators.FermionTerm(self.n_qubits,
                                                         1.0,
                                                         [(i, 1), (j, 1),
                                                          (l, 0), (k, 0)]). \
            jordan_wigner_transform()
        constraint_operator += transformed_term
        transformed_term = fermion_operators.FermionTerm(self.n_qubits,
                                                         1.0,
                                                         [(j, 1), (i, 1),
                                                          (l, 0), (k, 0)]). \
            jordan_wigner_transform()
        constraint_operator += transformed_term
        if not self.constraint_trivial(real_constraint):
          yield (real_constraint, 0.0)

        imaginary_constraint = self.extract_imaginary_operator(
            constraint_operator)
        if not self.constraint_trivial(imaginary_constraint):
          yield (imaginary_constraint, 0.0)

        constraint_operator = qubit_operators.QubitOperator(self.n_qubits)
        transformed_term = fermion_operators.FermionTerm(self.n_qubits,
                                                         1.0,
                                                         [(i, 1), (j, 1),
                                                          (l, 0), (k, 0)]). \
            jordan_wigner_transform()
        constraint_operator += transformed_term
        transformed_term = fermion_operators.FermionTerm(self.n_qubits,
                                                         1.0,
                                                         [(i, 1), (j, 1),
                                                          (k, 0), (l, 0)]). \
            jordan_wigner_transform()
        constraint_operator += transformed_term
        if not self.constraint_trivial(real_constraint):
          yield (real_constraint, 0.0)

        imaginary_constraint = self.extract_imaginary_operator(
            constraint_operator)
        if not self.constraint_trivial(imaginary_constraint):
          yield (imaginary_constraint, 0.0)

        constraint_operator = qubit_operators.QubitOperator(self.n_qubits)
        transformed_term = fermion_operators.FermionTerm(self.n_qubits,
                                                         1.0,
                                                         [(i, 1), (j, 1),
                                                          (l, 0), (k, 0)]). \
            jordan_wigner_transform()
        constraint_operator += transformed_term
        transformed_term = fermion_operators.FermionTerm(self.n_qubits,
                                                         1.0,
                                                         [(j, 1), (i, 1),
                                                          (k, 0), (l, 0)]). \
            jordan_wigner_transform()
        constraint_operator += transformed_term
        if not self.constraint_trivial(real_constraint):
          yield (real_constraint, 0.0)

        imaginary_constraint = self.extract_imaginary_operator(
            constraint_operator)
        if not self.constraint_trivial(imaginary_constraint):
          yield (imaginary_constraint, 0.0)

    # Contraction to 1 RDM from 2 RDM
    for i in range(self.n_qubits):
      for j in range(self.n_qubits):
        constraint_operator = qubit_operators.QubitOperator(self.n_qubits)
        for p in range(self.n_qubits):
          transformed_term = fermion_operators.FermionTerm(self.n_qubits,
                                                           1.0,
                                                           [(i, 1), (p, 1),
                                                            (p, 0), (j, 0)]). \
              jordan_wigner_transform()
          constraint_operator += transformed_term
        transformed_term = fermion_operators.FermionTerm(self.n_qubits,
                                                         -(self.n_fermions -
                                                           1),
                                                         [(i, 1), (j, 0)]). \
            jordan_wigner_transform()
        constraint_operator += transformed_term
        if not self.constraint_trivial(real_constraint):
          yield (real_constraint, 0.0)

        imaginary_constraint = self.extract_imaginary_operator(
            constraint_operator)
        if not self.constraint_trivial(imaginary_constraint):
          yield (imaginary_constraint, 0.0)

    # Linear relations between 2-particle matrices
    for ij in range(self.n_qubits**2):
      i, j = (ij / self.n_qubits), (ij % self.n_qubits)
      for kl in range(ij, self.n_qubits**2):
        k, l = (kl / self.n_qubits), (kl % self.n_qubits)
        # Q-Matrix
        constraint_operator = qubit_operators.QubitOperator(self.n_qubits)
        transformed_term = fermion_operators.FermionTerm(self.n_qubits,
                                                         -1.0 * (j == l),
                                                         [(i, 1), (k, 0)]). \
            jordan_wigner_transform()
        constraint_operator += transformed_term

        transformed_term = fermion_operators.FermionTerm(self.n_qubits,
                                                         1.0 * (i == l),
                                                         [(j, 1), (k, 0)]). \
            jordan_wigner_transform()
        constraint_operator += transformed_term

        transformed_term = fermion_operators.FermionTerm(self.n_qubits,
                                                         1.0 * (j == k),
                                                         [(i, 1), (l, 0)]). \
            jordan_wigner_transform()
        constraint_operator += transformed_term

        transformed_term = fermion_operators.FermionTerm(self.n_qubits,
                                                         -1.0 * (i == k),
                                                         [(j, 1), (l, 0)]). \
            jordan_wigner_transform()
        constraint_operator += transformed_term

        transformed_term = fermion_operators.FermionTerm(self.n_qubits,
                                                         1.0,
                                                         [(i, 1), (j, 1),
                                                          (l, 0), (k, 0)]). \
            jordan_wigner_transform()
        constraint_operator += transformed_term

        transformed_term = fermion_operators.FermionTerm(self.n_qubits,
                                                         -1.0,
                                                         [(i, 0), (j, 0),
                                                          (l, 1), (k, 1)]). \
            jordan_wigner_transform()
        constraint_operator += transformed_term

        constant_term = -2 * (i == k and j == l) if (i == k and j == l) else 0

        if not self.constraint_trivial(real_constraint):
          yield (real_constraint, constant_term)

        imaginary_constraint = self.extract_imaginary_operator(
            constraint_operator)
        if not self.constraint_trivial(imaginary_constraint):
          yield (imaginary_constraint, 0.0)

        # G-Matrix
        constraint_operator = qubit_operators.QubitOperator(self.n_qubits)
        transformed_term = fermion_operators.FermionTerm(self.n_qubits,
                                                         1.0 * (j == l),
                                                         [(i, 1), (k, 0)]). \
            jordan_wigner_transform()
        constraint_operator += transformed_term

        transformed_term = fermion_operators.FermionTerm(self.n_qubits,
                                                         1.0,
                                                         [(i, 1), (l, 1),
                                                          (k, 0), (j, 0)]). \
            jordan_wigner_transform()
        constraint_operator += transformed_term

        transformed_term = fermion_operators.FermionTerm(self.n_qubits,
                                                         -1.0,
                                                         [(i, 1), (j, 0),
                                                          (l, 1), (k, 0)]). \
            jordan_wigner_transform()
        constraint_operator += transformed_term

        if not self.constraint_trivial(real_constraint):
          yield (real_constraint, 0.0)

        imaginary_constraint = self.extract_imaginary_operator(
            constraint_operator)
        if not self.constraint_trivial(imaginary_constraint):
          yield (imaginary_constraint, 0.0)
