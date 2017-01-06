"""Constraints on fermionic reduced density matrices"""
import fermion_operators
import qubit_operators
import copy
import numpy
import scipy
import itertools
import molecular_operators


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

  @staticmethod
  def project_density_matrix(rho, new_trace=None):
    """Project a density matrix to the closest positive semi-definite matrix.
       Follows the algorithm in PhysRevLett.108.070502

    Args:
      rho: Numpy array containing the density matrix with dimension (N, N)
      new_trace(float): number to fix the new trace of the matrix to be

    Returns:
      rho_projected: The closest positive semi-definite matrix to rho.
    """
    # Rescale to unit trace if the matrix is not already
    rho_trace = numpy.trace(rho)
    rho_impure = rho / rho_trace

    dimension = rho_impure.shape[0]  # the dimension of the Hilbert space
    [eigvals, eigvecs] = scipy.linalg.eigh(rho_impure)

    # If matrix is already trace one PSD, we are done
    if numpy.min(eigvals) >= 0:
      if (new_trace is None):
        return rho
      else:
        return new_trace * rho_impure

    # Otherwise, continue finding closest trace one, PSD matrix
    eigvals = list(eigvals)
    eigvals.reverse()
    eigvals_new = [0.0] * len(eigvals)

    i = dimension
    accumulator = 0.0  # Accumulator
    while eigvals[i - 1] + accumulator / float(i) < 0:
      accumulator += eigvals[i - 1]
      i -= 1
    for j in range(i):
      eigvals_new[j] = eigvals[j] + accumulator / float(i)
    eigvals_new.reverse()

    # Reconstruct the matrix
    rho_projected = reduce(numpy.dot, (eigvecs,
                                       numpy.diag(eigvals_new),
                                       numpy.conj(eigvecs.T)))
    if (new_trace is None):
      rho_projected = rho_trace * rho_projected
    else:
      rho_projected = new_trace * rho_projected

    return rho_projected

  def perm_parity(self, perm):
    """Given a permutation of the digits 0..N in order as a list,
    returns its parity (or sign): +1 for even parity; -1 for odd."""
    lst = list(perm)
    parity = 1
    for i in range(0, len(lst) - 1):
      if lst[i] != i:
        parity *= -1
        mn = min(range(i, len(lst)), key=lst.__getitem__)
        lst[i], lst[mn] = lst[mn], lst[i]
    return parity

  def wedge_product(self, a, b, new_shape=None):
    """Compute the Grassmann Wedge Product between tensors a and b

    Args:
      a: numpy array of the first tensor
      b: numpy array of the second tensor
      new_shape: tuple describing the desired shape of the output

    Returns:
      Grassmann wedge product (a^b) as a numpy array
    """

    # Get number of upper and lower indices
    assert (len(a.shape) % 2 == 0 and len(b.shape) % 2 == 0)
    ka, kb = len(a.shape) / 2, len(b.shape) / 2
    N = ka + kb

    # Form initial tensor product
    ab = numpy.kron(a, b)
    ab = numpy.\
        reshape(ab,
                a.shape[:ka] + b.shape[:kb] + a.shape[ka:] + b.shape[kb:])

    # Make buffer and sum in permutations using numpy transpose
    result = numpy.zeros_like(ab)

    for perm1 in itertools.permutations(range(N)):
      for perm2 in itertools.permutations(range(N)):
        parity = self.perm_parity(perm1) * self.perm_parity(perm2)
        trans_list = [i for i in perm1] + [i + N for i in perm2]
        result += parity * numpy.transpose(ab, trans_list)

    if (new_shape is not None):
      result = result.reshape(new_shape)
    return (1.0 / scipy.math.factorial(N)) ** 2 * result

  def apply_positivity(self, density):
    """Project the 1- and 2-RDM to being positive semi-definite matrices

    Args:
      density(MolecularOperator): molecular operator holding the 1- and 2-
        reduced density matrices as their one and two body coefficients.

    Returns:
      projected_density(MolecularOperator): molecular operator holding the
        projected 1- and 2- reduced density matrices as their one and two
        body coefficients.
    """
    one_density = density.one_body_coefficients
    # Reorder for convention with PSD 2-RDM
    two_density = numpy.transpose(density.two_body_coefficients, [0, 1, 3, 2])
    constant = density.constant

    dimension = one_density.shape[0]
    two_density = two_density.reshape((dimension**2, dimension**2))

    # Project 1- and 2-RDMs
    projected_one_density = self.\
        project_density_matrix(one_density, new_trace=self.n_fermions)
    projected_two_density = self.\
        project_density_matrix(two_density,
                               new_trace=(self.n_fermions *
                                          (self.n_fermions - 1)))

    # Permute one-particle-hole matrices, then project again
    # not clear this functions as intended
    """one_hole = numpy.eye(dimension) - one_density
    projected_one_hole = self.\
      project_density_matrix(one_hole,
                             new_trace = self.n_qubits - self.n_fermions)
    projected_one_density = numpy.eye(dimension) -\
                            projected_one_hole

    # Do two-particle particle-hole permutations, then project
    # Q-Matrix
    two_hole_Q = 2.0 * numpy.eye(dimension**2) -\
                 4.0 * self.wedge_product(projected_one_density,
                                          numpy.eye(dimension),
                                          new_shape = ((dimension**2,
                                                        dimension**2))) + \
                 projected_two_density
    print("Q Trace: {}".format(numpy.trace(two_hole_Q)))
    projected_two_hole_Q = two_hole_Q #self.\
      #project_density_matrix(two_hole_Q)
    projected_two_density = projected_two_hole_Q -\
                            2.0 * numpy.eye(dimension**2) +\
                            4.0 * self.\
                              wedge_product(projected_one_density,
                                            numpy.eye(dimension),
                                            new_shape=((dimension**2,
                                                        dimension**2)))
    #G Matrix
    two_hole_G = numpy.kron(projected_one_density, numpy.eye(dimension)) +\
                 numpy.transpose(projected_two_density.
                                 reshape((dimension, ) * 4), [0, 2, 1, 3]).\
                   reshape((dimension**2, ) * 2)
    print("G Trace: {}".format(numpy.trace(two_hole_G)))
    projected_two_hole_G = two_hole_G #self.\
      #project_density_matrix(two_hole_G)
    projected_two_density = numpy.\
      transpose((projected_two_hole_G -
                 numpy.kron(projected_one_density,
                            numpy.eye(dimension))).
                reshape((dimension, ) * 4),
                [0, 2, 1, 3])"""

    # Permute back to ordering used by the rest of code for 2 RDM
    projected_two_density = numpy.transpose(projected_two_density.
                                            reshape((dimension, ) * 4),
                                            [0, 1, 3, 2])

    projected_density = molecular_operators.\
        MolecularOperator(constant,
                          projected_one_density,
                          projected_two_density)

    return projected_density

  def constraints_one_body(self):
    """A generator for one-body positivity constraints

      Yields:
        Constraint tuples of the form (QubitOperator, float), where
        the expectation value of the QubitOperator should be equal to the float
    """

    # One-RDM Trace Condition
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
