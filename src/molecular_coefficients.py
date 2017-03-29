"""Base class for representation of various molecule classes."""
from config import *
import itertools
import numpy
import copy


class MolecularCoefficientsError(Exception):
  pass


def one_body_basis_change(one_body_operator, rotation_matrix):
  """Change the basis of an operator represented by a 2D matrix, e.g. the
  1-RDM.

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
  """Change the basis of an operator represented by a 4D matrix, e.g. the
  2-RDM.

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


class MolecularCoefficients(object):
  """Class for storing the data structure with a constant, a 2D matrix, and a
  4D matrix. The most common examples of data that will use this structure are
  molecular Hamiltonians and molecular 2-RDM density operators. However, this
  class is able to exploit specific properties of molecular operators in order
  to enable more efficient manipulation of the data.

  Attributes:
    n_qubits: An int giving the number of qubits.
    constant: A constant term in the operator given as a float.
        For instance, the nuclear repulsion energy.
    one_body_coefficients: The coefficients of the 2D matrix terms. This is an
        n_qubits x n_qubits numpy array of floats.
        For instance, the one body term of MolecularOperator.
    two_body_coefficients: The coefficients of the 4D matrix terms. This is an
        n_qubits x n_qubits x n_qubits x n_qubits numpy array offloats.
        For instance, the two body term of MolecularOperator.
  """

  def __init__(self, constant, one_body_coefficients, two_body_coefficients):
    """Initialize the MolecularCoefficientsError class.

    Args:
      constant: A constant term in the operator given as a float.
          For instance, the nuclear repulsion energy.
      one_body_coefficients: The coefficients of the 2D matrix terms. This is
          an n_qubits x n_qubits numpy array of floats.
          For instance, the one body term of MolecularOperator.
      two_body_coefficients: The coefficients of the 4D matrix terms. This is
          an n_qubits x n_qubits x n_qubits x n_qubits numpy array offloats.
          For instance, the two body term of MolecularOperator.
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
    else:
      raise ValueError('args must be of length 0, 2, or 4.')

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

  def __eq__(self, molecular_coefficients):
    diff = max(abs(self.constant - molecular_coefficients.constant),
               numpy.amax(numpy.absolute(self.one_body_coefficients -
                          molecular_coefficients.one_body_coefficients)),
               numpy.amax(numpy.absolute(self.two_body_coefficients -
                          molecular_coefficients.two_body_coefficients)))
    return diff < EQUALITY_TOLERANCE

  def __neq__(self, molecular_coefficients):
    return not (self == molecular_coefficients)

  def __iter__(self):
    """Iterate over non-zero elements of MolecularCoefficients."""
    if self.constant:  # Constant.
      yield []
    for p in range(self.n_qubits):  # One-body terms.
      for q in range(self.n_qubits):
        if self.one_body_coefficients[p, q]:
          yield [p, q]
    for p in range(self.n_qubits):  # Two-body terms.
      for q in range(self.n_qubits):
        for r in range(self.n_qubits):
          for s in range(self.n_qubits):
            if self.two_body_coefficients[p, q, r, s]:
              yield [p, q, r, s]

  def __str__(self):
    """Print out the non-zero elements of MolecularCoefficients."""

    string = ''
    for key in self:
      if len(key) == 0:
        string += '[] {}\n'.format(self[key])
      elif len(key) == 2:
        string += '[{} {}] {}\n'.format(key[0], key[1], self[key])
      elif len(key) == 4:
        string += '[{} {} {} {}] {}\n'.format(key[0], key[1], key[2], key[3],
                                              self[key])
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
