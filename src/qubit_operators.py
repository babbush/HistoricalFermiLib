"""This files has utilities to read and store qubit Hamiltonians.
"""
from local_terms import LocalTerm, LocalTermError
from local_operators import LocalOperator
from sparse_operators import (qubit_term_sparse,
                              qubit_operator_sparse)
import fermion_operators
import molecular_operators
import itertools
import numpy
import copy


class QubitTermError(Exception):
  pass


class QubitOperatorError(Exception):
  pass


# Define products of all Pauli matrices for symbolic multiplication.
_PAULI_MATRIX_PRODUCTS = {('I', 'I'): (1., 'I'),
                          ('I', 'X'): (1., 'X'),
                          ('X', 'I'): (1., 'X'),
                          ('I', 'Y'): (1., 'Y'),
                          ('Y', 'I'): (1., 'Y'),
                          ('I', 'Z'): (1., 'Z'),
                          ('Z', 'I'): (1., 'Z'),
                          ('X', 'X'): (1., 'I'),
                          ('Y', 'Y'): (1., 'I'),
                          ('Z', 'Z'): (1., 'I'),
                          ('X', 'Y'): (1.j, 'Z'),
                          ('X', 'Z'): (-1.j, 'Y'),
                          ('Y', 'X'): (-1.j, 'Z'),
                          ('Y', 'Z'): (1.j, 'X'),
                          ('Z', 'X'): (1.j, 'Y'),
                          ('Z', 'Y'): (-1.j, 'X')}


def qubit_identity():
  return QubitTerm([], 1.)


class QubitTerm(LocalTerm):
  """Single term of a hamiltonian for a system of spin 1/2 particles or qubits.

  A Hamiltonian of qubits can be written as a sum of QubitTerm objects.
  Suppose you have n_qubits = 5 qubits a term of the Hamiltonian could
  be coefficient * X1 Z3 which we call a QubitTerm object. It means
  coefficient * (1 x PauliX x 1 x PauliZ x 1), where x is the tensor
  product, 1 the identity matrix, and the others are Pauli matrices. We
  only allow to apply one single Pauli Matrix to each qubit.

  Note: Always use the abstractions provided here to manipulate the
  .operators attribute. If ignoring this advice, an important thing to
  keep in mind is that the operators list is assumed to be sorted in order
  of the tensor factor on which the operator acts.

  Attributes:
    coefficient: A real or complex floating point number.
    operators: A sorted list of tuples. The first element of each tuple is an
      int indicating the qubit on which operators acts. The second element
      of each tuple is a string, either 'X', 'Y' or 'Z', indicating what
      acts on that tensor factor. The list is sorted by the first index.
  """
  def __init__(self, operators=None, coefficient=1.):
    """Inits PauliTerm.

    Specify to which qubits a Pauli X, Y, or Z is applied. To all not
    specified qubits (numbered 0, 1, ..., n_qubits-1) the identity is applied.
    Only one Pauli Matrix can be applied to each qubit.

    Args:
      coefficient: A real or complex floating point number.
      operators: A sorted list of tuples. The first element of each tuple is an
        int indicating the qubit on which operators acts, starting from zero.
        The second element of each tuple is a string, either 'X', 'Y' or 'Z',
        indicating what acts on that tensor factor.
        operators can also be specified by a string of the form '0X 2Z 5Y',
        indicating an X on qubit 0, Z on qubit 2, and Y on qubit 5.

    Raises:
      QubitTermError: Invalid operators provided to QubitTerm.
    """
    if operators is not None and not isinstance(operators, (tuple, list, str)):
      raise ValueError("Operators specified incorrectly.")

    if isinstance(operators, str):
      list_ops = []
      for el in operators.split():
        if len(el) < 2:
          raise ValueError("Operators specified incorrectly.")
        list_ops.append((int(el[:-1]), el[-1]))
      operators = list_ops

    super(QubitTerm, self).__init__(operators, coefficient)

    self.n_qubits = 0
    for operator in self:
      tensor_factor, action = operator
      if not isinstance(action, str) or action not in 'XYZ':
        raise ValueError("Invalid action provided: must be string 'X', "
                         "'Y', or 'Z'.")
      if not (isinstance(tensor_factor, int) and tensor_factor >= 0):
        raise QubitTermError('Invalid tensor factor provided to QubitTerm: '
                             'must be a non-negative integer.')
      if tensor_factor > self.n_qubits:
	self.n_qubits = tensor_factor

    # Make sure operators are sorted by tensor factor.
    self.operators.sort(key=lambda operator: operator[0])

  def __add__(self, addend):
    """Compute self + addend for a QubitTerm.

    Note that we only need to handle the case of adding other qubit terms.

    Args:
      addend: A QubitTerm.

    Returns:
      summand: A new instance of QubitOperator.

    Raises:
      TypeError: Object of invalid type cannot be added to QubitTerm.
    """
    if not issubclass(type(addend), (QubitTerm, QubitOperator)):
      raise TypeError('Cannot add term of invalid type to QubitTerm.')
    return QubitOperator([self]) + addend

  def __imul__(self, multiplier):
    """Multiply terms with scalar or QubitTerm using *=.

    Note that the "self" term is on the left of the multiply sign.

    Args:
      multiplier: Another QubitTerm object.
    """
    # Handle scalars.
    if (isinstance(multiplier, (int, float, complex)) or
       numpy.isscalar(multiplier)):
      self.coefficient *= multiplier
      return self

    # Handle QubitTerms.
    elif issubclass(type(multiplier), QubitTerm):
      # Relabel self * qubit_term as left_term * right_term.
      left_term = self
      right_term = multiplier
      self.coefficient *= multiplier.coefficient

      # Loop through terms and create new sorted list of operators.
      product_operators = []
      left_operator_index = 0
      right_operator_index = 0
      n_operators_left = len(left_term)
      n_operators_right = len(right_term)
      while (left_operator_index < n_operators_left and
             right_operator_index < n_operators_right):
        (left_qubit, left_matrix) = left_term[left_operator_index]
        (right_qubit, right_matrix) = right_term[right_operator_index]

        # Multiply matrices if tensor factors are the same.
        if left_qubit == right_qubit:
          left_operator_index += 1
          right_operator_index += 1
          (scalar, matrix) = _PAULI_MATRIX_PRODUCTS[(left_matrix,
                                                     right_matrix)]

          # Add new term.
          if matrix != 'I':
            product_operators += [(left_qubit, matrix)]
            self.coefficient *= scalar

        # If left_qubit > right_qubit, add right_matrix; else, add left_matrix.
        elif left_qubit > right_qubit:
          product_operators += [(right_qubit, right_matrix)]
          right_operator_index += 1
        else:
          product_operators += [(left_qubit, left_matrix)]
          left_operator_index += 1

      # If either term_index exceeds the number of operators, finish.
      if left_operator_index == n_operators_left:
        product_operators += right_term[right_operator_index::]
      elif right_operator_index == n_operators_right:
        product_operators += left_term[left_operator_index::]

      # We should now have gone through all operators.
      self.operators = product_operators
      return self

  def reverse_jordan_wigner(self):
    """Transforms a QubitTerm into an instance of FermionOperator using JW.

    Operators are mapped as follows:
    Z_j -> I - 2 a^\dagger_j a_j
    X_j -> (a^\dagger_j + a_j) Z_{j-1} Z_{j-2} .. Z_0
    Y_j -> i (a^\dagger_j - a_j) Z_{j-1} Z_{j-2} .. Z_0

    Returns:
      transformed_term: An instance of the FermionOperator class.

    Raises:
      QubitTermError: Invalid operator provided: must be 'X', 'Y' or 'Z'.
    """
    # Initialize transformed operator.
    identity = fermion_operators.fermion_identity()
    transformed_term = fermion_operators.FermionOperator(identity)
    working_term = QubitTerm(self.operators, 1.0)

    # Loop through operators.
    if working_term.operators:
      operator = working_term.operators[-1]
      while operator is not None:

        # Handle Pauli Z.
        if operator[1] == 'Z':
          number_operator = fermion_operators.number_operator(
              self.n_qubits, operator[0], -2.)
          transformed_operator = identity + number_operator

        else:
          raising_term = fermion_operators.FermionTerm([(operator[0], 1)])
          lowering_term = fermion_operators.FermionTerm([(operator[0], 0)])

          # Handle Pauli X, Y, Z.
          if operator[1] == 'Y':
            raising_term *= 1j
            lowering_term *= -1j

          elif operator[1] != 'X':
            # Raise for invalid operator.
            raise QubitTermError("Invalid operator provided: "
                                 "must be 'X', 'Y' or 'Z'")

          # Account for the phase terms.
          for j in reversed(range(operator[0])):
            z_term = QubitTerm(coefficient=1.0,
                               operators=[(j, 'Z')])
            z_term *= working_term
            working_term = copy.deepcopy(z_term)
          transformed_operator = fermion_operators.FermionOperator(
	      [raising_term, lowering_term])
          transformed_operator *= working_term.coefficient
          working_term.coefficient = 1.0

        # Get next non-identity operator acting below the 'working_qubit'.
        working_qubit = operator[0] - 1
        for working_operator in working_term[::-1]:
          if working_operator[0] <= working_qubit:
            operator = working_operator
            break
          else:
            operator = None

        # Multiply term by transformed operator.
        transformed_term *= transformed_operator

    # Account for overall coefficient
    transformed_term *= self.coefficient

    return transformed_term

  def __str__(self):
    """Return an easy-to-read string representation of the term."""
    string_representation = '{:+}'.format(self.coefficient)
    if self.operators == []:
      string_representation += ' I'
    for operator in self:
      if operator[1] == 'X':
        string_representation += ' X{}'.format(operator[0])
      elif operator[1] == 'Y':
        string_representation += ' Y{}'.format(operator[0])
      else:
        assert operator[1] == 'Z'
        string_representation += ' Z{}'.format(operator[0])
    return string_representation

  def get_sparse_operator(self):
    """Map the QubitTerm to a SparseOperator instance."""
    return qubit_term_sparse(self)

  def eigenspectrum(self):
    return self.get_sparse_operator().eigenspectrum()


class QubitOperator(LocalOperator):
  """A collection of QubitTerm objects acting on same number of qubits.

  Note that to be a Hamiltonian which is a hermitian operator, the individual
  QubitTerm objects need to have only real valued coefficients.

  Attributes:
    terms: Dictionary of QubitTerm objects. The dictionary key is
        QubitTerm.key() and the dictionary value is the QubitTerm.
  """
  def __init__(self, terms=None):
    """Init a QubitOperator.

    Args:
      terms: Dictionary or list of QubitTerm objects.

    Raises:
      QubitOperatorError: Invalid QubitTerms provided to QubitOperator.
    """
    super(QubitOperator, self).__init__(terms)
    for term in self:
      if not isinstance(term, QubitTerm):
        raise QubitTermError('Invalid QubitTerms provided to QubitOperator.')

  def __setitem__(self, operators, coefficient):
    if operators in self:
      self.terms[tuple(operators)].coefficient = coefficient
    else:
      new_term = QubitTerm(operators, coefficient)
      self.terms[tuple(operators)] = new_term

  def reverse_jordan_wigner(self):
    transformed_operator = fermion_operators.FermionOperator()
    for term in self:
      transformed_operator += term.reverse_jordan_wigner()
    return transformed_operator

  def get_sparse_operator(self):
    return qubit_operator_sparse(self)

  def eigenspectrum(self):
    return self.get_sparse_operator().eigenspectrum()

  def expectation(self, qubit_operator):
    """Take the expectation value of self with another qubit operator.

    Args:
      qubit_operator: An instance of the QubitOperator class.

    Returns:
      expectation: A float, giving the expectation value.
    """
    expectation = 0.
    for term in qubit_operator:
      expectation += term.coefficient * self[term.operators]
    return expectation

  def get_molecular_rdm(self):
    """Build a MolecularOperator from measured qubit operators.

    Returns: A MolecularOperator object.
    """
    one_rdm = numpy.zeros((self.n_qubits,) * 2, dtype=complex)
    two_rdm = numpy.zeros((self.n_qubits,) * 4, dtype=complex)

    # One-RDM.
    for i, j in itertools.product(range(self.n_qubits), repeat=2):
      transformed_operator = fermion_operators.FermionTerm(
          [(i, 1), (j, 0)]).jordan_wigner_transform()
      for term in transformed_operator:
        if tuple(term.operators) in self.terms:
          one_rdm[i, j] += term.coefficient * self[term.operators]

    # Two-RDM.
    for i, j, k, l in itertools.product(range(self.n_qubits), repeat=4):
      transformed_operator = fermion_operators.FermionTerm(
          [(i, 1), (j, 1), (k, 0), (l, 0)]).jordan_wigner_transform()
      for term in transformed_operator:
        if tuple(term.operators) in self.terms:
          two_rdm[i, j, k, l] += term.coefficient * self[term.operators]

    return molecular_operators.MolecularOperator(1.0, one_rdm, two_rdm)
