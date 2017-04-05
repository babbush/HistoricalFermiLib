"""Class to store and transform fermion operators.
"""
from __future__ import absolute_import

import copy

import numpy

from fermilib import qubit_operators
from fermilib.fenwick_tree import FenwickTree
from fermilib.local_terms import LocalTerm, LocalTermError
from fermilib.local_operators import LocalOperator, LocalOperatorError
from fermilib.sparse_operators import (jordan_wigner_term_sparse,
                                       jordan_wigner_operator_sparse)


class JordanWignerError(Exception):
  pass


class FermionTermError(LocalTermError):
  pass


class FermionOperatorError(LocalOperatorError):
  pass


def fermion_identity(coefficient=1.):
  return FermionTerm([], coefficient)


def one_body_term(p, q, coefficient=1.):
  """Return one-body operator which conserves particle number.

  Args:
    p, q: The sites between which the hopping occurs.
    coefficient: Optional float giving coefficient of term.
  """
  return FermionTerm([(p, 1), (q, 0)], coefficient)


def two_body_term(p, q, r, s, coefficient=1.):
  """Return two-body operator which conserves particle number.

  Args:
    p, q, r, s: The sites between which the hopping occurs.
    coefficient: Optional float giving coefficient of term.
  """
  return FermionTerm([(p, 1), (q, 1), (r, 0), (s, 0)], coefficient)


def number_operator(n_qubits, site=None, coefficient=1.):
  """Return a number operator.

  Args:
    n_qubits: An int giving the number of spin-orbitals in the system.
    site: The site on which to return the number operator.
      If None, return total number operator on all sites.
  """
  if site is None:
    operator = FermionOperator()
    for spin_orbital in range(n_qubits):
      operator += number_operator(n_qubits, spin_orbital)
  else:
    operator = FermionTerm([(site, 1), (site, 0)], coefficient)
  return operator


class FermionTerm(LocalTerm):
  """Stores a single term composed of products of fermionic ladder operators.

  Attributes:
    operators: A list of tuples. The first element of each tuple is an
      int indicating the site on which operators acts. The second element
      of each tuple is boole, indicating whether raising (1) or lowering (0).
    coefficient: A complex valued float giving the term coefficient.

    Example usage:
      Consider the term 6.7 * a_3^\dagger a_1 a_7^\dagger
      This object would have the attributes:
      term.coefficient = 6.7
      term.operators = [(3, 1), (1, 0), (7, 1)]
  """
  def __init__(self, operators=None, coefficient=1.):
    """Init a FermionTerm.

    There are two ways to initialize the FermionTerm a^\dagger_2 a_7
    Way one is to provide the operators list, e.g. [(2, 1), (7, 0)]
    The other way is to provide a string '2^ 7'

    Args:
      operators: A list of tuples. The first element of each tuple is an
          int indicating the site on which operators acts. The second element
          of each tuple is an integer indicating raising (1) or lowering (0).
          Alternatively, a string can be provided.
      coefficient: A complex valued float giving the term coefficient.

    Raises:
      ValueError: Provided incorrect operator in list of operators.
      ValueError: Invalid action provided to FermionTerm. Must be 0
                  (lowering) or 1 (raising).
    """
    if operators is not None and not isinstance(operators, (tuple, list, str)):
      raise ValueError("Operators specified incorrectly.")

    # Parse string input.
    if isinstance(operators, str):
      list_ops = []
      for el in operators.split():
        if el[-1] == '^':
          list_ops.append((int(el[:-1]), 1))
        else:
          try:
            list_ops.append((int(el), 0))
          except ValueError:
            raise ValueError('Invalid action provided to FermionTerm.')
      operators = list_ops

    # Initialize.
    super(FermionTerm, self).__init__(operators, coefficient)

    # Check type.
    for operator in self:
      if not isinstance(operator, tuple):
        raise ValueError('Provided incorrect operator in list of operators.')
      tensor_factor, action = operator
      if not (isinstance(tensor_factor, int) and tensor_factor >= 0):
        raise ValueError('Invalid tensor factor provided to FermionTerm: '
                         'must be a non-negative integer.')
      if action not in (0, 1):
        raise ValueError('Invalid action provided to FermionTerm. '
                         'Must be 0 (lowering) or 1 (raising).')

  def n_qubits(self):
    highest_qubit = 0
    for operator in self.operators:
      if operator[0] + 1 > highest_qubit:
        highest_qubit = operator[0] + 1
    return highest_qubit

  def __add__(self, addend):
    """Compute self + addend for a FermionTerm.

    Note that we only need to handle the case of adding other fermionic terms
    or operators.

    Args:
      addend: A FermionTerm or FermionOperator.

    Returns:
      summand: A new instance of FermionOperator.

    Raises:
      TypeError: Object of invalid type cannot be added to FermionTerm.
    """
    if not issubclass(type(addend), (FermionTerm, FermionOperator)):
      raise TypeError('Cannot add term of invalid type to FermionTerm.')

    return FermionOperator(self) + addend

  def __str__(self):
    """Return an easy-to-read string representation of the term."""
    string_representation = '{} ['.format(self.coefficient)
    for operator in self:
      string_representation += str(operator[0]) + '^' * operator[1] + ' '

    if self:
      string_representation = string_representation[:-1]
    string_representation += ']'
    return string_representation

  def hermitian_conjugate(self):
    """Hermitian conjugate this fermionic term."""
    self.coefficient = numpy.conjugate(self.coefficient)
    self.operators.reverse()
    for tensor_factor in range(len(self)):
      self[tensor_factor] = (self[tensor_factor][0],
                             1 - self[tensor_factor][1])

  def hermitian_conjugated(self):
    """Calculate Hermitian conjugate of fermionic term.

    Returns:
      A new FermionTerm object which is the hermitian conjugate of this.
    """
    res = copy.deepcopy(self)
    res.hermitian_conjugate()
    return res

  def is_normal_ordered(self):
    """Return whether or not term is in normal order.

    In our convention, normal ordering implies terms are ordered
    from highest tensor factor (on left) to lowest (on right).
    Also, ladder operators come first.
    """
    for i in range(1, len(self)):
      for j in range(i, 0, -1):
        right_operator = self[j]
        left_operator = self[j - 1]
        if right_operator[1] and not left_operator[1]:
          return False
        elif (right_operator[1] == left_operator[1] and
              right_operator[0] >= left_operator[0]):
          return False
    return True

  def normal_ordered(self):
    """Compute and return the normal ordered form of a FermionTerm.

    Not an in-place method.

    In our convention, normal ordering implies terms are ordered
    from highest tensor factor (on left) to lowest (on right).
    Also, ladder operators come first.

    Returns:
      FermionOperator object in normal ordered form.

    Warning:
      Even assuming that each creation or annihilation operator appears
      at most a constant number of times in the original term, the
      runtime of this method is exponential in the number of qubits.
    """
    # Initialize output.
    normal_ordered_operator = FermionOperator()

    # Copy self.
    term = copy.deepcopy(self)

    # Iterate from left to right across operators and reorder to normal form.
    # Swap terms operators into correct position by moving left to right.
    for i in range(1, len(term)):
      for j in range(i, 0, -1):
        right_operator = term[j]
        left_operator = term[j - 1]

        # Swap operators if raising on right and lowering on left.
        if right_operator[1] and not left_operator[1]:
          term[j - 1] = right_operator
          term[j] = left_operator
          term *= -1.

          # Replace a a^\dagger with 1 - a^\dagger a if indices are same.
          if right_operator[0] == left_operator[0]:
            operators_in_new_term = term[:(j - 1)]
            operators_in_new_term += term[(j + 1)::]
            new_term = FermionTerm(operators_in_new_term,
                                   -1. * term.coefficient)

            # Recursively add the processed new term.
            normal_ordered_operator += new_term.normal_ordered()

          # Handle case when operator type is the same.
        elif right_operator[1] == left_operator[1]:

          # If same two operators are repeated, term evaluates to zero.
          if right_operator[0] == left_operator[0]:
            return normal_ordered_operator

            # Swap if same ladder type but lower index on left.
          elif right_operator[0] > left_operator[0]:
            term[j - 1] = right_operator
            term[j] = left_operator
            term *= -1.

    # Add processed term to output and return.
    normal_ordered_operator += term
    return normal_ordered_operator

  def bravyi_kitaev_transform(self, n_qubits=None):
    """ Apply the Bravyi-Kitaev transform and return qubit operator.

    Returns:
      transformed_term: An instance of the QubitOperator class.

    Warning:
      Likely greedy. At the moment the method gets the node sets for
      each fermionic operator. FenwickNodes are not neccessary in this
      onstruction, only the indices matter here. This may be optimized
      by removing the unnecessary structure.

    Note:
        Reference: Operator Locality of Quantum Simulation of Fermionic Models
            by Havlicek, Troyer, Whitfield (arXiv:1701.07072).
    """
    if n_qubits is None:
      n_qubits = self.n_qubits()
    if not n_qubits or n_qubits < self.n_qubits():
      raise ValueError("Invalid n_qubits.")

    # Build the Fenwick Tree
    fenwick_tree = FenwickTree(n_qubits)

    # Initialize identity matrix.
    transformed_term = qubit_operators.QubitOperator(
        [qubit_operators.QubitTerm([], self.coefficient)])

    # Build the Bravyi-Kitaev transformed operators.
    for operator in self:
      index = operator[0]

      # Parity set. Set of nodes to apply Z to.
      parity_set = [node.index for node in
                    fenwick_tree.get_parity_set(index)]

      # Update set. Set of ancestors to apply X to.
      ancestors = [node.index for node in fenwick_tree.get_update_set(index)]

      # The C(j) set.
      ancestor_children = [node.index for node in
                           fenwick_tree.get_remainder_set(index)]

      # Switch between lowering/raising operators.
      d_coeff = .5j
      if operator[1]:
        d_coeff = -d_coeff

      # The fermion lowering operator is given by
      # a = (c+id)/2 where c, d are the majoranas.
      d_majorana_component = qubit_operators.QubitTerm(
          ([(operator[0], 'Y')] +
           [(index, 'Z') for index in ancestor_children] +
           [(index, 'X') for index in ancestors]),
          d_coeff)

      c_majorana_component = qubit_operators.QubitTerm(
          ([(operator[0], 'X')] +
           [(index, 'Z') for index in parity_set] +
           [(index, 'X') for index in ancestors]),
          0.5)

      transformed_term *= qubit_operators.QubitOperator(
          [c_majorana_component, d_majorana_component])

    return transformed_term

  def jordan_wigner_transform(self):
    """Apply the Jordan-Wigner transform and return qubit operator.

    Returns:
      transformed_term: An instance of the qubit_operators.QubitOperator
                        class.

    Warning:
      The runtime of this method is exponential in the locality of the
      original FermionTerm.
    """
    # Initialize identity matrix.
    transformed_term = qubit_operators.QubitOperator(
        [qubit_operators.QubitTerm([], self.coefficient)])

    # Loop through operators, transform and multiply.
    for operator in self:
      z_factors = [(index, 'Z') for index in range(0, operator[0])]

      # Handle identity.
      pauli_x_component = qubit_operators.QubitTerm(
          z_factors + [(operator[0], 'X')], 0.5)
      if operator[1]:
        pauli_y_component = qubit_operators.QubitTerm(
            z_factors + [(operator[0], 'Y')], -0.5j)
      else:
        pauli_y_component = qubit_operators.QubitTerm(
            z_factors + [(operator[0], 'Y')], 0.5j)

      transformed_term *= qubit_operators.QubitOperator(
          [pauli_x_component, pauli_y_component])
    return transformed_term

  def jordan_wigner_sparse(self, n_qubits=None):
    """Return a sparse matrix representation of the JW transformed term."""
    if n_qubits is None:
      n_qubits = self.n_qubits()
    if n_qubits == 0:
      raise ValueError("Invalid n_qubits.")
    if n_qubits < self.n_qubits():
      n_qubits = self.n_qubits()
    return jordan_wigner_term_sparse(self, n_qubits)

  def is_molecular_term(self):
    """Query whether term has correct form to be from a molecular.

    Require that term is particle-number conserving (same number of
    raising and lowering operators). Require that term has 0, 2 or 4
    ladder operators. Require that term conserves spin (parity of
    raising operators equals parity of lowering operators)."""
    if len(self.operators) not in (0, 2, 4):
      return False

    # Make sure term conserves particle number and spin.
    spin = 0
    particles = 0
    for operator in self:
      particles += (-1) ** operator[1]  # add 1 if create, else subtract
      spin += (-1) ** (operator[0] + operator[1])

    return particles == spin == 0


class FermionOperator(LocalOperator):
  """Data structure which stores sums of FermionTerm objects.

  Attributes:
    terms: A dictionary of FermionTerm objects.
  """
  def __init__(self, terms=None):
    """Init a FermionOperator.

    Args:
      terms: An instance or dictionary or list of FermionTerm objects.

    Raises:
      FermionOperatorError: Invalid FermionTerms provided to FermionOperator.
    """
    super(FermionOperator, self).__init__(terms)
    for term in self:
      if not isinstance(term, FermionTerm):
        raise FermionOperatorError('Invalid FermionTerms provided to '
                                   'FermionOperator.')

  def __setitem__(self, operators, coefficient):
    if operators in self:
      self.terms[tuple(operators)].coefficient = coefficient
    else:
      new_term = FermionTerm(operators, coefficient)
      self.terms[tuple(operators)] = new_term

  def normal_order(self):
    """Normal order this FermionOperator.

    Warning:
      Even assuming that each creation or annihilation operator appears
      at most a constant number of times in the original term, the
      runtime of this method is exponential in the number of qubits.
    """
    self.terms = self.normal_ordered().terms

  def normal_ordered(self):
    """Compute and return the normal ordered form of this
    FermionOperator.

    Not an in-place method.

    Returns:
      FermionOperator object in normal ordered form.

    Warning:
      Even assuming that each creation or annihilation operator appears
      at most a constant number of times in the original term, the
      runtime of this method is exponential in the number of qubits.
    """
    normal_ordered_operator = FermionOperator()
    for term in self:
      normal_ordered_operator += term.normal_ordered()
    return normal_ordered_operator

  def hermitian_conjugate(self):
    for term in self:
      term.hermitian_conjugate()

  def hermitian_conjugated(self):
    new = copy.deepcopy(self)
    new.hermitian_conjugate()
    return new

  def jordan_wigner_transform(self):
    """Apply the Jordan-Wigner transform and return qubit operator.

    Returns:
      transformed_operator: An instance of the
          qubit_operators.QubitOperator class.

    Warning:
      The runtime of this method is exponential in the maximum locality
      of the FermionTerms in the original FermionOperator.
    """
    transformed_operator = qubit_operators.QubitOperator()
    for term in self:
      transformed_operator += term.jordan_wigner_transform()
    return transformed_operator

  def bravyi_kitaev_transform(self, n_qubits=None):
    """Apply the Bravyi-Kitaev transform and return qubit operator.

    Returns:
      transformed_operator: An instance of the QubitOperator class.
    """
    if n_qubits is None:
      n_qubits = self.n_qubits()
    if not n_qubits or n_qubits < self.n_qubits():
      raise ValueError("Invalid n_qubits.")
    transformed_operator = qubit_operators.QubitOperator()
    for term in self:
      transformed_operator += term.bravyi_kitaev_transform(n_qubits)
    return transformed_operator

  def jordan_wigner_sparse(self, n_qubits=None):
    """Apply Jordan-Wigner transform directly to sparse matrix form"""
    if n_qubits is None:
      n_qubits = self.n_qubits()
    if n_qubits == 0:
      raise ValueError("Invalid n_qubits.")
    if n_qubits < self.n_qubits():
      n_qubits = self.n_qubits()
    return jordan_wigner_operator_sparse(self, n_qubits)

  def get_interaction_operator(self):
    """Convert a 2-body fermionic operator to instance of InteractionOperator.

    This function should only be called on fermionic operators which consist
    of only a_p^\dagger a_q and a_p^\dagger a_q^\dagger a_r a_s terms.
    The one-body terms are stored in a matrix, one_body[p, q], and the
    two-body terms are stored in a tensor, two_body[p, q, r, s].

    Returns:
      interaction_operator: An instance of the InteractionOperator class.

    Raises:
      ErrorInteractionOperator: FermionOperator is not a molecular operator.

    Warning:
      Even assuming that each creation or annihilation operator appears
      at most a constant number of times in the original operator, the
      runtime of this method is exponential in the number of qubits.
    """
    # Import here to avoid circular dependency.
    from fermilib import interation_operators

    # Normal order the terms and initialize.
    self.normal_order()
    constant = 0.
    one_body = numpy.zeros((self.n_qubits(), self.n_qubits()), complex)
    two_body = numpy.zeros((
        self.n_qubits(), self.n_qubits(), self.n_qubits(), self.n_qubits()),
        complex)

    # Loop through terms and assign to matrix.
    for term in self:
      coefficient = term.coefficient

      # Handle constant shift.
      if len(term) == 0:
        constant = coefficient

      elif len(term) == 2:
        # Handle one-body terms.
        if [operator[1] for operator in term] == [1, 0]:
          p, q = [operator[0] for operator in term]
          one_body[p, q] = coefficient
        else:
          raise interaction_operators.InteractionOperatorError(
              'FermionOperator is not a molecular operator.')

      elif len(term) == 4:
        # Handle two-body terms.
        if [operator[1] for operator in term] == [1, 1, 0, 0]:
          p, q, r, s = [operator[0] for operator in term]
          two_body[p, q, r, s] = coefficient
        else:
          raise interaction_operators.InteractionOperatorError(
              'FermionOperator is not a molecular operator.')

      else:
        # Handle non-molecular Hamiltonian.
        raise interaction_operators.InteractionOperatorError(
            'FermionOperator is  not a molecular operator.')

    # Form InteractionOperator and return.
    interaction_operator = interaction_operators.InteractionOperator(
        constant, one_body, two_body)
    return interaction_operator
