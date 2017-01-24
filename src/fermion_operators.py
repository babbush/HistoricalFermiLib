"""Class to store and transform fermion operators.
"""
import molecular_operators
import qubit_operators
import local_operators
import local_terms
import numpy
import copy


class JordanWignerError(Exception):
  pass


class FermionTermError(local_terms.LocalTermError):
  pass


class FermionOperatorError(local_operators.LocalOperatorError):
  pass


def identity(n_qubits):
  return FermionTerm(n_qubits, 1.)


def number_operator(n_qubits, site=None, coefficient=1.):
  """Return a number operator.

  Args:
    n_qubits: An int giving the number of spin-orbitals in the system.
    site: The site on which to return the number operator.
      If None, return total number operator on all sites.
  """
  if site is None:
    operator = FermionOperator(n_qubits)
    for spin_orbital in range(n_qubits):
      operator += number_operator(n_qubits, spin_orbital)
  else:
    operator = FermionTerm(n_qubits, coefficient, [(site, 1), (site, 0)])
  return operator


class FermionTerm(local_terms.LocalTerm):
  """Stores a single term composed of products of fermionic ladder operators.

  Attributes:
    _n_qubits: An int giving the number of spin-orbitals in the system.
    coefficient: A complex valued float giving the term coefficient.
    operators: A list of tuples. The first element of each tuple is an
      int indicating the site on which operators acts. The second element
      of each tuple is boole, indicating whether raising (1) or lowering (0).

    Example usage:
      Consider the term 6.7 * a_3^\dagger a_1 a_7^\dagger
      This object would have the attributes:
      term.coefficient = 6.7
      term.operators = [(3, 1), (1, 0), (7, 1)]
  """
  def __init__(self, n_qubits, coefficient=0., operators=None):
    """Init a FermionTerm.

    Args:
      n_qubits: Int, the number of qubits in the system.
      coefficient: A complex valued float giving the term coefficient.
      operators: A list of tuples. The first element of each tuple is an
          int indicating the site on which operators acts. The second element
          of each tuple is an integer indicating raising (1) or lowering (0).

    Raises:
      ValueError: Provided incorrect operator in list of operators.
      ValueError: Invalid action provided to FermionTerm. Must be 0
                  (lowering) or 1 (raising).
      ValueError: Invalid tensor factor provided to FermionTerm.
                  Must be an integer between 0 and n_qubits-1.
    """
    super(FermionTerm, self).__init__(n_qubits, coefficient, operators)

    for operator in self:
      if not isinstance(operator, tuple):
        raise ValueError('Provided incorrect operator in list of operators.')

      tensor_factor, action = operator
      if action not in (0, 1):
        raise ValueError('Invalid action provided to FermionTerm. '
                         'Must be 0 (lowering) or 1 (raising).')
      if not (isinstance(tensor_factor, int) and
              0 <= tensor_factor < n_qubits):
        raise ValueError('Invalid tensor factor provided to FermionTerm. '
                         'Must be an integer between 0 and n_qubits-1.')

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
      FermionTermError: Cannot add terms acting on different Hilbert spaces.
    """
    if not issubclass(type(addend), (FermionTerm, FermionOperator)):
      raise TypeError('Cannot add term of invalid type to FermionTerm.')

    if self.n_qubits != addend.n_qubits:
      raise FermionTermError('Cannot add terms acting on different'
                             'Hilbert spaces.')

    return FermionOperator(self.n_qubits, self) + addend

  def __str__(self):
    """Return an easy-to-read string representation of the term."""
    string_representation = '{} ('.format(self.coefficient)
    for operator in self:
      string_representation += str(operator[0]) + '+' * operator[1] + ' '

    if self:
      string_representation = string_representation[:-1]
    string_representation += ')'
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
    normal_ordered_operator = FermionOperator(self._n_qubits)

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
            new_term = FermionTerm(term._n_qubits,
                                   -1. * term.coefficient,
                                   operators_in_new_term)

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

  def bravyi_kitaev_transform(self):
    # TODO Jarrod.
    return None

  def jordan_wigner_transform(self):
    """Apply the Jordan-Wigner transform and return qubit operator.

    Returns:
      transformed_term: An instance of the QubitOperator class.

    Warning:
      Even assuming that each creation or annihilation operator appears
      at most a constant number of times in the original term, the
      runtime of this method is exponential in the number of qubits.
    """
    # Initialize identity matrix.
    transformed_term = qubit_operators.QubitOperator(
        self._n_qubits, [qubit_operators.QubitTerm(self._n_qubits,
                                                   self.coefficient)])
    # Loop through operators, transform and multiply.
    for operator in self:

      # Handle identity.
      pauli_x_component = qubit_operators.QubitTerm(
          self._n_qubits, 0.5,
          [(operator[0], 'X')] +
          [(index, 'Z') for index in range(operator[0] - 1, -1, -1)])
      if operator[1]:
        pauli_y_component = qubit_operators.QubitTerm(
            self._n_qubits, -0.5j,
            [(operator[0], 'Y')] +
            [(index, 'Z') for index in range(operator[0] - 1, -1, -1)])
      else:
        pauli_y_component = qubit_operators.QubitTerm(
            self._n_qubits, 0.5j,
            [(operator[0], 'Y')] +
            [(index, 'Z') for index in range(operator[0] - 1, -1, -1)])
      transformed_term *= qubit_operators.QubitOperator(
          self._n_qubits, [pauli_x_component, pauli_y_component])
    return transformed_term

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

    return not (particles or spin)


class FermionOperator(local_operators.LocalOperator):
  """Data structure which stores sums of FermionTerm objects.

  Attributes:
    _n_qubits: An int giving the number of spin-orbitals in the system.
    terms: A dictionary of FermionTerm objects.
  """
  def __init__(self, n_qubits, terms=None):
    """Init a FermionOperator.

    Args:
      n_qubits: Int, the number of qubits in the system.
      terms: Dictionary or list of FermionTerm objects.

    Raises:
      FermionOperatorError: Invalid FermionTerms provided to FermionOperator.
    """
    super(FermionOperator, self).__init__(n_qubits, terms)
    for term in self:
      if term.n_qubits != n_qubits or not isinstance(term, FermionTerm):
          raise FermionOperatorError('Invalid FermionTerms provided to '
                                     'FermionOperator.')

  def __setitem__(self, operators, coefficient):
    if operators in self:
      self.terms[tuple(operators)].coefficient = coefficient
    else:
      new_term = FermionTerm(self.n_qubits, coefficient, operators)
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
    normal_ordered_operator = FermionOperator(self._n_qubits)
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
      transformed_operator: An instance of the QubitOperator class.

    Warning:
      Even assuming that each creation or annihilation operator appears
      at most a constant number of times in the original operator, the
      runtime of this method is exponential in the number of qubits.
    """
    transformed_operator = qubit_operators.QubitOperator(self._n_qubits)
    for term in self:
      transformed_operator += term.jordan_wigner_transform()
    return transformed_operator

  def get_molecular_operator(self):
    """Convert a 2-body fermionic operator to instance of MolecularOperator.

    This function should only be called on fermionic operators which consist
    of only a_p^\dagger a_q and a_p^\dagger a_q^\dagger a_r a_s terms.
    The one-body terms are stored in a matrix, one_body[p, q], and the
    two-body terms are stored in a tensor, two_body[p, q, r, s].

    Returns:
      molecular_operator: An instance of the MolecularOperator class.

    Raises:
      ErrorMolecularOperator: FermionOperator is not a molecular operator.

    Warning:
      Even assuming that each creation or annihilation operator appears
      at most a constant number of times in the original operator, the
      runtime of this method is exponential in the number of qubits.
    """
    # Normal order the terms and initialize.
    self.normal_order()
    constant = 0.
    one_body = numpy.zeros((self._n_qubits, self._n_qubits), complex)
    two_body = numpy.zeros((
        self._n_qubits, self._n_qubits, self._n_qubits, self._n_qubits),
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
          raise ErrorMolecularOperator('FermionOperator is not a '
                                       'molecular operator.')

      elif len(term) == 4:
        # Handle two-body terms.
        if [operator[1] for operator in term] == [1, 1, 0, 0]:
          p, q, r, s = [operator[0] for operator in term]
          two_body[p, q, r, s] = coefficient
        else:
          raise ErrorMolecularOperator('FermionOperator is not a'
                                       'molecular operator.')

      else:
        # Handle non-molecular Hamiltonian.
        raise ErrorMolecularOperator('FermionOperator is not a'
                                     'molecular operator.')

    # Form MolecularOperator and return.
    molecular_operator = molecular_operators.MolecularOperator(
        constant, one_body, two_body)
    return molecular_operator
