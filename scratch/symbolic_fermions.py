"""This module symbolically maps fermion operators to spin operators."""


# Define products of all Pauli matrices for symbolic multiplication.
_PAULI_SYMBOL_PRODUCTS = {'II': 'I', 'XX': 'I', 'YY': 'I', 'ZZ': 'I',
                          'XY': 'Z', 'XZ': 'Y',
                          'YX': 'Z', 'YZ': 'X',
                          'ZX': 'Y', 'ZY': 'X',
                          'IX': 'X', 'IY': 'Y', 'IZ': 'Z',
                          'XI': 'X', 'YI': 'Y', 'ZI': 'Z'}
_PAULI_SCALAR_PRODUCTS = {'II': 1., 'XX': 1., 'YY': 1., 'ZZ': 1.,
                          'XY': 1.j, 'XZ': -1.j,
                          'YX': -1.j, 'YZ': 1.j,
                          'ZX': 1.j, 'ZY': -1.j,
                          'IX': 1., 'IY': 1., 'IZ': 1.,
                          'XI': 1., 'YI': 1., 'ZI': 1.}


def EvaluatePauliProduct(pauli_product, coefficient=1.):
  """Multiply out a sequence of Pauli operators acting on one tensor factor.

  Args:
    pauli_product: A string of X, Y, Z and I characters.
    coefficient: The coefficient of the product (used for recursion only).

  Returns:
    coefficient: The new coefficient of the operator.
    pauli_product: The simplified operator.
  """
  # Recursively multiply symbolic operators until only one operator remains.
  if len(pauli_product) > 1:
    symbol_product = _PAULI_SYMBOL_PRODUCTS[pauli_product[:2]]
    scalar_product = _PAULI_SCALAR_PRODUCTS[pauli_product[:2]]
    return EvaluatePauliProduct(symbol_product + pauli_product[2::],
                                scalar_product * coefficient)
  else:
    return coefficient, pauli_product


# Reduce an entire term.
def SimplifyPauliString(pauli_string):
  """Take a string of products of Pauli operators and simplify it.

  Args:
    pauli_string: A list of strings. The strings represent products
      of Pauli operators. Each element of the list represents a different
      tensor factor on which operators act.

  Returns:
    simple_coefficient: The scalar coefficient of the simplified term.
    simple_string: The simplified Pauli string (there should be only
      one operator acting on each tensor factor).
  """
  simple_coefficient = 1.
  simple_string = []
  for pauli_product in pauli_string:
    coefficient, operator = EvaluatePauliProduct(pauli_product)
    simple_coefficient *= coefficient
    simple_string += [operator]
  return simple_coefficient, simple_string


def SymbolicOperatorJW(index, n_qubits):
  """Symbolically apply Jordan-Wigner transform to single fermionic operator.

  The Jordan-Wigner transform of fermionic operators acts as follows:
  a^dagger_j = 0.5 (X_j - sqrt{-1} Y_j) Z_{j-1} Z_{j-2} ... Z_1
  a_j = 0.5 (X_j + sqrt{-1} Y_j) Z_{j-1} Z_{j-2} ... Z_1.
  Notice there are two terms here: one that begins with X and one that
  begins with Y. This function computes those two strings.

  Args:
    index: An int indicating the orbital on which the operator acts.
    n_qubits: An int indicating how many qubits there are in the system.

  Returns:
    coefficients: A list of coefficients (floats) that weigh the various
      spin Hamiltonian components that come from the JW transform.
    pauli_strings: A list of pauli_strings (that are lists of strings)
      which correspond to the Pauli strings from the JW transform.
  """
  # Compute Pauli strings.
  x_string = ['Z'] * (abs(index) - 1) + ['X'] + (n_qubits - abs(index)) * ['I']
  y_string = ['Z'] * (abs(index) - 1) + ['Y'] + (n_qubits - abs(index)) * ['I']

  # Figure out coefficients.
  x_string_coefficient = .5
  if index > 0:
    y_string_coefficient = -.5j
  else:
    y_string_coefficient = .5j

  # Return.
  coefficients = [x_string_coefficient, y_string_coefficient]
  pauli_strings = [x_string, y_string]
  return coefficients, pauli_strings


def SymbolicTermJW(fermion_coefficient, fermion_term, n_qubits):
  """Symbolically apply Jordan-Wigner transformation to fermion term.

  Throughout this code, fermionic operators, i.e. creation and annilhlation
  operators, are represented by positive and negative integers, respectively.
  Usually, terms are stored as python lists; e.g. [2 1 -2 -3] means raising
  on tensor factor two, raising on tensor factor one, lowering on two, etc.
  The point of this function is convert such a string to a Pauli string.

  Args:
    fermion_coefficient: The (float) coefficient of the term.
    fermion_term: The list of ints which represents the term.
    n_qubits: An int giving the total number of qubits in the system.

  Returns:
    pauli_coefficients: A list of floats giving the coefficients
      of the output strings.
    pauli_strings: A list of pauli_strings (that are lists of strings)
      which correspond to the Pauli strings from the JW transform.
  """
  # Loop over the fermion operators and apply Jordan-Wigner transformation.
  pauli_coefficients = [fermion_coefficient]
  pauli_strings = [n_qubits * ['I']]
  for index in fermion_term:
    jw_coefficients, jw_strings = SymbolicOperatorJW(index, n_qubits)

    # Loop over the post-transformation Jordan-Wigner strings and multiply.
    new_pauli_coefficients = []
    new_pauli_strings = []
    for pauli_coefficient, pauli_string in zip(pauli_coefficients,
                                               pauli_strings):
      for jw_coefficient, jw_string in zip(jw_coefficients,
                                           jw_strings):
        new_pauli_coefficients += [pauli_coefficient * jw_coefficient]

        # Multiply to form Pauli products one tensor factor at a time.
        new_pauli_string = []
        for qubit in range(n_qubits):
          new_pauli_string += [pauli_string[qubit] + jw_string[qubit]]
        new_pauli_strings += [new_pauli_string]

    # Update Pauli strings after multiplication.
    pauli_coefficients = new_pauli_coefficients
    pauli_strings = new_pauli_strings

  # Reduce terms.
  for term_number in range(len(pauli_coefficients)):
    coefficient, reduced_term = SimplifyPauliString(pauli_strings[term_number])
    pauli_coefficients[term_number] *= coefficient
    pauli_strings[term_number] = reduced_term

  # Return.
  return pauli_coefficients, pauli_strings


def SymbolicTransformationJW(fermion_coefficients, fermion_terms, tol=1e-9):
  """Apply the Jordan-Wigner transformation to collection of fermion terms.

  Args:
    fermion_coefficients: A list of coefficients (floats) for the terms.
    fermion_terms: A list of the fermion terms (lists of ints).
    tol: An optional float giving a cutoff below which terms are neglected.

  Returns:
    unique_strings: A python dictionary. The keys of the dictionary are
      tuples which correspond to the simplified Pauli strings in the output.
      The values are the real coefficients (floats) of those terms.
  """
  # Loop over terms and apply Jordan-Wigner transformation.
  unique_strings = {}
  n_qubits = max([max(map(abs, term)) for term in fermion_terms])
  for fermion_coefficient, fermion_term in zip(fermion_coefficients,
                                               fermion_terms):
    pauli_coefficients, pauli_strings = SymbolicTermJW(fermion_coefficient,
                                                       fermion_term, n_qubits)

    # Simplify terms and update dictionary.
    for pauli_coefficient, pauli_string in zip(pauli_coefficients,
                                               pauli_strings):
      simple_coefficient, simple_string = SimplifyPauliString(pauli_string)
      key = tuple(simple_string)
      if key in unique_strings:
        unique_strings[key] += simple_coefficient * pauli_coefficient
      else:
        unique_strings[key] = simple_coefficient * pauli_coefficient

  # Remove zeros and return dictionary of unique strings.
  for pauli_string, pauli_coefficient in unique_strings.items():
    if abs(pauli_coefficient) < tol:
      del unique_strings[pauli_string]
    else:
      assert abs(pauli_coefficient.imag) < tol
      unique_strings[pauli_string] = pauli_coefficient.real
  return unique_strings


def PrintPauliHamiltonian(coefficients, terms):
  """Print out the terms in the spin Hamiltonian in a readable way.

  Args:
    coefficients: A list of fermion term coefficients (floats).
    terms: A list of fermion terms (list of ints).
  """
  # Get unique_strings.
  unique_strings = SymbolicTransformationJW(coefficients, terms)

  # Loop through strings and format term.
  print '\nNow printing Pauli strings and coefficients:'
  for pauli_string, pauli_coefficient in unique_strings.iteritems():
    readable_string = ''
    for tensor_factor, operator in enumerate(pauli_string):
      if operator != 'I':
        symbolic_operator = operator + str(tensor_factor) + ' '
        readable_string += symbolic_operator

    # Print out the shift differently from everything else.
    if readable_string:
      print '% .3f  I' % pauli_coefficient
    else:
      print '% .3f  %s' % (pauli_coefficient, readable_string)
