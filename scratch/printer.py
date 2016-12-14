from ast import literal_eval
import scipy.sparse.linalg
from sys import argv
import scipy.sparse
import commutators
import numpy


# Get operator from string.
def StringToMatrix(string, n_orbitals):
  terms = string.split()
  I = scipy.sparse.csr_matrix([[1, 0], [0, 1]], dtype=complex)
  X = scipy.sparse.csr_matrix([[0, 1], [1, 0]], dtype=complex)
  Y = scipy.sparse.csr_matrix([[0,-1j], [1j, 0]], dtype=complex)
  Z = scipy.sparse.csr_matrix([[1, 0], [0, -1]], dtype=complex)

  # Build vector.
  vector = n_orbitals * ['I']
  for term in terms:
    operator = term[0]
    if operator != 'I':
      tensor_factor = int(term[2::])
      vector[tensor_factor] = operator

  # Multiply terms together.
  matrix = 1
  for operator in vector:
    if operator == 'X':
      matrix = scipy.sparse.kron(matrix, X, 'csr')
    elif operator == 'Y':
      matrix = scipy.sparse.kron(matrix, Y, 'csr')
    elif operator == 'Z':
      matrix = scipy.sparse.kron(matrix, Z, 'csr')
    else:
      assert operator == 'I'
      matrix = scipy.sparse.kron(matrix, I, 'csr')
  return matrix


# Convert Peter's list into a Pauli operator.
def ConvertToString(term):
  string = ''
  for tensor_factor, operator in enumerate(term):
    if operator == 1:
      string += ' X_%s' % str(tensor_factor)
    elif operator == 2:
      string += ' Y_%s' % str(tensor_factor)
    elif operator == 3:
      string += ' Z_%s' % str(tensor_factor)
    else:
      assert not operator
  if not string:
    return 'I'
  else:
    return string[1::]


# Get Bravyi-Kitaev terms.
def GetBravyiKitaevTerms(molecule, basis):
  name = 'data/bk_terms/%s_%s.dat' % (molecule, basis)
  term_dictionary = {}

  # Go through all the lines.
  with open(name, 'r') as file:
    for line in file:
      if line[0] != 'i':

        # Get term.
        for i, char in enumerate(line):
          if char == ']':
            term = literal_eval(line[:(i + 1)])
            break

        # Get rest of data.
        if commutators.GetConjugate(term):
          data = literal_eval(line[(2 * i + 6)::])
          term_dictionary[tuple(commutators.GetConjugate(term))] = data
        else:
          data = literal_eval(line[(i + 3)::])
        term_dictionary[tuple(term)] = data

  return term_dictionary


# Run code.
def main():

  # Parameters.
  molecule = str(argv[1])
  basis = str(argv[2])
  coefficients, terms = commutators.GetHamiltonianTerms(molecule, basis)
  try:
    n_orbitals = int(argv[3])
  except:
    n_orbitals = commutators.OrbitalCount(terms)
  try:
    assert argv[4]
    flag = False
  except:
    flag = True

  # Organize terms.
  term_dictionary = GetBravyiKitaevTerms(molecule, basis)
  bk_dictionary = {}
  for coefficient, term in zip(coefficients, terms):
    if max(map(abs, term)) <= n_orbitals:
      data = term_dictionary[tuple(term)]
      for operator in data:
        value = coefficient * numpy.real(operator[0])
        string = ConvertToString(operator[1])
        if string in bk_dictionary:
          bk_dictionary[string] += value
        else:
          bk_dictionary[string] = value

  # Print shit out.
  if flag:
    print '# Terms for %s in the %s basis, truncated to %i orbitals.'\
        % (molecule, basis, n_orbitals)
    for term, coefficient in bk_dictionary.iteritems():
      print '%4f : %s' % (coefficient, term)

  # Diagonalize.
  else:
    hamiltonian = 0
    for term, coefficient in bk_dictionary.iteritems():
      operator = coefficient * StringToMatrix(term, n_orbitals)
      hamiltonian = hamiltonian + operator
    print hamiltonian
    values, vectors = scipy.sparse.linalg.eigsh(hamiltonian, 1, which='SA')
    eigenstate = vectors[:, 0]
    energy = values[0]
    print 'Ground state energy is %s.' % repr(energy)


# Run.
if __name__ == '__main__':
  main()
