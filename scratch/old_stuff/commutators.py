"""This code assists in the explicit computation of Trotterization errors.
Owners: Ryan Babbush (t-ryba).
"""
from csv import reader
from sys import argv
import random
import numpy
import time
import re


"""Note about ladder operator notation:
  Throughout this code ladder operators, i.e. creation and annilhlation
  operators, are represented by positive and negative integers, respectively.
  Usually, terms are stored as python lists; e.g. [2 1 -2 -3] means raising
  on tensor factor two, raisinng on tensor factor one, lowering on two, etc.
  Accordingly, we define 'normal form' to be when the term is ordered from
  greatest to smallest integer.
  """


# Return the number of orbitals given a list of Hamiltonian terms.
def OrbitalCount(terms):
  n_orbitals = 0
  for term in terms:
    n_orbitals = max(n_orbitals, max(map(abs, term)))
  return n_orbitals


# Number of electrons.
def ElectronCount(molecule):
  if molecule == 'H':
    n_electrons = 1
  elif molecule in ['H2', 'He', 'HeH+'] or molecule[:3] == 'H2_':
    n_electrons = 2
  elif molecule in ['Li']:
    n_electrons = 3
  elif molecule in ['Be', 'LiH']:
    n_electrons = 4
  elif molecule == 'B':
    n_electrons = 5
  elif molecule  in ['C', 'Li2', 'BeH2']:
    n_electrons = 6
  elif molecule == 'N':
    n_electrons = 7
  elif molecule in ['O', 'CH2']:
    n_electrons = 8
  elif molecule == 'F':
    n_electrons = 9
  elif molecule in ['HF', 'H2O', 'NH3', 'CH4', 'Ne'] or molecule[:3] == 'HF_':
    n_electrons = 10
  elif molecule == 'Na':
    n_electrons = 11
  elif molecule in ['Mg', 'LiF']:
    n_electrons = 12
  elif molecule == 'Al':
    n_electrons = 13
  elif molecule in ['Si', 'CO']:
    n_electrons = 14
  elif molecule == 'P':
    n_electrons = 15
  elif molecule in ['S', 'NF', 'O2']:
    n_electrons = 16
  elif molecule == 'Cl':
    n_electrons = 17
  elif molecule in ['F2', 'HCl', 'H2S', 'Ar']:
    n_electrons = 18
  return n_electrons


# Check if an operator in normal form is a number operator.
def IsNumberOperator(term):
  n_operators = len(term)
  assert not n_operators % 2
  for i in range(n_operators // 2):
    if term[i] != -term[-(1 + i)]:
      return False
  return True


# Check if the double commutator [A, [B, C]] is trivially zero.
def TriviallyCommutes(A, B, C):
  if set(B).intersection(set(C)):
    return False
  elif set(A).intersection(set(B + C)):
    return False
  else:
    return True


# Return conjugate of a term, if it exists.
def GetConjugate(term):
  conjugate = [-i for i in term[::-1]]
  if term == conjugate:
    return []
  else:
    return conjugate


# Convert an input form into normal form and update hash table.
def AddNormalForm(coefficient, term, sum_terms):
  """Add an input term to the dictionary of total terms in normal form.

  Args:
    coefficient: A float giving the coefficient of the input term.
    term: A python list of integers specifying the term.
    sum_terms: A hash table with keys that are tuples of integers in normal
      form and keys that are floats representing the coefficient of that term.
  """
  # The terms to process are held in a queue.
  # At this point, check for complex conjugates.
  input_coefficients = [coefficient]
  input_terms = [term]

  # Loop until queue is empty.
  while len(input_terms):

    # Pop queues.
    coefficient = input_coefficients.pop(0)
    term = input_terms.pop(0)

    # Iterate from left to right across terms and reorder to normal form.
    # Swap terms into correct position by moving left to right.
    for i in range(1, len(term)):
      for j in range(i, 0, -1):
        right = term[j]
        left = term[j - 1]

        # If right operator is larger than left, swap!
        if right > left:
          term[j - 1] = right
          term[j] = left

          # If swapping lowering and raising operators of same tensor factor,
          # add (1 - right * left) by pushing term without left and right.
          if left == -right:
            input_coefficients += [coefficient]
            input_terms += [term[:(j - 1)] + term[(j + 1)::]]

          # Flip product sign.
          coefficient *= -1

        # If left operator is larger than right operator, stop swapping.
        else:
          break

    # Add processed term to hash table.
    if len(term) == len(set(term)):
      try:
        sum_terms[tuple(term)] += coefficient
      except:
        sum_terms[tuple(term)] = coefficient


# Order terms for Trotter series.
def InterleaveTerms(unique_terms):
  """Put terms into a specific ordering.

  Args: A dictionary with keys that are tuples of ints giving
      the terms. The values are the term coefficients.

  Returns:
    ordered_terms: A list of terms in some special order.
    ordered_coefficients: A list of coefficients in some special order.
  """
  # Initialize ordered lists.
  ordered_coefficients = []
  ordered_terms = []
  n_orbitals = 0
  for term in unique_terms.keys():
    n_orbitals = max([n_orbitals, max(map(abs, term))])

  # First add all Hpp terms as they mutually commute.
  for p in range(1, n_orbitals + 1):
    term = (p, -p)
    coefficient = unique_terms.pop(term, False)
    if coefficient:
      ordered_coefficients += [coefficient]
      ordered_terms += [list(term)]

  # Now add the Hpqqp terms, which also mutually commute.
  for p in range(1, n_orbitals + 1):
    for q in range(1, p + 1):
      term = (p, q, -q, -p)
      coefficient = unique_terms.pop(term, False)
      if coefficient:
        ordered_coefficients += [coefficient]
        ordered_terms += [list(term)]

  # Interleave.
  for p in range(1, n_orbitals + 1):
    for q in range(1, n_orbitals + 1):
      term = (p, -q)
      coefficient = unique_terms.pop(term, False)
      if coefficient:
        ordered_coefficients += [coefficient]
        ordered_terms += [list(term)]

      for r in range(1, min([p, q]) + 1):
        term = (p, r, -r, -q)
        coefficient = unique_terms.pop(term, False)
        if coefficient:
          ordered_coefficients += [coefficient]
          ordered_terms += [list(term)]

  # Lexicographic ordering of PQRS terms.
  for p in range(1, n_orbitals + 1):
    for q in range(1, p + 1):
      for r in range(1, n_orbitals + 1):
        for s in range(r, n_orbitals + 1):
          term = (p, q, -r, -s)
          coefficient = unique_terms.pop(term, False)
          if coefficient:
            ordered_coefficients += [coefficient]
            ordered_terms += [list(term)]

  # Return.
  assert not len(unique_terms)
  return ordered_coefficients, ordered_terms


# Function to load Jarrod's Hamiltonians.
def GetHamiltonianTerms(molecule, integral_type,
                        add_conjugates=True, verbose=False):
  """Parse and load molecular integral data for nonzero terms in normal form.

  Args:
    molecule: A string giving the proper chemical name, e.g. HCl or Fe2S2.
    integral_type: e.g. 'OAO'
    add_conjugates: If True, explicitly include conjugates in term list.
    verbose: If True, print stuff.

  Returns:
    coefficients: A list of floats giving the coefficients of the terms.
    terms: A list of lists of ints giving the terms in normal form.
  """
  # Get the name of the file.
  if 1:
    name = 'data/from_jarrod/%s-%s.int' % (molecule, integral_type)
  else:
    name = 'data/distances/%s-%s.int' % (molecule, integral_type)

  # Get the dictionary of integrals.
  integrals = {}
  n_orbitals = 0
  expression = r'-?\d+\.\d+|\d+'
  with open(name, 'r') as file:
    for line in file:
      data = re.findall(expression, line)

      # Single electron integral.
      if len(data) == 3:
        term = tuple([i + 1 for i in map(int, data[:2])])
        coefficient = float(data[-1])
        integrals[term] = coefficient
        n_orbitals = max(n_orbitals, max(term))

      # Two electron integral.
      elif len(data) == 5:
        term = tuple([i + 1 for i in map(int, data[:4])])
        coefficient = float(data[-1])
        # There really shouldn't be a negative sign on next line but
        # there needs to be for Jarrod's early files.
        integrals[term] = -coefficient
        n_orbitals = max(n_orbitals, max(term))

  # Loop over all possible terms and add the normal form.
  unique_terms = {}
  for p in range(1, n_orbitals + 1):
    for q in range(1, n_orbitals + 1):

      # Add 1-electron terms.
      coefficient = integrals.pop((p, q), False)
      if coefficient:
        term = [p, -q]
        AddNormalForm(coefficient, term, unique_terms)

      # Add 2-electron terms.
      for r in range(1, n_orbitals + 1):
        for s in range(1, n_orbitals + 1):
          coefficient = integrals.pop((p, q, r, s), False) / 2.
          if coefficient:
            term = [p, q, -r, -s]
            AddNormalForm(coefficient, term, unique_terms)

  # Remove complex conjugates.
  if not add_conjugates:
    for p in range(1, n_orbitals + 1):
      for q in range(1, n_orbitals + 1):
        term = (p, -q)
        if term in unique_terms:
          conjugate = tuple(GetConjugate(list(term)))
          unique_terms.pop(conjugate, False)
        for r in range(1, n_orbitals + 1):
          for s in range(1, n_orbitals + 1):
            term = (p, q, -r, -s)
            if term in unique_terms:
              conjugate = tuple(GetConjugate(list(term)))
              unique_terms.pop(conjugate, False)

  # Order terms and return.
  assert not len(integrals)
  coefficients, terms = InterleaveTerms(unique_terms)
  if verbose:
    print '\nOrder of terms in Trotter series:'
    for coefficient, term in zip(coefficients, terms):
      print coefficient, term
  return coefficients, terms


# Function to exactly compute double commutator sum.
def DoubleCommutators(molecule, basis, verbose=False):
  """This function computes the sum of all the double commutators.

  Args:
    coefficients: A python list of floats giving all term coefficients.
      This list needs to be provided in the same order as "terms" and
      that order should reflect the order of operators in the Trotter series.
    terms: A python list of lists of ints specifying all valid terms.

  Returns: A hash table with keys that are tuples of integers.
    corresponding to all nonzero operators in the simplified commutator
    sum. The values are floats corresponding the operator coefficients.
  """
  # Initialize.
  sum_terms = {}
  coefficients, terms = GetHamiltonianTerms(molecule, basis, False, verbose)
  n_terms = len(coefficients)
  n_orbitals = OrbitalCount(terms)
  print '\nHamiltonian contains %i distinct terms and %i orbitals.'\
    % (n_terms, n_orbitals)
  for term, coefficient in zip(terms, coefficients):
    if max(map(abs, term)) <= 17:
      print term, coefficient

  # Count commutators.
  n_commutators = 0
  for b in range(1, n_terms):
    n_commutators += (b + 1) * b
  one_percent = round(n_commutators / 100.)
  print 'There are %i possible commutators.\n' % n_commutators

  # Loop over all possible combinations.
  # Compute the sum over a <= b, b, c < c of...
  # (1 / 12) * [A * (1 - delta(A, B)/2), [B, C]] = ...
  # ((1 - delta(A, B) / 2) / 12) * (A B C - A C B - B C A + C B A).
  start = time.clock()
  counter = 0

  # Loop over B.
  for b in xrange(1, n_terms):
    B = terms[b]
    B_conjugate = GetConjugate(B)
    B_coefficient = coefficients[b]
    if B_conjugate:
      B_terms = [B] + [B_conjugate]
    else:
      B_terms = [B]

    # Loop over A.
    for a in xrange(b + 1):
      A = terms[a]
      A_conjugate = GetConjugate(A)
      A_coefficient = coefficients[a]
      if A_conjugate:
        A_terms = [A] + [A_conjugate]
      else:
        A_terms = [A]

      # Loop over C.
      for c in xrange(b):
        C = terms[c]
        C_conjugate = GetConjugate(C)
        C_coefficient = coefficients[c]
        if C_conjugate:
          C_terms = [C] + [C_conjugate]
        else:
          C_terms = [C]

        # Get coefficient.
        counter += 1
        coefficient = A_coefficient * B_coefficient * C_coefficient
        if a == b:
          coefficient /= 24.
        else:
          coefficient /= 12.

        # Compute commutators.
        for A_ in A_terms:
          for B_ in B_terms:
            for C_ in C_terms:
              if not TriviallyCommutes(A_, B_, C_):
                AddNormalForm(coefficient, A_ + B_ + C_, sum_terms)
                AddNormalForm(-coefficient, A_ + C_ + B_, sum_terms)
                AddNormalForm(-coefficient, B_ + C_ + A_, sum_terms)
                AddNormalForm(coefficient, C_ + B_ + A_, sum_terms)

        # Report progress.
        if not (counter % one_percent):
          percent_complete = counter / one_percent
          elapsed = time.clock() - start
          rate = elapsed / percent_complete
          eta = rate * (100 - percent_complete)
          print('%s. Computation %i%% complete. Approximately %i '
                'minute(s) remaining.' % (time.strftime(
                    '%B %d at %H:%M:%S', time.localtime()),
                percent_complete, round(eta / 60)))

  # Return.
  for term, coefficient in sum_terms.items():
    if not coefficient:
      del sum_terms[term]
  return sum_terms


# Save error data.
def SaveData(molecule, basis, sum_terms):
  name = 'data/error_terms/%s_%s.txt' % (molecule, basis)
  with open(name, 'w') as file:
    for term, coefficient in sum_terms.items():
      for operator in term:
        file.write('%i ' % operator)
      file.write(repr(coefficient) + '\n')


# Load error terms.
def GetErrorTerms(molecule, basis):
  name = 'data/error_terms/%s_%s.txt' % (molecule, basis)
  coefficients = []
  terms = []
  with open(name, 'r') as file:
    for line in file:
      data = line.split()
      coefficients += [float(data[-1])]
      terms += [map(int, data[:(len(data) - 1)])]
  coefficients = numpy.array(coefficients)
  return coefficients, terms


# Run code.
def main():

  # Parameters.
  molecule = str(argv[1])
  basis = str(argv[2])
  verbose = True

  # Compute double commutators.
  start = time.clock()
  sum_terms = DoubleCommutators(molecule, basis)
  SaveData(molecule, basis, sum_terms)
  elapsed = time.clock() - start

  # Print out all terms.
  if verbose:
    print '\nPrinting error terms:'
    for term, coefficient in sum_terms.items():
      print coefficient, term
  print '\nNumber of non-zero terms: %i.' % len(sum_terms)
  print 'Elapsed time: %i seconds.' % elapsed


# Run.
if __name__ == '__main__':
  main()
