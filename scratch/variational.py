"""This module explicitly solves for the ghetto CC variational ansatz.
"""
import time
import numpy
import scipy
import warnings
import commutators
import scipy.sparse
import scipy.optimize
import scipy.sparse.linalg
from itertools import combinations
from sys import argv


# Return matrix representation of creation or annihilation operator.
def JordanWignerTerm(index, n_orbitals):
  """Make a matrix representation of a fermion operator.

  Args:
    index: This is a nonzero integer. The integer indicates the tensor
      factor and the sign indicates raising or lowering.
    n_orbitals: This int gives the number of orbitals.

  Returns: The corresponding fermion operator.
  """
  # Define operators.
  I = scipy.sparse.csr_matrix([[1, 0], [0, 1]], dtype=float)
  Z = scipy.sparse.csr_matrix([[1, 0], [0, -1]], dtype=float)
  Q_raise = scipy.sparse.csr_matrix([[0, 0], [1, 0]], dtype=float)
  Q_lower = scipy.sparse.csr_matrix([[0, 1], [0, 0]], dtype=float)

  # Construct fermionic operator.
  assert index
  orbital = abs(index)
  operator = 1
  for tensor_factor in range(orbital - 1):
    operator = scipy.sparse.kron(operator, I, 'csr')
  if index > 0:
    operator = scipy.sparse.kron(operator, Q_raise, 'csr')
  else:
    operator = scipy.sparse.kron(operator, Q_lower, 'csr')
  for tensor_factor in range(n_orbitals - orbital):
    operator = scipy.sparse.kron(operator, Z, 'csr')
  assert scipy.sparse.isspmatrix_csr(operator)
  return operator


# Initialize Jordan-Wigner dictionary so terms aren't constantly reformed.
def GetJordanWignerTerms(n_orbitals):
    jw_terms = {}
    for index in range(-n_orbitals, n_orbitals + 1):
      if index:
        jw_terms[index] = JordanWignerTerm(index, n_orbitals)
    return jw_terms


# Return a particular operator given the term and coefficient.
def MatrixForm(coefficient, term, jw_terms,
    add_conjugates=False, anti_hermitian=False):
  operator = coefficient
  for index in term:
    operator = operator * jw_terms[index]
  if add_conjugates and commutators.GetConjugate(term):
    if anti_hermitian:
      operator = 1j * (operator - operator.getH())
    else:
      operator = operator + operator.getH()
  return operator


# Expectation.
def Expectation(operator, psi):
  psi = scipy.sparse.csr_matrix(psi)
  operator = scipy.sparse.csr_matrix(operator)
  expectation = psi.getH() * operator * psi
  assert expectation.get_shape() == (1, 1)
  return numpy.real(expectation[0, 0])


# Get number operator.
def NumberOperator(n_orbitals):
  number_operator = 0
  jw_terms = GetJordanWignerTerms(n_orbitals)
  for tensor_factor in range(1, n_orbitals + 1):
    term = [tensor_factor, -tensor_factor]
    operator = MatrixForm(1, term, jw_terms)
    number_operator = number_operator + operator
  return number_operator


# Initialize Hartree-Fock state.
def HartreeFockState(n_electrons, n_orbitals):
  occupied = scipy.sparse.csr_matrix([[0], [1]], dtype=float)
  psi = 1.
  unoccupied = scipy.sparse.csr_matrix([[1], [0]], dtype=float)
  for orbital in range(n_electrons):
    psi = scipy.sparse.kron(psi, occupied, 'csr')
  for orbital in range(n_orbitals - n_electrons):
    psi = scipy.sparse.kron(psi, unoccupied, 'csr')
  return psi


# Test if matrix is Hermitian.
def IsHermitian(matrix):
  conjugate = matrix.getH()
  difference = matrix - conjugate
  if difference.nnz:
    discrepancy = max(map(abs, difference.data))
    if discrepancy > 1e-9:
      print 'Hermitian discrepancy = %s.' % repr(discrepancy)
      return False
  return True


# Construct an operator from terms and coefficients.
def MakeOperator(coefficients, terms, verbose=False):

  # Initialize.
  n_terms = len(coefficients)
  one_percent = numpy.rint(numpy.ceil(n_terms / 100.))
  start = time.clock()
  n_orbitals = commutators.OrbitalCount(terms)
  jw_terms = GetJordanWignerTerms(n_orbitals)

  # Loop over terms.
  operator = 0
  for i, (coefficient, term) in enumerate(zip(coefficients, terms)):
    operator = operator + MatrixForm(coefficient, term, jw_terms)

    # Report progress.
    if verbose and not (i + 1) % one_percent:
      percent_complete = numpy.rint(100. * (i + 1) / n_terms)
      elapsed = time.clock() - start
      rate = elapsed / percent_complete
      eta = rate * (100 - percent_complete)
      print('%s. Computation %i%% complete. Approximately %i '
            'minute(s) remaining.' % (time.strftime(
                '%B %d at %H:%M:%S', time.localtime()),
            percent_complete, round(eta / 60)))

  assert IsHermitian(operator)
  return operator


# Save a sparse matrix.
def SaveSparse(name, operator):
  numpy.savez(name, data=operator.data, indices=operator.indices,
              indptr=operator.indptr, shape=operator.shape)


# Load a sparse matrix.
def LoadSparse(name):
  loader = numpy.load(name)
  operator = scipy.sparse.csr_matrix((loader['data'], loader['indices'],
                                      loader['indptr']), shape=loader['shape'])
  return operator


# Get the relevant Slater determinants.
def GetDeterminants(n_orbitals, n_electrons, n_excitations):

  # Initialize reference state.
  reference = numpy.zeros(n_orbitals, int)
  reference[:n_electrons] = 1
  states = []

  # Loop over excitation level.
  for m in range(n_excitations + 1):
    for occupied in combinations(range(n_electrons), r=m):
      for unoccupied in combinations(range(n_electrons, n_orbitals), r=m):
        state = numpy.copy(reference)
        state[list(unoccupied)] = 1
        state[list(occupied)] = 0
        states += [state]
  return numpy.array(states)


# Return a projector into a restricted Fock space.
def ConfigurationProjector(n_orbitals, n_electrons, n_excitations):

  # Name.
  if n_excitations == 'all':
    n_excitations = n_orbitals - n_electrons
  name = 'data/operators/projector_%i_%i_%i.npz'\
      % (n_orbitals, n_electrons, n_excitations)

  # Attempt to load.
  try:
    projector = LoadSparse(name)

  except:
    # Initialize projector computation.
    n_hilbert = 2 ** n_orbitals
    states = GetDeterminants(n_orbitals, n_electrons, n_excitations)
    unoccupied = scipy.sparse.csr_matrix([[1], [0]], dtype=int)
    occupied = scipy.sparse.csr_matrix([[0], [1]], dtype=int)
    number_operator = NumberOperator(n_orbitals)

    # Construct projector.
    projector = 0
    for state in states:

      # Construct computational basis state in Hilbert space.
      ket = 1
      for qubit in state:
        if qubit:
          ket = scipy.sparse.kron(ket, occupied, 'csr')
        else:
          ket = scipy.sparse.kron(ket, unoccupied, 'csr')

      # Add to projector.
      density = ket * ket.getH()
      projector = projector + density

    # Save and return projector.
    SaveSparse(name, projector)
  return projector


# Return an operator.
def GetHamiltonian(molecule, basis, n_excitations='FCI'):

  # Try to load it.
  name = 'data/operators/%s_%s_hamiltonian_%s.npz'\
      % (molecule, basis, str(n_excitations))
  try:
    assert not recompute
    hamiltonian = LoadSparse(name)

  # Compute pre-projected Hamiltonian.
  except:
    coefficients, terms = commutators.GetHamiltonianTerms(molecule, basis)
    hamiltonian = MakeOperator(coefficients, terms)

    # Project, if necessary.
    if n_excitations != 'FCI':
      n_hilbert = hamiltonian.shape[0]
      n_electrons = commutators.ElectronCount(molecule)
      n_orbitals = int(numpy.rint(numpy.log2(n_hilbert)))
      if n_orbitals - n_electrons > n_excitations:
        projector = ConfigurationProjector(
            n_orbitals, n_electrons, n_excitations)
        hamiltonian = projector * hamiltonian * projector.getH()

    # Save and return.
    SaveSparse(name, hamiltonian)
  return hamiltonian


# Compute and save information about lowest or highest operator eigenvalue.
def SparseDiagonalize(operator, which='SA'):
  max_steps = 1e7
  values, vectors = scipy.sparse.linalg.eigsh(
      operator, 1, which=which, maxiter=max_steps)
  eigenstate = vectors[:, 0]
  eigenstate = scipy.sparse.csr_matrix(eigenstate)
  return eigenstate.getH()


# Return the unitary corresponding to evolution under hamiltonian.
def GetUnitary(time, term):
  exponent = scipy.sparse.csc_matrix(-1j * time * term)
  assert IsHermitian(1j * exponent)
  with warnings.catch_warnings(record=False):
    warnings.simplefilter('ignore')
  unitary = scipy.sparse.linalg.expm(exponent)
  unitary.eliminate_zeros()
  return unitary.tocsr()


# Make a function object to return energy of variational ansatz.
class Objective:

  # Initialize function object.
  def __init__(self, terms, reference_state, hamiltonian, cc_form=False):

    # Initialize dictionary of unitaries.
    self.terms = terms
    self.hamiltonian = hamiltonian
    self.reference_state = reference_state
    n_orbitals = commutators.OrbitalCount(terms)
    jw_terms = GetJordanWignerTerms(n_orbitals)
    self.matrices = {}
    for term in terms:
      self.matrices[tuple(term)] = MatrixForm(
          1., term, jw_terms, add_conjugates=True, anti_hermitian=cc_form)
      assert IsHermitian(self.matrices[tuple(term)])
    self.min_energy = numpy.inf
    self.calls = 0

  # Method to return unitary which prepares a state.
  def GetStatePrep(self, coefficients):
    unitary = 1.
    for coefficient, term in zip(coefficients, self.terms):
      term_matrix = self.matrices[tuple(term)].copy()
      unitary = GetUnitary(coefficient, term_matrix) * unitary
    return unitary

  # Method to return unitary which prepares a state using single unitary.
  def ModStatePrep(self, coefficients):
    hamiltonian = 0.
    for coefficient, term in zip(coefficients, self.terms):
      hamiltonian = hamiltonian + coefficient * self.matrices[tuple(term)].copy()
    unitary = GetUnitary(1., hamiltonian)
    return unitary

  # Write paren method which returns energy given variables.
  def __call__(self, variables):
    state_prep = self.GetStatePrep(variables)
    state = state_prep * self.reference_state
    energy = Expectation(self.hamiltonian, state)
    self.calls += 1
    if energy < self.min_energy:
      self.min_energy = energy
      print 'Lowest energy after %i calls is %f.' % (self.calls, energy)
    return energy


# Variationally determine optimal coefficients for state prep.
def VariationallyMinimize(terms, reference_state, initial_guess,
                          hamiltonian, cc_form=False):
  ObjectiveFunction = Objective(terms, reference_state, hamiltonian, cc_form)
  method = 'BFGS'
  solution = scipy.optimize.minimize(
      ObjectiveFunction, initial_guess, method=method).x
  energy = ObjectiveFunction(solution)
  return solution, energy


# Unit tests.
def main():

  # Test parameters.
  molecule = str(argv[1])
  basis = str(argv[2])
  try:
    cc_form = bool(argv[3])
  except:
    cc_form = False

  # Get Hamiltonian and terms.
  fci_hamiltonian = GetHamiltonian(molecule, basis)
  cisd_hamiltonian = GetHamiltonian(molecule, basis, 2)
  coefficients, ham_terms = commutators.GetHamiltonianTerms(molecule, basis,
      add_conjugates=False)
  n_electrons = commutators.ElectronCount(molecule)
  n_orbitals = commutators.OrbitalCount(ham_terms)
  initial_guess = []
  terms = []
  for coefficient, term in zip(coefficients, ham_terms):
    if commutators.GetConjugate(term):
      initial_guess += [coefficient]
      terms += [term]

  # Get states and energies.
  fci_state = SparseDiagonalize(fci_hamiltonian)
  cisd_state = SparseDiagonalize(cisd_hamiltonian)
  hf_state = HartreeFockState(n_electrons, n_orbitals)
  fci_energy = Expectation(fci_hamiltonian, fci_state)
  cisd_energy = Expectation(fci_hamiltonian, cisd_state)
  hf_energy = Expectation(fci_hamiltonian, hf_state)

  # Obtain variational solution.
  print 'Analyzing %s in the %s basis.\n' % (molecule, basis)
  solution, variational_energy = VariationallyMinimize(
      terms, hf_state, initial_guess, fci_hamiltonian, cc_form)
  print '\nExact (FCI) energy is %s.' % repr(float(fci_energy))
  print 'CISD energy is %s.' % repr(float(cisd_energy))
  print 'Hartree-Fock energy is %s.' % repr(float(hf_energy))
  print 'Variationally optimal energy is %s.\n' % repr(float(variational_energy))


# Run.
if __name__ == '__main__':
  main()
