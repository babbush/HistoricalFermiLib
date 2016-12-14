"""This module explicitly constructs Hamiltonians.
"""
import time
import numpy
import scipy
import warnings
import commutators
import scipy.sparse
import scipy.sparse.linalg
from itertools import combinations
from sys import argv
import pylab


# Set global plot parameters.
pylab.rcParams['text.usetex'] = True
pylab.rcParams['text.latex.unicode'] = True
pylab.rc('text', usetex=True)
pylab.rc('font', family='sans=serif')
marker_size = 6
line_width = 2
axis_size = 18
font_size = 24


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
def MatrixForm(coefficient, term, jw_terms, add_conjugates=False):
  operator = coefficient
  for index in term:
    operator = operator * jw_terms[index]
  if add_conjugates and commutators.GetConjugate(term):
    operator = operator + operator.getH()
  return operator


# Expectation.
def Expectation(operator, psi):
  psi = scipy.sparse.csr_matrix(psi)
  operator = scipy.sparse.csr_matrix(operator)
  expectation = psi.getH() * operator * psi
  assert expectation.get_shape() == (1, 1)
  return expectation[0, 0]


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
def MakeOperator(coefficients, terms, verbose=False, add_conjugates=False):

  # Initialize.
  n_terms = len(coefficients)
  one_percent = numpy.rint(numpy.ceil(n_terms / 100.))
  start = time.clock()
  n_orbitals = commutators.OrbitalCount(terms)
  jw_terms = GetJordanWignerTerms(n_orbitals)

  # Loop over terms.
  operator = 0
  for i, (coefficient, term) in enumerate(zip(coefficients, terms)):
    operator = operator + MatrixForm(coefficient, term, jw_terms,
        add_conjugates)

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
def GetOperator(molecule, basis, operator_type,
                n_excitations='FCI', recompute=False):

  # Try to load it.
  name = 'data/operators/%s_%s_%s_%s.npz'\
      % (molecule, basis, operator_type, str(n_excitations))
  try:
    assert not recompute
    operator = LoadSparse(name)

  # Compute pre-projected operators.
  except:
    if n_excitations == 'FCI':

      # Compute Hamiltonian.
      if operator_type == 'hamiltonian':
        coefficients, terms = commutators.GetHamiltonianTerms(molecule, basis)
        operator = MakeOperator(coefficients, terms, True)

      # Compute error operator.
      else:
        assert operator_type == 'error'
        coefficients, terms = commutators.GetErrorTerms(molecule, basis)

        # Truncate terms.
        cutoff = 12
        relevant = numpy.absolute(coefficients) > 10 ** (-cutoff)
        coefficients = coefficients[relevant]
        terms = [terms[i] for i in numpy.argwhere(relevant)]
        operator = MakeOperator(coefficients, terms, True)

    # Project, if necessary.
    else:
      assert operator_type == 'hamiltonian'
      operator = GetOperator(molecule, basis, 'hamiltonian', 'FCI', recompute)
      n_electrons = commutators.ElectronCount(molecule)
      n_hilbert = operator.shape[0]
      n_orbitals = int(numpy.rint(numpy.log2(n_hilbert)))
      if n_orbitals - n_electrons > n_excitations:
        projector = ConfigurationProjector(
            n_orbitals, n_electrons, n_excitations)
        operator = projector * operator * projector.getH()

    # Save and return.
    SaveSparse(name, operator)
  return operator


# Compute and save information about lowest or highest operator eigenvalue.
# Get the highest state of the error operator or lowest state of Hamiltonian.
def SparseDiagonalize(molecule, basis, operator_type,
                      n_excitations='FCI', recompute=False):
  state_name = 'data/eigensystems/eigenstate_%s_%s_%s_%s.npz'\
      % (molecule, basis, operator_type, str(n_excitations))
  try:
    # Load data if file exists. Otherwise, compute it.
    assert not recompute
    eigenstate = LoadSparse(state_name)

  except:
    # Compute operator and diagonalize.
    max_steps = 1e7
    operator = GetOperator(molecule, basis, operator_type,
                           n_excitations, recompute)
    if operator_type == 'error':
      values, vectors = scipy.sparse.linalg.eigsh(
          operator, 1, which='LM', maxiter=max_steps)
      eigenstate = vectors[:, 0]
    elif operator_type == 'hamiltonian':
      values, vectors = scipy.sparse.linalg.eigsh(
          operator, 1, which='SA', maxiter=max_steps)
      eigenstate = vectors[:, 0]

    # Save data and return.
    eigenstate = scipy.sparse.csr_matrix(eigenstate)
    SaveSparse(state_name, eigenstate)
  return eigenstate.getH()


# Get information about projected ground state at certain level of excitation.
# Specifically, return error, energy, (overlap | norm), and ground state.
def GetData(molecule, basis, n_excitations='FCI', recompute=False):

  # Attempt to load state and data.
  data_name = 'data/eigensystems/data_%s_%s_%s.npy'\
      % (molecule, basis, str(n_excitations))
  try:
    assert not recompute
    error, energy, other = numpy.load(data_name)

  # Compute state and data.
  except:
    fci_state = SparseDiagonalize(
        molecule, basis, 'hamiltonian', 'FCI', recompute)
    hamiltonian = GetOperator(
        molecule, basis, 'hamiltonian', n_excitations, recompute)
    error_operator = GetOperator(molecule, basis, 'error', 'FCI', recompute)

    # Get unrestricted information.
    if n_excitations == 'FCI':
      state = fci_state
      error = Expectation(error_operator, fci_state)
      energy = Expectation(hamiltonian, fci_state)
      error_state = SparseDiagonalize(
          molecule, basis, 'error', 'FCI', recompute)
      other = abs(Expectation(error_operator, error_state))

    # Get restricted information.
    else:
      n_electrons = commutators.ElectronCount(molecule)
      n_hilbert = fci_state.shape[0]
      n_orbitals = int(numpy.rint(numpy.log2(n_hilbert)))
      state = SparseDiagonalize(
          molecule, basis, 'hamiltonian', n_excitations, recompute)
      energy = Expectation(hamiltonian, state)
      error = Expectation(error_operator, state)
      other = Expectation(fci_state * fci_state.getH(), state)

    # Save and return.
    numpy.save(data_name, [error, energy, other])
  return error, energy, other


# Return all eigenvectors and eigenvalues of Hamiltonian or error operator.
def GetEigensystem(molecule, basis, operator_type, recompute=False):

  # Attempt to load.
  state_name = 'data/eigensystems/dense_states_%s_%s_%s.npy'\
      % (molecule, basis, operator_type)
  spectrum_name = 'data/eigensystems/spectrum_%s_%s_%s.npy'\
      % (molecule, basis, operator_type)
  try:
    assert not recompute
    eigenstates = numpy.load(state_name)
    eigenvalues = numpy.load(spectrum_name)

  except:
    # Get operator and diagonalize.
    operator = GetOperator(molecule, basis, operator_type, 'FCI', recompute)
    values, vectors = numpy.linalg.eigh(operator.todense())
    order = numpy.argsort(values)
    eigenvalues = values[order]
    eigenstates = vectors[:, order].transpose()

    # Save and return.
    numpy.save(state_name, eigenstates)
    numpy.save(spectrum_name, eigenvalues)
  return eigenvalues, numpy.matrix(eigenstates)


# Create a Haar random unitary.
def HaarRandom(n_hilbert, seed=0):

  # Load unitary, if it exists.
  name = 'data/operators/haar_unitary_%i_%i.npy' % (n_hilbert, seed)
  try:
    U = numpy.load(name)
  except:
    X = (numpy.random.randn(n_hilbert, n_hilbert) + 1j * numpy.random.randn(
        n_hilbert, n_hilbert)) / numpy.sqrt(2)
    Q, R = numpy.linalg.qr(X)
    R = numpy.matrix(numpy.diag(numpy.diag(R) / numpy.absolute(numpy.diag(R))))
    Q = numpy.matrix(Q)
    U = numpy.array(Q * R)
    numpy.save(name, U)
  return numpy.matrix(U).getH()


# Get the expectation of each Hamiltonian eigenstate with the error operator.
def Expectations(molecule, basis, seed=0, recompute=False):

  # Load expectations if it exists.
  if seed:
    name = 'data/contributions/%s_%s_expectations_%i.npy'\
        % (molecule, basis, seed)
  else:
    name = 'data/contributions/%s_%s_expectations.npy'\
        % (molecule, basis)
  try:
    assert not recompute
    expectations = numpy.load(name)

  # Compute expectations.
  except:

    # Load data.
    error_operator = GetOperator(molecule, basis, 'error')
    n_hilbert = error_operator.shape[0]
    if seed:
      wavefunctions = HaarRandom(n_hilbert, seed)
    else:
      wavefunctions = GetEigensystem(molecule, basis, 'hamiltonian')[1]
    expectations = numpy.zeros(n_hilbert)

    # Compute expectations.
    one_percent = numpy.rint(numpy.ceil(n_hilbert / 100.))
    start = time.clock()
    for i, wavefunction in enumerate(wavefunctions):
      expectations[i] = numpy.real(
          Expectation(error_operator, wavefunction.getH()))

      # Report progress.
      if not (i + 1) % one_percent:
        percent_complete = numpy.rint(100. * (i + 1) / n_hilbert)
        elapsed = time.clock() - start
        rate = elapsed / percent_complete
        eta = rate * (100 - percent_complete)
        print('%s. Computation %i%% complete. Approximately %i '
              'minute(s) remaining.' % (time.strftime(
                  '%B %d at %H:%M:%S', time.localtime()),
              percent_complete, round(eta / 60)))

    # Save and return.
    numpy.save(name, expectations)
  return expectations


# Compute contributions to ground state error.
def Contributions(molecule, basis, recompute=False):

  # Load error contribution data if it exists.
  coefficients, terms = commutators.GetErrorTerms(molecule, basis)
  name = 'data/contributions/%s_%s.npy' % (molecule, basis)
  try:
    assert not recompute
    contributions = numpy.load(name)

  except:
    # Compute wavefunction.
    wavefunction = SparseDiagonalize(molecule, basis, 'hamiltonian')
    n_electrons = commutators.ElectronCount(molecule)
    n_orbitals = commutators.OrbitalCount(terms)
    n_terms = len(coefficients)

    # Determine the FCI contribution of each error term.
    jw_terms = GetJordanWignerTerms(n_orbitals)
    contributions = numpy.zeros(n_terms)
    one_percent = numpy.rint(numpy.ceil(n_terms / 100.))
    start = time.clock()
    for i in range(n_terms):

      # Build operator using precomputed Jordan-Wigner terms.
      term = terms[i]
      coefficient = coefficients[i]
      operator = MatrixForm(coefficient, term, jw_terms)
      contributions[i] = Expectation(operator, wavefunction)

      # Report progress.
      if not (i + 1) % one_percent:
        percent_complete = numpy.rint(100. * (i + 1) / n_terms)
        elapsed = time.clock() - start
        rate = elapsed / percent_complete
        eta = rate * (100 - percent_complete)
        print('%s. Computation %i%% complete. Approximately %i '
              'minute(s) remaining.' % (time.strftime(
                  '%B %d at %H:%M:%S', time.localtime()),
              percent_complete, round(eta / 60)))

    # Save contribution data and return.
    numpy.save(name, contributions)
  return contributions, terms


# Compute the variance of a operator given a state.
def GetVariance(operator, psi):
  psi = scipy.sparse.csr_matrix(psi)
  operator = scipy.sparse.csr_matrix(operator)

  expect_op_squared = psi.getH() * operator * psi
  assert expect_op_squared.get_shape() == (1, 1)
  expect_op_squared = numpy.square(expect_op_squared[0, 0])

  expect_squared_op = psi.getH() * operator * operator * psi
  assert expect_squared_op.get_shape() == (1, 1)
  expect_squared_op = expect_squared_op[0, 0]

  return expect_squared_op - expect_op_squared


# Plot variance histogram.
def PlotVariance(molecule, basis, n_excitations):
  global axis_size
  global font_size

  # Get and report variances.
  print '\nThe following lists the term and its variance:'
  state = SparseDiagonalize(molecule, basis, 'hamiltonian', n_excitations)
  coefficients, terms = commutators.GetHamiltonianTerms(molecule, basis,
    add_conjugates=False)
  n_orbitals = commutators.OrbitalCount(terms)
  jw_terms = GetJordanWignerTerms(n_orbitals)
  variances = []
  epsilon = 0.01
  n_terms = len(coefficients)
  measurements = 0
  for coefficient, term in zip(coefficients, terms):
    operator = MatrixForm(
        coefficient, term, jw_terms, add_conjugates=True)
    variance = GetVariance(operator, state)
    print term, variance
    variances += [variance]
    measurements += numpy.floor(variance * n_terms / (epsilon * epsilon))
  variances = pylab.array(variances)
  print 'Mean variance is %f.' % (pylab.sqrt(variances.dot(variances)) / float(n_terms))
  print 'We will require %i measurements.' % int(measurements)

  # Plot.
  if 0:
    n_bins = 200
    pylab.figure(1)
    n, bins, patches = pylab.hist(
        variances, bins=n_bins, log=True, normed=False, histtype='stepfilled')
    pylab.xticks(size=axis_size)
    pylab.yticks(size=axis_size)
    pylab.xlim([0, pylab.amax(variances)])
    pylab.xlabel(r'Variance', fontsize=font_size)
    pylab.ylabel(r'Number of terms', fontsize=font_size)
    pylab.tight_layout()
    pylab.show()


# Get expected results for a term with a state.
def ExpectedResults(state, term):

  # Make operator.
  I = scipy.sparse.csr_matrix([[1, 0], [0, 1]], dtype=complex)
  X = scipy.sparse.csr_matrix([[0, 1], [1, 0]], dtype=complex)
  Y = scipy.sparse.csr_matrix([[0, -1j], [1j, 0]], dtype=complex)
  Z = scipy.sparse.csr_matrix([[1, 0], [0, -1]], dtype=complex)

  # Construct fermionic operator.
  matrix = 1
  for operator in term:
    if operator == 'I':
      matrix = scipy.sparse.kron(matrix, I, 'csr')
    elif operator == 'X':
      matrix = scipy.sparse.kron(matrix, X, 'csr')
    elif operator == 'Y':
      matrix = scipy.sparse.kron(matrix, Y, 'csr')
    elif operator == 'Z':
      matrix = scipy.sparse.kron(matrix, Z, 'csr')

  # Get expectation value and variance.
  with warnings.catch_warnings(record=False):
    warnings.simplefilter('ignore')
    expectation = float(Expectation(matrix, state))
    variance = float(GetVariance(matrix, state))
  print 'Expectation of %s is %f with variance of %f.'\
      % (str(term), expectation, variance)
  return None


# Unit tests.
def main():

  # Test parameters.
  molecule = str(argv[1])
  basis = str(argv[2])
  try:
    n_excitations = argv[3]
  except:
    n_excitations = 'FCI'
  recompute = 0

  hamiltonian = GetOperator(molecule, basis, 'hamiltonian')
  values, vectors = scipy.sparse.linalg.eigsh(
    hamiltonian, 1, which='LA')


  # Report data different depending on whether data is on FCI or truncated CI.
  if n_excitations == 'FCI':
    print '\nAnalyzing %s in the %s basis:' % (molecule, basis)
    error, energy, error_norm = GetData(
        molecule, basis, n_excitations, recompute)
    print 'Norm of error operator is %s.' % repr(error_norm)

  # Truncated CI.
  else:
    n_excitations = int(n_excitations)
    print '\nAnalyzing %s in the %s basis with restriction to %i excitations:'\
      % (molecule, basis, n_excitations)
    error, energy, overlap = GetData(
        molecule, basis, n_excitations, recompute)
    print 'Overlap with FCI state is %s.' % repr(overlap)

  # Print common data.
  print 'Energy is %s.' % repr(energy)
  print 'Error is %s.' % repr(error)

  ## Evaluate a term.
  #term = ['Y','X','X','Y']
  #state = SparseDiagonalize(molecule, basis, 'hamiltonian', n_excitations)
  #s = state.todense()
  #print s[abs(s) > 1e-3]
  #ExpectedResults(state, term)


  ## Plot variance.
  #if 0:
  #  PlotVariance(molecule, basis, n_excitations)

  ## Determine further calculation based on size.
  #n_electrons = commutators.ElectronCount(molecule)
  #if n_electrons <= 10 and 0:

  #  # Get spectra.
  #  hamiltonian_spectrum, wavefunctions = GetEigensystem(
  #      molecule, basis, 'hamiltonian', recompute)
  #  error_spectrum, error_functions = GetEigensystem(
  #      molecule, basis, 'error', recompute)
  #  expectations = Expectations(molecule, basis, 1, recompute)
  #  expectations = Expectations(molecule, basis, 2, recompute)
  #  expectations = Expectations(molecule, basis, 3, recompute)
  #  expectations = Expectations(molecule, basis, 0, recompute)

  #  # Analyze error operator.
  #  energy_sigma = numpy.std(hamiltonian_spectrum)
  #  error_sigma = numpy.std(error_spectrum)
  #  expectation_sigma = numpy.std(expectations)
  #  print '\nStandard deviation of energies is %s.' % repr(energy_sigma)
  #  print 'Standard deviation of error is %s.' % repr(error_sigma)
  #  print 'Standard deviation of error expectation is %s.'\
  #      % repr(expectation_sigma)

  #  # Get error contributions.
  #  tolerance = 1e-9
  #  if n_excitations == 'FCI':
  #    contributions = Contributions(molecule, basis, recompute)[0]
  #    contribution_error = numpy.sum(contributions)
  #    discrepancy = abs(contribution_error - error)
  #    if discrepancy > tolerance:
  #      print '\nWarnings: sum of contributions is off by %s.'\
  #          % repr(discrepancy)


# Run.
if __name__ == '__main__':
  main()
