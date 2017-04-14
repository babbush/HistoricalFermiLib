"""This code analyzes the Trotterization error terms.
Owners: Ryan Babbush (t-ryba).
"""
from scipy.misc import factorial
from sys import argv
import commutators
import operators
import numpy
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


# Truncate terms and values.
def Truncate(values, terms, cutoff):
  threshold = 10 ** (-cutoff)
  truncated_terms = []
  truncated_values = values[pylab.absolute(values) > threshold]
  for value, term in zip(values, terms):
    if abs(value) > threshold:
      truncated_terms += [term]
  return truncated_values, truncated_terms


# Plot a histogram of values.
def MakeHistogram(values, value_type, name, cutoff):
  global axis_size
  global font_size
  n_bins = 300
  log_bins = pylab.logspace(-cutoff, 0, n_bins)

  # Histogram log values.
  pylab.figure(0)
  positive_values = values[values > 0]
  negative_values = -values[values < 0]
  n, bins, patches = pylab.hist(positive_values, bins=log_bins,
                                normed=False, histtype='stepfilled',
                                lw=0, color='g')
  n, bins, patches = pylab.hist(negative_values, bins=log_bins,
                                normed=False, histtype='step', lw=2, color='b')
  pylab.xscale('log')
  pylab.xticks(size=axis_size)
  pylab.yticks(size=axis_size)
  pylab.xlabel(r'$|\textrm{%s}|$' % value_type, fontsize=font_size)
  pylab.ylabel(r'Number of terms', fontsize=font_size)
  pylab.legend(['positive values', 'negative values'])
  pylab.tight_layout()
  pylab.savefig(name + '_log_histogram.pdf')
  pylab.show()

  # Histogram true values.
  pylab.figure(1)
  var = pylab.var(values)
  n, bins, patches = pylab.hist(values, bins=n_bins, range=(-var, var),
                                log=False, normed=False, histtype='stepfilled')
  pylab.xticks(size=axis_size)
  pylab.yticks(size=axis_size)
  pylab.ylim([0, 500])
  pylab.xlabel(r'%s' % value_type, fontsize=font_size)
  pylab.ylabel(r'Number of terms', fontsize=font_size)
  pylab.tight_layout()
  pylab.savefig(name + '_histogram.pdf')
  pylab.show()


# Function to plot orbital contributions.
def OrbitalContributions(values, terms, value_type, name):
  global axis_size
  global font_size

  # Initialize.
  n_orbitals = commutators.OrbitalCount(terms)
  orbitals = range(1, n_orbitals + 1)
  local_contributions = pylab.zeros(n_orbitals)
  single_marginal = pylab.zeros(n_orbitals)
  double_marginal = pylab.zeros((n_orbitals + 1, n_orbitals + 1))

  # Determine the magnitude contributed by each orbital.
  for magnitude, term in zip(pylab.absolute(values), terms):
    tensor_factors = set(pylab.absolute(term))

    # Compute marginal one and two-body contributions.
    for p in tensor_factors:
      single_marginal[p - 1] += magnitude
      double_marginal[p, p] += magnitude
      for q in tensor_factors:
        if p != q:
          double_marginal[p, q] += magnitude
          double_marginal[q, p] += magnitude

    # Compute local contributions
    if len(tensor_factors) == 1:
      local_contributions[term[0] - 1] = magnitude

  # Plot local contributions.
  pylab.figure(0)
  pylab.plot(orbitals, local_contributions, lw=2, marker='o')
  pylab.xlim([1, n_orbitals])
  pylab.xticks(size=axis_size)
  pylab.yticks(size=axis_size)
  pylab.xlabel(r'Orbital number', fontsize=font_size)
  pylab.ylabel(r'Local %s' % value_type,
               fontsize=font_size)
  pylab.tight_layout()
  pylab.savefig(name + '_local_orbitals.pdf')
  pylab.show()

  # Plot marginal contributions.
  pylab.figure(1)
  pylab.plot(orbitals, single_marginal, lw=2, marker='o')
  pylab.xlim([1, n_orbitals])
  pylab.xticks(size=axis_size)
  pylab.yticks(size=axis_size)
  pylab.xlabel(r'Orbital number', fontsize=font_size)
  pylab.ylabel(r'Marginal %s' % value_type,
               fontsize=font_size)
  pylab.tight_layout()
  pylab.savefig(name + '_single_orbitals.pdf')
  pylab.show()

  # Plot two-body marginal contributions.
  pylab.figure(2)
  pylab.imshow(double_marginal, interpolation='nearest')
  pylab.colorbar()
  pylab.xlim([.5, n_orbitals + .5])
  pylab.ylim([.5, n_orbitals + .5])
  pylab.xticks(size=axis_size)
  pylab.yticks(size=axis_size)
  pylab.xlabel(r'Orbital number', fontsize=font_size)
  pylab.ylabel(r'Orbital number', fontsize=font_size)
  pylab.title(r'$|\textrm{%s}|$' % value_type, fontsize=font_size)
  pylab.tight_layout()
  pylab.savefig(name + '_double_orbitals.pdf')
  pylab.show()


# Plot error expectation by state.
def AnalyzeSpectrum(molecule, basis):

  # Load data.
  hamiltonian_spectrum, wavefunctions = operators.GetEigensystem(
      molecule, basis, 'hamiltonian')
  error_spectrum = operators.GetEigensystem(molecule, basis, 'error')[0]
  prefix = 'data/figures/%s_%s_' % (molecule, basis)
  hamiltonian_norm = pylab.amax(pylab.absolute(hamiltonian_spectrum))
  hamiltonian_sigma = pylab.std(hamiltonian_spectrum)
  error_norm = pylab.amax(pylab.absolute(error_spectrum))
  error_sigma = pylab.std(error_spectrum)
  print '\nNorm of Hamiltonian is %s and standard deviation is %s.'\
      % (repr(hamiltonian_norm), repr(hamiltonian_sigma))
  print 'Norm of error operator is %s and standard deviation is %s.'\
      % (repr(error_norm), repr(error_sigma))

  # Histrogram error spectrum.
  factor = 1.5
  pylab.figure(0)
  n_bins = error_spectrum.size
  n, bins, patches = pylab.hist(error_spectrum, bins=n_bins,
                                normed=False, histtype='stepfilled', color='k')
  pylab.xticks(size= factor * axis_size)
  pylab.yticks(size= factor * axis_size)
  pylab.xlabel('Error operator eigenvalues', fontsize= factor * font_size)
  pylab.ylabel('Number of states', fontsize=factor * font_size)
  pylab.tight_layout()
  axis_max = abs(max(error_spectrum))
  pylab.xlim([-axis_max, axis_max])
  pylab.ylim([0, pylab.amax(n)])
  pylab.savefig(prefix + '_error_spectrum.pdf')
  pylab.show()


# Get expectations of random and Hamiltonian states.
def AnalyzeExpectations(molecule, basis, n_unitaries=5):

  # Get data.
  hamiltonian_spectrum, wavefunctions = operators.GetEigensystem(
      molecule, basis, 'hamiltonian')
  wavefunctions = pylab.matrix(wavefunctions)
  error_operator = operators.GetOperator(molecule, basis, 'error')
  name = 'data/figures/%s_%s_expectations.pdf' % (molecule, basis)

  # Get random expectations.
  random_expectations = []
  for i in range(n_unitaries):
    random_expectations += list(operators.Expectations(molecule, basis, i))
  random_expectations = pylab.array(random_expectations)
  sigma_random = pylab.std(random_expectations)
  print 'Standard deviation of random expectations is %s.' % repr(sigma_random)

  # Get Hamiltonian expectations.
  n_hilbert = hamiltonian_spectrum.size
  hamiltonian_expectations = pylab.zeros(n_hilbert)
  for i, wavefunction in enumerate(wavefunctions):
    hamiltonian_expectations[i] = operators.Expectation(
        error_operator, wavefunction.getH())
  sigma_ham = pylab.std(hamiltonian_expectations)
  print 'Standard deviation of Hamiltonian expectations is %s.' % repr(sigma_ham)

  # Histogram error expectations.
  pylab.figure(0)
  n_bins = n_hilbert // 10
  n, bins, patches = pylab.hist(random_expectations, bins=n_bins,
                                normed=True, histtype='step',
                                lw=2, color='b')
  n, bins, patches = pylab.hist(hamiltonian_expectations, bins=n_bins,
                                normed=True, histtype='step', lw=2, color='r')
  pylab.xticks(size=axis_size)
  pylab.yticks(size=axis_size)
  pylab.xlabel('Expected error', fontsize=font_size)
  pylab.ylabel('State propability', fontsize=font_size)
  pylab.legend(['Haar random vectors', 'Hamiltonian eigenvectors'])
  pylab.tight_layout()
  pylab.savefig(name)
  pylab.show()


# Show basis set progression for molecular hydrogen.
def BasisProgression(basis):

  # Specify basis sets.
  spatial_sets = ['STO6G', '321G', '631G', '631++G']
  spatial_sizes = [4, 8, 8, 12]
  if 0:
    spatial_sets += ['631Gss', '631G++Gss', '6311++Gss']
    spatial_sizes += [20, 24, 28]
  orbital_sets = [spatial_set + '-' + basis for spatial_set in spatial_sets]

  # Load data for sets.
  molecule = 'H2'
  n_excitations = 'FCI'
  prefix = 'data/figures/%s_basis_progression' % basis
  n_sizes = len(orbital_sets)
  errors = pylab.zeros(n_sizes)
  energies = pylab.zeros(n_sizes)
  error_norms = pylab.zeros(n_sizes)
  for i, orbital_basis in enumerate(orbital_sets):
    error, energy, error_norm = operators.GetData(
        molecule, orbital_basis, n_excitations)
    errors[i] = error
    energies[i] = energy
    error_norms[i] = error_norm

  # Plot errors.
  pylab.figure(0)
  pylab.plot(spatial_sizes, errors, lw=0, marker='o')
  for i, orbital_basis in enumerate(spatial_sets):
    pylab.annotate(orbital_basis,
               xy=(spatial_sizes[i] + .5, errors[i]),
               xytext=(spatial_sizes[i] + .5, errors[i]),
               fontsize=axis_size)
  pylab.xticks(size=axis_size)
  pylab.yticks(size=axis_size)
  pylab.xlabel('Number of spin-orbitals', fontsize=font_size)
  pylab.ylabel('Error in ground state', fontsize=font_size)
  pylab.xlim([2, spatial_sizes[-1] + 5])
  pylab.tight_layout()
  pylab.savefig(prefix + '_errors.pdf')
  pylab.show()

  # Plot error norms.
  pylab.figure(1)
  pylab.plot(spatial_sizes, error_norms, lw=0, marker='o')
  for i, orbital_basis in enumerate(spatial_sets):
    pylab.annotate(orbital_basis,
               xy=(spatial_sizes[i] + .5, error_norms[i]),
               xytext=(spatial_sizes[i] + .5, error_norms[i]),
               fontsize=axis_size)
  pylab.xticks(size=axis_size)
  pylab.yticks(size=axis_size)
  pylab.xlabel('Number of spin-orbitals', fontsize=font_size)
  pylab.ylabel('Norm of error operator', fontsize=font_size)
  pylab.xlim([2, spatial_sizes[-1] + 5])
  pylab.ylim([.5 * pylab.amin(error_norms), 1.1 * pylab.amax(error_norms)])
  pylab.tight_layout()
  pylab.savefig(prefix + '_norms.pdf')
  pylab.show()

  # Plot ratio.
  pylab.figure(2)
  ratio = errors / error_norms
  pylab.plot(spatial_sizes, ratio, lw=0, marker='o')
  for i, orbital_basis in enumerate(spatial_sets):
    pylab.annotate(orbital_basis,
               xy=(spatial_sizes[i] + .5, ratio[i]),
               xytext=(spatial_sizes[i] + .5, ratio[i]),
               fontsize=axis_size)
  pylab.xticks(size=axis_size)
  pylab.yticks(size=axis_size)
  pylab.xlabel('Number of spin-orbitals', fontsize=font_size)
  pylab.ylabel('Ground state error / error norm', fontsize=font_size)
  pylab.xlim([2, spatial_sizes[-1] + 5])
  pylab.ylim([.5 * pylab.amin(ratio), 1.1 * pylab.amax(ratio)])
  pylab.tight_layout()
  pylab.savefig(prefix + '_ratio.pdf')
  pylab.show()


# Show basis set progression for molecular hydrogen.
def AllBasisProgression(molecule):

  # Specify basis sets.
  pylab.figure(0)
  spatial_sets = ['STO6G', '321G', '631G', '631++G']
  spatial_sizes = [4, 8, 8, 12]

  # Loop over molecules.
  for basis in ['OAO', 'CMO', 'NMO']:
    orbital_sets = [spatial_set + '-' + basis for spatial_set in spatial_sets]

    # Load data for sets.
    sizes = []
    ratios = []
    spatials = []
    n_excitations = 'FCI'
    prefix = 'data/figures/%s_basis_progression' % basis
    for i in range(len(orbital_sets)):
      try:
        error, energy, error_norm = operators.GetData(
            molecule, orbital_sets[i], n_excitations)
        spatials += [spatial_sets[i]]
        ratios += [abs(error / error_norm)]
        sizes += [spatial_sizes[i]]
        print molecule, basis, spatial_sets[i], ratios[i]
      except:
        pass

    # Plot ratio.
    sizes = pylab.array(sizes)
    ratios = pylab.array(ratios)
    pylab.plot(sizes, ratios, lw=1, marker='o')
    for i, orbital_basis in enumerate(spatials):
      pylab.annotate(orbital_basis,
                 xy=(sizes[i] + .5, ratios[i]),
                 xytext=(spatial_sizes[i] + .5, ratios[i]),
                 fontsize=axis_size)

  # Finish plot.
  pylab.legend(['local basis', 'canonical basis', 'natural basis'])
  pylab.xticks(size=axis_size)
  pylab.yticks(size=axis_size)
  pylab.xlabel('Number of spin-orbitals', fontsize=font_size)
  pylab.ylabel('Ground state error / error norm', fontsize=font_size)
  pylab.xlim([3.5, 15])
  pylab.ylim([0, .25])
  pylab.tight_layout()
  pylab.savefig('data/figures/basis_ratio_all.pdf')
  pylab.show()


# Main.
def main():

  # Parameters.
  molecule = str(argv[1])
  basis = str(argv[2])
  try:
    term_type = str(argv[3])
  except:
    term_type = 'exact'
  cutoff = 18

  # Plots.
  expectations = 0
  basis_prog = 0
  histogram = 0
  orbitals = 0
  spectrum = 1
  verbose = 0

  # Set data to FCI.
  if term_type == 'exact':
    value_type = 'Ground state contributions'
    fci_contributions, fci_terms = operators.Contributions(molecule, basis)
    values, terms = Truncate(fci_contributions, fci_terms, cutoff)

  # Set data to initial Hamiltonian data.
  elif term_type == 'initial':
    value_type = 'Hamiltonian coefficients'
    coefficients, terms = commutators.GetHamiltonianTerms(
        molecule, basis, verbose=False, add_conjugates=True)
    values = pylab.array(coefficients)

  # Set data to error term coefficients.
  elif term_type == 'coefficients':
    value_type = 'Error coefficients'
    coefficients, error_terms = commutators.GetErrorTerms(molecule, basis)
    values, terms = Truncate(coefficients, error_terms, cutoff)

  # Set name.
  name = 'data/figures/%s_%s_%s' % (molecule, basis, term_type)

  # Print out terms.
  if verbose:
    print '%s and terms are as follow:' % value_type
    for i in pylab.argsort(pylab.absolute(values)):
      print terms[i], repr(values[i])

  # Make histogram.
  if histogram:
    MakeHistogram(values, value_type, name, cutoff)

  # Plot orbital contributions.
  if orbitals:
    OrbitalContributions(values, terms, value_type, name)

  # Plot eigenspectra of Hamiltonian and error operator.
  if spectrum:
    AnalyzeSpectrum(molecule, basis)

  # Plot expected errors of Hamiltonian and Haar ensembles.
  if expectations:
    AnalyzeExpectations(molecule, basis)

  # Plot basis set progression for molecular hydrogen.
  if basis_prog:
    if 0:
      BasisProgression(basis)
    else:
      AllBasisProgression(molecule)


# Run.
if __name__ == '__main__':
  main()
