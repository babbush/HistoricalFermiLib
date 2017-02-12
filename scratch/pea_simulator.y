"""This file computes the energy estimated from Trotterization and PEA.
"""
from sys import argv
import commutators
import operators
import warnings
import scipy
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


# View sparsity.
def Spy(A):
  try:
    A = A.todense()
  except:
    pass
  A[A != 0] = 1
  A = pylab.real(A)
  pylab.matshow(A)
  pylab.show()


# Return the unitary corresponding to evolution under hamiltonian.
def Unitary(hamiltonian, time):
  exponent = scipy.sparse.csc_matrix(-1j * time * hamiltonian)
  assert operators.IsHermitian(exponent)
  with warnings.catch_warnings(record=False):
    warnings.simplefilter('ignore')
  unitary = scipy.sparse.linalg.expm(exponent)
  unitary.eliminate_zeros()
  return unitary.tocsr()


# Return the 2nd order Trotter operator.
def Trotterize(coefficients, terms, trotter_slices, time):
  
  # Initialize.
  n_orbitals = commutators.OrbitalCount(terms)
  jw_terms = operators.GetJordanWignerTerms(n_orbitals)
  delta_t = time / trotter_slices
  duration = delta_t / 2.
  n_terms = len(coefficients)
  one_percent = pylab.rint(pylab.ceil(n_terms / 100.))

  # Assemble single Trotter series.
  circuit = 1.
  start = commutators.time.clock()
  for i, (coefficient, term) in enumerate(zip(coefficients, terms)):
    hamiltonian = operators.MatrixForm(coefficient, term, jw_terms, True)
    gate = Unitary(hamiltonian, duration)
    circuit = gate * circuit * gate

    # Report progress.
    if not (i + 1) % one_percent:
      percent_complete = pylab.rint(100. * (i + 1) / n_terms)
      elapsed = commutators.time.clock() - start
      rate = elapsed / percent_complete
      eta = rate * (100 - percent_complete)
      print('%s. Computation %i%% complete. Approximately %i '
            'minute(s) remaining.' % (commutators.time.strftime(
                '%B %d at %H:%M:%S', commutators.time.localtime()),
            percent_complete, round(eta / 60)))

  # Exponentiate circuit and return.
  circuit = circuit ** trotter_slices
  circuit.eliminate_zeros()
  return circuit


# Phase estimation assuming oracular ground state.
def EstimateEnergy(circuit, oracle, time):
  eigenvalue = operators.Expectation(circuit, oracle)
  phase = pylab.angle(eigenvalue)
  energy = - phase / time
  return energy


# Get error data via simulation.
def SimulatedTrotterError(molecule, basis, trotter_number, recompute=False):
  
  # Name and load data.
  name = 'data/trotter/%s_%s_%i.npy' % (molecule, basis, trotter_number)
  try:
    assert not recompute
    simulation_error = pylab.load(name)

  except:
    # Get Hamiltonian and optimal time-scale.
    coefficients, terms = commutators.GetHamiltonianTerms(
        molecule, basis, add_conjugates=False)
    energy, norm, wavefunction = operators.SparseDiagonalize(
        molecule, basis, 'hamiltonian')
    upper_bound = pylab.ceil(norm)
    lower_bound = pylab.floor(energy)
    time = 1. / (upper_bound - lower_bound)

    # Loop through data and compute simulated energy and return.
    circuit = Trotterize(coefficients, terms, trotter_number, time)
    simulation_energy = EstimateEnergy(circuit, wavefunction, time)
    simulation_error = simulation_energy - energy
    pylab.save(name, simulation_error)
  return simulation_error


# Plot Trotter error data.
def PlotTrotterErrors(molecule, basis, trotter_numbers, recompute=False):
  global marker_size
  global line_width
  global font_size
  
  # Initialize.
  n_points = trotter_numbers.size
  energy, norm, wavefunction = operators.SparseDiagonalize(
      molecule, basis, 'hamiltonian')
  upper_bound = pylab.ceil(norm)
  lower_bound = pylab.floor(energy)
  time = 1. / (upper_bound - lower_bound)
  delta_ts = float(time) / trotter_numbers
 
  # Get predicted errors.
  error_operator = operators.GetOperator(molecule, basis, 'error')
  ground_error = operators.Expectation(error_operator, wavefunction)
  predicted_errors = ground_error * pylab.square(delta_ts)

  # Loop through data and compute simulated errors.
  simulated_errors = pylab.zeros_like(predicted_errors)
  discrepancy = pylab.zeros_like(predicted_errors)
  for i, M in enumerate(trotter_numbers):
    print '\nComputing simulated Trotter error with Trotter number %i.' % M
    simulated_errors[i] = SimulatedTrotterError(molecule, basis, M, recompute)
    print 'Simulated error = %s. Predicted error = %s.'\
        % (repr(simulated_errors[i]), repr(predicted_errors[i]))
    discrepancy[i] = abs(simulated_errors[i] - predicted_errors[i])
    print 'discrepancy per time step squared is %s.'\
        % repr(discrepancy[i] / pylab.square(delta_ts[i]))

  # Plot errors.
  pylab.figure(0)
  pylab.plot(trotter_numbers, pylab.absolute(simulated_errors), 'o-',
      markersize=marker_size, lw=line_width)
  pylab.xticks(fontsize=font_size)
  pylab.yticks(fontsize=font_size)
  pylab.xscale('log')
  pylab.yscale('log')
  pylab.xlabel(r'Trotter number', fontsize=font_size)
  pylab.ylabel(r'Error in simulation (Hartree)', fontsize=font_size)
  pylab.xlim([trotter_numbers[0], trotter_numbers[-1]])
  pylab.tight_layout()
  name = 'data/figures/%s_%s_trotter_error.pdf' % (molecule, basis)
  pylab.savefig(name)
  pylab.show()

  # Plot discrepancy.
  pylab.figure(1)
  pylab.plot(trotter_numbers, discrepancy, 'o-',
      markersize=marker_size, lw=line_width)
  pylab.xticks(fontsize=font_size)
  pylab.yticks(fontsize=font_size)
  pylab.xscale('log')
  pylab.yscale('log')
  pylab.xlabel(r'Trotter number', fontsize=font_size)
  pylab.ylabel(r'discrepancy in error prediction', fontsize=font_size)
  pylab.xlim([trotter_numbers[0], trotter_numbers[-1]])
  pylab.tight_layout()
  name = 'data/figures/%s_%s_trotter_discrepancy.pdf' % (molecule, basis)
  pylab.savefig(name)
  pylab.show()


# Unit tests.
def main():
  
  # Test parameters.
  molecule = str(argv[1])
  basis = str(argv[2])
  trotter_numbers = 2 ** pylab.arange(8)
  recompute = False

  # Do it.
  PlotTrotterErrors(molecule, basis, trotter_numbers, recompute)


# Run.
if __name__ == '__main__':
  main()
