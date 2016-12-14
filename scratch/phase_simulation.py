from sys import argv
import numpy
import numpy.random
import commutators
import operators
import bayes


# Get angles and ancilla expectation values.
def GenerateData(molecule, basis, times, n_measurements, verbose=False):
  """Uses molecular Hamiltonian to simulate experimental data.

  Args:
    molecule: String giving molecule, e.g. "H2O".
    basis: String giving basis, either "OAO", "CMO" or "NMO".
    n_measurements: Int giving how many measurements per angle.
    times: A 1D array of floats giving the lengths of evolutions.
    verbose: Boole, whether to print eigenvalues or not.

  Returns:
    angles: The experimental angles.
    ancilla states: The number of times the ancilla are in 0 or 1.
  """
  # Get spectra.
  n_experiments = times.size
  n_electrons = commutators.ElectronCount(molecule)
  eigenvalues, eigenstates = operators.GetEigensystem(
      molecule, basis, 'hamiltonian')
  n_hilbert = eigenstates.shape[1]
  n_orbitals = int(numpy.rint(numpy.log2(n_hilbert)))
  tolerance = 1e-8

  # Get overlaps with Hartree-Fock.
  hf_state = operators.HartreeFockState(n_electrons, n_orbitals).todense()
  amplitudes = []
  phases = []
  for i, eigenstate in enumerate(eigenstates):
    assert abs(eigenstate * eigenstate.getH() - 1) < tolerance
    overlap = (eigenstate * hf_state)[0, 0]
    if abs(overlap) > tolerance:
      amplitudes += [overlap]
      phases += [eigenvalues[i]]
      if verbose:
        print amplitudes[-1], phases[-1]

  # Make sure the amplitudes add up.
  amplitudes = numpy.array(amplitudes)
  phases = numpy.array(phases)
  assert amplitudes.dot(amplitudes) - 1. < tolerance
  probabilities = amplitudes * amplitudes
  expectation_value = probabilities.dot(phases)
  if verbose:
    print 'expectation value is %f.\n' % expectation_value

  # Generate data.
  if 0:
    angles = 2. * numpy.pi * numpy.random.rand(n_experiments)
  else:
    sigma = 0.1
    angles = numpy.empty((n_experiments), float)
    for i, time in enumerate(times):
      if 0:
        guess = expectation_value
      else:
        guess = -1.83
        mu = (numpy.pi / 2.) - (time * guess % (2. * numpy.pi))
      angles[i] = mu + sigma * numpy.random.randn()
  sampler = bayes.GeneralSampler(phases, amplitudes)
  ancilla_states = numpy.zeros((n_experiments, 2), int)
  for i in range(n_experiments):
    for j in range(n_measurements):
      if sampler(times[i], angles[i]):
        ancilla_states[i, 1] += 1
      else:
        ancilla_states[i, 0] += 1
  return angles, ancilla_states


# Test.
def main():

  # Parameters.
  molecule = str(argv[1])
  basis = str(argv[2])
  n_measurements = int(argv[3])
  n_experiments = int(argv[4])
  n_times = 10
  verbose = 0
  seed = 8
  if seed:
    numpy.random.seed(seed)
  n_repeats = n_experiments // n_times
  times = numpy.repeat(numpy.linspace(1., 2., n_times), n_repeats)
  assert times.size == n_experiments

  # Print data.
  angles, ancilla_states = GenerateData(
    molecule, basis, times, n_measurements, verbose)
  if not verbose:
    for i in range(n_experiments):
      print times[i], angles[i], ancilla_states[i, 0], ancilla_states[i, 1]


# Run.
if __name__ == '__main__':
  main()
