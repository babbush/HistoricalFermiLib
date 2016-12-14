"""Class for simulating continuous evolutions on Gmon chip.

For details about CRaB scheme see arXiv:1103.0855.
"""
import sys
import numpy
import numpy.random
import pylab
import scipy
import scipy.sparse
import scipy.optimize
import scipy.sparse.linalg
import fermi_hubbard


# Class to form representation of circuit. Supports both 'bang-bang' and CRaB.
class GmonCircuit:

  # Method to represent single-qubit gate as operator on full register.
  def Expand(self, operator, tensor_factor):
    """This function expands a local operator to act on entire Hilbert space.

    Args:
      operator: This is a 2 by 2 sparse matrix in scipy.sparse 'csc' format.
      tensor_factor: This integer indicates the qubit index on which
        'operator' acts. The first qubit has 'tensor_factor' = 0.

    Returns: A scipy.sparse 'csc' matrix that acts on entire Hilbert space.
    """
    front = scipy.sparse.eye(2 ** tensor_factor, format='csc')
    back = scipy.sparse.eye(
        2 ** (self.n_qubits - tensor_factor - 1), format='csc')
    return scipy.sparse.kron(
        front, scipy.sparse.kron(operator, back, 'csc'), 'csc')

  # Initantiate a Pauli-X operator that acts on register.
  def X(self, tensor_factor):
    x = scipy.sparse.csc_matrix([[0, 1], [1, 0]], dtype=complex)
    return self.Expand(x, tensor_factor)

  # Initantiate a Pauli-Y operator that acts on register.
  def Y(self, tensor_factor):
    y = scipy.sparse.csc_matrix([[0, -1j], [1j, 0]], dtype=complex)
    return self.Expand(y, tensor_factor)

  # Initantiate a Pauli-Z operator that acts on register.
  def Z(self, tensor_factor):
    z = scipy.sparse.csc_matrix([[1, 0], [0, -1]], dtype=complex)
    return self.Expand(z, tensor_factor)

  # Instantiate Gmon two-qubit interaction (XX + YY) for full register.
  def XXYY(self, qubit_i, qubit_j):
    """This function represents an XX + YY interaction.

    Args:
      qubit_i: An integer specifying the tensor_factor of the first qubit.
      qubit_j: An integer specifying the tensor_factor of the second qubit.

    Returns: A scipy.sparse 'csc' matrix that acts on entire Hilbert space.
    """
    return self.X(qubit_i) * self.X(qubit_j) + self.Y(qubit_i) * self.Y(qubit_j)

  # Test if matrix is Hermitian.
  def IsHermitian(self, matrix):
    conjugate = matrix.getH()
    difference = matrix - conjugate
    if difference.nnz:
      discrepancy = max(map(abs, difference.data))
      if discrepancy > 1e-12:
        print 'Hermitian discrepancy = %s.' % repr(discrepancy)
        return False
    return True

  # Initialization.
  def __init__(self, total_time, initial_state, hamiltonian,
               n_steps=0, noise=1e-5, use_crab=1, use_xy=1, verbose=0):
    """This function initializes the GmonCircuit class.

    As an important part of the class initialization we instantiate self.terms
    which is a python list. The elements of self.terms represent the
    individually programmable parts of the Hamiltonian. One can recover the
    entire Hamiltonian by summing together the elements of self.terms,
    multiplied by the appropriate magnitudes of each term at given time.

    Args:
      total_time: A float indicating the total evolution time.
      initial_state: A scipy.sparse 'csc' matrix giving the initial state.
      hamiltonian: A scipy.sparse 'csc' matrix giving the target Hamiltonian.
      n_steps: The number of Trotter slices.
      noise: A float scaling the noise added to objective values.
      use_crab: If True, is CRaB. If False, use bang-bang.
      use_xy: An optional boolean to use local x and y terms.
      verbose: Option to print out energy each time function is called.

    Returns: An instance of GmonCircuit.
    """
    # Check initial state and Hamiltonian.
    tolerance = 1e-12
    normalization = (initial_state.getH() * initial_state)[0, 0]
    assert abs(numpy.real(1 - normalization)) < tolerance
    assert self.IsHermitian(hamiltonian)

    # Initialize circuit object attributes.
    self.total_time = total_time
    self.initial_state = scipy.sparse.csc_matrix(initial_state)
    self.hamiltonian = scipy.sparse.csc_matrix(hamiltonian)
    self.n_steps = n_steps
    self.noise = noise
    self.use_crab = use_crab
    self.verbose = verbose
    self.n_hilbert = hamiltonian.shape[0]
    self.n_qubits = int(numpy.rint(numpy.log2(self.n_hilbert)))
    self.n_calls = 0
    self.lowest_energy = numpy.inf

    # Initialize Hamiltonian terms.
    self.terms = []
    if use_xy:
      for i in range(self.n_qubits):
        self.terms += [self.X(i)]
      for i in range(self.n_qubits):
        self.terms += [self.Y(i)]
    for i in range(self.n_qubits):
      self.terms += [self.Z(i)]
    for i in range(self.n_qubits - 1):
      self.terms += [self.XXYY(i, i + 1)]
    self.n_terms = len(self.terms)

  # Method to compute the unitary corresponding to the quantum evolution.
  def GetUnitary(self, time_series):
    """Compute a unitary matrix to perform evolution specified by time series.

    Args: A 2D array of floats corresponding to the amplitude of each
        term in self.terms at each time step.

    Returns: A scipy.sparse 'csc' unitary matrix.
    """
    # Initialize.
    n_terms, n_steps = time_series.shape
    dt = self.total_time / float(n_steps)
    assert n_terms == self.n_terms
    unitary = scipy.sparse.eye(self.n_hilbert)

    # Plot pulses.
    if 0:
      line_width = 1
      font_size = 18
      pylab.figure(0)
      pylab.xlabel('Time', fontsize=font_size)
      pylab.ylabel('Amplitude', fontsize=font_size)
      times = numpy.linspace(0., self.total_time, n_steps)
      for term in range(self.n_terms):
        pylab.plot(times, time_series[term], lw=line_width)
      pylab.show()

    # Loop through time steps.
    for step in range(n_steps):

      # Get the Hamiltonian and the exponent of the instantaneous unitary.
      hamiltonian = scipy.sparse.csc_matrix((self.n_hilbert, self.n_hilbert))
      for term in range(n_terms):
        amplitude = time_series[term, step]
        if amplitude:
          hamiltonian = hamiltonian + amplitude * self.terms[term]
      assert self.IsHermitian(hamiltonian)
      exponent = - 1j * hamiltonian * dt
      gate = scipy.sparse.linalg.expm(exponent)
      unitary = gate * unitary
    return unitary

  # Evaluate expectation value of evolved state given unitary.
  def Objective(self, unitary):
    """Evaluate the objective value defined by the evolution and Hamiltonian.

    Args: A scipy.sparse 'csc' unitary matrix

    Returns: A float giving the objective values (i.e. the energy).
    """
    # Get the unitary and evaluate its action.
    tolerance = 1e-12
    final_state = unitary * self.initial_state
    normalization = (final_state.getH() * final_state)[0, 0]
    assert abs(numpy.real(1 - normalization)) < tolerance

    # Compute the expectation value and make sure it is real.
    expectation = final_state.getH() * self.hamiltonian * final_state
    assert expectation.get_shape() == (1, 1)
    objective = expectation[0, 0]
    assert abs(numpy.imag(objective)) < tolerance
    return numpy.real(objective)

  # Method to compute objective value under 'bang-bang' scheme.
  def BangBang(self, parameters):
    """This is called to simulate the bang-bang scheme and return objective.

    Args: The input parameters should be a python array of floats which
      indicate the start and stop times for the bangs of each pulse.
      The array should have dimension (n_terms, 2 * n_bangs) where n_bangs
      is the number of bangs. Each variable in the second dimension should
      be a positive float that indicates the duration between start/stop
      times. Thus, the sum of the second dimension should always be less
      than the total_time.

    Returns: A float representing the objective value.
    """
    # Initialize bang times and phases.
    bang_times = numpy.cumsum(numpy.absolute(parameters), 1)
    time_breaks = numpy.sort(bang_times.flatten())
    time_breaks = time_breaks[time_breaks <= self.total_time]
    n_steps = time_breaks.size - 1
    dt = self.total_time / float(n_steps)
    bang_durations = time_breaks[-n_steps:] - time_breaks[:n_steps]

    # Loop through times and determine the phase of each term.
    time = 0.
    time_series = numpy.zeros((self.n_terms, n_steps), float)
    for step, duration in enumerate(bang_durations):
      time += duration
      for term in range(self.n_terms):
        phase = numpy.searchsorted(bang_times[term], time) % 2
        time_series[term, step] = (-1.) ** phase * duration / dt

    # Compute the unitary, compute objective, return.
    unitary = self.GetUnitary(time_series)
    energy = self.Objective(unitary)
    return energy

  # Method to compute objective value under CRaB scheme.
  def Crab(self, parameters):
    """This is called to simulate the CRaB scheme and return objective.

    Args: A 2D array of floats that stores the Fourier coefficients
        for the elements of self.terms. Has dimensions of (n_terms, n_modes).

    Returns: A float giving the objective values (i.e. the energy).
    """
    # Initialize time vector.
    w0 = 2. * numpy.pi / self.total_time
    times = numpy.linspace(0., self.total_time, self.n_steps)
    time_series = numpy.zeros((self.n_terms, self.n_steps), float)
    n_modes = parameters.shape[1] // 2

    # Loop through Fourier modes and sum together to recover time series.
    for term in range(self.n_terms):
      for n in range(n_modes):
        n_a, a = 2 * n + 1, parameters[term, 2 * n]
        n_b, b = 2 * n, parameters[term, 2 * n + 1]
        time_series[term] += a * numpy.sin(n_a * w0 * times)
        time_series[term] += b * numpy.cos(n_b * w0 * times)

    # Get the unitary and energy.
    unitary = self.GetUnitary(time_series)
    energy = self.Objective(unitary)
    return energy

  # Call method to combute objective value.
  def __call__(self, parameters):

    # Get objective.
    if self.use_crab:
      objective = self.Crab(numpy.reshape(parameters, (self.n_terms, -1)))
    else:
      objective = self.BangBang(numpy.reshape(parameters, (self.n_terms, -1)))

    # Add noise.
    if self.noise:
      objective += self.noise * numpy.random.randn()

    # Report energy and return.
    if self.verbose:
      self.n_calls += 1
      if objective < self.lowest_energy:
        self.lowest_energy = objective
      print 'Query number %i. Current value is %f. Lowest value is %f'\
          % (self.n_calls, objective, self.lowest_energy)
    return objective


# Unit tests.
def main():

  # Fermi-Hubbard model parameters.
  x_dimension = 2
  y_dimension = 1
  t = 1.
  u = 2.
  epsilon = 0.
  penalty = 3.
  periodic = 1
  spinless = 0
  half_filling = 0

  # General variational scheme parameters.
  total_time = 1e0
  n_parameters = 1e1
  n_steps = 2 * n_parameters
  use_crab = 1
  use_xy = 1
  parameter_scale = 1e1
  perturbation_scale = 1e-1
  random_seed = 8

  # Optimization parameters.
  method = 'COBYLA'
  verbose = 1
  noise = 1e-6
  tolerance = 1e-4
  maxiter = 1e5

  # Initialize the Hamiltonian.
  n_orbitals = x_dimension * y_dimension
  n_qubits = 2 * n_orbitals - n_orbitals * spinless
  number_operator = fermi_hubbard.NumberOperator(n_qubits)
  hamiltonian = fermi_hubbard.FermiHubbardHamiltonian(x_dimension, y_dimension,
                                                      t, u, epsilon, penalty,
                                                      periodic, spinless)
  if half_filling:
    projector = fermi_hubbard.ConfigurationProjector(n_qubits, n_orbitals)
    hamiltonian = projector * hamiltonian * projector.getH()

  # Compute ground state and print energy.
  energy, ground_state = fermi_hubbard.SparseDiagonalize(hamiltonian)
  print '\nThe ground state energy is %f.' % energy
  n_electrons = fermi_hubbard.Expectation(number_operator, ground_state)
  print 'The ground state has %f electrons.' % n_electrons
  n_hilbert = ground_state.shape[0]

  # Compute initial state as random perturbation to ground state.
  numpy.random.seed(random_seed)
  if perturbation_scale:
    perturbation = scipy.sparse.csc_matrix(numpy.random.randn(n_hilbert)).getH()
    perturbation = perturbation_scale * perturbation
    initial_state = perturbation + ground_state
    normalization = numpy.sqrt((initial_state.getH() * initial_state)[0, 0])
    initial_state = initial_state / normalization
  else:
    # Use physically inspired initial state.
    simple_h = fermi_hubbard.FermiHubbardHamiltonian(x_dimension, y_dimension,
                                                      t, 0, epsilon, penalty,
                                                      periodic, spinless)
    projector = fermi_hubbard.ConfigurationProjector(n_qubits, n_orbitals)
    simple_h = projector * simple_h * projector.getH()
    initial_state = fermi_hubbard.SparseDiagonalize(simple_h)[1]
  initial_energy = fermi_hubbard.Expectation(hamiltonian, initial_state)
  print '\nThe reference state energy is %f.' % initial_energy
  initial_electrons = fermi_hubbard.Expectation(number_operator, initial_state)
  print 'The reference state has %f electrons.' % initial_electrons

  # Initialize class.
  system = GmonCircuit(total_time, initial_state, hamiltonian, n_steps,
                       noise, use_crab, use_xy, verbose)
  initial_guess = parameter_scale * numpy.random.randn(
    system.n_terms, n_parameters).flatten()

  # Run optimization or just readout initial guess.
  print '\nSystem has %i parameters.' % (system.n_terms * n_parameters)
  print 'Now printing energies during optimization:'
  solution = scipy.optimize.minimize(system, initial_guess,
                                     method=method, tol=tolerance,
                                     options={'maxiter':maxiter}).x


# Run.
if __name__ == '__main__':
  main()
