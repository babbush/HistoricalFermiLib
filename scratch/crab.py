"""Class for simulating continuous evolutions on Gmon chip with CRaB scheme.

For details about CRaB scheme see arXiv:1103.0855.
"""
import itertools
import numpy
import pylab
import scipy
import scipy.sparse
import scipy.sparse.linalg
import fermi_hubbard

#import google3
#from google3.pyglib import gfile


# Class to form representation of circuit.
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


  # Initialization.
  def __init__(self, n_modes, n_steps, total_time,
               initial_state, hamiltonian, short_time=False):
    """This function initializes the GmonCircuit class.

    As an important part of the class initialization we instantiate self.terms
    which is a python list. The elements of self.terms represent the
    individually programmable parts of the Hamiltonian. One can recover the
    entire Hamiltonian by summing together the elements of self.terms,
    multiplied by the appropriate magnitudes of each term at given time.

    Args:
      n_modes: An integer indicating the number of Fourier modes per term.
      n_steps: An integer indicating the number of discrete time slices.
      total_time: A float indicating the total evolution time.
      initial_state: A scipy.sparse 'csc' matrix giving the initial state.
      hamiltonian: A scipy.sparse 'csc' matrix giving the target Hamiltonian.
      short_time: An optional boolean. A "True" value requests that the unitary
        is computed by repeated application of the short-time propagator. A
        "False" value uses matrix exponentiation.

    Returns: An instance of GmonCircuit.
    """
    # Initialize circuit object attributes.
    self.n_modes = n_modes
    self.n_steps = n_steps
    self.total_time = total_time
    self.short_time = short_time
    self.hamiltonian = scipy.sparse.csc_matrix(hamiltonian)
    self.initial_state = initial_state
    self.dt = self.total_time / self.n_steps
    self.n_hilbert = initial_state.shape[0]
    self.n_qubits = int(numpy.rint(numpy.log2(self.n_hilbert)))
    self.n_terms = 3 * self.n_qubits + (self.n_qubits - 1)
    self.n_parameters = self.n_modes * self.n_terms

    # Initialize Hamiltonian terms.
    self.terms = []
    for i in range(self.n_qubits):
      self.terms += [self.X(i)]
    for i in range(self.n_qubits):
      self.terms += [self.Y(i)]
    for i in range(self.n_qubits):
      self.terms += [self.Z(i)]
    for i in range(self.n_qubits - 1):
      self.terms += [self.XXYY(i, i + 1)]
    assert len(self.terms) == self.n_terms


  # Method to compute pulse sequence for single term.
  def GetTimeSeries(self, amplitudes):
    """This function makes a time series vector from Fourier amplitudes.

    Args: A 1D array of floats giving the coefficients of the harmonics.
    Returns: A 1D array of floats giving the corresponding time series.
    """
    # Initialize time vector.
    w0 = 2. * numpy.pi / self.total_time
    times = numpy.linspace(0., self.total_time, self.n_steps)
    time_series = numpy.zeros_like(times)

    # Loop through Fourier modes and sum together to recover time series.
    assert amplitudes.size == self.n_modes
    for n, amplitude in enumerate(amplitudes):
      time_series += amplitude * numpy.sin((n + 1) * w0 * times)
    return time_series


  # Method to compute pulses for all terms.
  def GetPulses(self, parameters):
    """Compute all the pulses for all parts of the Hamiltonian.

    We expect that parameters is a vector that sequentially stores the Fourier
    modes for all of the elements of self.terms.

    Args: A 1D array of floats giving the coefficients of the harmonics.
    Returns: A 2D array of floats giving the corresponding time series for each
    component. The dimensions of this array are (n_terms, n_steps).
    """
    # Compute pulses.
    assert parameters.size == self.n_parameters
    pulses = numpy.zeros((self.n_terms, self.n_steps), float)
    for i in range(self.n_terms):
      start = i * self.n_modes
      stop = (i + 1) * self.n_modes
      amplitudes = parameters[start:stop]
      pulses[i] = self.GetTimeSeries(amplitudes)

    # Plot pulses.
    if 1:
      line_width = 1
      font_size = 18
      pylab.figure(0)
      pylab.xlabel('Time', fontsize=font_size)
      pylab.ylabel('Amplitude', fontsize=font_size)
      times = numpy.linspace(0., self.total_time, self.n_steps)
      for i in range(self.n_terms):
        pylab.plot(times, pulses[i], lw=line_width)
      pylab.show()
    return pulses


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


  # Method to compute the unitary corresponding to the quantum evolution.
  def GetUnitary(self, parameters, short_time):
    """Compute a unitary matrix to perform evolution specified by time series.

    We expect that parameters is a vector that sequentially stores the Fourier
    modes for all of the elements of self.terms.

    Args:
      parameters: A 1D array of floats which specifies all the Fourier amplitudes.
      short_time: A "True" value requests that the unitary is computed by repeated
        application of the short-time propagator. A "False" value uses exponentiation.

    Returns: A scipy.sparse 'csc' unitary matrix.
    """
    pulses = self.GetPulses(parameters)
    unitary = scipy.sparse.eye(self.n_hilbert)
    for step in range(self.n_steps):

      # Get the Hamiltonian and the exponent of the instantaneous unitary.
      hamiltonian = 0
      for i in range(self.n_terms):
        hamiltonian = hamiltonian + pulses[i, step] * self.terms[i]
      assert self.IsHermitian(hamiltonian)
      exponent = - 1j * hamiltonian * self.dt

      # Use either the short-time propagator or compute the exponential.
      if short_time:
        unitary = unitary + exponent * unitary
      else:
        unitary = scipy.sparse.linalg.expm_multiply(exponent, unitary)
    return unitary


  # Evaluate expectation value of evolved state with Hamiltonian.
  def __call__(self, parameters):
    """Evaluate the objective value defined by the evolution and Hamiltonian.

    Args: A 1D array of floats which specifies all the Fourier amplitudes.
    Returns: A float giving the objective values (i.e. the energy).
    """
    # Get the unitary and evaluate its action.
    unitary = self.GetUnitary(parameters, self.short_time)
    final_state = unitary * self.initial_state

    # Compute the expectation value and make sure it is real.
    expectation = final_state.getH() * self.hamiltonian * final_state
    assert expectation.get_shape() == (1, 1)
    objective = expectation[0, 0]
    assert abs(numpy.imag(objective)) < 1e-12
    return numpy.real(objective)


# Tests.
if __name__ == '__main__':

  # Test parameters.
  import numpy.random
  import operators
  import pylab
  n_modes = 3
  n_steps = 100
  total_time = 10
  molecule = 'LiH'
  basis = 'CMO-6'
  n_orbitals = 6
  n_electrons = 3
  short_time = 0
  scale = 1e2
  seed = 3

  # Get Hamiltonian and initial state.
  hamiltonian = operators.GetOperator(molecule, basis, 'hamiltonian')
  initial_state = operators.HartreeFockState(n_electrons, n_orbitals)

  # Evaluate once or multiple times.
  numpy.random.seed(seed)
  if n_steps:
    objective = GmonCircuit(n_modes, n_steps, total_time,
                            initial_state, hamiltonian, short_time)
    initial_guess = numpy.random.randn(objective.n_parameters)
    initial_guess /= numpy.sum(numpy.absolute(initial_guess))
    initial_guess *= scale
    print 'With %i Trotter steps, energy is %f.'\
        % (n_steps, objective(initial_guess))

  else:
    # Test with different number of time steps.
    n_points = 10
    step_vector = map(int, numpy.logspace(1, 3.5, n_points))
    energies = numpy.zeros(n_points, float)
    for i, n_steps in enumerate(step_vector):

      # Initialize.
      if not i:
        objective = GmonCircuit(n_modes, n_steps, total_time,
                                initial_state, hamiltonian, short_time)
        initial_guess = numpy.random.randn(objective.n_parameters)
        initial_guess /= numpy.sum(numpy.absolute(initial_guess))
        initial_guess *= scale
      else:
        objective.n_steps = n_steps

      # Evaluate.
      energies[i] = objective(initial_guess)
      print 'With %i Trotter steps, energy is %f.' % (n_steps, energies[i])

    # Plot.
    line_width = 2
    font_size = 18
    pylab.figure(0)
    pylab.xlabel('Trotter Number', fontsize=font_size)
    pylab.ylabel('Energy', fontsize=font_size)
    pylab.xscale('log')
    pylab.plot(step_vector, energies, lw=line_width, marker='o')
    pylab.show()
