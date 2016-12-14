"""Class for simulating evolutions on Gmon chip.
"""
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

  # Initialization.
  def __init__(self, total_time, initial_state, hamiltonian):
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

    Returns: An instance of GmonCircuit.
    """
    # Initialize circuit object attributes.
    self.total_time = total_time
    self.initial_state = scipy.sparse.csc_matrix(initial_state)
    self.hamiltonian = scipy.sparse.csc_matrix(hamiltonian)
    self.n_hilbert = hamiltonian.shape[0]
    self.n_qubits = int(numpy.rint(numpy.log2(self.n_hilbert)))
    self.calls = 0

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
    self.n_terms = len(self.terms)

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
    final_state = unitary * self.initial_state

    # Compute the expectation value and make sure it is real.
    expectation = final_state.getH() * self.hamiltonian * final_state
    assert expectation.get_shape() == (1, 1)
    objective = expectation[0, 0]
    assert abs(numpy.imag(objective)) < 1e-12
    return numpy.real(objective)

  # Method to compute pulses for all terms.
  def GetPulses(self, parameters):
    """Compute all the pulses for all parts of the Hamiltonian.

    Args: A 2D array of floats that stores the Fourier coefficients
        for the elements of self.terms. Has dimensions of (n_terms, n_modes).

    Returns: A 2D array of floats giving the corresponding time series for each
      component. The dimensions of this array are (n_modes, n_steps).
    """
    # Initialize time vector and make n_steps = n_modes.
    n_terms, n_steps = parameters.shape
    w0 = 2. * numpy.pi / self.total_time
    times = numpy.linspace(0., self.total_time, n_steps)
    time_series = numpy.zeros((self.n_terms, n_steps), float)

    # Loop through Fourier modes and sum together to recover time series.
    for term in range(self.n_terms):
      for n, amplitude in enumerate(parameters[term]):
        time_series[term] += amplitude * numpy.sin((n + 1) * w0 * times)

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
    return time_series

  # Method to compute objective value under CRaB scheme.
  def __call__(self, parameters):
    """This is called to simulate the CRaB scheme and return objective.

    Args: A 2D array of floats that stores the Fourier coefficients
        for the elements of self.terms. Has dimensions of (n_terms, n_modes).

    Returns: A float giving the objective values (i.e. the energy).
    """
    n_variables = parameters.size
    n_modes = n_variables // self.n_terms
    variables = numpy.reshape(parameters, (self.n_terms, n_modes))
    #time_series = self.GetPulses(variables)
    unitary = self.GetUnitary(variables)
    energy = self.Objective(unitary)
    self.calls += 1
    if 1:
      print self.calls, energy
    return energy


# Unit tests.
def main():

  # Fermi-Hubbard model parameters.
  x_dimension = 2
  y_dimension = 1
  t = 1.
  u = 2.
  epsilon = 0.
  penalty = 0.
  periodic = True
  spinless = False
  half_filling = False

  # General variational scheme parameters.
  total_time = 1e0
  n_modes = 1e2
  energy_scale = 1e0
  perturbation_scale = 1e-1
  method = 'Nelder-Mead'
  random_seed = 1

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
  n_hilbert = ground_state.size

  # Compute initial state as random perturbation to ground state.
  numpy.random.seed(random_seed)
  perturbation = scipy.sparse.csc_matrix(numpy.random.randn(n_hilbert)).getH()
  perturbation = perturbation_scale * perturbation
  initial_state = perturbation + ground_state
  normalization = numpy.sqrt((initial_state.getH() * initial_state)[0, 0])
  initial_state = initial_state / normalization

  # Initialize.
  n_terms = 3 * n_qubits + (n_qubits - 1)
  initial_guess = energy_scale * numpy.random.randn(n_terms, n_modes)
  system = GmonCircuit(total_time, initial_state, hamiltonian)

  # Run optimization.
  if 0:
    print 'Now printing energies from random guesses:'
    for i in range(100):
      initial_guess = energy_scale * numpy.random.randn(n_terms, n_modes)
      system(initial_guess)

  else:
    print 'Now printing energies during optimization:'
    solution = scipy.optimize.minimize(
        system, initial_guess.flatten(), method=method).x


# Run.
if __name__ == '__main__':
  main()
