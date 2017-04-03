"""This module constructs Hamiltonians for the uniform electron gas (jellium).
"""
import fermion_operators
import itertools
import numpy
import copy


# Global intra-spatial-orbital coupling constant.
_COUPLING_CONSTANT_2D = 1.486605
_COUPLING_CONSTANT_3D = 0.941156


class CouplingConstantError(Exception):
  pass


class QubitSpecificationError(Exception):
  pass


def qubit_id(grid_length, coordinates, spin=None):
  """Return the tensor factor of a qubit with given coordinates and spin.

  Args:
    grid_length: Int, the number of points in one dimension of the grid.
    coordinates: List or tuple of ints giving Cartesian coordinates of
        grid element. Note that periodic boundary conditions are enforced.
        Acceptable to provide an int (instead of tuple or list) for 1D case.
    spin: Boole, 0 means spin down and 1 means spin up.
        If None, assume spinless model.

  Returns:
    tensor_factor: The tensor factor associated with provided qubit label.

  Raises:
    QubitSpecificiationError: Invalid qubit coordinates provided.
  """
  # Initialize.
  tensor_factor = 0
  if isinstance(coordinates, int):
    coordinates = [coordinates]

  # Loop through dimensions of coordinate tuple.
  for dimension, coordinate in enumerate(coordinates):

    # Make sure coordinate is an integer in the correct bounds.
    if isinstance(coordinate, int) and (-1 <= coordinate <= grid_length):

      # Enforce periodic boundary conditions and update tensor_factors.
      coordinate %= grid_length
      tensor_factor += coordinate * (grid_length ** dimension)

    else:
      # Raise for invalid model.
      raise QubitSpecificationError('Invalid qubit coordinates provided.')

  # Account for spin and return.
  if spin is None:
    return tensor_factor
  else:
    tensor_factor *= 2
    tensor_factor += spin
    return tensor_factor


def kinetic_operator(n_dimensions, grid_length, spinless=False):
  """Return an instance of the FermionOperator class cooresponding to T.

  WARNING: This function is not the usual jellium kinetic operator.
  This is the discretized Laplacian with periodic boundary conditions.

  Args:
    n_dimensions: An int giving the number of dimensions for the model.
    grid_length: Int, the number of points in one dimension of the grid.
    spinless: Boole, whether to use the spinless model or not.

  Returns:
    operator: An instance of the FermionOperator class.
  """
  # Initialize.
  n_points = grid_length ** n_dimensions
  n_qubits = n_points * (2 ** (not spinless))
  n_neighbors = float(2 * n_dimensions)
  operator = fermion_operators.FermionOperator(n_qubits)
  if spinless:
    spins = [None]
  else:
    spins = [0, 1]

  # Loop once through all lattice sites.
  for coordinates in itertools.product(range(grid_length),
                                       repeat=n_dimensions):
    coordinates = list(coordinates)

    # Loop over spins.
    for spin in spins:

      # Add the diagonal element.
      central_qubit = qubit_id(grid_length, coordinates, spin)
      diagonal_element = fermion_operators.number_operator(
          n_qubits, central_qubit, n_neighbors / 2.)
      operator += diagonal_element

      # Loop over neighboring qubits.
      for dimension in range(n_dimensions):
        for direction in [-1, 1]:
          neighbor_coordinates = copy.deepcopy(coordinates)
          neighbor_coordinates[dimension] += direction
          neighbor_qubit = qubit_id(grid_length, neighbor_coordinates, spin)

          # Add transition to neighboring qubits.
          operators = [(neighbor_qubit, 1), (central_qubit, 0)]
          transition_element = fermion_operators.FermionTerm(
              n_qubits, -0.5, operators)
          operator += transition_element

  # Return.
  return operator


def coulomb_interaction(coordinates_a, coordinates_b, length_scale):
  """Compute the non-periodic Coulomb potential 1 / r_{ab}."""
  positions_a = length_scale * numpy.array(coordinates_a)
  positions_b = length_scale * numpy.array(coordinates_b)
  difference_vector = positions_a - positions_b
  displacement = numpy.sqrt(difference_vector.dot(difference_vector))
  coulomb_coupling = 1. / displacement
  return coulomb_coupling


def potential_operator(n_dimensions, grid_length, length_scale,
                       spinless=False, coupling_constant=None):
  """Return an instance of the FermionOperator class cooresponding to V.

  WARNING: This function is not the usual jellium potential operator.
  This is the potential operator without any periodic boundary conditions.

  Args:
    n_dimensions: An int giving the number of dimensions for the model.
    grid_length: Int, the number of points in one dimension of the grid.
    length_scale: Float, the real space length of a box dimension.
    spinless: Boole, whether to use the spinless model or not.
    coupling_constant: Float, the intra-orbital-coupling constant.
        If None, use globals defined at top of file for 2D and 3D.

  Returns:
    operator: An instance of the FermionOperator class.

  Raises:
    CouplingConstantError: Must specify coupling constant for model.
  """
  # Initialize.
  n_points = grid_length ** n_dimensions
  n_qubits = n_points * (2 ** (not spinless))
  n_neighbors = float(2 * n_dimensions)
  operator = fermion_operators.FermionOperator(n_qubits)
  if spinless:
    spins = [None]
  else:
    spins = [0, 1]

    # Set default coupling constant for 2D and 3D.
    if coupling_constant is None:
      if n_dimensions == 2:
        coupling_constant = _COUPLING_CONSTANT_2D
      elif n_dimensions == 3:
        coupling_constant = _COUPLING_CONSTANT_3D
      else:
        raise CouplingConstantError(
            'Must specify coupling constant for model in chosen dimension.')
    coupling_constant /= length_scale

  # Loop once through all lattice sites.
  for coordinates in itertools.product(range(grid_length),
                                       repeat=n_dimensions):
    coordinates = list(coordinates)

    # Add the intra-orbital-coupling.
    if not spinless:
      up_spin = qubit_id(grid_length, coordinates, 1)
      down_spin = qubit_id(grid_length, coordinates, 0)
      operators = [(up_spin, 1), (up_spin, 0), (down_spin, 1), (down_spin, 0)]
      intra_orbital_term = fermion_operators.FermionTerm(
          n_qubits, coupling_constant, operators)
      operator += intra_orbital_term

    # Loop over spins.
    for spin in spins:

      # Loop over all other qubits with same spin.
      for neighbor_coordinates in itertools.product(range(grid_length),
                                                    repeat=n_dimensions):

        # Skip coordinates if same as central qubit.
        neighbor_coordinates = list(neighbor_coordinates)
        if neighbor_coordinates == coordinates:
          continue

        # Compute interaction.
        coulomb_coupling = coulomb_interaction(
            coordinates, neighbor_coordinates, length_scale)

        # Loop over spins.
        for neighbor_spin in spins:

          # Identify interacting qubits.
          central_qubit = qubit_id(grid_length, coordinates, spin)
          neighbor_qubit = qubit_id(
              grid_length, neighbor_coordinates, neighbor_spin)

          # Add interaction term.
          operators = [(central_qubit, 1), (central_qubit, 0),
                       (neighbor_qubit, 1), (neighbor_qubit, 0)]
          coulomb_term = fermion_operators.FermionTerm(
              n_qubits, coulomb_coupling, operators)
          operator += coulomb_term

  # Return.
  return operator


def jellium_model(n_dimensions, grid_length,
                  length_scale, spinless=False):
  """Return instance of the FermionOperator class representing Hamiltonian.

  Args:
    n_dimensions: An int giving the number of dimensions for the model.
    grid_length: Int, the number of points in one dimension of the grid.
    length_scale: Float, the real space length of a box dimension.
    spinless: Boole, whether to use the spinless model or not.

  Returns:
    hamiltonian: An instance of the FermionOperator class.
  """
  hamiltonian = kinetic_operator(n_dimensions,
                                 grid_length,
                                 length_scale,
                                 spinless)
  hamiltonian += potential_operator(n_dimensions,
                                    grid_length,
                                    length_scale,
                                    spinless)
  return hamiltonian
