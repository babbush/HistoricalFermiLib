"""This module constructs Hamiltonians for the uniform electron gas
(jellium)."""
from __future__ import absolute_import

import itertools

import numpy

from fermilib.ops import FermionOperator
from projectqtemp.ops._qubit_operator import QubitOperator


# Exceptions.
class OrbitalSpecificationError(Exception):
    pass


def orbital_id(grid_length, grid_coordinates, spin=None):
    """Return the tensor factor of a orbital with given coordinates and spin.

    Args:
        grid_length: Int, the number of points in one dimension of the grid.
        grid_coordinates: List or tuple of ints giving coordinates of grid
            element. Acceptable to provide an int (instead of tuple or list)
            for 1D case.
        spin: Boole, 0 means spin down and 1 means spin up.
            If None, assume spinless model.

    Returns:
        tensor_factor: The tensor factor associated with provided orbital
        label.

    Raises:
        OrbitalSpecificiationError: Invalid orbital coordinates provided.
    """
    # Initialize.
    if isinstance(grid_coordinates, int):
        grid_coordinates = [grid_coordinates]

    # Loop through dimensions of coordinate tuple.
    tensor_factor = 0
    for dimension, grid_coordinate in enumerate(grid_coordinates):

        # Make sure coordinate is an integer in the correct bounds.
        if isinstance(grid_coordinate, int) and grid_coordinate < grid_length:
            tensor_factor += grid_coordinate * (grid_length ** dimension)

        else:
            # Raise for invalid model.
            raise OrbitalSpecificationError(
                'Invalid orbital coordinates provided.')

    # Account for spin and return.
    if spin is None:
        return tensor_factor
    else:
        tensor_factor *= 2
        tensor_factor += spin
        return tensor_factor


def grid_indices(qubit_id, n_dimensions, grid_length, spinless):
    """
    This function is the inverse of orbital_id.

    Args:
        qubit_id: The tensor factor to map to grid indices.
        n_dimensions: An int giving the number of dimensions for the model.
        grid_length (int): The number of points in one dimension of the grid.
        spinless (bool): Whether to use the spinless model or not.

    Returns:
        grid_indices: The location of the qubit on the grid.
    """
    # Remove spin degree of freedom.
    orbital_id = qubit_id
    if not spinless:
        if (orbital_id % 2):
            orbital_id -= 1
        orbital_id /= 2

    # Get grid indices.
    grid_indices = []
    for dimension in range(n_dimensions):
        remainder = orbital_id % (grid_length ** (dimension + 1))
        grid_index = remainder // (grid_length ** dimension)
        grid_indices += [grid_index]
    return grid_indices


def position_vector(position_indices, grid_length, length_scale):
    """
    Given grid point coordinate, return position vector with dimensions.

    Args:
        position_indices: List or tuple of integers giving grid point
            coordinate. Allowed values are ints in [0, grid_length).
        grid_length (int): The number of points in one dimension of the grid.
        length_scale (float): The real space length of a box dimension.

    Returns:
        position_vector: A numpy array giving the position vector with
        dimensions.

    Raises:
        orbitalSpecificationError: Position indices must be integers
            in [0, grid_length).

    """
    # Raise exceptions.
    if isinstance(position_indices, int):
        position_indices = [position_indices]
    if (not isinstance(grid_length, int) or
        max(position_indices) >= grid_length or
            min(position_indices) < 0.):
        raise orbitalSpecificationError(
            'Position indices must be integers in [0, grid_length).')

    # Compute position vector.
    shift = float(grid_length - 1) / 2.
    adjusted_vector = numpy.array(position_indices, float) - shift
    position_vector = length_scale * adjusted_vector / float(grid_length)
    return position_vector


def momentum_vector(momentum_indices, grid_length, length_scale):
    """
    Given grid point coordinate, return momentum vector with dimensions.

    Args:
        momentum_indices: List or tuple of integers giving momentum indices.
            Allowed values are ints in [0, grid_length).
        grid_length: Int, the number of points in one dimension of the grid.
        length_scale: Float, the real space length of a box dimension.

        Returns:
            momentum_vector: A numpy array giving the momentum vector with
                dimensions.

    Raises:
        OrbitalSpecificationError: Momentum indices must be integers
            in [0, grid_length).
    """
    # Raise exceptions.
    if isinstance(momentum_indices, int):
        momentum_indices = [momentum_indices]
    if (not isinstance(grid_length, int) or
        max(momentum_indices) >= grid_length or
            min(momentum_indices) < 0.):
        raise OrbitalSpecificationError(
            'Momentum indices must be integers in [0, grid_length).')

    # Compute momentum vector.
    shift = float(grid_length - 1) / 2.
    adjusted_vector = numpy.array(momentum_indices, float) - shift
    momentum_vector = 2. * numpy.pi * adjusted_vector / length_scale
    return momentum_vector


def momentum_kinetic_operator(n_dimensions, grid_length,
                              length_scale, spinless=False):
    """
    Return the kinetic energy operator in momentum second quantization.

    Args:
        n_dimensions: An int giving the number of dimensions for the model.
        grid_length: Int, the number of points in one dimension of the grid.
        length_scale: Float, the real space length of a box dimension.
        spinless: Bool, whether to use the spinless model or not.

    Returns:
        operator: An instance of the FermionOperator class.
    """
    # Initialize.
    n_points = grid_length ** n_dimensions
    operator = FermionOperator((), 0.0)
    if spinless:
        spins = [None]
    else:
        spins = [0, 1]

    # Loop once through all plane waves.
    for grid_indices in itertools.product(range(grid_length),
                                          repeat=n_dimensions):
        momenta = momentum_vector(grid_indices, grid_length, length_scale)
        coefficient = momenta.dot(momenta) / 2.

        # Loop over spins.
        for spin in spins:
            orbital = orbital_id(grid_length, grid_indices, spin)

            # Add interaction term.
            operators = ((orbital, 1), (orbital, 0))
            operator += FermionOperator(operators, coefficient)

    return operator


def momentum_potential_operator(n_dimensions, grid_length,
                                length_scale, spinless=False):
    """
    Return the potential operator in momentum second quantization.

    Args:
        n_dimensions: An int giving the number of dimensions for the model.
        grid_length: Int, the number of points in one dimension of the grid.
        length_scale: Float, the real space length of a box dimension.
        spinless: Boole, whether to use the spinless model or not.

    Returns:
        operator: An instance of the FermionOperator class.

    Raises:
        OrbitalSpecificationError: 'Must use an odd number of momentum modes.'
    """
    # Make sure number of orbitals is odd.
    if not (grid_length % 2):
        raise OrbitalSpecificationError(
            'Must use an odd number of momentum modes.')

    # Initialize.
    n_points = grid_length ** n_dimensions
    volume = length_scale ** float(n_dimensions)
    prefactor = 2. * numpy.pi / volume
    operator = FermionOperator((), 0.0)
    if spinless:
        spins = [None]
    else:
        spins = [0, 1]

    # Loop once through all plane waves.
    for grid_indices_a in itertools.product(range(grid_length),
                                            repeat=n_dimensions):
        for grid_indices_b in itertools.product(range(grid_length),
                                                repeat=n_dimensions):
            for omega_indices in itertools.product(range(grid_length),
                                                   repeat=n_dimensions):

                # Compute the shifted indices.
                shifted_omega_indices = [index - grid_length // 2 for
                                         index in omega_indices]
                shifted_indices_d = [
                    (grid_indices_a[i] - shifted_omega_indices[i]) %
                    grid_length for i in range(n_dimensions)]
                shifted_indices_c = [
                    (grid_indices_b[i] + shifted_omega_indices[i]) %
                    grid_length for i in range(n_dimensions)]

                # Skip omega indices that cannot conserve momentum.
                if (min(shifted_indices_d) < 0 or
                        max(shifted_indices_c) >= grid_length):
                    continue

                # Get the momenta vectors.
                momenta_a = momentum_vector(
                    grid_indices_a, grid_length, length_scale)
                momenta_b = momentum_vector(
                    grid_indices_b, grid_length, length_scale)
                omega_momenta = momentum_vector(
                    omega_indices, grid_length, length_scale)

                # Skip if omega momentum is zero.
                if not omega_momenta.any():
                    continue

                # Loop over spins.
                for spin_a in spins:
                    for spin_b in spins:

                        # Compute coefficient.
                        coefficient = prefactor / \
                            omega_momenta.dot(omega_momenta)

                        # Get orbitals.
                        orbital_a = orbital_id(
                            grid_length, grid_indices_a, spin_a)
                        orbital_b = orbital_id(
                            grid_length, grid_indices_b, spin_b)
                        orbital_c = orbital_id(
                            grid_length, shifted_indices_c, spin_b)
                        orbital_d = orbital_id(
                            grid_length, shifted_indices_d, spin_a)

                        # Add interaction term.
                        if (orbital_a != orbital_b) and \
                                (orbital_c != orbital_d):
                            operators = ((orbital_a, 1), (orbital_b, 1),
                                         (orbital_c, 0), (orbital_d, 0))
                            operator += FermionOperator(operators, coefficient)

    # Return.
    return operator


def position_kinetic_operator(n_dimensions, grid_length,
                              length_scale, spinless=False):
    """
    Return the kinetic operator in position space second quantization.

    Args:
        n_dimensions: An int giving the number of dimensions for the model.
        grid_length: Int, the number of points in one dimension of the grid.
        length_scale: Float, the real space length of a box dimension.
        spinless: Bool, whether to use the spinless model or not.

    Returns:
        operator: An instance of the FermionOperator class.
    """
    # Initialize.
    n_points = grid_length ** n_dimensions
    operator = FermionOperator((), 0.0)
    if spinless:
        spins = [None]
    else:
        spins = [0, 1]

    # Loop once through all lattice sites.
    for grid_indices_a in itertools.product(range(grid_length),
                                            repeat=n_dimensions):
        for grid_indices_b in itertools.product(range(grid_length),
                                                repeat=n_dimensions):
            coordinates_a = position_vector(
                grid_indices_a, grid_length, length_scale)
            coordinates_b = position_vector(
                grid_indices_b, grid_length, length_scale)
            differences = coordinates_b - coordinates_a

            # Compute coefficient.
            coefficient = 0.
            for momenta_indices in itertools.product(range(grid_length),
                                                     repeat=n_dimensions):
                momenta = momentum_vector(
                    momenta_indices, grid_length, length_scale)
                if momenta.any():
                    coefficient += (
                        numpy.cos(momenta.dot(differences)) *
                        momenta.dot(momenta) / (2. * float(n_points)))

            # Loop over spins and identify interacting orbitals.
            for spin in spins:
                orbital_a = orbital_id(grid_length, grid_indices_a, spin)
                orbital_b = orbital_id(grid_length, grid_indices_b, spin)

                # Add interaction term.
                operators = ((orbital_a, 1), (orbital_b, 0))
                operator += FermionOperator(operators, coefficient)

    # Return.
    return operator


def position_potential_operator(n_dimensions, grid_length,
                                length_scale, spinless=False):
    """
    Return the potential operator in position space second quantization.

    Args:
        n_dimensions: An int giving the number of dimensions for the model.
        grid_length: Int, the number of points in one dimension of the grid.
        length_scale: Float, the real space length of a box dimension.
        spinless: Boole, whether to use the spinless model or not.

    Returns:
        operator: An instance of the FermionOperator class.

    """
    # Initialize.
    n_points = grid_length ** n_dimensions
    volume = length_scale ** float(n_dimensions)
    prefactor = 2. * numpy.pi / volume
    operator = FermionOperator((), 0.0)
    if spinless:
        spins = [None]
    else:
        spins = [0, 1]

    # Loop once through all lattice sites.
    for grid_indices_a in itertools.product(range(grid_length),
                                            repeat=n_dimensions):
        for grid_indices_b in itertools.product(range(grid_length),
                                                repeat=n_dimensions):
            coordinates_a = position_vector(
                grid_indices_a, grid_length, length_scale)
            coordinates_b = position_vector(
                grid_indices_b, grid_length, length_scale)
            differences = coordinates_b - coordinates_a

            # Compute coefficient.
            coefficient = 0.
            for momenta_indices in itertools.product(range(grid_length),
                                                     repeat=n_dimensions):
                momenta = momentum_vector(
                    momenta_indices, grid_length, length_scale)
                if momenta.any():
                    coefficient += (
                        prefactor * numpy.cos(momenta.dot(differences)) /
                        momenta.dot(momenta))

            # Loop over spins and identify interacting orbitals.
            for spin_a in spins:
                for spin_b in spins:
                    orbital_a = orbital_id(grid_length, grid_indices_a, spin_a)
                    orbital_b = orbital_id(grid_length, grid_indices_b, spin_b)

                    # Add interaction term.
                    if orbital_a != orbital_b:
                        operators = ((orbital_a, 1), (orbital_a, 0),
                                     (orbital_b, 1), (orbital_b, 0))
                        operator += FermionOperator(operators, coefficient)

    return operator


def jellium_model(n_dimensions, grid_length, length_scale,
                  spinless=False, momentum_space=False):
    """
    Return jellium Hamiltonian as FermionOperator class.

    Args:
        n_dimensions: An int giving the number of dimensions for the model.
        grid_length: Int, the number of points in one dimension of the grid.
        length_scale: Float, the real space length of a box dimension.
        spinless: Bool, whether to use the spinless model or not.
        momentum_space: Boole, whether to return in momentum space (True)
            or position space (False).

    Returns:
        hamiltonian: An instance of the FermionOperator class.
    """
    if momentum_space:
        hamiltonian = momentum_kinetic_operator(n_dimensions,
                                                grid_length,
                                                length_scale,
                                                spinless)
        hamiltonian += momentum_potential_operator(n_dimensions,
                                                   grid_length,
                                                   length_scale,
                                                   spinless)
    else:
        hamiltonian = position_kinetic_operator(n_dimensions,
                                                grid_length,
                                                length_scale,
                                                spinless)
        hamiltonian += position_potential_operator(n_dimensions,
                                                   grid_length,
                                                   length_scale,
                                                   spinless)
    return hamiltonian


def jordan_wigner_position_jellium(n_dimensions, grid_length,
                                   length_scale, spinless=False):
    """
    Return the position space jellium Hamiltonian as QubitOperator.

    Args:
        n_dimensions: An int giving the number of dimensions for the model.
        grid_length: Int, the number of points in one dimension of the grid.
        length_scale: Float, the real space length of a box dimension.
        spinless: Bool, whether to use the spinless model or not.

    Returns:
        hamiltonian: An instance of the QubitOperator class.
    """
    # Initialize.
    n_orbitals = grid_length ** n_dimensions
    volume = length_scale ** float(n_dimensions)
    if spinless:
        spins = [None]
        n_qubits = n_orbitals
    else:
        spins = [0, 1]
        n_qubits = 2 * n_orbitals
    hamiltonian = QubitOperator((), 0.0)

    # Compute the identity coefficient.
    identity_coefficient = 0.
    for k_indices in itertools.product(range(grid_length),
                                       repeat=n_dimensions):
        momenta = momentum_vector(k_indices, grid_length, length_scale)
        if momenta.any():
            identity_coefficient += momenta.dot(momenta) / 2.
            identity_coefficient -= (numpy.pi * float(n_orbitals) /
                                     (momenta.dot(momenta) * volume))
    if spinless:
        identity_coefficient /= 2.

    # Add identity term.
    identity_term = identity_coefficient * QubitOperator()
    hamiltonian += identity_term

    # Compute coefficient of local Z terms.
    z_coefficient = 0.
    for k_indices in itertools.product(range(grid_length),
                                       repeat=n_dimensions):
        momenta = momentum_vector(k_indices, grid_length, length_scale)
        if momenta.any():
            z_coefficient += numpy.pi / (momenta.dot(momenta) * volume)
            z_coefficient -= momenta.dot(momenta) / (4. * float(n_orbitals))

    # Add local Z terms.
    for qubit in range(n_qubits):
        qubit_term = QubitOperator(((qubit, 'Z'),), z_coefficient)
        hamiltonian += qubit_term

    # Add ZZ terms.
    prefactor = numpy.pi / volume
    for p in range(n_qubits):
        for q in range(p + 1, n_qubits):

            # Get positions.
            index_p = grid_indices(p, n_dimensions, grid_length, spinless)
            index_q = grid_indices(q, n_dimensions, grid_length, spinless)
            position_p = position_vector(index_p, grid_length, length_scale)
            position_q = position_vector(index_q, grid_length, length_scale)
            differences = position_p - position_q

            # Loop through momenta.
            zpzq_coefficient = 0.
            for k_indices in itertools.product(range(grid_length),
                                               repeat=n_dimensions):
                momenta = momentum_vector(k_indices, grid_length, length_scale)
                if momenta.any():
                    zpzq_coefficient += prefactor * numpy.cos(
                        momenta.dot(differences)) / momenta.dot(momenta)

            # Add term.
            qubit_term = QubitOperator(((p, 'Z'), (q, 'Z')), zpzq_coefficient)
            hamiltonian += qubit_term

    # Add XZX + YZY terms.
    prefactor = .25 / float(n_orbitals)
    for p in range(n_qubits):
        for q in range(p + 1, n_qubits):
            if not spinless and (p + q) % 2:
                continue

            # Get positions.
            index_p = grid_indices(p, n_dimensions, grid_length, spinless)
            index_q = grid_indices(q, n_dimensions, grid_length, spinless)
            position_p = position_vector(index_p, grid_length, length_scale)
            position_q = position_vector(index_q, grid_length, length_scale)
            differences = position_p - position_q

            # Loop through momenta.
            term_coefficient = 0.
            for k_indices in itertools.product(range(grid_length),
                                               repeat=n_dimensions):
                momenta = momentum_vector(k_indices, grid_length, length_scale)
                if momenta.any():
                    term_coefficient += prefactor * momenta.dot(momenta) * \
                        numpy.cos(momenta.dot(differences))

            # Add term.
            z_string = tuple((i, 'Z') for i in range(p + 1, q))
            xzx_operators = ((p, 'X'),) + z_string + ((q, 'X'),)
            yzy_operators = ((p, 'Y'),) + z_string + ((q, 'Y'),)
            hamiltonian += QubitOperator(xzx_operators, term_coefficient)
            hamiltonian += QubitOperator(yzy_operators, term_coefficient)

    # Return Hamiltonian.
    return hamiltonian


def fourier_transform(jellium_model, n_dimensions, grid_length, length_scale,
                      spinless):
    """
    Apply Fourier tranform to change the jellium model in momentum space.
    c^\dagger_\nu = sqrt(1/N) \sum_m {a^\dagger_m exp[-i k_\nu r_m]}
    c_\nu = sqrt(1/N) \sum_m {a_m exp[i k_\nu r_m]}

    Args:
        jellium_model: The jellium model in momentum space.
        n_dimensions: An int giving the number of dimensions for the model.
        grid_length: Int, the number of points in one dimension of the grid.
        length_scale: Float, the real space length of a box dimension.
        spinless: Bool, whether to use the spinless model or not.

    Returns:
        hamiltonian: An instance of the FermionOperator class.
    """
    if spinless:
        spins = [None]
    else:
        spins = [0, 1]

    hamiltonian = None

    for term in jellium_model.terms:
        transformed_term = None
        for ladder_operator in term:
            momentum_indices = grid_indices(ladder_operator[0], n_dimensions,
                                            grid_length, spinless)
            momentum_vec = momentum_vector(momentum_indices, grid_length,
                                           length_scale)
            new_basis = None
            for position_indices in itertools.product(range(grid_length),
                                                      repeat=n_dimensions):
                position_vec = position_vector(position_indices, grid_length,
                                               length_scale)
                if spinless:
                    spin = None
                else:
                    spin = ladder_operator[0] % 2
                orbital = orbital_id(grid_length, position_indices, spin)
                exp_index = 1.0j * numpy.dot(momentum_vec, position_vec)
                if ladder_operator[1] == 1:
                    exp_index *= -1.0

                element = FermionOperator(((orbital, ladder_operator[1]),),
                                          numpy.exp(exp_index))
                if new_basis == None:
                    new_basis = element
                else:
                    new_basis += element

            new_basis *= numpy.sqrt(1.0/float(grid_length**n_dimensions))

            if transformed_term == None:
                transformed_term = new_basis
            else:
                transformed_term *= new_basis
        if transformed_term is None:
            continue

        transformed_term *= jellium_model.terms[term] # Coefficient.

        if hamiltonian == None:
            hamiltonian = transformed_term
        else:
            hamiltonian += transformed_term

    return hamiltonian
