"""This module constructs Hamiltonians in plan wave basis and plan wave dual
basis for 3D cubic structure."""
from __future__ import absolute_import

import itertools
import numpy

from fermilib.config import *
from fermilib.ops import FermionOperator
from fermilib.utils._jellium import orbital_id, grid_indices, position_vector, \
         momentum_vector, jellium_model
from projectqtemp.ops._qubit_operator import QubitOperator


#Exceptions.
class ValueError(Exception):
    pass


def dual_basis_u_operator(n_dimensions, grid_length, length_scale,
                          nuclear_charges, spinless):
    """
    Return the external potential operator in plane wave dual basis.

    Args:
        n_dimensions: An int giving the number of dimensions for the model.
        grid_length: Int, the number of points in one dimension of the grid.
        length_scale: Float, the real space length of a box dimension.
        nuclear_charges: 3D int array, the nuclear charges.
        spinless: Bool, whether to use the spinless model or not.

    Returns:
        operator: An instance of the FermionOperator class.
    """
    n_points = grid_length ** n_dimensions
    volume = length_scale ** float(n_dimensions)
    prefactor = -4.0 * numpy.pi / volume
    operator = None 
    if spinless:
        spins = [None]
    else:
        spins = [0, 1]

    for grid_indices_p in itertools.product(range(grid_length),
                                            repeat=n_dimensions):
        coordinate_p = position_vector(grid_indices_p, grid_length,
                                       length_scale)
        for grid_indices_j in itertools.product(range(grid_length),
                                                repeat=n_dimensions):
            coordinate_j = position_vector(grid_indices_j, grid_length,
                                           length_scale)
            for momenta_indices in itertools.product(range(grid_length),
                                                     repeat=n_dimensions):
                momenta = momentum_vector(momenta_indices, grid_length,
                                          length_scale)
                momenta_squred = momenta.dot(momenta)
                if momenta_squred < EQ_TOLERANCE:
                    continue
                exp_index = 1.0j * momenta.dot(coordinate_j - coordinate_p)
                coefficient = prefactor / momenta_squred * \
                        nuclear_charges[grid_indices_j] * numpy.exp(exp_index)

                for spin_p in spins:
                    orbital_p = orbital_id(
                            grid_length, grid_indices_p, spin_p)
                    operators = ((orbital_p, 1), (orbital_p, 0))
                    if operator is None:
                        operator = FermionOperator(operators, coefficient)
                    else:
                        operator += FermionOperator(operators, coefficient)

    return operator


def plane_wave_u_operator(n_dimensions, grid_length, length_scale,
                          nuclear_charges, spinless):
    """
    Return the external potential operator in plane wave basis.

    Args:
        n_dimensions: An int giving the number of dimensions for the model.
        grid_length: Int, the number of points in one dimension of the grid.
        length_scale: Float, the real space length of a box dimension.
        nuclear_charges: 3D int array, the nuclear charges.
        spinless: Bool, whether to use the spinless model or not.

    Returns:
        operator: An instance of the FermionOperator class.
    """
    n_points = grid_length ** n_dimensions
    volume = length_scale ** float(n_dimensions)
    prefactor = -4.0 * numpy.pi / volume
    operator = None 
    if spinless:
        spins = [None]
    else:
        spins = [0, 1]

    for grid_indices_p in itertools.product(range(grid_length),
                                            repeat=n_dimensions):
        momenta_p = momentum_vector(grid_indices_p, grid_length,
                                    length_scale)
        for grid_indices_q in itertools.product(range(grid_length),
                                                repeat=n_dimensions):
            if grid_indices_p == grid_indices_q:
                continue
            momenta_q = momentum_vector(grid_indices_q, grid_length,
                                        length_scale)
            momenta_p_q = momenta_p - momenta_q
            for grid_indices_j in itertools.product(range(grid_length),
                                                    repeat=n_dimensions):
                coordinate_j = position_vector(grid_indices_j, grid_length,
                                               length_scale)
                exp_index = 1.0j * momenta_p_q.dot(coordinate_j)
                coefficient = prefactor / momenta_p_q.dot(momenta_p_q) * \
                        nuclear_charges[grid_indices_j] * numpy.exp(exp_index)

                for spin_p in spins:
                    for spin_q in spins:
                        orbital_p = orbital_id(
                                grid_length, grid_indices_p, spin_p)
                        orbital_q = orbital_id(
                                grid_length, grid_indices_q, spin_q)
                        operators = ((orbital_p, 1), (orbital_q, 0))
                        if operator is None:
                            operator = FermionOperator(operators, coefficient)
                        else:
                            operator += FermionOperator(operators, coefficient)

    return operator


def get_hamiltonian(grid_length, length_scale, nuclear_charges, spinless=False,
                    use_dual_basis=True):
    """
    Returns Hamiltonian as FermionOperator class.

    Args:
        grid_length: Int, the number of points in one dimension of the grid.
        length_scale: Float, the real space length of a box dimension.
        nuclear_charges: 3D int array, the nuclear charges.
        spinless: Bool, whether to use the spinless model or not.
        use_dual_basis: Boole, whether to return in plane wave basis (False)
                or plane wave dual basis (True).

    Returns:
        hamiltonian: An instance of the FermionOperator class.
    """
    if nuclear_charges.shape != (grid_length, grid_length, grid_length):
        raise ValueError('Invalid nuclear charges array shape.')

    if use_dual_basis:
        return jellium_model(3, grid_length, length_scale, spinless, False) + \
            dual_basis_u_operator(3, grid_length, length_scale, nuclear_charges,
                                  spinless)
    else:
        return jellium_model(3, grid_length, length_scale, spinless, True) + \
            plane_wave_u_operator(3, grid_length, length_scale, nuclear_charges,
                                  spinless)
