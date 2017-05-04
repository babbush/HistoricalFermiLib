from __future__ import absolute_import

import unittest
import numpy

import fermilib.ops as ops
import fermilib.utils._plane_wave_hamiltonian as plane_wave_hamiltonian 
import fermilib.utils._jellium as jellium
from fermilib.transforms import jordan_wigner, get_eigenspectrum


class PlaneWaveHamiltonianTest(unittest.TestCase):

    def test_fourier_transform_u_operator(self):
        n_dimensions = 1
        length_scale = 1
        grid_length = 3
        spinless_set = [True, False]
        nuclear_charges = numpy.empty((3))
        nuclear_charges[0] = 1
        nuclear_charges[1] = -3 
        nuclear_charges[2] = 2
        for spinless in spinless_set:
            h_plane_wave = plane_wave_hamiltonian.plane_wave_u_operator(
                    n_dimensions, grid_length, length_scale, nuclear_charges,
                    spinless)
            h_dual_basis = plane_wave_hamiltonian.dual_basis_u_operator(
                    n_dimensions, grid_length, length_scale, nuclear_charges,
                    spinless)
            h_plane_wave_t = jellium.fourier_transform(
                    h_plane_wave, n_dimensions, grid_length, length_scale,
                    spinless)
            assert ops.normal_ordered(h_plane_wave_t).isclose(
                    ops.normal_ordered(h_dual_basis))


# Run test.
if __name__ == '__main__':
    unittest.main()
