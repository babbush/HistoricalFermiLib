from __future__ import absolute_import

import unittest
import numpy

import fermilib.ops as ops
import fermilib.utils._plane_wave_hamiltonian as plane_wave_hamiltonian
import fermilib.utils._jellium as jellium
from fermilib.transforms import jordan_wigner
from fermilib.utils import eigenspectrum


class PlaneWaveHamiltonianTest(unittest.TestCase):

    def test_fourier_transform(self):
        n_dimensions = 1
        length_scale = 1
        grid_length = 3
        spinless_set = [True, False]
        nuclear_charges = numpy.empty((3))
        nuclear_charges[0] = 1
        nuclear_charges[1] = -3
        nuclear_charges[2] = 2
        for spinless in spinless_set:
            h_plane_wave = plane_wave_hamiltonian.get_hamiltonian(
                n_dimensions, grid_length, length_scale, nuclear_charges,
                spinless, False)
            h_dual_basis = plane_wave_hamiltonian.get_hamiltonian(
                n_dimensions, grid_length, length_scale, nuclear_charges,
                spinless, True)
            h_plane_wave_t = jellium.fourier_transform(
                h_plane_wave, n_dimensions, grid_length, length_scale,
                spinless)
            assert ops.normal_ordered(h_plane_wave_t).isclose(
                ops.normal_ordered(h_dual_basis))

    def test_u_operator_integration(self):
        n_dimensions = 1
        length_scale = 1
        grid_length = 3
        spinless_set = [True, False]
        nuclear_charges = numpy.empty((3))
        nuclear_charges[0] = 1
        nuclear_charges[1] = -3
        nuclear_charges[2] = 2
        for spinless in spinless_set:
            u_plane_wave = plane_wave_hamiltonian.plane_wave_u_operator(
                n_dimensions, grid_length, length_scale, nuclear_charges,
                spinless)
            u_dual_basis = plane_wave_hamiltonian.dual_basis_u_operator(
                n_dimensions, grid_length, length_scale, nuclear_charges,
                spinless)
            jw_u_plane_wave = jordan_wigner(u_plane_wave)
            jw_u_dual_basis = jordan_wigner(u_dual_basis)
            u_plane_wave_spectrum = eigenspectrum(jw_u_plane_wave)
            u_dual_basis_spectrum = eigenspectrum(jw_u_dual_basis)

            diff = numpy.amax(numpy.absolute(
                u_plane_wave_spectrum - u_dual_basis_spectrum))
            self.assertAlmostEqual(diff, 0)

# Run test.
if __name__ == '__main__':
    unittest.main()
