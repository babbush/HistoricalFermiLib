import sparse_operators
import qubit_operators
import itertools
import unittest
import jellium
import numpy


class JelliumTest(unittest.TestCase):

  def test_orbital_id(self):

    # Test in 1D with spin.
    grid_length = 5
    input_coords = [0, 1, 2, 3, 4]
    tensor_factors_up = [1, 3, 5, 7, 9]
    tensor_factors_down = [0, 2, 4, 6, 8]

    test_output_up = [jellium.orbital_id(
        grid_length, i, 1) for i in input_coords]
    test_output_down = [jellium.orbital_id(
        grid_length, i, 0) for i in input_coords]

    self.assertEqual(test_output_up, tensor_factors_up)
    self.assertEqual(test_output_down, tensor_factors_down)

    with self.assertRaises(jellium.OrbitalSpecificationError):
      jellium.orbital_id(5, 6, 1)

    # Test in 2D without spin.
    grid_length = 3
    input_coords = [(0, 0), (0, 1), (1, 2)]
    tensor_factors = [0, 3, 7]
    test_output = [jellium.orbital_id(
        grid_length, i) for i in input_coords]
    self.assertEqual(test_output, tensor_factors)

  def test_position_vector(self):

    # Test in 1D.
    grid_length = 4
    length_scale = 4.
    test_output = [jellium.position_vector(i, grid_length, length_scale) for
                   i in range(grid_length)]
    correct_output = [-1.5, -.5, .5, 1.5]
    self.assertEqual(correct_output, test_output)

    grid_length = 11
    length_scale = 2. * numpy.pi
    for i in range(grid_length):
      self.assertAlmostEqual(
          -jellium.position_vector(i, grid_length, length_scale),
          jellium.position_vector(
              grid_length - i - 1, grid_length, length_scale))

    # Test in 2D.
    grid_length = 3
    length_scale = 3.
    test_input = []
    test_output = []
    for i in range(3):
      for j in range(3):
        test_input += [(i, j)]
        test_output += [jellium.position_vector(
            (i, j), grid_length, length_scale)]
    correct_output = numpy.array([[-1., -1.], [-1., 0.], [-1., 1.],
                                 [0., -1.], [0., 0.], [0., 1.],
                                 [1., -1.], [1., 0.], [1., 1.]])
    self.assertAlmostEqual(0., numpy.amax(test_output - correct_output))

  def test_momentum_vector(self):
    grid_length = 3
    length_scale = 2. * numpy.pi
    test_output = [jellium.momentum_vector(i, grid_length, length_scale) for
                   i in range(grid_length)]
    correct_output = [-1., 0, 1.]
    self.assertEqual(correct_output, test_output)

    grid_length = 11
    length_scale = 2. * numpy.pi
    for i in range(grid_length):
      self.assertAlmostEqual(
          -jellium.momentum_vector(i, grid_length, length_scale),
          jellium.momentum_vector(
              grid_length - i - 1, grid_length, length_scale))

    # Test in 2D.
    grid_length = 3
    length_scale = 2. * numpy.pi
    test_input = []
    test_output = []
    for i in range(3):
      for j in range(3):
        test_input += [(i, j)]
        test_output += [jellium.momentum_vector(
            (i, j), grid_length, length_scale)]
    correct_output = numpy.array([[-1, -1], [-1, 0], [-1, 1],
                                 [0, -1], [0, 0], [0, 1],
                                 [1, -1], [1, 0], [1, 1]])
    self.assertAlmostEqual(0., numpy.amax(test_output - correct_output))

  def test_kinetic_integration(self):

    # Compute kinetic energy operator in both momentum and position space.
    n_dimensions = 2
    grid_length = 2
    length_scale = 3.
    spinless = 0
    momentum_kinetic = jellium.momentum_kinetic_operator(
        n_dimensions, grid_length, length_scale, spinless)
    position_kinetic = jellium.position_kinetic_operator(
        n_dimensions, grid_length, length_scale, spinless)

    # Diagonalize.
    sparse_momentum = (momentum_kinetic.jordan_wigner_transform()).\
        get_sparse_operator()
    sparse_position = (position_kinetic.jordan_wigner_transform()).\
        get_sparse_operator()
    momentum_spectrum = sparse_momentum.eigenspectrum()
    position_spectrum = sparse_position.eigenspectrum()

    # Confirm spectra are the same.
    difference = numpy.amax(
        numpy.absolute(momentum_spectrum - position_spectrum))
    self.assertAlmostEqual(difference, 0.)

  def test_potential_integration(self):

    # Compute potential energy operator in both momentum and position space.
    n_dimensions = 2
    grid_length = 3
    length_scale = 2.
    spinless = 1
    momentum_potential = jellium.momentum_potential_operator(
        n_dimensions, grid_length, length_scale, spinless)
    position_potential = jellium.position_potential_operator(
        n_dimensions, grid_length, length_scale, spinless)

    # Diagonalize and confirm the same energy.
    sparse_momentum = (momentum_potential.jordan_wigner_transform()).\
        get_sparse_operator()
    sparse_position = (position_potential.jordan_wigner_transform()).\
        get_sparse_operator()
    momentum_spectrum = sparse_momentum.eigenspectrum()
    position_spectrum = sparse_position.eigenspectrum()

    # Confirm spectra are the same.
    difference = numpy.amax(
        numpy.absolute(momentum_spectrum - position_spectrum))
    self.assertAlmostEqual(difference, 0.)

  def test_jellium_model_integration(self):

    # Compute jellium Hamiltonian in both momentum and position space.
    n_dimensions = 2
    grid_length = 3
    length_scale = 1.
    spinless = 1
    momentum_hamiltonian = jellium.jellium_model(
        n_dimensions, grid_length, length_scale, spinless, 1)
    position_hamiltonian = jellium.jellium_model(
        n_dimensions, grid_length, length_scale, spinless, 0)

    # Diagonalize and confirm the same energy.
    sparse_momentum = (momentum_hamiltonian.jordan_wigner_transform()).\
        get_sparse_operator()
    sparse_position = (position_hamiltonian.jordan_wigner_transform()).\
        get_sparse_operator()
    momentum_spectrum = sparse_momentum.eigenspectrum()
    position_spectrum = sparse_position.eigenspectrum()

    # Confirm spectra are the same.
    difference = numpy.amax(
        numpy.absolute(momentum_spectrum - position_spectrum))
    self.assertAlmostEqual(difference, 0.)

  def test_coefficients(self):

    # Test that the coefficients post-JW transform are as claimed in paper.
    n_dimensions = 2
    grid_length = 3
    length_scale = 2.
    spinless = 1
    n_orbitals = grid_length ** n_dimensions
    n_qubits = (2 ** (1 - spinless)) * n_orbitals
    volume = length_scale ** n_dimensions

    # Kinetic operator.
    kinetic = jellium.position_kinetic_operator(
        n_dimensions, grid_length, length_scale, spinless)
    qubit_kinetic = kinetic.jordan_wigner_transform()

    # Potential operator.
    potential = jellium.position_potential_operator(
        n_dimensions, grid_length, length_scale, spinless)
    qubit_potential = potential.jordan_wigner_transform()

    # Total.
    qubit_jellium = qubit_kinetic + qubit_potential

    # Check identity.
    identity = qubit_operators.qubit_identity()
    kinetic_coefficient = qubit_kinetic[identity]
    potential_coefficient = qubit_potential[identity]

    paper_kinetic_coefficient = 0.
    paper_potential_coefficient = 0.
    for indices in itertools.product(range(grid_length),
                                     repeat=n_dimensions):
      momenta = jellium.momentum_vector(
          indices, grid_length, length_scale)
      paper_kinetic_coefficient += float(
          n_qubits) * momenta.dot(momenta) / float(4. * n_orbitals)

      if momenta.any():
        potential_contribution = -numpy.pi * float(n_qubits) / float(
            2. * momenta.dot(momenta) * volume)
        paper_potential_coefficient += potential_contribution

    self.assertAlmostEqual(
        kinetic_coefficient, paper_kinetic_coefficient)
    self.assertAlmostEqual(
        potential_coefficient, paper_potential_coefficient)

    # Check Zp.
    for p in range(n_qubits):
      zp = qubit_operators.QubitTerm([(p, 'Z')])
      kinetic_coefficient = qubit_kinetic[zp]
      potential_coefficient = qubit_potential[zp]

      paper_kinetic_coefficient = 0.
      paper_potential_coefficient = 0.
      for indices in itertools.product(range(grid_length),
                                       repeat=n_dimensions):
        momenta = jellium.momentum_vector(
            indices, grid_length, length_scale)
        paper_kinetic_coefficient -= momenta.dot(
            momenta) / float(4. * n_orbitals)

        if momenta.any():
          potential_contribution = numpy.pi / float(
              momenta.dot(momenta) * volume)
          paper_potential_coefficient += potential_contribution

      self.assertAlmostEqual(
          kinetic_coefficient, paper_kinetic_coefficient)
      self.assertAlmostEqual(
          potential_coefficient, paper_potential_coefficient)

    # Check Zp Zq.
    if spinless:
      spins = [None]
    else:
      spins = [0, 1]

    for indices_a in itertools.product(range(grid_length),
                                       repeat=n_dimensions):
      for indices_b in itertools.product(range(grid_length),
                                         repeat=n_dimensions):

        paper_kinetic_coefficient = 0.
        paper_potential_coefficient = 0.

        position_a = jellium.position_vector(
            indices_a, grid_length, length_scale)
        position_b = jellium.position_vector(
            indices_b, grid_length, length_scale)
        differences = position_b - position_a

        for spin_a in spins:
          for spin_b in spins:

            p = jellium.orbital_id(
                grid_length, indices_a, spin_a)
            q = jellium.orbital_id(
                grid_length, indices_b, spin_b)

            if p == q:
              continue

            zpzq = qubit_operators.QubitTerm([(p, 'Z'), (q, 'Z')])
            potential_coefficient = qubit_potential[zpzq]

            for indices_c in itertools.product(range(grid_length),
                                               repeat=n_dimensions):
              momenta = jellium.momentum_vector(
                  indices_c, grid_length, length_scale)

              if momenta.any():
                potential_contribution = numpy.pi * numpy.cos(
                    differences.dot(momenta)) / float(
                    momenta.dot(momenta) * volume)
                paper_potential_coefficient += potential_contribution

            self.assertAlmostEqual(
                potential_coefficient, paper_potential_coefficient)

  def test_jordan_wigner_position_jellium(self):

    # Parameters.
    n_dimensions = 2
    grid_length = 3
    length_scale = 1.
    spinless = 1

    # Compute fermionic jellium Hamiltonian.
    fermion_hamiltonian = jellium.jellium_model(
        n_dimensions, grid_length, length_scale, spinless, 0)
    qubit_hamiltonian = fermion_hamiltonian.jordan_wigner_transform()

    # Compute Jordan-Wigner jellium Hamiltonian.
    test_hamiltonian = jellium.jordan_wigner_position_jellium(
        n_dimensions, grid_length, length_scale, spinless)

    # Make sure Hamiltonians are the same.
    self.assertTrue(test_hamiltonian == qubit_hamiltonian)

    # Check number of terms.
    n_qubits = qubit_hamiltonian.n_qubits()
    if spinless:
      paper_n_terms = 1 - .5 * n_qubits + 1.5 * (n_qubits ** 2)
    else:
      paper_n_terms = 1 - .5 * n_qubits + n_qubits ** 2
    self.assertTrue(len(qubit_hamiltonian) <= paper_n_terms)


# Run test.
if __name__ == '__main__':
  unittest.main()
