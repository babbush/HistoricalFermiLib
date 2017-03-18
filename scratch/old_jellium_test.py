import unittest
import old_jellium as jellium


class JelliumTest(unittest.TestCase):

  def setUp(self):
    self.grid_length = 4

  def test_qubit_id(self):

    # Test in 1D with spin.
    input_coords = [-1, 0, 1, 2, 3, 4]
    tensor_factors_up = [7, 1, 3, 5, 7, 1]
    tensor_factors_down = [6, 0, 2, 4, 6, 0]
    test_output_up = [jellium.qubit_id(
        self.grid_length, i, 1) for i in input_coords]
    test_output_down = [jellium.qubit_id(
        self.grid_length, i, 0) for i in input_coords]
    self.assertEqual(test_output_up, tensor_factors_up)
    self.assertEqual(test_output_down, tensor_factors_down)

    # Test in 2D without spin.
    input_coords = [(0, 2), (2, 4)]
    tensor_factors = [8, 2]
    test_output = [jellium.qubit_id(
        self.grid_length, i) for i in input_coords]
    self.assertEqual(test_output, tensor_factors)

  def test_kinetic_operator(self):

    # Test in 1D with spin.
    spinless = False
    n_dimensions = 1
    n_qubits = 2 * self.grid_length ** n_dimensions
    operator = jellium.kinetic_operator(
        n_dimensions, self.grid_length, spinless)
    self.assertEqual(operator.n_qubits, n_qubits)
    for term in operator:
      self.assertEqual(len(term.operators), 2)
      raising, lowering = term.operators
      self.assertTrue(raising[1])
      self.assertFalse(lowering[1])
      self.assertEqual(raising[0] % 2, lowering[0] % 2)
      if raising[0] == lowering[0]:
        self.assertAlmostEqual(term.coefficient, 1.)
      else:
        self.assertAlmostEqual(term.coefficient, -.5)

    # Test in 2D without spin.
    spinless = True
    n_dimensions = 2
    n_qubits = self.grid_length ** n_dimensions
    operator = jellium.kinetic_operator(
        n_dimensions, self.grid_length, spinless)
    self.assertEqual(operator.n_qubits, n_qubits)

    # Test diagonal elements.
    for qubit in xrange(n_qubits):
      operators = [(qubit, 1), (qubit, 0)]
      self.assertAlmostEqual(operator[operators], 2.)

    # Test hopping at position 2.
    self.assertAlmostEqual(operator[[(1, 1), (2, 0)]], -.5)
    self.assertAlmostEqual(operator[[(2, 1), (1, 0)]], -.5)
    self.assertAlmostEqual(operator[[(3, 1), (2, 0)]], -.5)
    self.assertAlmostEqual(operator[[(2, 1), (3, 0)]], -.5)
    self.assertAlmostEqual(operator[[(14, 1), (2, 0)]], -.5)
    self.assertAlmostEqual(operator[[(2, 1), (14, 0)]], -.5)
    self.assertAlmostEqual(operator[[(6, 1), (2, 0)]], -.5)
    self.assertAlmostEqual(operator[[(2, 1), (6, 0)]], -.5)

    # Count terms.
    n_terms = ((2 * n_dimensions + 1) *
               (self.grid_length ** n_dimensions) *
               2 ** (not spinless))
    self.assertEqual(n_terms, len(operator))

  def test_coulomb_interaction(self):
    length_scale = 10.
    coordinates_a = [0, 0, 0]
    coordinates_b = [0, 0, -1]
    expected_coupling = 1. / 10.
    coulomb_coupling = jellium.coulomb_interaction(
        coordinates_a, coordinates_b, length_scale)
    self.assertAlmostEqual(expected_coupling, coulomb_coupling)

  def test_potential_operator(self):

    # Test in 1D without spin.
    spinless = True
    n_dimensions = 1
    length_scale = 1.
    grid_length = self.grid_length
    operator = jellium.potential_operator(
        n_dimensions, grid_length, length_scale, spinless)
    for i in range(grid_length):
      for j in range(grid_length):
        if i != j:
          operators = [(i, 1), (i, 0), (j, 1), (j, 0)]
          coupling = 1. / abs(float(i - j))
          self.assertAlmostEqual(operator[operators], coupling)


# Run test.
if __name__ == '__main__':
  unittest.main()
