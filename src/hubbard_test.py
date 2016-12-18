import unittest
import hubbard


class FermiHubbardTest(unittest.TestCase):

  def setUp(self):
    self.x_dimension = 2
    self.y_dimension = 2
    self.tunneling = 2.
    self.coulomb = 1.
    self.magnetic_field = 0.5
    self.chemical_potential = 0.25
    self.periodic = 0
    self.spinless = 0

  def test_two_by_two_spinful(self):

    # Initialize the Hamiltonian.
    hubbard_model = hubbard.fermi_hubbard(
        self.x_dimension, self.y_dimension, self.tunneling, self.coulomb,
        self.chemical_potential, self.magnetic_field,
        self.periodic, self.spinless)

    # Check up spin on site terms.
    self.assertAlmostEqual(hubbard_model([(0, 1), (0, 0)]), -.75)
    self.assertAlmostEqual(hubbard_model([(2, 1), (2, 0)]), -.75)
    self.assertAlmostEqual(hubbard_model([(4, 1), (4, 0)]), -.75)
    self.assertAlmostEqual(hubbard_model([(6, 1), (6, 0)]), -.75)

    # Check down spin on site terms.
    self.assertAlmostEqual(hubbard_model([(1, 1), (1, 0)]), .25)
    self.assertAlmostEqual(hubbard_model([(3, 1), (3, 0)]), .25)
    self.assertAlmostEqual(hubbard_model([(5, 1), (5, 0)]), .25)
    self.assertAlmostEqual(hubbard_model([(7, 1), (7, 0)]), .25)

    # Check up right/left hopping terms.
    self.assertAlmostEqual(hubbard_model([(0, 1), (2, 0)]), -2.)
    self.assertAlmostEqual(hubbard_model([(2, 1), (0, 0)]), -2.)
    self.assertAlmostEqual(hubbard_model([(4, 1), (6, 0)]), -2.)
    self.assertAlmostEqual(hubbard_model([(6, 1), (4, 0)]), -2.)

    # Check up top/bottom hopping terms.
    self.assertAlmostEqual(hubbard_model([(0, 1), (4, 0)]), -2.)
    self.assertAlmostEqual(hubbard_model([(4, 1), (0, 0)]), -2.)
    self.assertAlmostEqual(hubbard_model([(2, 1), (6, 0)]), -2.)
    self.assertAlmostEqual(hubbard_model([(6, 1), (2, 0)]), -2.)

    # Check down right/left hopping terms.
    self.assertAlmostEqual(hubbard_model([(1, 1), (3, 0)]), -2.)
    self.assertAlmostEqual(hubbard_model([(3, 1), (1, 0)]), -2.)
    self.assertAlmostEqual(hubbard_model([(5, 1), (7, 0)]), -2.)
    self.assertAlmostEqual(hubbard_model([(7, 1), (5, 0)]), -2.)

    # Check down top/bottom hopping terms.
    self.assertAlmostEqual(hubbard_model([(1, 1), (5, 0)]), -2.)
    self.assertAlmostEqual(hubbard_model([(5, 1), (1, 0)]), -2.)
    self.assertAlmostEqual(hubbard_model([(3, 1), (7, 0)]), -2.)
    self.assertAlmostEqual(hubbard_model([(7, 1), (3, 0)]), -2.)

    # Check on site interaction term.
    self.assertAlmostEqual(hubbard_model([(0, 1), (0, 0), (1, 1), (1, 0)]), 1.)
    self.assertAlmostEqual(hubbard_model([(2, 1), (2, 0), (3, 1), (3, 0)]), 1.)
    self.assertAlmostEqual(hubbard_model([(4, 1), (4, 0), (5, 1), (5, 0)]), 1.)
    self.assertAlmostEqual(hubbard_model([(6, 1), (6, 0), (7, 1), (7, 0)]), 1.)


# Run test.
if __name__ == '__main__':
  unittest.main()
