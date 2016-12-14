import fermi_hubbard
import fermion_tools
import unittest


class FermiHubbardTest(unittest.TestCase):


  def setUp(self):

    # Initialize a 2 by 2 spinful model.
    self.x_dimension = 2
    self.y_dimension = 2
    self.tunneling = 2.
    self.coulomb = 1.
    self.magnetic_field = 0.5
    self.chemical_potential = 0.25
    self.penalty = None
    self.verbose = 0
    self.periodic = 0
    self.spinless = 0
    self.n_electrons = 4
    self.tolerance = 1e-6


  def testCheckEnergyTwoByTwo(self):

    # Initialize the Hamiltonian.
    n_orbitals = self.x_dimension * self.y_dimension
    n_qubits = 2 * n_orbitals - n_orbitals * self.spinless
    hamiltonian = fermi_hubbard.FermiHubbardHamiltonian(
        self.x_dimension, self.y_dimension, self.tunneling, self.coulomb,
        self.chemical_potential, self.magnetic_field, self.penalty,
        self.periodic, self.spinless, self.verbose)

    # Compute ground state energy.
    projector = fermion_tools.ConfigurationProjector(n_qubits,
                                                     self.n_electrons)
    hamiltonian = projector * hamiltonian * projector.getH()
    energy, _ = fermion_tools.SparseDiagonalize(hamiltonian)
    self.assertAlmostEqual(energy, -9.27051036)


# Run test.
if __name__ == '__main__':
  unittest.main()
