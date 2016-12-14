"""This module constructions Hamiltonians for the Fermi-Hubbard model.

The idea is that some fermions move around on a grid and the energy of the model
depends on where the fermions are. In the standard Fermi-Hubbard model (which we
call the "spinful" model), there is room for an "up" fermion and a "down"
fermion at each site on the grid. Accordingly, the Hamiltonian is

H = - tunneling sum_{<i,j>} sum_sigma (a^dagger_{i, sigma} a_{j, sigma}
  + a^dagger_{j, sigma} a_{i, sigma})
  + coulomb sum_{i} a^dagger_{i, up} a_{i, up} a^dagger_{j, down} a_{j, down}
  + chemical_potential sum_i (a^dagger_{i, up} a_{i, up}
  + a^dagger_{i, down} a_{i, down})
  + magnetic_field sum_i (a^dagger_{i, up} a_{i, up}
  - a^dagger_{i, down} a_{i, down}).

There are N sites and 2*N spin-orbitals. The operators a^dagger_i and a_i are
fermionic creation and annihilation operators. One can transform these operators
to qubit operator using the Jordan-Wigner transformation:

a^dagger_j = 0.5 (X - i Y) prod_{k = 1}^{j - 1} Z_k
a_j = 0.5 (X + i Y) prod_{k = 1}^{j - 1} Z_k

The code also allows one to construct the spinless Fermi-Hubbard model,
H = - tunneling sum_{k=1}^{N-1} (a_k^dagger a_{k + 1} + a_{k+1}^dagger a_k)
  + coulomb sum_{k=1}^{N-1} a_k^dagger a_k a_{k+1}^dagger a_{k+1}
  + magnetic_field sum_{k=1}^N (-1)^k a_k^dagger a_k
  - chemical_potential sum_{k=1}^N a_k^dagger a_k.

These Hamiltonians live a square lattice which has dimensions of
x_dimension by y_dimension. They can have periodic boundary conditions or not.
"""

import numpy
import scipy
import scipy.sparse
import fermion_tools


# Function to return up-orbital index given orbital index.
def Up(index):
  return 2 * index - 1


# Function to return down-orbital index given orbital index.
def Down(index):
  return 2 * index


def FermiHubbardSymbolic(x_dimension, y_dimension, tunneling, coulomb,
                         chemical_potential=None, magnetic_field=None,
                         periodic=True, spinless=False, verbose=False):
  """Return symbolic representation of a Fermi-Hubbard Hamiltonian.

  Throughout this code ladder operators, i.e. creation and annilhlation
  operators, are represented by positive and negative integers, respectively.
  Usually, terms are stored as python lists; e.g. [2 1 -2 -3] means raising
  on tensor factor two, raising on tensor factor one, lowering on two, etc.

  Args:
    x_dimension: An integer giving the number of sites in width.
    y_dimension: An integer giving the number of sites in height.
    tunneling: A float giving the tunneling amplitude.
    coulomb: A float giving the attractive local interaction strength.
    chemical_potential: An optional float giving the potential of each site.
      Default value is None.
    magnetic_field: An optional float giving a magnetic field at each site.
      Default value is None.
    periodic: If True, add periodic boundary conditions.
    spinless: An optional Boolean. If False, each site has spin up orbitals and
      spin down orbitals. If True, return a spinless Fermi-Hubbard model.
    verbose: An optional Boolean. If True, print all second quantized terms.

  Returns:
    coefficients: A list of floats giving the coefficients of the terms.
    terms: A list of lists of ints giving the terms in normal form.
  """
  # Loop through sites and add terms.
  n_orbitals = x_dimension * y_dimension
  coefficients = []
  terms = []
  for orbital in xrange(1, n_orbitals + 1):

    # Add chemical potential and magnetic field terms.
    if chemical_potential and spinless:
      x_index = orbital % x_dimension
      y_index = (orbital - 1) // x_dimension
      sign = (-1.) ** (x_index + y_index)
      coefficients += [sign * chemical_potential]
      terms += [[orbital, -orbital]]
    if chemical_potential and not spinless:
      coefficients += [-1. * chemical_potential, -1. * chemical_potential]
      terms += [[Up(orbital), -Up(orbital)], [Down(orbital), -Down(orbital)]]
    if magnetic_field and not spinless:
      coefficients += [-1. * magnetic_field, magnetic_field]
      terms += [[Up(orbital), -Up(orbital)], [Down(orbital), -Down(orbital)]]

    # Add local pair interaction terms.
    if not spinless:
      coefficients += [coulomb]
      terms += [[Up(orbital), -Up(orbital), Down(orbital), -Down(orbital)]]

    # Index coupled orbitals.
    right_neighbor = orbital + 1
    bottom_neighbor = orbital + x_dimension

    # Account for periodic boundaries.
    if periodic:
      if (x_dimension > 2) and (orbital % x_dimension == 0):
        right_neighbor -= x_dimension
      if (y_dimension > 2) and (orbital + x_dimension > n_orbitals):
        bottom_neighbor -= x_dimension * y_dimension

    # Add transition to neighbor on right.
    if orbital % x_dimension or (periodic and x_dimension > 2):
      if spinless:
        coefficients += [coulomb, -tunneling, -tunneling]
        terms += [[orbital, -orbital, right_neighbor, -right_neighbor]]
        terms += [[orbital, -right_neighbor]]
        terms += [[right_neighbor, -orbital]]
      else:
        coefficients += [-tunneling, -tunneling, -tunneling, -tunneling]
        terms += [[Up(orbital), -Up(right_neighbor)]]
        terms += [[Up(right_neighbor), -Up(orbital)]]
        terms += [[Down(orbital), -Down(right_neighbor)]]
        terms += [[Down(right_neighbor), -Down(orbital)]]

    # Add transition to neighbor below.
    if orbital + x_dimension <= n_orbitals or (periodic and y_dimension > 2):
      if spinless:
        coefficients += [coulomb, -tunneling, -tunneling]
        terms += [[orbital, -orbital, bottom_neighbor, -bottom_neighbor]]
        terms += [[orbital, -bottom_neighbor]]
        terms += [[bottom_neighbor, -orbital]]
      else:
        coefficients += [-tunneling, -tunneling, -tunneling, -tunneling]
        terms += [[Up(orbital), -Up(bottom_neighbor)]]
        terms += [[Up(bottom_neighbor), -Up(orbital)]]
        terms += [[Down(orbital), -Down(bottom_neighbor)]]
        terms += [[Down(bottom_neighbor), -Down(orbital)]]

  # Print out all second quantized terms and return.
  if verbose:
    print "\nNow printing all second quantized terms and their coefficients:"
    for coefficient, term in zip(coefficients, terms):
      print coefficient, term
  return coefficients, terms


def FermiHubbardHamiltonian(x_dimension, y_dimension, tunneling, coulomb,
                            chemical_potential=None, magnetic_field=None,
                            penalty=None, periodic=True,
                            spinless=False, verbose=False):
  """Return a Fermi-Hubbard Hamiltonian in scipy.sparse 'csc' format.

  Args:
    x_dimension: An integer giving the number of sites in width.
    y_dimension: An integer giving the number of sites in height.
    tunneling: A float giving the tunneling amplitude.
    coulomb: A float giving the attractive local interaction strength.
    chemical_potential: An optional float for chemical potential of each site.
      Default value is None.
    magnetic_field: An optional float for magnetic field at each site.
      Default value is None.
    penalty: The penalty of a Lagragian multiplier which restricts state
        to support of the half-full particle manifold. Default value is None.
    periodic: If True, add periodic boundary conditions.
    spinless: An optional Boolean. If False, each site has spin up orbitals and
        spin down orbitals. If True, return a spinless Fermi-Hubbard model.
    verbose: An optional Boolean. If True, print all second quantized terms.

  Returns:
    The Hamiltonian matrix in scipy.sparse 'csc' format.
  """
  # Initialize Jordan-Wigner terms.
  n_orbitals = x_dimension * y_dimension
  n_qubits = 2 * n_orbitals - n_orbitals * spinless
  jw_terms = fermion_tools.GetJordanWignerTerms(n_qubits)
  coefficients, terms = FermiHubbardSymbolic(x_dimension, y_dimension,
                                             tunneling, coulomb,
                                             chemical_potential,
                                             magnetic_field, periodic,
                                             spinless, verbose)

  # Compute matrix form of each term and sum together.
  hamiltonian = 0
  for coefficient, term in zip(coefficients, terms):
    operator = fermion_tools.MatrixForm(coefficient, term, jw_terms)
    hamiltonian = hamiltonian + operator

  # Add penalty for not being half-full.
  if penalty:
    identity = scipy.sparse.identity(2 ** n_qubits, format="csc")
    operator = penalty * (n_orbitals * identity - fermion_tools.NumberOperator(
        n_qubits)) ** 2
    hamiltonian = hamiltonian + operator

  # Make sure its Hermitian and return.
  assert fermion_tools.IsHermitian(hamiltonian)
  return hamiltonian


def WriteFermiHubbard(x_dimension, y_dimension, tunneling, coulomb,
                      chemical_potential, magnetic_field, penalty,
                      periodic, spinless, verbose, n_electrons, output_name):
  """Return a Fermi-Hubbard Hamiltonian in a protocol buffer.

  Args:
    x_dimension: An integer giving the number of sites in width.
    y_dimension: An integer giving the number of sites in height.
    tunneling: A float giving the tunneling amplitude.
    coulomb: A float giving the attractive local interaction strength.
    chemical_potential: A float giving the chemical potential of each site.
    magnetic_field: A float giving a local magnetic field at each site.
    penalty: The penalty of a Lagragian multiplier which restricts ground state
        to the support of the half-full particle-number manifold.
    periodic: If True, add periodic boundary conditions.
    spinless: If False, each site has spin up orbitals and
        spin down orbitals. If True, return a spinless Fermi-Hubbard model.
    verbose: If True, print all second quantized terms.
    n_electrons: An int giving the number of electrons to project to.
    output_name: A float giving the name of the output file.
  """
  # Get the data.
  hamiltonian = FermiHubbardHamiltonian(x_dimension, y_dimension,
                                        tunneling, coulomb, chemical_potential,
                                        magnetic_field, penalty,
                                        periodic, spinless, verbose)
  n_orbitals = x_dimension * y_dimension
  n_qubits = 2 * n_orbitals - n_orbitals * spinless
  if n_electrons:
    projector = fermion_tools.ConfigurationProjector(n_qubits, n_electrons)
    hamiltonian = projector * hamiltonian * projector.getH()
  hamiltonian = hamiltonian.tocoo()
  values = hamiltonian.data
  row_indices = hamiltonian.row
  col_indices = hamiltonian.col
  dimension = 2 ** n_qubits

  # Put data in protocol buffer.
  hamiltonian_pb = sparsemat_pb2.SparseMat()
  hamiltonian_pb.size_x = dimension
  hamiltonian_pb.size_y = dimension
  for value, row_index, col_index in zip(values, row_indices, col_indices):
    hamiltonian_pb.value.append(float(numpy.real(value)))
    hamiltonian_pb.y.append(int(row_index))
    hamiltonian_pb.x.append(int(col_index))

  # Write protocol buffer.
  with gfile.GFile(output_name, "w") as f_out:
    f_out.write(text_format.MessageToString(hamiltonian_pb))


def SpinOperator(n_orbitals):
  """Operator to measure total angular momentum.

  Args:
    n_orbitals: An int giving the number of spin-orbitals in system.

  Returns:
    A scipy.sparse csc matrix operator.
  """
  n_qubits = 2 * n_orbitals
  up_operator = 0
  down_operator = 0
  jw_terms = fermion_tools.GetJordanWignerTerms(n_qubits)
  for orbital in range(1, n_orbitals + 1):
    up_term = [Up(orbital), -Up(orbital)]
    down_term = [Down(orbital), -Down(orbital)]
    up_operator = up_operator + fermion_tools.MatrixForm(
        0.5, up_term, jw_terms)
    down_operator = down_operator + fermion_tools.MatrixForm(
        0.5, down_term, jw_terms)
  return up_operator - down_operator


def DoubleOccupancyOperator(n_orbitals):
  """Operator to measure double occupancy.

  Args:
    n_orbitals: An int giving the number of spin-orbitals in system.

  Returns:
    A scipy.sparse csc matrix operator.
  """
  n_qubits = 2 * n_orbitals
  jw_terms = fermion_tools.GetJordanWignerTerms(n_qubits)
  operator = 0
  for orbital in range(1, n_orbitals + 1):
    term = [Up(orbital), -Up(orbital), Down(orbital), -Down(orbital)]
    operator = operator + fermion_tools.MatrixForm(1, term, jw_terms)
  return 2. * operator / float(n_orbitals)
