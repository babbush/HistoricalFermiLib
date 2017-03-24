"""Class and functions to store quantum chemistry data."""
from molecular_data import MolecularData
from pyscf import ci, cc, fci, mp
import pyscf
import numpy


def prepare_pyscf_molecule(molecule):
  """This function creates and saves a psi4 input file.

  Args:
    molecule: An instance of the MolecularData class.

  Returns:
    pyscf_molecule: A pyscf molecule instance.
  """
  pyscf_molecule = pyscf.gto.Mole()
  pyscf_molecule.atom = molecule.geometry
  pyscf_molecule.basis = molecule.basis
  pyscf_molecule.spin = molecule.multiplicity - 1
  pyscf_molecule.charge = molecule.charge
  pyscf_molecule.build()
  return pyscf_molecule


def compute_scf(pyscf_molecule):
  """Perform a Hartree-Fock calculation.

  Args:
    pyscf_molecule: A pyscf molecule instance.

  Returns:
    pyscf_scf: A PySCF "SCF" calculation object.
  """
  if pyscf_molecule.spin:
    pyscf_scf = pyscf.scf.ROHF(pyscf_molecule)
  else:
    pyscf_scf = pyscf.scf.RHF(pyscf_molecule)
  return pyscf_scf


def compute_integrals(pyscf_molecule, pyscf_scf):
  """Compute the 1-electron and 2-electron integrals.

  Args:
    pyscf_molecule: A pyscf molecule instance.
    pyscf_scf: A PySCF "SCF" calculation object.

  Returns:
    one_electron_integrals: An N by N array storing h_{pq}
    two_electron_integrals: An N by N by N by N array storing h_{pqrs}.
  """
  # Get one electrons integrals.
  n_orbitals = pyscf_molecule.nbas
  one_electron_compressed = reduce(numpy.dot, (pyscf_scf.mo_coeff.T,
                                               pyscf_scf.get_hcore(),
                                               pyscf_scf.mo_coeff))
  one_electron_integrals = one_electron_compressed.reshape(n_orbitals,
                                                           n_orbitals)

  # Get two electron integrals in compressed format.
  two_electron_compressed = pyscf.ao2mo.kernel(pyscf_molecule,
                                               pyscf_scf.mo_coeff)
  two_electron_integrals = numpy.empty((n_orbitals, n_orbitals,
                                        n_orbitals, n_orbitals))

  # Unpack symmetry.
  n_pairs = n_orbitals * (n_orbitals + 1) // 2
  if two_electron_compressed.ndim == 2:

    # Case of 4-fold symmetry.
    assert(two_electron_compressed.size == n_pairs ** 2)
    pq = 0
    for p in range(n_orbitals):
      for q in range(0, p + 1):
        rs = 0
        for r in range(0, n_orbitals):
          for s in range(0, r + 1):
            pqrs_value = two_electron_compressed[pq, rs]
            two_electron_integrals[p, q, r, s] = pqrs_value
            rs += 1
        pq += 1
  else:

    # Case of 8-fold symmetry.
    assert(two_electron_compressed.size == n_pairs * (n_pairs + 1) // 2)
    pq = 0
    pqrs = 0
    for p in range(n_orbitals):
      for q in range(0, p + 1):
        rs = 0
        for r in range(0, p + 1):
          for s in range(0, r + 1):
            if pq >= rs:
              pqrs_value = two_electron_compressed[pqrs]
              two_electron_integrals[p, q, r, s] = pqrs_value
              pqrs += 1
            rs += 1
        pq += 1

  # Return.
  return one_electron_integrals, two_electron_integrals


def run_pyscf(molecule,
              run_scf=True,
              run_mp2=False,
              run_cisd=False,
              run_ccsd=False,
              run_fci=False):
  """This function runs a Psi4 calculation.

  Args:
    molecule: An instance of the MolecularData class.
    run_scf: Optional boolean to run SCF calculation.
    run_cisd: Optional boolean to run CISD calculation.
    run_ccsd: Optional boolean to run CCSD calculation.
    run_fci: Optional boolean to FCI calculation.

  Returns:
    molecule: The updated MolecularData object.
  """
  # Prepare pyscf molecule.
  pyscf_molecule = prepare_pyscf_molecule(molecule)
  molecule.n_orbitals = pyscf_molecule.nbas
  molecule.n_qubits = 2 * molecule.n_orbitals
  molecule.nuclear_repulsion = pyscf.gto.energy_nuc(pyscf_molecule)

  # Run SCF.
  if run_scf or run_cisd or run_ccsd or run_fci:
    pyscf_scf = compute_scf(pyscf_molecule)
    molecule.hf_energy = pyscf_scf.kernel()
    one_body_integrals, two_body_integrals = compute_integrals(
        pyscf_molecule, pyscf_scf)
    molecule.one_body_integrals = one_body_integrals
    integrals_name = molecule.data_handle() + '_eri'
    numpy.save(integrals_name, two_body_integrals)

  # Run MP2.
  if run_mp2:
    pyscf_mp2 = pyscf.mp.MP2(pyscf_scf)
    molecule.mp2_energy = pyscf_mp2.kernel()

  # Run CISD.
  if run_cisd:
    pyscf_cisd = pyscf.ci.CISD(pyscf_scf)
    molecule.cisd_energy = pyscf_cisd.kernel()

  # Run CCSD.
  if run_ccsd:
    pyscf_ccsd = pyscf.cc.CCSD(pyscf_scf)
    molecule.ccsd_energy = pyscf_ccsd.kernel()

  # Run FCI.
  if run_fci:
    pyscf_fci = pyscf.fci.FCI(pyscf_molecule, pyscf_scf.mo_coeff)
    molecule.fci_energy = pyscf_fci.kernel()[0]

  # Return updated molecule instance.
  molecule.save()
  return molecule


# Test.
if __name__ == '__main__':

  # Molecule parameters.
  basis = 'sto-3g'
  multiplicity = 1
  geometry = [['H', (0, 0, 0.7414 * x)] for x in range(2)]
  description = 'scf_tests'

  # Calculation parameters.
  run_scf = 1
  run_mp2 = 1
  run_cisd = 1
  run_ccsd = 1
  run_fci = 1

  # Get molecule and run calculation.
  molecule = MolecularData(
      geometry, basis, multiplicity, description=description)
  if 1:
    molecule = run_pyscf(
        molecule, run_scf, run_mp2, run_cisd, run_ccsd, run_fci)
  else:
    import run_psi4
    molecule = run_psi4.run_psi4(
        molecule, run_scf, run_mp2, run_cisd, run_ccsd, run_fci)

  # Get molecular Hamiltonian.
  molecular_hamiltonian = molecule.get_molecular_hamiltonian()
  print molecular_hamiltonian

  # Get eigenspectrum.
  sparse_operator = molecular_hamiltonian.get_sparse_operator()
  print '\nEigenspectrum follows:'
  for eigenvalue in sparse_operator.eigenspectrum():
    print eigenvalue
