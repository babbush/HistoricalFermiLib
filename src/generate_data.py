"""This is a development-only script for generating data."""
import chemical_series
import molecular_data
import run_psi4


if __name__ == '__main__':

  # Set chemical parameters.
  basis = 'sto-3g'
  max_electrons = 10
  spacing = 0.7414
  periodic_table = 1

  # Select calculations.
  force_recompute = 1
  run_scf = 1
  run_mp2 = 1
  run_cisd = 1
  run_ccsd = 1
  run_fci = 1
  tolerate_error = 1

  # Generate data.
  for n_electrons in range(2, max_electrons + 1):

    # Initialize.
    if periodic_table:
      atomic_symbol = molecular_data._PERIODIC_TABLE[n_electrons]
      molecule = chemical_series.make_atom(atomic_symbol, basis)
    else:
      molecule = chemical_series.make_atomic_ring(n_electrons, spacing, basis)

    # To run or not to run.
    if not molecule.hf_energy:
      run_job = 1
    elif run_mp2 and not molecule.mp2_energy:
      run_job = 1
    elif run_cisd and not molecule.cisd_energy:
      run_job = 1
    elif run_ccsd and not molecule.ccsd_energy:
      run_job = 1
    elif run_fci and not molecule.fci_energy:
      run_job = 1
    else:
      run_job = force_recompute

    # Run.
    if run_job:
      molecule = run_psi4.run_psi4(molecule,
                                   run_scf=run_scf,
                                   run_mp2=run_mp2,
                                   run_cisd=run_cisd,
                                   run_ccsd=run_ccsd,
                                   run_fci=run_fci,
                                   tolerate_error=tolerate_error)
