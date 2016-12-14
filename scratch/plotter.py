"""These functions compare properties of different molecules.
"""
from scipy.stats import linregress
from scipy.misc import factorial
from sys import argv
import commutators
import operators
import pylab
import re


# Set global plot parameters.
pylab.rcParams['text.usetex'] = True
pylab.rcParams['text.latex.unicode'] = True
pylab.rc('text', usetex=True)
pylab.rc('font', family='sans=serif')
marker_size = 6
line_width = 2
axis_size = 18
font_size = 24


# Indicate whether atom or molecule.
def IsAtom(molecule):
  atoms = ['H', 'He',
           'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
           'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar']
  if molecule in set(atoms):
    return True
  else:
    return False


# Return the nuclear charges in a molecule.
def NuclearCharges(molecule):
  if IsAtom(molecule):
    charges = pylab.array(commutators.ElectronCount(molecule))
  else:
    components = re.findall(r'[A-Z][a-z]\d?|[A-Z]\d?', molecule)
    atoms = []
    for component in components:
      if component[-1].isdigit():
        multiplicity = int(component[-1])
        atom = component[:(len(component) - 1)]
        atoms += multiplicity * [atom]
      else:
        atoms += [component]
    charges = pylab.array(map(commutators.ElectronCount, atoms))
  return charges


# Format molecule name.
def LatexName(molecule):
  name = ''
  for char in molecule:
    try:
      number = int(char)
      name += '$_%i$' % number
    except:
      name += char
  return name


# Plot molecules.
def PlotMolecules(molecules, basis):
  global line_width
  global axis_size
  global font_size

  # Initialize.
  n_excitations = None
  n_molecules = len(molecules)
  x_axis = pylab.zeros(n_molecules)
  y_axis = pylab.zeros(n_molecules)

  # Assign number of spin orbitals to each molecule and compure error.
  for i, molecule in enumerate(molecules):
    print '\nAnalyzing %s, molecule %i of %i.' % (molecule, i + 1, n_molecules)

    # Analyze nuclei.
    n_electrons = commutators.ElectronCount(molecule)
    charges = NuclearCharges(molecule)
    z_max = pylab.amax(charges)
    z_mean = pylab.mean(charges)
    z_rms = pylab.sqrt(charges.dot(charges)/charges.size)

    # Get Hamiltonian statistics.
    integrals, terms = commutators.GetHamiltonianTerms(molecule, basis)
    integrals = pylab.array(integrals)
    n_orbitals = commutators.OrbitalCount(terms)
    n_csfs = factorial(n_orbitals) / (factorial(
        n_electrons) * factorial(n_orbitals - n_electrons))
    integral_rms = pylab.sqrt(integrals.dot(integrals)/integrals.size)
    integral_max = pylab.amax(pylab.absolute(integrals))

    # Get error operator information.
    fci_error, fci_energy, error_norm = operators.GetData(
        molecule, basis, 'FCI')
    cancellation = abs(fci_error / error_norm)

    # Get triangle inequality information.
    if 0:
      error_coefficients = commutators.GetErrorTerms(molecule, basis)[0]
      triangle = pylab.sum(pylab.absolute(error_coefficients))

    # Get information in different basis.
    if 0:

      # Get Hartree-Fock information.
      if n_orbitals - n_electrons:
        hf_error, hf_energy, hf_overlap = operators.GetData(
            molecule, basis, 2)
        hf_discrepancy = abs(fci_error - hf_error)
        hf_reduction = abs(hf_discrepancy / fci_error)

      # Get CISD information.
      if n_orbitals - n_electrons > 2:
        cisd_error, cisd_energy, cisd_overlap = operators.GetData(
            molecule, basis, 2)
        cisd_discrepancy = abs(fci_error - cisd_error)
        cisd_reduction = abs(cisd_discrepancy / fci_error)

      # Get CISDT information.
      if 1:
        if n_orbitals - n_electrons > 3:
          cisdt_error, cisdt_energy, cisdt_overlap = operators.GetData(
              molecule, basis, 3)
          cisdt_discrepancy = abs(fci_error - cisdt_error)
          cisdt_reduction = abs(cisdt_discrepancy / fci_error)


    # Get spectra.
    if 0:
      hamiltonian_spectrum, wavefunctions = operators.GetEigensystem(
          molecule, basis, 'hamiltonian')
      hamiltonian_spread = pylab.std(hamiltonian_spectrum)
      error_spectrum, error_functions = operators.GetEigensystem(
          molecule, basis, 'error')
      error_spread = pylab.std(error_spectrum)

      # Get expectation values of Hamiltonian states with error operator.
      expectations = operators.Expectations(molecule, basis)
      expectation_spread = pylab.std(expectations)
      mean_expectation = pylab.mean(pylab.absolute(expectations))
      max_expectation = pylab.amax(pylab.absolute(expectations))

    # Set the x-axis here!
    if 0:
      x_type = 'z_rms'
      x_axis[i] = z_rms
      x_label = 'RMS nuclear charge'
      x_log = False
    if 0:
      x_type = 'z_mean'
      x_axis[i] = z_mean
      x_label = 'Mean nuclear charge'
      x_log = False
    if 0:
      x_type = 'z_max'
      x_axis[i] = z_max
      x_label = 'Max nuclear charge'
      x_log = True
    if 0:
      x_type = 'triangle'
      x_axis[i] = triangle
      x_label = 'Norm from triangle inequality'
      x_log = True
    if 0:
      x_type = 'hf_overlap'
      x_axis[i] = hf_overlap
      x_label = 'Overlap with reference state'
      x_log = False
    if 0:
      x_type = 'filling_ratio'
      x_axis[i] = float(n_electrons) / float(n_orbitals)
      x_label = 'Filling ratio'
      x_log = False
    if 0:
      x_type = 'csfs'
      x_axis[i] = n_csfs
      x_label = 'Number of CSFs'
      x_log = True
    if 0:
      x_type = 'cisd_error'
      x_axis[i] = abs(cisd_error)
      x_label = 'CISD error (Hartree / $\Delta_t^2$)'
      x_log = True
    if 0:
      x_type = 'hf_error'
      x_axis[i] = hf_error
      x_label = 'Hartree-Fock error (Hartree)'
      x_log = True
    if 0:
      x_type = 'integral_rms'
      x_axis[i] = integral_rms
      x_label = 'RMS value of integrals'
      x_log = False
    if 0:
      x_type = 'orbitals'
      x_axis[i] = n_orbitals
      x_label = 'Number of spin-orbitals'
      x_log = False
    if 0:
      x_type = 'electrons'
      x_axis[i] = n_electrons
      x_label = 'Number of electrons'
      x_log = False
    if 0:
      x_type = 'empty_orbitals'
      x_axis[i] = n_orbitals - n_electrons
      x_label = 'Number of empty orbitals'
      x_log = False
    if 0:
      x_type = 'cisd_overlap'
      x_axis[i] = cisd_overlap
      x_label = 'CISD overlap'
      x_log = True
    if 0:
      x_type = 'error_spread'
      x_axis[i] = error_spread
      x_label = 'Std. dev. of error eigenspectrum'
      x_log = True
    if 0:
      x_type = 'fci_error'
      x_axis[i] = abs(fci_error)
      x_label = r'Trotter error (Hartree / $\Delta_t^2$)'
      x_log = True
    if 1:
      x_type = 'error_norm'
      x_axis[i] = error_norm
      x_label = r'Norm of error operator (hartree / $\Delta_t^2$)'
      x_log = True
    if 0:
      x_type = 'error_norm'
      x_axis[i] = error_norm
      x_label = r'Norm of error operator'
      x_log = True
    if 0:
      x_type = 'composite'
      x_axis[i] = n_electrons * z_max
      x_label = 'Composite metric'
      x_log = True
    if 0:
      x_type = 'hf_reduction'
      x_axis[i] = hf_reduction
      x_label = 'Fraction of remaining error'
      x_log = True

    # Set the y-axis here!
    if 0:
      y_type = 'mean_expectation'
      y_axis[i] = mean_expectation
      y_label = 'Mean magnitude of error'
      y_log = True
    if 0:
      y_type = 'max_expectation'
      y_axis[i] = max_expectation
      y_label = 'Max expected error'
      y_log = True
    if 0:
      y_type = 'expectation_spread'
      y_axis[i] = expectation_spread
      y_label = 'Std. dev. of expected error'
      y_log = True
    if 0:
      y_type = 'error_spread'
      y_axis[i] = error_spread
      y_label = 'Std. dev. of error eigenspectrum'
      y_log = True
    if 0:
      y_type = 'csfs'
      y_axis[i] = n_csfs
      y_label = 'Number of CSFs'
      y_log = True
    if 0:
      y_type = 'hf_overlap'
      y_axis[i] = hf_overlap
      y_label = 'Overlap with reference state'
      y_log = False
    if 0:
      y_type = 'integral_rms'
      y_axis[i] = integral_rms
      y_label = 'RMS value of integrals'
      y_log = False
    if 0:
      y_type = 'cisd_overlap'
      y_axis[i] = cisd_overlap
      y_label = 'CISD overlap'
      y_log = True
    if 0:
      y_type = 'csfs'
      y_axis[i] = n_csfs
      y_label = 'Number of CSFs'
      y_log = True
    if 1:
      y_type = 'fci_error'
      y_axis[i] = abs(fci_error)
      y_label = r'Trotter error (hartree / $\Delta_t^2$)'
      y_log = True
    if 0:
      y_type = 'fci_error'
      y_axis[i] = abs(fci_error)
      y_label = r'FCI error (Hartree / $\Delta_t^2$)'
      y_log = True
    if 0:
      y_type = 'fci_energy'
      y_axis[i] = fci_energy
      y_label = 'Exact energy (Hartree)'
      y_log = False
    if 0:
      y_type = 'cisd_discrepancy'
      y_axis[i] = cisd_discrepancy
      y_label = 'discrepancy in CISD error'
      y_log = True
    if 0:
      y_type = 'cisd_reduction'
      y_axis[i] = cisd_reduction
      y_label = 'Fraction of remaining error'
      y_log = True
    if 0:
      y_type = 'cisdt_reduction'
      y_axis[i] = cisdt_reduction
      y_label = 'Fraction of remaining error'
      y_log = True
    if 0:
      y_type = 'hf_reduction'
      y_axis[i] = hf_reduction
      y_label = 'Fraction of remaining error'
      y_log = True
    if 0:
      y_type = 'cancellation'
      y_axis[i] = cancellation
      y_label = 'Ground error / error norm'
      y_log = True
    if 0:
      y_type = 'error_norm'
      y_axis[i] = error_norm
      y_label = r'Norm of error operator'
      y_log = True
    if 0:
      y_type = 'triangle'
      y_axis[i] = triangle
      y_label = 'Triangle norm'
      y_log = True
    if 0:
      y_type = 'triangle_ratio'
      y_axis[i] = triangle / error_norm
      y_label = 'Triangle norm / exact norm'
      y_log = True

    # Remove molecules, if necessary.
    if bool(x_type[:2] == 'ci') ^ bool(y_type[:2] == 'ci'):
      if n_orbitals - n_electrons <= 2:
        y_axis[i] = pylab.nan

    # Print while we wait.
    print '%s equals %s.' % (x_label, repr(x_axis[i]))
    print '%s equals %s.' % (y_label, repr(y_axis[i]))

  # Plot.
  pylab.figure(0)
  molecules = pylab.array(molecules)[y_axis == y_axis]
  x_axis = x_axis[y_axis == y_axis]
  y_axis = y_axis[y_axis == y_axis]
  atoms = pylab.array(map(IsAtom, molecules))
  not_atoms = pylab.array(1 - atoms, bool)
  pylab.plot(x_axis[atoms], y_axis[atoms], lw=0, marker='o', color='r')
  pylab.plot(x_axis[not_atoms], y_axis[not_atoms], lw=0, marker='o', color='b')

  # Add nice looking annotations.
  for i, molecule in enumerate(molecules):
    if x_log:
      pylab.xscale('log')
      pylab.annotate(r'%s' % LatexName(molecule),
                     xy=(x_axis[i], y_axis[i]),
                     xytext=(1.1 * x_axis[i], y_axis[i]),
                     fontsize=axis_size - 2)
      pylab.xlim([min(x_axis) / 1.1, 1.2 * max(x_axis)])

    else:
      delta = 0.04 * (max(x_axis) - min(x_axis))
      pylab.annotate(r'%s' % LatexName(molecule),
                     xy=(x_axis[i], y_axis[i]),
                     xytext=(delta + x_axis[i], y_axis[i]),
                     fontsize=axis_size - 2)
      pylab.xlim([min(x_axis) - 4. * delta, max(x_axis) + 4. * delta])

  # Set y-scale.
  if y_log:
    pylab.yscale('log')
    pylab.ylim([min(y_axis) / 5., 5. * max(y_axis)])
    #pylab.ylim([min(y_axis) / 5., 1.4])
  else:
    delta = 0.1 * (max(y_axis) - min(y_axis))
    pylab.ylim([min(y_axis) - delta, max(y_axis) + delta])

  # Least square fit.
  if x_log and y_log and 1:
    import numpy
    import numpy.linalg
    log_y = pylab.log10(y_axis)
    log_x = pylab.log10(x_axis)
    print '\nPerforming least squares regression on data:'
    vander = numpy.ones((x_axis.size, 1))
    solution = numpy.linalg.lstsq(vander, log_y - 6. * log_x)
    intercept = solution[0]
    line = intercept + 6. * log_x
    SS_ave = pylab.sum(pylab.square(pylab.mean(log_y) - log_y))
    SS_res = pylab.sum(pylab.square(line - log_y))
    r_value = 1 - SS_res / SS_ave
    fit = 10. ** line
    pylab.plot(x_axis, fit, color='k')

    # Print out regression stats.
    print 'R-squared = %s.' % repr(r_value * r_value)
    pylab.annotate(r'$\textrm{r}^2-\textrm{value} = %f$' % (r_value * r_value),
                 xy=(3, 10 ** 4.5),
                 xytext = (3, 10 ** 4.5),
                 fontsize=axis_size)

  # Fit in to log-log plot.
  elif x_log and y_log and 0:
    log_y = pylab.log10(y_axis)
    log_x = pylab.log10(x_axis)
    slope, intercept, r_value, p_value, sigma = linregress(log_x, log_y)
    print '\nPerforming linear regression on log-log plot:'
    print 'R-squared = %s.' % repr(r_value * r_value)
    print 'Slope of %s with standard deviation of %s.'\
        % (repr(slope), repr(sigma))
    line = 10. ** (slope * log_x + intercept)
    pylab.plot(x_axis, line, color='k')

    # Print out regression stats.
    pylab.annotate(r'$\textrm{slope} = %f \pm %f$' % (slope, sigma),
                 xy=(3, 10 ** 1.5),
                 xytext = (3, 10 ** 1.5),
                 fontsize=axis_size)
    pylab.annotate(r'$\textrm{r}^2-\textrm{value} = %f$' % (r_value * r_value),
                 xy=(3, 10 ** 1.25),
                 xytext = (3, 10 ** 1.25),
                 fontsize=axis_size)

  # Finish making the plot.
  pylab.xticks(size=axis_size)
  pylab.yticks(size=axis_size)
  pylab.xlabel(r'%s' % x_label, fontsize=font_size)
  pylab.ylabel(r'%s' % y_label, fontsize=font_size)
  pylab.tight_layout()
  pylab.savefig('data/figures/%s_vs_%s_%s.pdf' % (x_type, y_type, basis))
  pylab.show()


# Compare different basis sets.
def PlotBasisSets(molecules):
  global line_width
  global axis_size
  global font_size

  # Initialize.
  n_molecules = len(molecules)
  x_cmo = pylab.zeros(n_molecules)
  x_nmo = pylab.zeros(n_molecules)
  y_cmo = pylab.zeros(n_molecules)
  y_nmo = pylab.zeros(n_molecules)

  # Assign number of spin orbitals to each molecule and compure error.
  for i, molecule in enumerate(molecules):
    print '\nAnalyzing %s, molecule %i of %i.' % (molecule, i + 1, n_molecules)

    # Get error operator information.
    cmo_error, cmo_energy, cmo_norm = operators.GetData(
        molecule, 'CMO', 'FCI')
    nmo_error, nmo_energy, nmo_norm = operators.GetData(
        molecule, 'NMO', 'FCI')
    if 1:
      integrals, terms = commutators.GetHamiltonianTerms(molecule, 'OAO')
      n_orbitals = commutators.OrbitalCount(terms)
      cmo_norm = n_orbitals
      nmo_norm = n_orbitals

    # Set axis.
    x_cmo[i] = cmo_norm
    x_nmo[i] = nmo_norm
    y_cmo[i] = abs(cmo_error)
    y_nmo[i] = abs(nmo_error)

  # Add Dave's results.
  if 1:
    dave_molecules = ['HCl', 'F2', 'NH3', 'H2S', 'CO', 'CH4']
    dave_orbitals = [20, 20, 16, 22, 20, 18]
    dave_errors = [0.005, 0.013, 0.027, 0.017, 0.036, 0.033]
    cmo_molecules = molecules + dave_molecules
    x_cmo = pylab.append(x_cmo, dave_orbitals)
    y_cmo = pylab.append(y_cmo, dave_errors)
  else:
    cmo_molecules = molecules

  # Add second period.
  if 1:
    oao_molecules = molecules + ['Na', 'Mg', 'Al', 'Si', 'S']
    x_oao = pylab.zeros(len(oao_molecules))
    y_oao = pylab.zeros(len(oao_molecules))
    for i, molecule in enumerate(oao_molecules):
      oao_error, oao_energy, oao_norm = operators.GetData(
          molecule, 'OAO', 'FCI')
      integrals, terms = commutators.GetHamiltonianTerms(molecule, 'OAO')
      n_orbitals = commutators.OrbitalCount(terms)
      if 1:
        x_oao[i] = n_orbitals
      else:
        x_oao[i] = oao_norm
      y_oao[i] = abs(oao_error)
  else:
    oao_molecules = molecules

  # Plot.
  pylab.figure(0)
  colors = ['r', 'g', 'b']
  xs = [x_oao, x_cmo, x_nmo]
  ys = [y_oao, y_cmo, y_nmo]
  ms = [oao_molecules, cmo_molecules, molecules]
  markers = ['o', '^', 's']
  for x_axis, y_axis, color, marker, mols in zip(xs, ys, colors, markers, ms):
    pylab.plot(x_axis, y_axis, lw=0, marker=marker, color=color)
    for i, molecule in enumerate(mols):
      if 1:
        pylab.annotate(r'%s' % LatexName(molecule),
                       xy=(x_axis[i], y_axis[i]),
                       xytext=(1.1 * x_axis[i], y_axis[i]),
                       color=color,
                       fontsize=axis_size - 2)
      else:
        pass

  # Set limits.
  pylab.xscale('log')
  pylab.yscale('log')
  max_y = max(map(max, ys))
  min_y = min(map(min, ys))
  max_x = max(map(max, xs))
  min_x = min(map(min, xs))
  pylab.xlim([min_x / 1.5, 5 * max_x])
  pylab.ylim([min_y / 5, 7 * max_y])

  # Finish making the plot.
  x_label = r'Number of spin-orbitals in minimal basis'
  y_label = r'Trotter error (Hartree / $\Delta_t^2$)'
  pylab.xticks(size=axis_size)
  pylab.yticks(size=axis_size)
  pylab.xlabel(r'%s' % x_label, fontsize=font_size)
  pylab.ylabel(r'%s' % y_label, fontsize=font_size)
  pylab.legend(['local basis', 'canonical basis', 'natural basis'], loc=3,
      prop={'size':20})
  pylab.tight_layout()
  pylab.savefig('data/figures/basis_set_orbitals.pdf')
  pylab.show()


# Compare different configuration sets.
def PlotConfigurationSets(molecules, basis):
  global line_width
  global axis_size
  global font_size

  # Initialize.
  n_molecules = len(molecules)
  fci_error = pylab.zeros(n_molecules)
  hf_error = pylab.nan * pylab.ones(n_molecules)
  cisd_error = pylab.nan * pylab.ones(n_molecules)
  cisdt_error = pylab.nan * pylab.ones(n_molecules)
  cisdtq_error = pylab.nan * pylab.ones(n_molecules)

  # Assign number of spin orbitals to each molecule and compure error.
  for i, molecule in enumerate(molecules):
    print '\nAnalyzing %s, molecule %i of %i.' % (molecule, i + 1, n_molecules)

    # Get FCI error.
    fci_error[i] = abs(operators.GetData(molecule, basis, 'FCI')[0])
    integrals, terms = commutators.GetHamiltonianTerms(molecule, basis)
    n_electrons = commutators.ElectronCount(molecule)
    n_orbitals = commutators.OrbitalCount(terms)

    # Get HF error.
    if n_orbitals - n_electrons > 0:
      hf_error[i] = abs(operators.GetData(molecule, basis, 0)[0])

      # Get CISD error.
      if n_orbitals - n_electrons > 2:
        cisd_error[i] = abs(operators.GetData(molecule, basis, 2)[0])

        # Get CISDT error.
        if n_orbitals - n_electrons > 3:
          cisdt_error[i] = abs(operators.GetData(molecule, basis, 3)[0])

          # Get CISDTQ error.
          if n_orbitals - n_electrons > 4:
            cisdtq_error[i] = abs(operators.GetData(molecule, basis, 4)[0])
            print molecule, cisdtq_error[i]

  # Plot.
  pylab.figure(0)
  colors = ['r', 'g', 'b']
  markers = ['o', '^', 's']
  hf_dis = 100. * abs(fci_error - hf_error) / fci_error
  cisd_dis = 100. * abs(fci_error - cisd_error) / fci_error
  cisdt_dis = 100. * abs(fci_error - cisdt_error) / fci_error
  cisdtq_dis = 100. * abs(fci_error - cisdtq_error) / fci_error
  #ys = [cisd_dis, cisdt_dis, cisdtq_dis]
  ys = [hf_error, cisd_error, cisdt_error]
  for y_axis, color, marker in zip(ys, colors, markers):
    pylab.plot(fci_error[y_axis == y_axis], y_axis[y_axis == y_axis],
        lw=0, marker=marker, color=color)
    for i, molecule in enumerate(molecules):
      print molecule, hf_dis[i], cisd_dis[i], cisdt_dis[i], cisdtq_dis[i]
      if y_axis[i] == y_axis[i]:
        pylab.annotate(r'%s' % LatexName(molecule),
                      xy=(fci_error[i], y_axis[i]),
                      xytext=(1.1 * fci_error[i], y_axis[i]),
                      color=color,
                      fontsize=axis_size - 2)

  # Set limits.
  line = pylab.logspace(-20, 20, 10)
  pylab.plot(line, line, color='k', lw=1)
  #pylab.plot(line, 10 * [100], color='k', lw=1)
  pylab.xscale('log')
  pylab.yscale('log')
  pylab.xlim([1e-3, 1e1])
  pylab.ylim([1e-3, 1e1])

  # Finish making the plot.
  x_label = r'Exact error'
  y_label = r'Ansatz error'
  pylab.xticks(size=axis_size)
  pylab.yticks(size=axis_size)
  pylab.xlabel(r'%s' % x_label, fontsize=font_size)
  pylab.ylabel(r'%s' % y_label, fontsize=font_size)
  pylab.legend(['Hartree-Fock Ansatz', 'CISD ansatz','CISDT ansatz'], loc=0)
  pylab.tight_layout()
  pylab.savefig('data/figures/configuration_compare_%s.pdf' % basis)
  pylab.show()


# Define main.
def main():

  # Parameters.
  basis = str(argv[1])
  molecules = []

  # Select atoms.
  molecules += ['Li', 'Be']
  molecules += ['B', 'C', 'N']
  #molecules += ['O', 'F', 'Ne']
  if 0:
    molecules += ['Na', 'Mg']
    molecules += ['Al', 'Si', 'P']
    molecules += ['S', 'Cl', 'Ar']
    del molecules[-1]
    del molecules[-1]

  ## Select molecules.
  molecules += ['H2', 'HeH+']
  #del molecules[-2]
  #molecules += ['LiH', 'HF']
  #molecules += ['BeH2', 'CH2', 'H2O']

  # Plot.
  if 0:
    PlotMolecules(molecules, basis)
  elif 1:
    PlotBasisSets(molecules)
  else:
    PlotConfigurationSets(molecules, basis)


# Run.
if __name__ == '__main__':
  main()
