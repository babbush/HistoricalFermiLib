import jellium
import numpy
import pylab
import scipy
import scipy.stats


# Plot norm as function of system size.

# Parameters.
n_dimensions = 1
spinless = 1
if 1:
  length_scale = 1.
  grid_length_min = 3
  grid_length_max = 51
  grid_lengths = [grid_length for grid_length in
                  range(grid_length_min, grid_length_max) if (grid_length % 2)]
  x_label = 'Log of M in {}D'.format(n_dimensions)
  x_values = numpy.log10(grid_lengths)
else:
  n_points = 50
  grid_length = 5
  length_scale_min = -2
  length_scale_max = 3.
  length_scales = [length_scale for length_scale in
                   numpy.logspace(length_scale_min,
                                  length_scale_max, n_points)]
  x_label = 'Log of Omega in {}D'.format(n_dimensions)
  x_values = float(n_dimensions) * numpy.log10(length_scales)

# Loop over different sizes.
kinetic_norms = []
potential_norms = []
#for length_scale in length_scales:
for grid_length in grid_lengths:
  print('Grid length is {}. Length scale is {}.'.format(
      grid_length, length_scale))

  # Get qubit Hamiltonian.
  qubit_hamiltonian = jellium.jordan_wigner_position_jellium(
      n_dimensions, grid_length, length_scale, spinless)

  # Compute norms.
  kinetic_norm = 0.
  potential_norm = 0.
  for qubit_term in qubit_hamiltonian:
    if not len(qubit_term):
      continue
    elif qubit_term[0][1] == 'Z':
      potential_norm += abs(qubit_term.coefficient)
    else:
      kinetic_norm += abs(qubit_term.coefficient)
  print('Kinetic norm is {}.\nPotential norm is {}.\n'.format(
      kinetic_norm, potential_norm))
  kinetic_norms += [kinetic_norm]
  potential_norms += [potential_norm]

# Initialize post-processing.
kinetic_norms = numpy.array(kinetic_norms)
potential_norms = numpy.array(potential_norms)

# Perform kinetic norm regression.
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
    x_values, numpy.log10(kinetic_norms))
kinetic_norm_y = intercept + slope * x_values
print('Slope for kinetic norm is {} +/- {}.\n'.format(slope, std_err))

# Plot.
pylab.figure(0)
pylab.plot(x_values, numpy.log10(kinetic_norms), 'x')
pylab.plot(x_values, kinetic_norm_y)
pylab.title('Slope is {} +/- {}.'.format(slope, std_err))
pylab.xlabel(x_label)
pylab.ylabel('Log of Kinetic Norm')
pylab.show()

# Perform potential norm regression.
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
    x_values, numpy.log10(potential_norms))
potential_norm_y = intercept + slope * x_values
print('Slope for potential norm is {} +/- {}.\n'.format(slope, std_err))

# Plot.
pylab.figure(1)
pylab.plot(x_values, numpy.log10(potential_norms), 'x')
pylab.plot(x_values, potential_norm_y)
pylab.title('Slope is {} +/- {}.'.format(slope, std_err))
pylab.xlabel(x_label)
pylab.ylabel('Log of Potential Norm')
pylab.show()
