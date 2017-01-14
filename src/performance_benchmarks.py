"""This file contains tests of code performance to reveal bottlenecks."""
import molecular_operators
import numpy
import time


def artificial_molecular_operator(n_qubits):
  """Make an artificial random MolecularOperator for testing purposes."""

  # Initialize.
  constant = numpy.random.randn()
  one_body_coefficients = numpy.zeros((n_qubits, n_qubits), float)
  two_body_coefficients = numpy.zeros((n_qubits, n_qubits,
                                       n_qubits, n_qubits), float)

  # Randomly generate the one-body and two-body integrals.
  for p in xrange(n_qubits):
    for q in xrange(n_qubits):

      # One-body terms.
      if (p <= p) and (p % 2 == q % 2):
        one_body_coefficients[p, q] = numpy.random.randn()
        one_body_coefficients[q, p] = one_body_coefficients[p, q]

      # Keep looping.
      for r in xrange(n_qubits):
        for s in xrange(n_qubits):

          # Skip zero terms.
          if (p == q) or (r == s):
            continue

          # Identify and skip one of the complex conjugates.
          if [p, q, r, s] != [s, r, q, p]:
            unique_indices = len(set([p, q, r, s]))

            # srqp srpq sprq spqr sqpr sqrp rsqp rspq rpsq rpqs rqps rqsp.
            if unique_indices == 4:
              if min(r, s) <= min(p, q):
                continue

            # qqpp.
            elif unique_indices == 2:
              if q < p:
                continue

          # Add the two-body coefficients.
          two_body_coefficients[p, q, r, s] = numpy.random.randn()
          two_body_coefficients[s, r, q, p] = two_body_coefficients[p, q, r, s]

  # Build the molecular operator and return.
  molecular_operator = molecular_operators.MolecularOperator(
      constant, one_body_coefficients, two_body_coefficients)
  return molecular_operator


def benchmark_molecular_operator_jordan_wigner(n_qubits):
  """Test speed with which molecular operators transform to qubit operators.

  Args:
    n_qubits: The size of the molecular operator instance. Ideally, we would
        be able to transform to a qubit operator for 50 qubit instances in
        much less than a minute. We are way too slow right now.

  Returns:
    runtime: The number of seconds required to make the conversion.
  """
  # Get an instance of MolecularOperator.
  molecular_operator = artificial_molecular_operator(n_qubits)

  # Convert to a qubit operator.
  start = time.time()
  qubit_operator = molecular_operator.jordan_wigner_transform()
  end = time.time()

  # Return runtime.
  runtime = end - start
  return runtime


# Run benchmarks.
if __name__ == '__main__':

  # Run MolecularOperator.jordan_wigner_transform() benchmark.
  if 1:
    n_qubits = 18
    print('Starting test on MolecularOperator.jordan_wigner_transform()')
    runtime = benchmark_molecular_operator_jordan_wigner(n_qubits)
    print('MolecularOperator.jordan_wigner_transform() ' +
          'takes {} seconds on {} qubits.'.format(runtime, n_qubits))