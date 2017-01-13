"""The purpose of this file is to document and demonstrate known bugs."""
import unittest
import local_operators
import local_terms
import numpy
import copy


class NumpyScalarBug(unittest.TestCase):
  """Multiplying certain classes by numpy scalars returns invalid objects.

  When a FermionOpeator, QubitOperator or LocalOperator is multiplied by
  a scalar it is supposed to return that operator times the scalar.
  Something unexpected happens when the scalar is from a numpy data type,
  for instance when the scalar is a numpy.float64 instead of a "float".
  I have tried to change the __rmul__ methods in local_operators and
  qubit_operators to handle numpy scalar types by recognizing them with
  numpy.isscalar() but to no avail. Note that the bug is only with the
  __rmul__ and not with __mul__/__lmul__. Note that the bug may also
  need to be fixed in the QubitTerm which overwrites the __imul__ class
  from its parent, LocalTerm."""

  def setUp(self):

    # Get the numpy scalar and also a normal one for comparison.
    self.good_scalar = -2.9
    self.bad_scalar = numpy.float64(self.good_scalar)

    # Get a LocalTerm to test things out.
    self.n_qubits = 8
    self.coefficient = 1.
    self.operators = [3, 4, 5]
    self.local_term = local_terms.LocalTerm(
        self.n_qubits, self.coefficient, self.operators)

    # Make a LocalOperator to test things out.
    new_term = copy.deepcopy(self.local_term)
    self.local_operator = local_operators.LocalOperator(
        self.n_qubits, [new_term, self.local_term])

  def test_right_local_term_multiply(self):
    correct = self.good_scalar * self.local_term
    buggy = self.bad_scalar * self.local_term
    print 'The correct result for a single term follows:\n{}'.format(correct)
    print 'The buggy results for a single term follows:\n{}'.format(buggy)
    self.assertEqual(correct, buggy)

  def test_right_local_operator_multiply(self):
    correct = self.good_scalar * self.local_operator
    buggy = self.bad_scalar * self.local_operator
    print 'The correct result for an operator follows:\n{}'.format(correct)
    print 'The buggy results for an operator follows:\n{}'.format(buggy)
    self.assertEqual(correct, buggy)


# Run test.
if __name__ == '__main__':
  unittest.main()
