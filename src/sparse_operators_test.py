"""Tests for sparse_operators.py"""
from sparse_operators import *
import unittest


# Make global definitions.
_IDENTITY_CSC = scipy.sparse.identity(2, format='csr', dtype=complex)
_PAULI_X_CSC = scipy.sparse.csc_matrix([[0., 1.], [1., 0.]], dtype=complex)
_PAULI_Y_CSC = scipy.sparse.csc_matrix([[0., -1.j], [1.j, 0.]], dtype=complex)
_PAULI_Z_CSC = scipy.sparse.csc_matrix([[1., 0.], [0., -1.]], dtype=complex)
_Q_RAISE_CSC = (_PAULI_X_CSC - 1.j * _PAULI_Y_CSC) / 2.
_Q_LOWER_CSC = (_PAULI_X_CSC + 1.j * _PAULI_Y_CSC) / 2.
_PAULI_MATRIX_MAP = {'I': _IDENTITY_CSC, 'X': _PAULI_X_CSC,
                     'Y': _PAULI_Y_CSC, 'Z': _PAULI_Z_CSC}


class SparseOperatorTest(unittest.TestCase):

  def test_kronecker_operators(self):

    self.assertAlmostEqual(
        0, numpy.amax(numpy.absolute(
            kronecker_operators(3 * [_IDENTITY_CSC]) -
            kronecker_operators(3 * [_PAULI_X_CSC]) ** 2)))


if __name__ == '__main__':
  unittest.main()
