"""Tests for sparse_operators.py."""
from __future__ import absolute_import

import unittest

import numpy
import scipy.sparse

from fermilib import fermion_operators
from fermilib.sparse_operators import kronecker_operators
from fermilib.transforms import (jordan_wigner, jordan_wigner_sparse)


# Make copy definitions over from module.
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

    def test_qubit_jw_fermion_integration(self):

        # Initialize a random fermionic operator.
        fermion_operator = fermion_operators.FermionTerm(
            [(3, 1), (2, 1), (1, 0), (0, 0)], -4.3)
        fermion_operator += fermion_operators.FermionTerm(
            [(3, 1), (1, 0)], 8.17)
        fermion_operator += 3.2 * fermion_operators.fermion_identity()
        fermion_operator **= 3

        # Map to qubits and compare matrix versions.
        qubit_operator = jordan_wigner(fermion_operator)
        qubit_sparse = qubit_operator.get_sparse_operator()
        qubit_spectrum = qubit_sparse.eigenspectrum()
        fermion_sparse = jordan_wigner_sparse(fermion_operator)
        fermion_spectrum = fermion_sparse.eigenspectrum()
        self.assertAlmostEqual(0., numpy.amax(
            numpy.absolute(fermion_spectrum - qubit_spectrum)))


if __name__ == '__main__':
    unittest.main()
