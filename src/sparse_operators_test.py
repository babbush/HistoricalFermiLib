"""Tests for sparse_operators.py"""
from sparse_operators import *
import fermion_operators
import unittest


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
    n_qubits = 5
    fermion_operator = fermion_operators.FermionTerm(
        n_qubits, -4.3, [(3, 1), (2, 1), (1, 0), (0, 0)])
    fermion_operator += fermion_operators.FermionTerm(
        n_qubits, 8.17, [(3, 1), (1, 0)])
    fermion_operator += 3.2 * fermion_operators.fermion_identity(n_qubits)
    fermion_operator **= 3

    # Map to qubits and compare matrix versions.
    qubit_operator = fermion_operator.jordan_wigner_transform()
    qubit_sparse = qubit_operator.get_sparse_operator()
    qubit_spectrum = qubit_sparse.get_eigenspectrum()
    fermion_sparse = fermion_operator.jordan_wigner_sparse()
    fermion_spectrum = fermion_sparse.get_eigenspectrum()
    self.assertAlmostEqual(0., numpy.amax(
        numpy.absolute(fermion_spectrum - qubit_spectrum)))


if __name__ == '__main__':
  unittest.main()