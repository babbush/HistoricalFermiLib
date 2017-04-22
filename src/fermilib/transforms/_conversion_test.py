"""Tests for _conversion.py."""
from __future__ import absolute_import

import copy
import unittest

import numpy

from fermilib import fermion_operators as fo
from fermilib.fermion_operators import (FermionTerm, FermionOperator,
                                        number_operator, one_body_term,
                                        two_body_term)
from fermilib.interaction_operators import InteractionOperator
from fermilib.qubit_operators import QubitTerm, QubitOperator, qubit_identity
from ._conversion import (eigenspectrum, get_fermion_operator,
                          get_interaction_operator, get_sparse_operator)


class GetInteractionOperatorTest(unittest.TestCase):

    def test_get_molecular_operator(self):
        coefficient = 3.
        operators = [(2, 1), (3, 0), (0, 0), (3, 1)]
        term = FermionTerm(operators, coefficient)
        op = FermionOperator(term)

        molecular_operator = get_interaction_operator(op)
        fermion_operator = get_fermion_operator(molecular_operator)
        fermion_operator.normal_order()
        op.normal_order()
        self.assertEqual(op, fermion_operator)


class GetSparseOperatorQubitTest(unittest.TestCase):

    def test_sparse_matrix_Y(self):
        term = QubitTerm([(0, 'Y')])
        sparse_operator = get_sparse_operator(term)
        self.assertEqual(list(sparse_operator.matrix.data), [1j, -1j])
        self.assertEqual(list(sparse_operator.matrix.indices), [1, 0])
        self.assertTrue(sparse_operator.is_hermitian())

    def test_sparse_matrix_ZX(self):
        coefficient = 2.
        operators = [(0, 'Z'), (1, 'X')]
        term = QubitTerm(operators, coefficient)
        sparse_operator = get_sparse_operator(term)
        self.assertEqual(list(sparse_operator.matrix.data), [2., 2., -2., -2.])
        self.assertEqual(list(sparse_operator.matrix.indices), [1, 0, 3, 2])
        self.assertTrue(sparse_operator.is_hermitian())

    def test_sparse_matrix_ZIZ(self):
        operators = [(0, 'Z'), (2, 'Z')]
        term = QubitTerm(operators)
        sparse_operator = get_sparse_operator(term)
        self.assertEqual(list(sparse_operator.matrix.data),
                         [1, -1, 1, -1, -1, 1, -1, 1])
        self.assertEqual(list(sparse_operator.matrix.indices), list(range(8)))
        self.assertTrue(sparse_operator.is_hermitian())

    def test_sparse_matrix_combo(self):
        qop = (QubitTerm([(0, 'Y'), (1, 'X')], -0.1j) +
               QubitTerm([(0, 'X'), (1, 'Z')], 3. + 2.j))
        sparse_operator = get_sparse_operator(qop)

        self.assertEqual(list(sparse_operator.matrix.data),
                         [3 + 2j, 0.1, 0.1, -3 - 2j,
                          3 + 2j, -0.1, -0.1, -3 - 2j])
        self.assertEqual(list(sparse_operator.matrix.indices),
                         [2, 3, 2, 3, 0, 1, 0, 1])

    def test_sparse_matrix_zero_1qubit(self):
        sparse_operator = get_sparse_operator(QubitOperator(), 1)
        sparse_operator.eliminate_zeros()
        self.assertEqual(len(list(sparse_operator.matrix.data)), 0)
        self.assertEqual(sparse_operator.matrix.shape, (2, 2))

    def test_sparse_matrix_zero_5qubit(self):
        sparse_operator = get_sparse_operator(QubitOperator(), 5)
        sparse_operator.eliminate_zeros()
        self.assertEqual(len(list(sparse_operator.matrix.data)), 0)
        self.assertEqual(sparse_operator.matrix.shape, (32, 32))

    def test_sparse_matrix_identity_1qubit(self):
        sparse_operator = get_sparse_operator(QubitOperator(qubit_identity()),
                                              1)
        self.assertEqual(list(sparse_operator.matrix.data), [1] * 2)
        self.assertEqual(sparse_operator.matrix.shape, (2, 2))

    def test_sparse_matrix_identity_5qubit(self):
        sparse_operator = get_sparse_operator(QubitOperator(qubit_identity()),
                                              5)
        self.assertEqual(list(sparse_operator.matrix.data), [1] * 32)
        self.assertEqual(sparse_operator.matrix.shape, (32, 32))

    def test_sparse_matrix_linearity(self):
        identity = QubitOperator(qubit_identity())
        zzzz = QubitOperator(QubitTerm([(i, 'Z') for i in range(4)], 1.0))

        sparse1 = get_sparse_operator(identity + zzzz)
        sparse2 = get_sparse_operator(identity, 4) + get_sparse_operator(zzzz)

        self.assertEqual(list(sparse1.matrix.data), [2] * 8)
        self.assertEqual(list(sparse1.matrix.indices),
                         [0, 3, 5, 6, 9, 10, 12, 15])
        self.assertEqual(list(sparse2.matrix.data), [2] * 8)
        self.assertEqual(list(sparse2.matrix.indices),
                         [0, 3, 5, 6, 9, 10, 12, 15])


if __name__ == '__main__':
    unittest.main()
