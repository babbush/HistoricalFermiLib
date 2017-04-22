"""Tests for _jordan_wigner_interaction_op.py."""
from __future__ import absolute_import
import numpy
import unittest

from fermilib.interaction_operators import InteractionOperator
from projectqtemp.ops._fermion_operator import FermionOperator
from projectqtemp.ops import _fermion_operator as fo

from transforms._jordan_wigner import jordan_wigner
from transforms._jordan_wigner_interaction_op import (jordan_wigner_one_body,
                                                      jordan_wigner_two_body)

class InteractionOperatorsJWTest(unittest.TestCase):

    def setUp(self):
        self.n_qubits = 5
        self.constant = 0.
        self.one_body = numpy.zeros((self.n_qubits, self.n_qubits), float)
        self.two_body = numpy.zeros((self.n_qubits, self.n_qubits,
                                     self.n_qubits, self.n_qubits), float)
        self.interaction_operator = InteractionOperator(self.constant,
                                                        self.one_body,
                                                        self.two_body)

    def test_jordan_wigner_one_body(self):
        # Make sure it agrees with jordan_wigner(FermionTerm).
        for p in range(self.n_qubits):
            for q in range(self.n_qubits):
                # Get test qubit operator.
                test_operator = jordan_wigner_one_body(p, q)

                # Get correct qubit operator.
                fermion_term = FermionOperator(((p, 1), (q, 0)))
                correct_op = jordan_wigner(fermion_term)
                hermitian_conjugate = fo.hermitian_conjugated(fermion_term)
                if not fermion_term.isclose(hermitian_conjugate):
                    correct_op += jordan_wigner(hermitian_conjugate)

                self.assertEqual(test_operator, correct_op)

    def test_jordan_wigner_two_body(self):
        # Make sure it agrees with jordan_wigner(FermionTerm).
        for p in range(self.n_qubits):
            for q in range(self.n_qubits):
                for r in range(self.n_qubits):
                    for s in range(self.n_qubits):
                        # Get test qubit operator.
                        test_operator = jordan_wigner_two_body(p, q, r, s)

                        # Get correct qubit operator.
                        fermion_term = FermionOperator(((p, 1), (q, 1),
                                                        (r, 0), (s, 0)))
                        correct_op = jordan_wigner(fermion_term)
                        hermitian_conjugate = fo.hermitian_conjugated(
                            fermion_term)
                        if not fermion_term.isclose(hermitian_conjugate):
                            correct_op += jordan_wigner(hermitian_conjugate)

                        self.assertEqual(test_operator, correct_op)

if __name__ == '__main__':
    unittest.main()