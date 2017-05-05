#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""Tests for operator_utils."""
from __future__ import absolute_import

import numpy
import unittest

from fermilib.ops import *
from fermilib.transforms import jordan_wigner, get_interaction_operator
from fermilib.utils import eigenspectrum, is_identity, count_qubits

from projectqtemp.ops._qubit_operator import QubitOperator

class OperatorUtilsTest(unittest.TestCase):

    def setUp(self):
        self.n_qubits = 5
        self.fermion_term = FermionOperator('1^ 2^ 3 4', -3.17)
        self.fermion_operator = self.fermion_term + hermitian_conjugated(
                self.fermion_term)
        self.qubit_operator = jordan_wigner(self.fermion_operator)
        self.interaction_operator = get_interaction_operator(
                self.fermion_operator)

    def test_n_qubits(self):
        self.assertEqual(self.n_qubits,
                         count_qubits(self.fermion_term))
        self.assertEqual(self.n_qubits,
                         count_qubits(self.fermion_operator))
        self.assertEqual(self.n_qubits,
                         count_qubits(self.qubit_operator))
        self.assertEqual(self.n_qubits,
                         count_qubits(self.interaction_operator))

    def test_eigenspectrum(self):
        fermion_eigenspectrum = eigenspectrum(self.fermion_operator)
        qubit_eigenspectrum = eigenspectrum(self.qubit_operator)
        interaction_eigenspectrum = eigenspectrum(self.interaction_operator)
        for i in range(2 ** self.n_qubits):
            self.assertAlmostEqual(fermion_eigenspectrum[i],
                                   qubit_eigenspectrum[i])
            self.assertAlmostEqual(fermion_eigenspectrum[i],
                                   interaction_eigenspectrum[i])

    def test_is_identity(self):
        self.assertTrue(is_identity(FermionOperator(())))
        self.assertTrue(is_identity(2. * FermionOperator(())))
        self.assertTrue(is_identity(QubitOperator(())))
        self.assertTrue(is_identity(QubitOperator((), 2.)))
        self.assertFalse(is_identity(FermionOperator('1^')))
        self.assertFalse(is_identity(QubitOperator('X1')))
        self.assertFalse(is_identity(FermionOperator()))
        self.assertFalse(is_identity(QubitOperator()))


if __name__ == '__main__':
    unittest.main()


