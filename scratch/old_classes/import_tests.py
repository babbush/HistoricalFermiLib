"""Tests of different circular dependency errors."""
from __future__ import absolute_import

import unittest


class ImportTest(unittest.TestCase):
    def test_fqmll_from(self):
        from fermilib.fermion_operators import FermionOperator
        from fermilib.qubit_operators import QubitOperator
        from fermilib.interaction_operators import InteractionOperator
        from fermilib.local_terms import LocalTerm
        from fermilib.local_operators import LocalOperator

    def test_lmqfl_from1(self):
        from fermilib.local_terms import LocalTerm
        from fermilib.interaction_operators import InteractionOperator
        from fermilib.qubit_operators import QubitOperator
        from fermilib.fermion_operators import FermionOperator
        from fermilib.local_operators import LocalOperator

    def test_lmqfl_from2(self):
        from fermilib.local_operators import LocalOperator
        from fermilib.interaction_operators import InteractionOperator
        from fermilib.qubit_operators import QubitOperator
        from fermilib.fermion_operators import FermionOperator
        from fermilib.local_terms import LocalTerm

    def test_llqmf_from(self):
        from fermilib.local_operators import LocalOperator
        from fermilib.local_terms import LocalTerm
        from fermilib.interaction_operators import InteractionOperator
        from fermilib.qubit_operators import QubitOperator
        from fermilib.fermion_operators import FermionOperator

    def test_empty(self):
        with self.assertRaises(NameError):
            f = FermionOperator(3)


if __name__ == '__main__':
    unittest.main()
