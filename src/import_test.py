"""Tests of different circular dependency errors."""
import unittest


class ImportTest(unittest.TestCase):
  def test_fqmll_from(self):
    from fermion_operators import FermionOperator
    from qubit_operators import QubitOperator
    from molecular_operators import MolecularOperator
    from local_terms import LocalTerm
    from local_operators import LocalOperator

  def test_lmqfl_from1(self):
    from local_terms import LocalTerm
    from molecular_operators import MolecularOperator
    from qubit_operators import QubitOperator
    from fermion_operators import FermionOperator
    from local_operators import LocalOperator

  def test_lmqfl_from2(self):
    from local_operators import LocalOperator
    from molecular_operators import MolecularOperator
    from qubit_operators import QubitOperator
    from fermion_operators import FermionOperator
    from local_terms import LocalTerm

  def test_llqmf_from(self):
    from local_operators import LocalOperator
    from local_terms import LocalTerm
    from molecular_operators import MolecularOperator
    from qubit_operators import QubitOperator
    from fermion_operators import FermionOperator

  def test_empty(self):
    with self.assertRaises(NameError):
      f = FermionOperator(3)

if __name__ == '__main__':
    unittest.main()
