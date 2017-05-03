"""Tests for _trotter.py."""

import numpy
import pytest
import transforms
from utils import _trotter

from future.utils import iteritems
from math import sqrt
from scipy.linalg import expm

from fermilib.transforms import get_sparse_operator
from fermilib.config import *
from fermilib.ops import normal_ordered
from fermilib.utils import MolecularData
from projectqtemp.ops import QubitOperator
from psi4tmp import run_psi4


def test_commutator_commutes():
    zero = QubitOperator((), 0.0)
    assert _trotter.commutator(QubitOperator(),
                               QubitOperator('X3')).isclose(zero)


def test_commutator_single_pauli():
    com = _trotter.commutator(QubitOperator('X3'), QubitOperator('Y3'))
    expected = 2j * QubitOperator('Z3')
    assert expected.isclose(com)


def test_commutator_multi_pauli():
    com = _trotter.commutator(QubitOperator('Z1 X2 Y4'),
                              QubitOperator('X1 Z2 X4'))
    expected = -2j * QubitOperator('Y1 Y2 Z4')
    assert expected.isclose(com)


def test_trivially_commutes_id_id():
    assert _trotter.trivially_commutes(QubitOperator(), 3 * QubitOperator())


def test_trivially_commutes_id_x():
    assert _trotter.trivially_commutes(QubitOperator(), QubitOperator('X1'))


def test_trivially_commutes_id_xx():
    assert _trotter.trivially_commutes(QubitOperator(),
                                       QubitOperator('X1 X3'))


def test_trivially_commutes_nonid_with_id():
    assert _trotter.trivially_commutes(QubitOperator('X1 Z5 Y9 Z11'),
                                       QubitOperator())


def test_trivially_commutes_no_intersect():
    assert _trotter.trivially_commutes(QubitOperator('X1 Y3 Z6'),
                                       QubitOperator('Z0 Z2 X4 Y5'))


def test_trivially_commutes_allsame_oddintersect():
    assert _trotter.trivially_commutes(QubitOperator('X1 X3 X4 Z6 X8'),
                                       QubitOperator('X1 X3 X4 Z7 Y9'))


def test_trivially_commutes_even_anti():
    assert _trotter.trivially_commutes(QubitOperator('X1 Z2 Z3 X10'),
                                       QubitOperator('Y1 X2 Z3 Y9'))


def test_no_trivial_commute_odd_anti():
    assert not _trotter.trivially_commutes(QubitOperator('X1'),
                                           QubitOperator('Y1'))


def test_no_trivial_commute_triple_anti_intersect():
    assert not _trotter.trivially_commutes(QubitOperator('X0 Z2 Z4 Z9 Y17'),
                                           QubitOperator('Y0 X2 Y4 Z9 Z16'))


def test_no_trivial_commute_mostly_commuting():
    assert not _trotter.trivially_commutes(QubitOperator('X0 Y1 Z2 X4 Y5 Y6'),
                                           QubitOperator('X0 Y1 Z2 X4 Z5 Y6'))


def test_trivial_double_commute_no_intersect():
    assert _trotter.trivially_double_commutes(QubitOperator('X1 Z2 Y4'),
                                              QubitOperator('Y0 X3 Z6'),
                                              QubitOperator('Y5'))


def test_trivial_double_commute_no_intersect_a_bc():
    assert _trotter.trivially_double_commutes(QubitOperator('X1 Z2 Y4'),
                                              QubitOperator('Y0 X3 Z6'),
                                              QubitOperator('Z3 Y5'))


def test_trivial_double_commute_bc_intersect_commute():
    assert _trotter.trivially_double_commutes(QubitOperator('X1 Z2 Y4'),
                                              QubitOperator('X0 Z3'),
                                              QubitOperator('Y0 X3'))


def test_error_operator_bad_order():
    with pytest.raises(NotImplementedError):
        _trotter.error_operator([QubitOperator], 1)


def test_error_operator_all_diagonal():
    terms = [QubitOperator(), QubitOperator('Z0 Z1 Z2'),
             QubitOperator('Z0 Z3'), QubitOperator('Z0 Z1 Z2 Z3')]
    zero = QubitOperator((), 0.0)
    assert zero.isclose(_trotter.error_operator(terms))


def test_error_operator_xyz():
    terms = [QubitOperator('X1'), QubitOperator('Y1'), QubitOperator('Z1')]
    expected = numpy.array([[-2./3, 1./3, 0., 0.],
                            [1./3, 2./3, 0., 0.],
                            [0., 0., -2./3, 1./3],
                            [0., 0., 1./3, 2./3]])
    assert numpy.allclose(
        get_sparse_operator(_trotter.error_operator(terms)).to_dense(),
        expected)


def test_error_bound_xyz_tight():
    terms = [QubitOperator('X1'), QubitOperator('Y1'), QubitOperator('Z1')]
    expected = sqrt(5. / 9)  # norm of [[-2/3, 1/3], [1/3, 2/3]]
    assert numpy.isclose(_trotter.error_bound(terms, tight=True),
                         expected)


def test_error_bound_xyz_loose():
    terms = [QubitOperator('X1'), QubitOperator('Y1'), QubitOperator('Z1')]
    assert numpy.isclose(_trotter.error_bound(terms, tight=False),
                         4. * (2**2 + 1**2))


def test_H2_integration():
    return  # figure out a test for H2 that can be done sensibly
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 1.))]
    basis = 'sto-3g'
    multiplicity = 1
    molecule = MolecularData(geometry, basis, multiplicity)
    
    # Run calculations.
    run_scf = 1
    run_ccsd = 1
    run_fci = 1
    verbose = 0
    delete_input = 1
    delete_output = 0
    molecule = run_psi4(molecule, run_scf=True, run_ccsd=True, run_fci=True,
                        verbose=False, delete_input=True, delete_output=False)
    
    molecular_hamiltonian = molecule.get_molecular_hamiltonian()
    fermion_hamiltonian = transforms.get_fermion_operator(
        molecular_hamiltonian)
    fermion_hamiltonian = normal_ordered(fermion_hamiltonian)
    
    # Get qubit Hamiltonian.
    qubit_hamiltonian = transforms.jordan_wigner(fermion_hamiltonian)
    print qubit_hamiltonian
    terms = []
    for term, coefficient in iteritems(qubit_hamiltonian.terms):
        if coefficient:
            terms.append(QubitOperator(term, coefficient))
    
    import time
    start = time.time()
    
    print("\nFor LiH at bond length 1.45, with %i terms acting on %i qubits:"
          % (len(terms), qubit_hamiltonian.n_qubits()))
    print("Loose error bound = %f" % _trotter.error_bound(terms))
    print "Took ", time.time() - start, " to compute"
    start = time.time()
    print "Tight error bound = %f" % _trotter.error_bound(terms, tight=True)
    print "Took ", time.time() - start, " to compute"  
    

      