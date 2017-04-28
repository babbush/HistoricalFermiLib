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

"""Tests for _qubit_operator.py."""
import copy

import numpy
import pytest

from projectqtemp.ops import _qubit_operator


def test_pauli_operator_product_unchanged():
    correct = {('I', 'I'): (1., 'I'),
               ('I', 'X'): (1., 'X'),
               ('X', 'I'): (1., 'X'),
               ('I', 'Y'): (1., 'Y'),
               ('Y', 'I'): (1., 'Y'),
               ('I', 'Z'): (1., 'Z'),
               ('Z', 'I'): (1., 'Z'),
               ('X', 'X'): (1., 'I'),
               ('Y', 'Y'): (1., 'I'),
               ('Z', 'Z'): (1., 'I'),
               ('X', 'Y'): (1.j, 'Z'),
               ('X', 'Z'): (-1.j, 'Y'),
               ('Y', 'X'): (-1.j, 'Z'),
               ('Y', 'Z'): (1.j, 'X'),
               ('Z', 'X'): (1.j, 'Y'),
               ('Z', 'Y'): (-1.j, 'X')}
    assert _qubit_operator._PAULI_OPERATOR_PRODUCTS == correct


def test_init_defaults():
    loc_op = _qubit_operator.QubitOperator()
    assert len(loc_op.terms) == 1
    assert loc_op.terms[()] == 1.0


@pytest.mark.parametrize("coefficient", [0.5, 0.6j, numpy.float64(2.303),
                         numpy.complex128(-1j)])
def test_init_tuple(coefficient):
    loc_op = ((0, 'X'), (5, 'Y'), (6, 'Z'))
    qubit_op = _qubit_operator.QubitOperator(loc_op, coefficient)
    assert len(qubit_op.terms) == 1
    assert qubit_op.terms[loc_op] == coefficient


def test_init_str():
    qubit_op = _qubit_operator.QubitOperator('X0 Y5 Z12', -1.)
    correct = ((0, 'X'), (5, 'Y'), (12, 'Z'))
    assert correct in qubit_op.terms
    assert qubit_op.terms[correct] == -1.0


def test_init_str_identity():
    qubit_op = _qubit_operator.QubitOperator('')
    assert () in qubit_op.terms


def test_init_bad_term():
    with pytest.raises(ValueError):
        qubit_op = _qubit_operator.QubitOperator(list())


def test_init_bad_coefficient():
    with pytest.raises(ValueError):
        qubit_op = _qubit_operator.QubitOperator('X0', "0.5")


def test_init_bad_action():
    with pytest.raises(ValueError):
        qubit_op = _qubit_operator.QubitOperator('Q0')


def test_init_bad_tuple():
    with pytest.raises(ValueError):
        qubit_op = _qubit_operator.QubitOperator(((0, 1, 'X'),))


def test_init_bad_str():
    with pytest.raises(ValueError):
        qubit_op = _qubit_operator.QubitOperator('X')


def test_init_bad_qubit_num():
    with pytest.raises(_qubit_operator.QubitOperatorError):
        qubit_op = _qubit_operator.QubitOperator('X-1')


def test_isclose_abs_tol():
    a = _qubit_operator.QubitOperator('X0', -1.)
    b = _qubit_operator.QubitOperator('X0', -1.05)
    c = _qubit_operator.QubitOperator('X0', -1.11)
    assert a.isclose(b, rel_tol=1e-14, abs_tol=0.1)
    assert not a.isclose(c, rel_tol=1e-14, abs_tol=0.1)
    a = _qubit_operator.QubitOperator('X0', -1.0j)
    b = _qubit_operator.QubitOperator('X0', -1.05j)
    c = _qubit_operator.QubitOperator('X0', -1.11j)
    assert a.isclose(b, rel_tol=1e-14, abs_tol=0.1)
    assert not a.isclose(c, rel_tol=1e-14, abs_tol=0.1)


def test_isclose_rel_tol():
    a = _qubit_operator.QubitOperator('X0', 1)
    b = _qubit_operator.QubitOperator('X0', 2)
    assert a.isclose(b, rel_tol=2.5, abs_tol=0.1)
    # Test symmetry
    assert a.isclose(b, rel_tol=1, abs_tol=0.1)
    assert b.isclose(a, rel_tol=1, abs_tol=0.1)


def test_isclose_zero_terms():
    op = _qubit_operator.QubitOperator(((1, 'Y'), (0, 'X')), -1j) * 0
    assert op.isclose(_qubit_operator.QubitOperator((), 0.0),
                      rel_tol=1e-12, abs_tol=1e-12)
    assert _qubit_operator.QubitOperator((), 0.0).isclose(
        op, rel_tol=1e-12, abs_tol=1e-12)


def test_isclose_different_terms():
    a = _qubit_operator.QubitOperator(((1, 'Y'),), -0.1j)
    b = _qubit_operator.QubitOperator(((1, 'X'),), -0.1j)
    assert a.isclose(b, rel_tol=1e-12, abs_tol=0.2)
    assert not a.isclose(b, rel_tol=1e-12, abs_tol=0.05)
    assert b.isclose(a, rel_tol=1e-12, abs_tol=0.2)
    assert not b.isclose(a, rel_tol=1e-12, abs_tol=0.05)


def test_isclose_different_num_terms():
    a = _qubit_operator.QubitOperator(((1, 'Y'),), -0.1j)
    a += _qubit_operator.QubitOperator(((2, 'Y'),), -0.1j)
    b = _qubit_operator.QubitOperator(((1, 'X'),), -0.1j)
    assert not b.isclose(a, rel_tol=1e-12, abs_tol=0.05)
    assert not a.isclose(b, rel_tol=1e-12, abs_tol=0.05)


def test_nqubits_0():
    op = _qubit_operator.QubitOperator()
    assert op.n_qubits() == 0


def test_nqubits_1():
    op = _qubit_operator.QubitOperator('X0', 3)
    assert op.n_qubits() == 1


def test_nqubits_doubledigit():
    op = _qubit_operator.QubitOperator('X5 Z11 Y27')
    assert op.n_qubits() == 28


def test_nqubits_multiterm():
    op = (_qubit_operator.QubitOperator() +
          _qubit_operator.QubitOperator('X1 X2 X3') +
          _qubit_operator.QubitOperator())
    assert op.n_qubits() == 4


def test_imul_inplace():
    qubit_op = _qubit_operator.QubitOperator("X1")
    prev_id = id(qubit_op)
    qubit_op *= 3.
    assert id(qubit_op) == prev_id


@pytest.mark.parametrize("multiplier", [0.5, 0.6j, numpy.float64(2.303),
                         numpy.complex128(-1j)])
def test_imul_scalar(multiplier):
    loc_op = ((1, 'X'), (2, 'Y'))
    qubit_op = _qubit_operator.QubitOperator(loc_op)
    qubit_op *= multiplier
    assert qubit_op.terms[loc_op] == pytest.approx(multiplier)


def test_imul_qubit_op():
    op1 = _qubit_operator.QubitOperator(((0, 'Y'), (3, 'X'),
                                        (8, 'Z'), (11, 'X')), 3.j)
    op2 = _qubit_operator.QubitOperator(((1, 'X'), (3, 'Y'), (8, 'Z')), 0.5)
    op1 *= op2
    correct_coefficient = 1.j * 3.0j * 0.5
    correct_term = ((0, 'Y'), (1, 'X'), (3, 'Z'), (11, 'X'))
    assert len(op1.terms) == 1
    assert correct_term in op1.terms


def test_imul_qubit_op_2():
    op3 = _qubit_operator.QubitOperator(((1, 'Y'), (0, 'X')), -1j)
    op4 = _qubit_operator.QubitOperator(((1, 'Y'), (0, 'X'), (2, 'Z')), -1.5)
    op3 *= op4
    op4 *= op3
    assert ((2, 'Z'),) in op3.terms
    assert op3.terms[((2, 'Z'),)] == 1.5j


def test_imul_bidir():
    op_a = _qubit_operator.QubitOperator(((1, 'Y'), (0, 'X')), -1j)
    op_b = _qubit_operator.QubitOperator(((1, 'Y'), (0, 'X'), (2, 'Z')), -1.5)
    op_a *= op_b
    op_b *= op_a
    assert ((2, 'Z'),) in op_a.terms
    assert op_a.terms[((2, 'Z'),)] == 1.5j
    assert ((0, 'X'), (1, 'Y')) in op_b.terms
    assert op_b.terms[((0, 'X'), (1, 'Y'))] == -2.25j


def test_imul_bad_multiplier():
    op = _qubit_operator.QubitOperator(((1, 'Y'), (0, 'X')), -1j)
    with pytest.raises(TypeError):
        op *= "1"


def test_mul_by_scalarzero():
    op = _qubit_operator.QubitOperator(((1, 'Y'), (0, 'X')), -1j) * 0
    assert ((0, 'X'), (1, 'Y')) in op.terms
    assert op.terms[((0, 'X'), (1, 'Y'))] == pytest.approx(0.0)


def test_mul_bad_multiplier():
    op = _qubit_operator.QubitOperator(((1, 'Y'), (0, 'X')), -1j)
    with pytest.raises(TypeError):
        op = op * "0.5"


def test_mul_out_of_place():
    op1 = _qubit_operator.QubitOperator(((0, 'Y'), (3, 'X'),
                                        (8, 'Z'), (11, 'X')), 3.j)
    op2 = _qubit_operator.QubitOperator(((1, 'X'), (3, 'Y'), (8, 'Z')), 0.5)
    op3 = op1 * op2
    correct_coefficient = 1.j * 3.0j * 0.5
    correct_term = ((0, 'Y'), (1, 'X'), (3, 'Z'), (11, 'X'))
    assert op1.isclose(_qubit_operator.QubitOperator(
        ((0, 'Y'), (3, 'X'), (8, 'Z'), (11, 'X')), 3.j))
    assert op2.isclose(_qubit_operator.QubitOperator(((1, 'X'), (3, 'Y'),
                                                     (8, 'Z')), 0.5))
    assert op3.isclose(_qubit_operator.QubitOperator(correct_term,
                                                    correct_coefficient))


def test_mul_npfloat64():
    op = _qubit_operator.QubitOperator(((1, 'X'), (3, 'Y')), 0.5)
    res = op * numpy.float64(0.5)
    assert res.isclose(_qubit_operator.QubitOperator(((1, 'X'), (3, 'Y')),
                                                    0.5 * 0.5))


def test_mul_multiple_terms():
    op = _qubit_operator.QubitOperator(((1, 'X'), (3, 'Y'), (8, 'Z')), 0.5)
    op += _qubit_operator.QubitOperator(((1, 'Z'), (3, 'X'), (8, 'Z')), 1.2)
    op += _qubit_operator.QubitOperator(((1, 'Z'), (3, 'Y'), (9, 'Z')), 1.4j)
    res = op * op
    correct = _qubit_operator.QubitOperator((), 0.5 ** 2 + 1.2 ** 2 + 1.4j ** 2)
    correct += _qubit_operator.QubitOperator(((1, 'Y'), (3, 'Z')),
                                2j * 1j * 0.5 * 1.2)
    assert res.isclose(correct)


@pytest.mark.parametrize("multiplier", [0.5, 0.6j, numpy.float64(2.303),
                         numpy.complex128(-1j)])
def test_rmul_scalar(multiplier):
    op = _qubit_operator.QubitOperator(((1, 'X'), (3, 'Y'), (8, 'Z')), 0.5)
    res1 = op * multiplier
    res2 = multiplier * op
    assert res1.isclose(res2)


def test_rmul_bad_multiplier():
    op = _qubit_operator.QubitOperator(((1, 'X'), (3, 'Y'), (8, 'Z')), 0.5)
    with pytest.raises(TypeError):
        op = "0.5" * op


@pytest.mark.parametrize("divisor", [0.5, 0.6j, numpy.float64(2.303),
                         numpy.complex128(-1j), 2])
def test_truediv_and_div(divisor):
    op = _qubit_operator.QubitOperator(((1, 'X'), (3, 'Y'), (8, 'Z')), 0.5)
    original = copy.deepcopy(op)
    res = op / divisor
    correct = op * (1. / divisor)
    assert res.isclose(correct)
    # Test if done out of place
    assert op.isclose(original)


def test_truediv_bad_divisor():
    op = _qubit_operator.QubitOperator(((1, 'X'), (3, 'Y'), (8, 'Z')), 0.5)
    with pytest.raises(TypeError):
        op = op / "0.5"


@pytest.mark.parametrize("divisor", [0.5, 0.6j, numpy.float64(2.303),
                         numpy.complex128(-1j), 2])
def test_itruediv_and_idiv(divisor):
    op = _qubit_operator.QubitOperator(((1, 'X'), (3, 'Y'), (8, 'Z')), 0.5)
    original = copy.deepcopy(op)
    correct = op * (1. / divisor)
    op /= divisor
    assert op.isclose(correct)
    # Test if done in-place
    assert not op.isclose(original)


def test_itruediv_bad_divisor():
    op = _qubit_operator.QubitOperator(((1, 'X'), (3, 'Y'), (8, 'Z')), 0.5)
    with pytest.raises(TypeError):
        op /= "0.5"


def test_iadd_different_term():
    term_a = ((1, 'X'), (3, 'Y'), (8, 'Z'))
    term_b = ((1, 'Z'), (3, 'Y'), (8, 'Z'))
    a = _qubit_operator.QubitOperator(term_a, 1.0)
    a += _qubit_operator.QubitOperator(term_b, 0.5)
    assert len(a.terms) == 2
    assert a.terms[term_a] == pytest.approx(1.0)
    assert a.terms[term_b] == pytest.approx(0.5)
    a += _qubit_operator.QubitOperator(term_b, 0.5)
    assert len(a.terms) == 2
    assert a.terms[term_a] == pytest.approx(1.0)
    assert a.terms[term_b] == pytest.approx(1.0)


def test_iadd_bad_addend():
    op = _qubit_operator.QubitOperator((), 1.0)
    with pytest.raises(TypeError):
        op += "0.5"


def test_add():
    term_a = ((1, 'X'), (3, 'Y'), (8, 'Z'))
    term_b = ((1, 'Z'), (3, 'Y'), (8, 'Z'))
    a = _qubit_operator.QubitOperator(term_a, 1.0)
    b = _qubit_operator.QubitOperator(term_b, 0.5)
    res = a + b + b
    assert len(res.terms) == 2
    assert res.terms[term_a] == pytest.approx(1.0)
    assert res.terms[term_b] == pytest.approx(1.0)
    # Test out of place
    assert a.isclose(_qubit_operator.QubitOperator(term_a, 1.0))
    assert b.isclose(_qubit_operator.QubitOperator(term_b, 0.5))


def test_add_bad_addend():
    op = _qubit_operator.QubitOperator((), 1.0)
    with pytest.raises(TypeError):
        op = op + "0.5"


def test_sub():
    term_a = ((1, 'X'), (3, 'Y'), (8, 'Z'))
    term_b = ((1, 'Z'), (3, 'Y'), (8, 'Z'))
    a = _qubit_operator.QubitOperator(term_a, 1.0)
    b = _qubit_operator.QubitOperator(term_b, 0.5)
    res = a - b
    assert len(res.terms) == 2
    assert res.terms[term_a] == pytest.approx(1.0)
    assert res.terms[term_b] == pytest.approx(-0.5)
    res2 = b - a
    assert len(res2.terms) == 2
    assert res2.terms[term_a] == pytest.approx(-1.0)
    assert res2.terms[term_b] == pytest.approx(0.5)


def test_sub_bad_subtrahend():
    op = _qubit_operator.QubitOperator((), 1.0)
    with pytest.raises(TypeError):
        op = op - "0.5"


def test_isub_different_term():
    term_a = ((1, 'X'), (3, 'Y'), (8, 'Z'))
    term_b = ((1, 'Z'), (3, 'Y'), (8, 'Z'))
    a = _qubit_operator.QubitOperator(term_a, 1.0)
    a -= _qubit_operator.QubitOperator(term_b, 0.5)
    assert len(a.terms) == 2
    assert a.terms[term_a] == pytest.approx(1.0)
    assert a.terms[term_b] == pytest.approx(-0.5)
    a -= _qubit_operator.QubitOperator(term_b, 0.5)
    assert len(a.terms) == 2
    assert a.terms[term_a] == pytest.approx(1.0)
    assert a.terms[term_b] == pytest.approx(-1.0)


def test_isub_bad_addend():
    op = _qubit_operator.QubitOperator((), 1.0)
    with pytest.raises(TypeError):
        op -= "0.5"


def test_neg():
    op = _qubit_operator.QubitOperator(((1, 'X'), (3, 'Y'), (8, 'Z')), 0.5)
    -op
    # out of place
    assert op.isclose(_qubit_operator.QubitOperator(((1, 'X'), (3, 'Y'),
                                                     (8, 'Z')), 0.5))
    correct = -1.0 * op
    assert correct.isclose(-op)


def test_isidentity_identity():
    assert _qubit_operator.QubitOperator().is_identity()


def test_isidentity_mulidentity():
    op = _qubit_operator.QubitOperator() * 2
    assert op.is_identity()


def test_isidentity_zero():
    op = (0 * _qubit_operator.QubitOperator() +
          _qubit_operator.QubitOperator('X2', 0.0))
    assert op.is_identity()


def test_isidentity_zeroX():
    op = (-2 * _qubit_operator.QubitOperator() +
          _qubit_operator.QubitOperator('X2', 0.0))
    assert op.is_identity()


def test_isidentity_IX():
    op = (-2 * _qubit_operator.QubitOperator() +
          _qubit_operator.QubitOperator('X2', 0.03j))
    assert not op.is_identity()


def test_str():
    op = _qubit_operator.QubitOperator(((1, 'X'), (3, 'Y'), (8, 'Z')), 0.5)
    assert str(op) == "0.5 X1 Y3 Z8\n"
    op2 = _qubit_operator.QubitOperator((), 2)
    assert str(op2) == "2 I\n"


def test_rep():
    op = _qubit_operator.QubitOperator(((1, 'X'), (3, 'Y'), (8, 'Z')), 0.5)
    # Not necessary, repr could do something in addition
    assert repr(op) == str(op)
