"""Tests  _fermion_operator.py."""
import copy
import numpy
import pytest

from ._fermion_operator import (fermion_identity,
                                FermionOperator,
                                FermionOperatorError,
                                hermitian_conjugated,
                                number_operator,
                                one_body_term,
                                two_body_term)


def test_init_defaults():
    loc_op = FermionOperator()
    assert len(loc_op.terms) == 1
    assert loc_op.terms[()] == 1.0


@pytest.mark.parametrize("coefficient", [0.5, 0.6j, numpy.float64(2.303),
                         numpy.complex128(-1j)])
def test_init_tuple(coefficient):
    loc_op = ((0, 1), (5, 0), (6, 1))
    fermion_op = FermionOperator(loc_op, coefficient)
    assert len(fermion_op.terms) == 1
    assert fermion_op.terms[loc_op] == coefficient


def test_init_str():
    fermion_op = FermionOperator('0^ 5 12^', -1.)
    correct = ((0, 1), (5, 0), (12, 1))
    assert correct in fermion_op.terms
    assert fermion_op.terms[correct] == -1.0


def test_init_str_identity():
    fermion_op = FermionOperator('')
    assert () in fermion_op.terms


def test_init_bad_term():
    with pytest.raises(ValueError):
        fermion_op = FermionOperator(list())


def test_init_bad_coefficient():
    with pytest.raises(ValueError):
        fermion_op = FermionOperator('0^', "0.5")


def test_init_bad_action_str():
    with pytest.raises(ValueError):
        fermion_op = FermionOperator('0-')


def test_init_bad_action_tuple():
    with pytest.raises(ValueError):
        FermionOperator(((0, 2),))


def test_init_bad_tuple():
    with pytest.raises(ValueError):
        fermion_op = FermionOperator(((0, 1, 1),))


def test_init_bad_str():
    with pytest.raises(ValueError):
        fermion_op = FermionOperator('^')


def test_init_bad_mode_num():
    with pytest.raises(FermionOperatorError):
        fermion_op = FermionOperator('-1^')


def test_fermion_identity():
    op = fermion_identity(3.)
    assert op.isclose(FermionOperator() * 3.)


def test_one_body_term():
    op = one_body_term(1, 3, coefficient=-1j+2)
    assert op.isclose((2 - 1j) * FermionOperator(((1, 1), (3, 0))))


def test_two_body_term():
    op = two_body_term(4, 11, 7, 4, 0.5)
    assert op.isclose(0.5 * FermionOperator(((4, 1), (11, 1),
                                             (7, 0), (4, 0))))


def test_number_operator_site():
    op = number_operator(3, 2, 1j)
    assert op.isclose(FermionOperator(((2, 1), (2, 0))) * 1j)


def test_number_operator_nosite():
    op = number_operator(4)
    expected = (FermionOperator(((0, 1), (0, 0))) +
                FermionOperator(((1, 1), (1, 0))) +
                FermionOperator(((2, 1), (2, 0))) +
                FermionOperator(((3, 1), (3, 0))))
    assert op.isclose(expected)


def test_nqubits_0():
    op = FermionOperator()
    assert op.n_qubits() == 0


def test_nqubits_1():
    op = FermionOperator('0', 3)
    assert op.n_qubits() == 1


def test_nqubits_doubledigit():
    op = FermionOperator('27 5^ 11^')
    assert op.n_qubits() == 28


def test_nqubits_multiterm():
    op = (FermionOperator() + FermionOperator('1 2 3') +
          FermionOperator())
    assert op.n_qubits() == 4


def test_isclose_abs_tol():
    a = FermionOperator('0^', -1.)
    b = FermionOperator('0^', -1.05)
    c = FermionOperator('0^', -1.11)
    assert a.isclose(b, rel_tol=1e-14, abs_tol=0.1)
    assert not a.isclose(c, rel_tol=1e-14, abs_tol=0.1)
    a = FermionOperator('0^', -1.0j)
    b = FermionOperator('0^', -1.05j)
    c = FermionOperator('0^', -1.11j)
    assert a.isclose(b, rel_tol=1e-14, abs_tol=0.1)
    assert not a.isclose(c, rel_tol=1e-14, abs_tol=0.1)


def test_isclose_rel_tol():
    a = FermionOperator('0', 1)
    b = FermionOperator('0', 2)
    assert a.isclose(b, rel_tol=2.5, abs_tol=0.1)
    # Test symmetry
    assert a.isclose(b, rel_tol=1, abs_tol=0.1)
    assert b.isclose(a, rel_tol=1, abs_tol=0.1)


def test_isclose_zero_terms():
    op = FermionOperator('1^ 0', -1j) * 0
    assert op.isclose(FermionOperator((), 0.0),
                      rel_tol=1e-12, abs_tol=1e-12)
    assert FermionOperator((), 0.0).isclose(
        op, rel_tol=1e-12, abs_tol=1e-12)


def test_isclose_different_terms():
    a = FermionOperator(((1, 0),), -0.1j)
    b = FermionOperator(((1, 1),), -0.1j)
    assert a.isclose(b, rel_tol=1e-12, abs_tol=0.2)
    assert not a.isclose(b, rel_tol=1e-12, abs_tol=0.05)
    assert b.isclose(a, rel_tol=1e-12, abs_tol=0.2)
    assert not b.isclose(a, rel_tol=1e-12, abs_tol=0.05)


def test_isclose_different_num_terms():
    a = FermionOperator(((1, 0),), -0.1j)
    a += FermionOperator(((1, 1),), -0.1j)
    b = FermionOperator(((1, 0),), -0.1j)
    assert not b.isclose(a, rel_tol=1e-12, abs_tol=0.05)
    assert not a.isclose(b, rel_tol=1e-12, abs_tol=0.05)


def test_imul_inplace():
    fermion_op = FermionOperator("1^")
    prev_id = id(fermion_op)
    fermion_op *= 3.
    assert id(fermion_op) == prev_id
    print(fermion_op.terms.keys())
    assert fermion_op.terms[((1, 1),)] == 3.


@pytest.mark.parametrize("multiplier", [0.5, 0.6j, numpy.float64(2.303),
                         numpy.complex128(-1j)])
def test_imul_scalar(multiplier):
    loc_op = ((1, 0), (2, 1))
    fermion_op = FermionOperator(loc_op)
    fermion_op *= multiplier
    assert fermion_op.terms[loc_op] == pytest.approx(multiplier)


def test_imul_fermion_op():
    op1 = FermionOperator(((0, 1), (3, 0), (8, 1), (8, 0), (11, 1)), 3.j)
    op2 = FermionOperator(((1, 1), (3, 1), (8, 0)), 0.5)
    op1 *= op2
    correct_coefficient = 1.j * 3.0j * 0.5
    correct_term = ((0, 1), (3, 0), (8, 1), (8, 0), (11, 1),
                    (1, 1), (3, 1), (8, 0))
    assert len(op1.terms) == 1
    assert correct_term in op1.terms


def test_imul_fermion_op_2():
    op3 = FermionOperator(((1, 1), (0, 0)), -1j)
    op4 = FermionOperator(((1, 0), (0, 1), (2, 1)), -1.5)
    op3 *= op4
    op4 *= op3
    assert ((1, 1), (0, 0), (1, 0), (0, 1), (2, 1)) in op3.terms
    assert op3.terms[((1, 1), (0, 0), (1, 0), (0, 1), (2, 1))] == 1.5j


def test_imul_bidir():
    op_a = FermionOperator(((1, 1), (0, 0)), -1j)
    op_b = FermionOperator(((1, 1), (0, 1), (2, 1)), -1.5)
    op_a *= op_b
    op_b *= op_a
    assert ((1, 1), (0, 0), (1, 1), (0, 1), (2, 1)) in op_a.terms
    assert op_a.terms[((1, 1), (0, 0), (1, 1), (0, 1), (2, 1))] == 1.5j
    assert (((1, 1), (0, 1), (2, 1),
             (1, 1), (0, 0), (1, 1), (0, 1), (2, 1)) in op_b.terms)
    assert op_b.terms[((1, 1), (0, 1), (2, 1),
                       (1, 1), (0, 0),
                       (1, 1), (0, 1), (2, 1))] == -2.25j


def test_imul_bad_multiplier():
    op = FermionOperator(((1, 1), (0, 1)), -1j)
    with pytest.raises(TypeError):
        op *= "1"


def test_mul_by_scalarzero():
    op = FermionOperator(((1, 1), (0, 1)), -1j) * 0
    assert ((0, 1), (1, 1)) not in op.terms
    assert ((1, 1), (0, 1)) in op.terms
    assert op.terms[((1, 1), (0, 1))] == pytest.approx(0.0)


def test_mul_bad_multiplier():
    op = FermionOperator(((1, 1), (0, 1)), -1j)
    with pytest.raises(TypeError):
        op = op * "0.5"


def test_mul_out_of_place():
    op1 = FermionOperator(((0, 1), (3, 1), (3, 0), (11, 1)), 3.j)
    op2 = FermionOperator(((1, 1), (3, 1), (8, 0)), 0.5)
    op3 = op1 * op2
    correct_coefficient = 3.0j * 0.5
    correct_term = ((0, 1), (3, 1), (3, 0), (11, 1), (1, 1), (3, 1), (8, 0))
    assert op1.isclose(FermionOperator(
        ((0, 1), (3, 1), (3, 0), (11, 1)), 3.j))
    assert op2.isclose(FermionOperator(((1, 1), (3, 1), (8, 0)), 0.5))
    assert op3.isclose(FermionOperator(correct_term, correct_coefficient))


def test_mul_npfloat64():
    op = FermionOperator(((1, 0), (3, 1)), 0.5)
    res = op * numpy.float64(0.5)
    assert res.isclose(FermionOperator(((1, 0), (3, 1)), 0.5 * 0.5))


def test_mul_multiple_terms():
    op = FermionOperator(((1, 0), (8, 1)), 0.5)
    op += FermionOperator(((1, 1), (9, 1)), 1.4j)
    res = op * op
    correct = FermionOperator(((1, 0), (8, 1), (1, 0), (8, 1)), 0.5 ** 2)
    correct += (FermionOperator(((1, 0), (8, 1), (1, 1), (9, 1)), 0.7j) +
                FermionOperator(((1, 1), (9, 1), (1, 0), (8, 1)), 0.7j))
    correct += FermionOperator(((1, 1), (9, 1), (1, 1), (9, 1)), 1.4j ** 2)
    assert res.isclose(correct)


@pytest.mark.parametrize("multiplier", [0.5, 0.6j, numpy.float64(2.303),
                         numpy.complex128(-1j)])
def test_rmul_scalar(multiplier):
    op = FermionOperator(((1, 1), (3, 0), (8, 1)), 0.5)
    res1 = op * multiplier
    res2 = multiplier * op
    assert res1.isclose(res2)


def test_rmul_bad_multiplier():
    op = FermionOperator(((1, 1), (3, 0), (8, 1)), 0.5)
    with pytest.raises(TypeError):
        op = "0.5" * op


@pytest.mark.parametrize("divisor", [0.5, 0.6j, numpy.float64(2.303),
                         numpy.complex128(-1j), 2])
def test_truediv_and_div(divisor):
    op = FermionOperator(((1, 1), (3, 0), (8, 1)), 0.5)
    original = copy.deepcopy(op)
    res = op / divisor
    correct = op * (1. / divisor)
    assert res.isclose(correct)
    # Test if done out of place
    assert op.isclose(original)


def test_truediv_bad_divisor():
    op = FermionOperator(((1, 1), (3, 0), (8, 1)), 0.5)
    with pytest.raises(TypeError):
        op = op / "0.5"


@pytest.mark.parametrize("divisor", [0.5, 0.6j, numpy.float64(2.303),
                         numpy.complex128(-1j), 2])
def test_itruediv_and_idiv(divisor):
    op = FermionOperator(((1, 1), (3, 0), (8, 1)), 0.5)
    original = copy.deepcopy(op)
    correct = op * (1. / divisor)
    op /= divisor
    assert op.isclose(correct)
    # Test if done in-place
    assert not op.isclose(original)


def test_itruediv_bad_divisor():
    op = FermionOperator(((1, 1), (3, 0), (8, 1)), 0.5)
    with pytest.raises(TypeError):
        op /= "0.5"


def test_iadd_different_term():
    term_a = ((1, 1), (3, 0), (8, 1))
    term_b = ((1, 1), (3, 1), (8, 0))
    a = FermionOperator(term_a, 1.0)
    a += FermionOperator(term_b, 0.5)
    assert len(a.terms) == 2
    assert a.terms[term_a] == pytest.approx(1.0)
    assert a.terms[term_b] == pytest.approx(0.5)
    a += FermionOperator(term_b, 0.5)
    assert len(a.terms) == 2
    assert a.terms[term_a] == pytest.approx(1.0)
    assert a.terms[term_b] == pytest.approx(1.0)


def test_iadd_bad_addend():
    op = FermionOperator((), 1.0)
    with pytest.raises(TypeError):
        op += "0.5"


def test_add():
    term_a = ((1, 1), (3, 0), (8, 1))
    term_b = ((1, 0), (3, 0), (8, 1))
    a = FermionOperator(term_a, 1.0)
    b = FermionOperator(term_b, 0.5)
    res = a + b + b
    assert len(res.terms) == 2
    assert res.terms[term_a] == pytest.approx(1.0)
    assert res.terms[term_b] == pytest.approx(1.0)
    # Test out of place
    assert a.isclose(FermionOperator(term_a, 1.0))
    assert b.isclose(FermionOperator(term_b, 0.5))


def test_add_bad_addend():
    op = FermionOperator((), 1.0)
    with pytest.raises(TypeError):
        op = op + "0.5"


def test_sub():
    term_a = ((1, 1), (3, 1), (8, 1))
    term_b = ((1, 0), (3, 1), (8, 1))
    a = FermionOperator(term_a, 1.0)
    b = FermionOperator(term_b, 0.5)
    res = a - b
    assert len(res.terms) == 2
    assert res.terms[term_a] == pytest.approx(1.0)
    assert res.terms[term_b] == pytest.approx(-0.5)
    res2 = b - a
    assert len(res2.terms) == 2
    assert res2.terms[term_a] == pytest.approx(-1.0)
    assert res2.terms[term_b] == pytest.approx(0.5)


def test_sub_bad_subtrahend():
    op = FermionOperator((), 1.0)
    with pytest.raises(TypeError):
        op = op - "0.5"


def test_isub_different_term():
    term_a = ((1, 1), (3, 1), (8, 0))
    term_b = ((1, 0), (3, 1), (8, 1))
    a = FermionOperator(term_a, 1.0)
    a -= FermionOperator(term_b, 0.5)
    assert len(a.terms) == 2
    assert a.terms[term_a] == pytest.approx(1.0)
    assert a.terms[term_b] == pytest.approx(-0.5)
    a -= FermionOperator(term_b, 0.5)
    assert len(a.terms) == 2
    assert a.terms[term_a] == pytest.approx(1.0)
    assert a.terms[term_b] == pytest.approx(-1.0)


def test_isub_bad_addend():
    op = FermionOperator((), 1.0)
    with pytest.raises(TypeError):
        op -= "0.5"


def test_neg():
    op = FermionOperator(((1, 1), (3, 1), (8, 1)), 0.5)
    -op
    # out of place
    assert op.isclose(FermionOperator(((1, 1), (3, 1), (8, 1)), 0.5))
    correct = -1.0 * op
    assert correct.isclose(-op)


def test_pow_square_term():
    coeff = 6.7j
    ops = ((3, 1), (1, 0), (4, 1))
    term = FermionOperator(ops, coeff)
    squared = term ** 2
    expected = FermionOperator(ops + ops, coeff ** 2)
    assert squared.isclose(term * term)
    assert squared.isclose(expected)


def test_pow_zero_term():
    coeff = 6.7j
    ops = ((3, 1), (1, 0), (4, 1))
    term = FermionOperator(ops, coeff)
    zerod = term ** 0
    expected = fermion_identity()
    assert expected.isclose(zerod)


def test_pow_one_term():
    coeff = 6.7j
    ops = ((3, 1), (1, 0), (4, 1))
    term = FermionOperator(ops, coeff)
    assert term.isclose(term ** 1)


def test_pow_high_term():
    coeff = 6.7j
    ops = ((3, 1), (1, 0), (4, 1))
    term = FermionOperator(ops, coeff)
    high = term ** 10
    expected = FermionOperator(ops * 10, coeff ** 10)
    assert expected.isclose(high)


def test_pow_neg_error():
    with pytest.raises(ValueError):
        fermion_identity() ** -1


def test_pow_nonint_error():
    with pytest.raises(ValueError):
        FermionOperator('3 2^') ** 0.5


def test_hermitian_conjugate_empty():
    op = FermionOperator()
    op.hermitian_conjugate()
    assert op.isclose(FermionOperator())


def test_hermitian_conjugate_simple():
    op = FermionOperator('1^')
    op_hc = FermionOperator('1')
    op.hermitian_conjugate()
    assert op.isclose(op_hc)


def test_hermitian_conjugate_complex_const():
    op = FermionOperator('1^ 3', 3j)
    op_hc = -3j * FermionOperator('3^ 1')
    op.hermitian_conjugate()
    assert op.isclose(op_hc)


def test_hermitian_conjugate_notordered():
    op = FermionOperator('1 3^ 3 3^', 3j)
    op_hc = -3j * FermionOperator('3 3^ 3 1^')
    op.hermitian_conjugate()
    assert op.isclose(op_hc)


def test_hermitian_conjugate_semihermitian():
    op = (FermionOperator() + 2j * FermionOperator('1^ 3') +
          FermionOperator('3^ 1') * -2j + FermionOperator('2^ 2', 0.1j))
    op_hc = (FermionOperator() + FermionOperator('1^ 3', 2j) +
             FermionOperator('3^ 1', -2j) +
             FermionOperator('2^ 2', -0.1j))
    op.hermitian_conjugate()
    assert op.isclose(op_hc)


def test_hermitian_conjugated_empty():
    op = FermionOperator()
    assert op.isclose(hermitian_conjugated(op))


def test_hermitian_conjugated_simple():
    op = FermionOperator('0')
    op_hc = FermionOperator('0^')
    assert op_hc.isclose(hermitian_conjugated(op))


def test_hermitian_conjugated_complex_const():
    op = FermionOperator('2^ 2', 3j)
    op_hc = FermionOperator('2^ 2', -3j)
    assert op_hc.isclose(hermitian_conjugated(op))


def test_hermitian_conjugated_multiterm():
    op = FermionOperator('1^ 2') + FermionOperator('2 3 4')
    op_hc = FermionOperator('2^ 1') + FermionOperator('4^ 3^ 2^')
    assert op_hc.isclose(hermitian_conjugated(op))


def test_hermitian_conjugated_semihermitian():
    op = (FermionOperator() + 2j * FermionOperator('1^ 3') +
          FermionOperator('3^ 1') * -2j + FermionOperator('2^ 2', 0.1j))
    op_hc = (FermionOperator() + FermionOperator('1^ 3', 2j) +
             FermionOperator('3^ 1', -2j) +
             FermionOperator('2^ 2', -0.1j))
    assert op_hc.isclose(hermitian_conjugated(op))


def test_is_normal_ordered_empty():
    op = FermionOperator() * 2
    assert op.is_normal_ordered()


def test_is_normal_ordered_number():
    op = FermionOperator('2^ 2') * -1j
    assert op.is_normal_ordered()


def test_is_normal_ordered_reversed():
    assert not FermionOperator('2 2^').is_normal_ordered()


def test_is_normal_ordered_create():
    assert FermionOperator('11^').is_normal_ordered()


def test_is_normal_ordered_annihilate():
    assert FermionOperator('0').is_normal_ordered()


def test_is_normal_ordered_long_not():
    assert not FermionOperator('0 5^ 3^ 2^ 1^').is_normal_ordered()


def test_is_normal_ordered_long_descending():
    assert FermionOperator('5^ 3^ 2^ 1^ 0').is_normal_ordered()


def test_is_normal_ordered_multi():
    op = FermionOperator('4 3 2^ 2') + FermionOperator('1 2')
    assert not op.is_normal_ordered()


def test_is_normal_ordered_multiorder():
    op = FermionOperator('4 3 2 1') + FermionOperator('3 2')
    assert op.is_normal_ordered()


def test_normal_ordered_single_term():
    op = FermionOperator('4 3 2 1') + FermionOperator('3 2')
    assert op.isclose(op.normal_ordered())


def test_normal_ordered_two_term():
    op_b = FermionOperator(((2, 0), (4, 0), (2, 1)), -88.)
    normal_ordered_b = op_b.normal_ordered()
    expected = (FermionOperator(((4, 0),), 88.) +
                FermionOperator(((2, 1), (4, 0), (2, 0)), 88.))
    assert normal_ordered_b.isclose(expected)


def test_normal_ordered_number():
    number_op2 = FermionOperator(((2, 1), (2, 0)))
    assert number_op2.isclose(number_op2.normal_ordered())


def test_normal_ordered_number_reversed():
    n_term_rev2 = FermionOperator(((2, 0), (2, 1)))
    number_op2 = number_operator(3, 2)
    expected = fermion_identity() - number_op2
    assert n_term_rev2.normal_ordered().isclose(expected)


def test_normal_ordered_offsite():
    op = FermionOperator(((3, 1), (2, 0)))
    assert op.isclose(op.normal_ordered())


def test_normal_ordered_offsite_reversed():
    op = FermionOperator(((3, 0), (2, 1)))
    expected = -FermionOperator(((2, 1), (3, 0)))
    assert expected.isclose(op.normal_ordered())


def test_normal_ordered_double_create():
    op = FermionOperator(((2, 0), (3, 1), (3, 1)))
    expected = FermionOperator((), 0.0)
    assert expected.isclose(op.normal_ordered())


def test_normal_ordered_double_create_separated():
    op = FermionOperator(((3, 1), (2, 0), (3, 1)))
    expected = FermionOperator((), 0.0)
    assert expected.isclose(op.normal_ordered())


def test_normal_ordered_multi():
    op = FermionOperator(((2, 0), (1, 1), (2, 1)))
    expected = (-FermionOperator(((2, 1), (1, 1), (2, 0))) -
                FermionOperator(((1, 1),)))
    assert expected.isclose(op.normal_ordered())


def test_normal_ordered_triple():
    op_132 = FermionOperator(((1, 1), (3, 0), (2, 0)))
    op_123 = FermionOperator(((1, 1), (2, 0), (3, 0)))
    op_321 = FermionOperator(((3, 0), (2, 0), (1, 1)))

    assert op_132.isclose(-op_123.normal_ordered())
    assert op_132.isclose(op_132.normal_ordered())
    assert op_132.isclose(op_321.normal_ordered())


def test_is_molecular_term_fermion_identity():
    op = FermionOperator()
    assert op.is_molecular_term()


def test_is_molecular_term_number():
    op = number_operator(n_orbitals=5, orbital=3)
    assert op.is_molecular_term()


def test_is_molecular_term_updown():
    op = FermionOperator(((2, 1), (4, 0)))
    assert op.is_molecular_term()


def test_is_molecular_term_downup():
    op = FermionOperator(((2, 0), (4, 1)))
    assert op.is_molecular_term()


def test_is_molecular_term_downup_badspin():
    op = FermionOperator(((2, 0), (3, 1)))
    assert not op.is_molecular_term()


def test_is_molecular_term_three():
    op = FermionOperator(((0, 1), (2, 1), (4, 0)))
    assert not op.is_molecular_term()


def test_is_molecular_term_r():
    op = FermionOperator(((0, 1), (2, 0), (1, 1), (3, 0)))
    assert op.is_molecular_term()


def test_str():
    op = FermionOperator(((1, 1), (3, 0), (8, 1)), 0.5)
    assert str(op) == "0.5 [1^ 3 8^]\n"
    op2 = FermionOperator((), 2)
    print(str(op2))
    assert str(op2) == "2 []\n"


def test_rep():
    op = FermionOperator(((1, 1), (3, 0), (8, 1)), 0.5)
    # Not necessary, repr could do something in addition
    assert repr(op) == str(op)
