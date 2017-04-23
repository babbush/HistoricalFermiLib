"""FermionOperator stores a sum of creation and annihilation operators acting
on some set of modes."""
import copy
import numpy as np


class FermionOperatorError(Exception):
    pass


def fermion_identity(coefficient=1.):
    return coefficient * FermionOperator()


def one_body_term(p, q, coefficient=1.):
    """Return one-body operator which conserves particle number.

    Args:   p, q: The sites between which the hopping occurs.
    coefficient: Optional float giving coefficient of term.

    """
    return FermionOperator(((p, 1), (q, 0)), coefficient)


def two_body_term(p, q, r, s, coefficient=1.):
    """Return two-body operator which conserves particle number.

    Args:   p, q, r, s: The sites between which the hopping occurs.
    coefficient: Optional float giving coefficient of term.

    """
    return FermionOperator(((p, 1), (q, 1), (r, 0), (s, 0)), coefficient)


def number_operator(n_qubits, site=None, coefficient=1.):
    """Return a number operator.

    Args:   n_qubits: An int giving the number of spin-orbitals in the
    system.   site: The site on which to return the number operator.
    If None, return total number operator on all sites.

    """
    if site is None:
        operator = FermionOperator((), 0.0)
        for spin_orbital in range(n_qubits):
            operator += number_operator(n_qubits, spin_orbital, coefficient)
    else:
        operator = FermionOperator(((site, 1), (site, 0)), coefficient)
    return operator


def hermitian_conjugated(fermionoperator):
    """Calculate Hermitian conjugate of fermionic term.

    Returns:   A new FermionTerm object which is the hermitian
    conjugate of this.

    """
    res = copy.deepcopy(fermionoperator)
    res.hermitian_conjugate()
    return res


class FermionOperator(object):
    """
    A sum of terms acting on qubits, e.g., 0.5 * '0^ 5' + 0.3 * '1^ 2^'.

    A term is an operator acting on n modes and can be represented as:

    coefficent * local_operator[0] x ... x local_operator[n-1]

    where x is the tensor product. A local operator is creation or
    annihilation operator acting on a single mode. In math notation a
    term is, for example, 0.5 * '0^ 5', which means that a creation
    operator acts on mode 0 and an annihilation operator acts on mode 5,
    while the identity operator acts on all other qubits.

    A FermionOperator represents a sum of terms acting on modes and
    overloads operations for easy manipulation of these objects by the
    user.

    Attributes:
      terms (dict): key: A term represented by a tuple of tuples. Each
                         tuple represents a local operator and is an
                         operator which acts on one mode stored as a
                         tuple. The first element is an integer
                         indicating the mode on which a non-trivial
                         local operator acts and the second element is a
                         string, either '0' indicating annihilation, or
                         '1' indicating creation in that mode.
                         E.g. '2^ 5' is ((2, 1), (5, 0))
                         The tuples are sorted according to the qubit number
                         they act on, starting from 0.
                    value: Coefficient of this term as a (complex) float
    """

    def __init__(self, term=(), coefficient=1.):
        """
        Inits a FermionOperator.

        The init function only allows to initialize one term. Additional
        terms have to be added using += (which is fast) or using + of
        two FermionOperator objects:

        Example:
            .. code-block:: python

            ham = (QubitOperator('0^ 3', 0.5) + 0.5 * QubitOperator('3^ 0'))
            # Equivalently
            ham2 = QubitOperator('0^ 3', 0.5)
            ham2 += 0.5 * QubitOperator('3^ 0')

        Note:
            Adding terms to FermionOperator is faster using += (as this
            is done by in-place addition). Specifying the coefficient in
            the __init__ is faster than by multiplying a QubitOperator
            with a scalar as calls an out-of-place multiplication.

        Args:
            coefficient (complex float, optional): The coefficient of the
                first term of this FermionOperator. Default is 1.0.
            term (tuple of tuples, a string, or optional):
                1) A tuple of tuples. The first element of each tuple is
                   an integer indicating the mode on which a non-trivial
                   local operator acts, starting from zero. The second
                   element of each tuple is an integer, either 1 or 0,
                   indicating whether creation or annihilation acts on
                   that mode.
                2) A string of the form '0^ 2', indicating creation in
                   mode 0 and annihilation in mode 2. The string should
                   be sorted by the mode number.
                3) default will result in identity operations on all
                   modes, which is just an empty tuple '()'.

        Raises:
          FermionOperatorError: Invalid term provided to FermionOperator.

        """
        if term is not None and not isinstance(term, (tuple, str)):
            raise ValueError('Operators specified incorrectly.')
        if not isinstance(coefficient, (int, float, complex)):
            raise ValueError('Coefficient must be scalar.')

        self.terms = {}

        # Parse string input.
        if isinstance(term, str):
            list_ops = []
            for el in term.split():
                if el[-1] == '^':
                    list_ops.append((int(el[:-1]), 1))
                else:
                    try:
                        list_ops.append((int(el), 0))
                    except ValueError:
                        raise ValueError(
                            'Invalid action provided to FermionTerm.')
            self.terms[tuple(list_ops)] = coefficient

        # Tuple input.
        elif isinstance(term, tuple) and len(term) != 0:
            term = list(term)
            self.terms[tuple(term)] = coefficient
        else:
            self.terms[()] = coefficient

        # Check type.
        for term in self.terms:
            for operator in term:
                tensor_factor, action = operator
                if not (isinstance(tensor_factor, int) and tensor_factor >= 0):
                    raise FermionOperatorError('Invalid tensor factor provided'
                                               ' to FermionTerm: must be a '
                                               'non-negative int.')
                if action not in (0, 1):
                    raise ValueError('Invalid action provided to FermionTerm. '
                                     'Must be 0 (lowering) or 1 (raising).')

    def n_qubits(self):
        """Return the minimum number of qubits this FermionOperator must
        act on."""
        highest_mode = 0
        for term in self.terms:
            term_modes = max(term or ((-1, -1),), key=lambda t: t[0])[0] + 1
            if term_modes > highest_mode:
                highest_mode = term_modes
        return highest_mode

    def isclose(self, other, rel_tol=1e-12, abs_tol=1e-12):
        """
        Returns True if other (FermionOperator) is close to self.

        Comparison is done for each term individually. Return True
        if the difference between each terms in self and other is
        less than the relative tolerance w.r.t. either other or self
        (symmetric test) or if the difference is less than the absolute
        tolerance.

        Args:
            other(FermionOperator): FermionOperator to compare against.
            rel_tol(float): Relative tolerance, must be greater than 0.0
            abs_tol(float): Absolute tolerance, must be at least 0.0

        """
        # terms which are in both:
        for term in set(self.terms).intersection(set(other.terms)):
            a = self.terms[term]
            b = other.terms[term]
            # math.isclose does this in Python >=3.5
            if not abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol):
                return False
        # terms only in one (compare to 0.0 so only abs_tol)
        for term in set(self.terms).symmetric_difference(set(other.terms)):
            if term in self.terms:
                if not abs(self.terms[term]) <= abs_tol:
                    return False
            elif not abs(other.terms[term]) <= abs_tol:
                return False
        return True

    def __imul__(self, multiplier):
        """
        In-place multiply (*=) terms with scalar or FermionOperator.

        Args:
          multiplier(complex float, or FermionOperator): multiplier

        """
        # Handle scalars.
        if isinstance(multiplier, (int, float, complex)):
            for term in self.terms:
                self.terms[term] *= multiplier
            return self

        # Handle FermionOperator.
        elif isinstance(multiplier, FermionOperator):
            result_terms = dict()
            for left_term in self.terms:
                for right_term in multiplier.terms:
                    new_coefficient = (self.terms[left_term] *
                                       multiplier.terms[right_term])

                    product_operators = left_term + right_term

                    # Add to result dict
                    result_terms[tuple(product_operators)] = new_coefficient
            self.terms = result_terms
            return self
        else:
            raise TypeError('Cannot in-place multiply term of invalid type '
                            'to FermionOperator.')

    def __mul__(self, multiplier):
        """
        Return self * multiplier for a scalar, or a FermionOperator.

        Args:
          multiplier: A scalar, or a FermionOperator.

        Returns:
          product: A FermionOperator.

        Raises:
          TypeError: Invalid type cannot be multiply with FermionOperator.

        """
        if isinstance(multiplier, (int, float, complex, FermionOperator)):
            product = copy.deepcopy(self)
            product *= multiplier
            return product
        else:
            raise TypeError(
                'Object of invalid type cannot multiply with FermionOperator.')

    def __rmul__(self, multiplier):
        """
        Return multiplier * self for a scalar.

        We only define __rmul__ for scalars because the left multiply
        exist for  FermionOperator and left multiply
        is also queried as the default behavior.

        Args:
          multiplier: A scalar to multiply by.

        Returns:
          product: A new instance of FermionOperator.

        Raises:
          TypeError: Object of invalid type cannot multiply FermionOperator.

        """
        if not isinstance(multiplier, (int, float, complex)):
            raise TypeError(
                'Object of invalid type cannot multiply with FermionOperator.')
        return self * multiplier

    def __div__(self, divisor):
        """
        Return self / divisor for a scalar.

        Note:
            This is always floating point division.

        Args:
          divisor: A scalar to divide by.

        Returns:
          A new instance of FermionOperator.

        Raises:
          TypeError: Cannot divide local operator by non-scalar type.

        """
        if not isinstance(divisor, (int, float, complex)):
            raise TypeError('Cannot divide QubitOperator by non-scalar type.')
        return self * (1.0 / divisor)

    def __idiv__(self, divisor):
        self *= (1.0 / divisor)
        return self

    def __iadd__(self, addend):
        """In-place method for += addition of FermionOperator.

        Args:
          addend: A FermionOperator.

        Raises:
          TypeError: Cannot add invalid type.

        """
        if isinstance(addend, FermionOperator):
            for term in addend.terms:
                if term in self.terms:
                    self.terms[term] += addend.terms[term]
                else:
                    self.terms[term] = addend.terms[term]
        else:
            raise TypeError('Cannot add invalid type to FermionOperator.')
        return self

    def __add__(self, addend):
        """ Return self + addend for a FermionOperator. """
        summand = copy.deepcopy(self)
        summand += addend
        return summand

    def __sub__(self, subtrahend):
        """Return self - subtrahend for a FermionOperator."""
        if not isinstance(subtrahend, FermionOperator):
            raise TypeError('Cannot subtract invalid type to FermionOperator.')
        return self + (-1. * subtrahend)

    def __neg__(self):
        return -1 * self

    def hermitian_conjugate(self):
        """Hermitian conjugate this fermionic term."""
        conj_terms = {}
        for term in self.terms:
            conj_coeff = np.conjugate(self.terms[term])
            conj_term = list(term)
            conj_term.reverse()
            for local_op in range(len(conj_term)):
                conj_term[local_op] = (conj_term[local_op][0],
                                       1 - conj_term[local_op][1])

            conj_terms[tuple(conj_term)] = conj_coeff
        self.terms = conj_terms

    def is_normal_ordered(self):
        """Return whether or not term is in normal order.

        In our convention, normal ordering implies terms are ordered
        from highest tensor factor (on left) to lowest (on right). Also,
        ladder operators come first.

        """
        n_qubits = self.n_qubits()
        for term in self.terms:
            creating = True  # normal ordered must start with creation ops
            pos = n_qubits
            for i in range(len(term)):
                if creating and not term[i][1]:
                    creating = False
                    pos = term[i][0]
                elif term[i][0] >= pos or term[i][1] != creating:
                    return False
                pos = term[i][0]
        return True

    def normal_ordered(self):
        """Compute and return the normal ordered form of a FermionOperator.

        Not an in-place method.

        In our convention, normal ordering implies terms are ordered
        from highest tensor factor (on left) to lowest (on right).
        Also, ladder operators come first.

        Returns:
          FermionOperator object in normal ordered form.

        Warning:
          Even assuming that each creation or annihilation operator appears
          at most a constant number of times in the original term, the
          runtime of this method is exponential in the number of qubits.

        """
        # Initialize output.
        normal_ordered_op = FermionOperator((), 0.0)

        for term in self.terms:
            new_ops = list(term)
            new_term = FermionOperator(tuple(new_ops), self.terms[term])
            new_coeff = self.terms[term]

            # Iterate from left to right across operators and reorder to normal
            # form. Swap terms operators into correct position by moving left
            # to right.
            for i in range(1, len(new_ops)):
                for j in range(i, 0, -1):
                    right_op = new_ops[j]
                    left_op = new_ops[j - 1]

                    # Swap operators if raising on right and lowering on left.
                    if right_op[1] and not left_op[1]:
                        new_ops[j-1], new_ops[j] = right_op, left_op
                        new_coeff *= -1

                        # Replace a a^\dagger with 1 - a^\dagger a if indices
                        # are same.
                        if right_op[0] == left_op[0]:
                            new_ops2 = new_ops[:j-1] + new_ops[j+1:]
                            new_coeff2 = -new_coeff
                            new_term2 = FermionOperator(tuple(new_ops2),
                                                        new_coeff2)

                            # Recursively add the processed new term.
                            normal_ordered_op += new_term2.normal_ordered()

                    # Handle case when operator type is the same.
                    elif right_op[1] == left_op[1]:

                        # If same two operators are repeated, term evaluates to
                        # zero.
                        if right_op[0] == left_op[0]:
                            new_coeff = 0.0

                        # Swap if same ladder type but lower index on left.
                        elif right_op[0] > left_op[0]:
                            new_ops[j-1], new_ops[j] = right_op, left_op
                            new_coeff *= -1

            # Add processed new term to output and return.
            if new_coeff:
                normal_ordered_op += FermionOperator(tuple(new_ops), new_coeff)

        return normal_ordered_op

    def is_molecular_term(self):
        """Query whether term has correct form to be from a molecular.

        Require that term is particle-number conserving (same number of
        raising and lowering operators). Require that term has 0, 2 or 4
        ladder operators. Require that term conserves spin (parity of
        raising operators equals parity of lowering operators).

        """
        for term in self.terms:
            if len(term) not in (0, 2, 4):
                return False

            # Make sure term conserves particle number and spin.
            spin = 0
            particles = 0
            for operator in term:
                particles += (-1) ** operator[1]  # add 1 if create, else -1
                spin += (-1) ** (operator[0] + operator[1])
            if not (particles == spin == 0):
                return False

        return True

    def __str__(self):
        """Return an easy-to-read string representation."""
        string_rep = ''
        for term in self.terms:
            tmp_string = '{} ['.format(self.terms[term])
            for operator in term:
                if operator[1] == 1:
                    tmp_string += '{}^ '.format(operator[0])
                elif operator[1] == 0:
                    tmp_string += '{} '.format(operator[0])
            string_rep += '{}]\n'.format(tmp_string.strip())
        return string_rep

    def __repr__(self):
        return str(self)