import projectq

from fermilib.qubit_operators import QubitTerm, QubitOperator
from math import sqrt


def error_operator(terms, series_order=2):
    """Determine the difference between the exact generator of unitary
    evolution and the approximate generator given by Trotter-Suzuki
    to the given order.

    Args:
      terms: a list of QubitTerms in the Hamiltonian to be simulated.
      series_order: the order at which to compute the BCH expansion.
                    Only the second order formula is currently
                    implemented (corresponding to Equation 9 of the
                    paper).

    Returns:
      The difference between the true and effective generators of time
      evolution for a single Trotter step.

    Notes: follows Equation 9 of Poulin et al.'s work in "The Trotter
           Step Size Required for Accurate Quantum Simulation of Quantum
           Chemistry".

      """

    if series_order != 2:
        raise NotImplementedError

    error_op = QubitOperator()

    for alpha in range(len(terms)):
        for beta in range(alpha, len(terms)):
            for alpha_prime in range(beta - 1):
                double_com = terms[alpha].commutator(
                    terms[beta].commutator(terms[alpha_prime]))
                error_op += double_com
                if alpha == beta:
                    error_op -= double_com / 2.0

    return -error_op / 12.0


def error_bound(terms, tight=False):
    """Numerically upper bound the error in the ground state energy
    for the second order Trotter-Suzuki expansion.

    Args:
      terms: a list of QubitTerms in the Hamiltonian to be simulated.
      tight: whether to use the triangle inequality to give a loose
             upper bound on the error (default) or to calculate the
             norm of the error operator.

    Returns:
      A float upper bound on the norm of the error in the ground state
      energy.

    Notes: follows Poulin et al.'s work in "The Trotter Step Size
           Required for Accurate Quantum Simulation of Quantum
           Chemistry". In particular, Equation 16 is used for a loose
           upper bound, and the norm of Equation 9 is calculated for
           a tighter bound using the error operator from error_operator.

           Possible extensions of this function would be to get the
           expectation value of the error operator with the Hartree-Fock
           state or CISD state, which can scalably bound the error in
           the ground state but much more accurately than the triangle
           inequality.

    """
    zero = QubitOperator()
    error = 0.0

    if tight:
        # return the Frobenius norm of the error operator
        # (upper bound on error)
        for term in error_operator(terms):
            error += abs(term.coefficient) ** 2
        error = sqrt(error)

    elif not tight:
        for term_a in terms:
            error_a = sum([abs(term_b.coefficient) for term_b in terms
                           if term_a.commutator(term_b) != zero])
            error += 4.0 * abs(term_a.coefficient) * error_a ** 2

    return error

if __name__ == '__main__':
    terms = [2 * QubitTerm('X2 Y1'), QubitTerm('Z1 X2'), QubitTerm('Z2'),
             QubitTerm('Y4 Z5 Z0'), 5 * QubitTerm('Z1 Z4')]

    print error_operator(terms)
    print error_bound(terms)
    print error_bound(terms, tight=True)

    print "\nXYZ only:"
    print error_bound([QubitTerm('X1'), QubitTerm('Y1'), QubitTerm('Z1')])
    print error_bound([QubitTerm('X1'), QubitTerm('Y1'), QubitTerm('Z1')],
                       tight=True)
