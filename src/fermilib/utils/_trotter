import numpy
import transforms

from future.utils import iteritems
from math import sqrt
from scipy.linalg import expm

from fermilib.config import *
from fermilib.ops import normal_ordered
from fermilib.utils import MolecularData
from projectqtemp.ops import QubitOperator
from psi4tmp import run_psi4


def commutator(op1, op2):
    return op1 * op2 - op2 * op1


def trivially_commutes(term_a, term_b):
    position_a = 0
    position_b = 0
    commutes = True

    len_a = len(term_a.terms.keys()[0])
    len_b = len(term_b.terms.keys()[0])

    while position_a < len_a and position_b < len_b:
        qubit_a = term_a.terms.keys()[0][position_a][0]
        qubit_b = term_b.terms.keys()[0][position_b][0]

        if qubit_a > qubit_b:
            position_a += 1
        elif qubit_a < qubit_b:
            position_b += 1
        else:
            action_a = term_a.terms.keys()[0][position_a][1]
            action_b = term_a.terms.keys()[0][position_b][1]
            if action_a != action_b:
                commutes = not commutes
            position_a += 1
            position_b += 1

    return bool((commutes + len_a + len_b - position_a - position_b) % 2)


def trivially_double_commutes(term_a, term_b, term_c):
    """
    Check if the double commutator [term_a, [term_b, term_c]] is
    trivially zero.

    Args:
      term_a, term_b, term_c: Single-term QubitOperators.

    Notes:
      If the sets of qubits which term_b and term_c act on, or if the
      intersection of term_a's qubits with (term_b's qubits U term_c's
      qubits) is empty, then the double commutator is trivially zero.
    """

    # determine the set of qubits each term acts on
    qubits_a = set([term_a.terms.keys()[0][i][0]
                    for i in range(len(term_a.terms.keys()[0]))])
    qubits_b = set([term_b.terms.keys()[0][i][0]
                    for i in range(len(term_b.terms.keys()[0]))])
    qubits_c = set([term_c.terms.keys()[0][i][0]
                    for i in range(len(term_c.terms.keys()[0]))])

    return not (trivially_commutes(term_b, term_c) or
                qubits_a.intersection(set(qubits_b.union(qubits_c))))


def error_operator(terms, series_order=2):
    """
    Determine the difference between the exact generator of unitary
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

    error_operator = QubitOperator((), 0.0)

    for alpha in range(len(terms)):
        if terms[alpha].terms.values()[0]:
            for beta in range(alpha, len(terms)):
                    if terms[beta].terms.values()[0]:
                        for alpha_prime in range(beta - 1):
                            if not trivially_double_commutes(
                                    terms[alpha], terms[beta],
                                    terms[alpha_prime]):
                                double_com = commutator(
                                    terms[alpha],
                                    commutator(terms[beta],
                                               terms[alpha_prime]))
                                error_operator += double_com
                                if alpha == beta:
                                    error_operator -= double_com / 2.0

    return -error_operator / 12.0


def error_bound(terms, tight=False):
    """
    Numerically upper bound the error in the ground state energy
    for the second order Trotter-Suzuki expansion.

    Args:
        terms: a list of single-term QubitOperators in the Hamiltonian
               to be simulated.
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
    zero = QubitOperator((), 0.0)
    error = 0.0

    if tight:
        # return the Frobenius norm of the error operator
        # (upper bound on error)
        for coefficient in error_operator(terms).terms.values():
            error += abs(coefficient) ** 2
        error = sqrt(error)

    elif not tight:
        for term_a in terms:
            coefficient_a = term_a.terms.values()[0]
            if coefficient_a:
                error_a = 0.

                for term_b in terms:
                    coefficient_b = term_b.terms.values()[0]
                    if coefficient_b and not trivially_commutes(term_a,
                                                                term_b):
                        error_a += abs(coefficient_b)

                error += 4.0 * abs(coefficient_a) * error_a ** 2

    return error

if __name__ == '__main__':
    terms = [2 * QubitOperator('X2 Y1'), QubitOperator('Z1 X2'),
             QubitOperator('Z2'), QubitOperator('Y4 Z5 Z0'),
             5 * QubitOperator('Z1 Z4')]

    for term_a in terms:
        for term_b in terms:
            print (str(term_a) + " and " + str(term_b) + " do" +
                   " not" * (not trivially_commutes(term_a, term_b)) +
                   " trivially commute.")

    print error_operator(terms)
    print error_bound(terms)
    print error_bound(terms, tight=True)

    print "\nXYZ only:"
    print error_bound([QubitOperator('X1'), QubitOperator('Y1'),
                       QubitOperator('Z1')])
    print error_bound([QubitOperator('X1'), QubitOperator('Y1'),
                       QubitOperator('Z1')], tight=True)

    geometry = [('Li', (0., 0., 0.)), ('H', (0., 0., 1.45))]
    basis = 'sto-3g'
    multiplicity = 1
    filename = THIS_DIRECTORY + '/tests/testdata/H1-Li1_sto-3g_singlet'
    molecule = MolecularData(geometry, basis, multiplicity, filename=filename)
    molecule.load()

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
    terms = []
    for term, coefficient in iteritems(qubit_hamiltonian.terms):
        if coefficient:
            terms.append(QubitOperator(term, coefficient))

    import time
    start = time.time()

    print("\nFor LiH at bond length 1.45, with %i terms acting on %i qubits:"
          % (len(terms), qubit_hamiltonian.n_qubits()))
    print("Loose error bound = %f" % error_bound(terms))
    print "Took ", time.time() - start, " to compute"
    start = time.time()
    print "Tight error bound = %f" % error_bound(terms, tight=True)
    print "Took ", time.time() - start, " to compute"
