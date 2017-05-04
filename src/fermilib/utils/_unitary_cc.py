"""Module to create and manipulate unitary coupled cluster operators."""
from __future__ import absolute_import

import copy
import itertools
import projectq
import projectq.setups
import projectq.setups.decompositions
import projectq.ops
import projectq.backends
import projectq.cengines
import projectq.meta

from fermilib.ops import FermionOperator
from fermilib.transforms import jordan_wigner

def pack_uccsd_amplitudes(single_amplitudes, double_amplitudes):
    """Pack full CC amplitudes into 1D anti-Hermitian form"""
    assert(False)

    n_orbitals = single_amplitudes.shape[0]

    t1_size = n_orbitals * (n_orbitals - 1) / 2
    t2_size = n_orbitals**2 * (n_orbitals**2 - 1) / 2
    packed_amplitudes = numpy.zeros(t1_size + t2_size)

    for i in range(n_orbitals):
        for j in range(i):
            packed_amplitudes[i * (i - 1) / 2 + j] = single_amplitudes[i,j]

    for i in range(n_orbitals):
        for j in range(i):
            for k in range(j):
                for l in range(k):
                    packed_amplitudes[i] = double_amplitudes[i,j,k,l]


def uccsd_operator(single_amplitudes, double_amplitudes, anti_hermitian=True):
    """
    Create a fermionic operator that is the generator of uccsd.

    Args:
        single_amplitudes(ndarray): [NxN] array storing single excitation
            amplitudes corresponding to t[i,j] * (a_i^\dagger a_j + H.C.)
        double_amplitudes(ndarray): [NxNxNxN] array storing double excitation
            amplitudes corresponding to
            t[i,j,k,l] * (a_i^\dagger a_j a_k^\dagger a_l + H.C.)
        anti_hermitian(Bool): Flag to generate only normal CCSD operator
            rather than unitary variant, primarily for testing

    Returns:
        uccsd_operator(FermionOperator): Anti-hermitian fermion operator that
        is the generator for the uccsd wavefunction.
    """
    n_orbitals = single_amplitudes.shape[0]
    assert(n_orbitals == double_amplitudes.shape[0])
    uccsd_operator = FermionOperator((), 0.0)

    # Add single excitations
    for i, j in itertools.product(range(n_orbitals), repeat=2):
        if single_amplitudes[i, j] == 0.:
            continue
        uccsd_operator += FermionOperator(
            ((i, 1), (j, 0)), single_amplitudes[i, j])
        if (anti_hermitian):
            uccsd_operator += FermionOperator(
                ((j, 1), (i, 0)), -single_amplitudes[i, j])

        # Add double excitations
    for i, j, k, l in itertools.product(range(n_orbitals), repeat=4):
        if double_amplitudes[i, j, k, l] == 0.:
            continue
        uccsd_operator += FermionOperator(
            ((i, 1), (j, 0), (k, 1), (l, 0)),
            double_amplitudes[i, j, k, l])
        if (anti_hermitian):
            uccsd_operator += FermionOperator(
                ((l, 1), (k, 0), (j, 1), (i, 0)),
                -double_amplitudes[i, j, k, l])

    return uccsd_operator


def _identify_non_commuting(cmd):
    """
    Recognize all TimeEvolution gates with >1 terms but which don't all
    commute.
    """
    hamiltonian = cmd.gate.hamiltonian
    if len(hamiltonian.terms) == 1:
        return False
    else:
        id_op = projectq.ops.QubitOperator((), 0.0)
        for term in hamiltonian.terms:
            test_op = projectq.ops.QubitOperator(term, hamiltonian.terms[term])
            for other in hamiltonian.terms:
                other_op = projectq.ops.QubitOperator(other, hamiltonian.terms[other])
                commutator = test_op * other_op - other_op * test_op
                if not commutator.isclose(id_op,
                                          rel_tol=1e-9,
                                          abs_tol=1e-9):
                    return True
    return False

def _first_order_trotter(cmd):
    qureg = cmd.qubits
    eng = cmd.engine
    hamiltonian = cmd.gate.hamiltonian
    time = cmd.gate.time
    with projectq.meta.Control(eng, cmd.control_qubits):
        # First order Trotter splitting
            for term in hamiltonian.terms:
                ind_operator = projectq.ops.QubitOperator(term, hamiltonian.terms[term])
                projectq.ops.TimeEvolution(time, ind_operator) | qureg

def _two_gate_filter(self, cmd):
    if ((not isinstance(cmd.gate, projectq.ops.TimeEvolution)) and
        (len(cmd.qubits[0]) <= 2 or
             isinstance(cmd.gate, projectq.ops.ClassicalInstructionGate))):
        return True
    return False

def uccsd_trotter_engine(compiler_backend=projectq.backends.Simulator()):
    # Set ProjectQ rules to two-qubit gates with first order Trotter
    rule_set = \
        projectq.cengines. \
            DecompositionRuleSet(modules=[projectq.setups.decompositions])
    trotter_rule_set = projectq.cengines. \
        DecompositionRule(gate_class=projectq.ops.TimeEvolution,
                          gate_decomposer=_first_order_trotter,
                          gate_recognizer=_identify_non_commuting)
    rule_set.add_decomposition_rule(trotter_rule_set)
    replacer = projectq.cengines.AutoReplacer(rule_set)

    # Start the compiler engine with these rules
    compiler_engine = projectq.MainEngine(backend=compiler_backend,
                                          engine_list=[replacer,
                                                       projectq.cengines.
                                          InstructionFilter(_two_gate_filter)])
    return compiler_engine

def uccsd_circuit(single_amplitudes, double_amplitudes,
                  fermion_transform=jordan_wigner):
    """Create a uccsd ansatz circuit"""

    # Assess number of qubits
    n_qubits = single_amplitudes.shape[0]

    # Build UCCSD generator
    fermion_generator = uccsd_operator(single_amplitudes, double_amplitudes)

    # Transform generator to qubits
    qubit_generator = 1.0j * fermion_transform(fermion_generator)

    # Cast to real part only for compatibility with current ProjectQ routine
    for key in qubit_generator.terms:
        qubit_generator.terms[key] = float(qubit_generator.terms[key].real)

    # Allocate wavefunction and act evolution on gate according to compilation
    evolution_operator = projectq.ops.\
        TimeEvolution(time=1.,
                      hamiltonian=qubit_generator)

    return evolution_operator
