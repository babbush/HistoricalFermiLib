"""Class and functions to store reduced density matrices."""
from __future__ import absolute_import

import copy

import numpy

from fermilib.interaction_tensors import InteractionTensor


class InteractionRDMError(Exception):
    pass


def unpack_spatial_rdm(one_rdm_a,
                       one_rdm_b,
                       two_rdm_aa,
                       two_rdm_ab,
                       two_rdm_bb):
    """Covert from spin compact spatial format to spin-orbital format for RDM.

    Note: the compact 2-RDM is stored as follows where A/B are spin up/down:
    RDM[pqrs] = <| a_{p, A}^\dagger a_{r, A}^\dagger a_{q, A} a_{s, A} |>
      for 'AA'/'BB' spins.
    RDM[pqrs] = <| a_{p, A}^\dagger a_{r, B}^\dagger a_{q, B} a_{s, A} |>
      for 'AB' spins.

    Args:
      one_rdm_a: 2-index numpy array storing alpha spin
        sector of 1-electron reduced density matrix.
      one_rdm_b: 2-index numpy array storing beta spin
        sector of 1-electron reduced density matrix.
      two_rdm_aa: 4-index numpy array storing alpha-alpha spin
        sector of 2-electron reduced density matrix.
      two_rdm_ab: 4-index numpy array storing alpha-beta spin
        sector of 2-electron reduced density matrix.
      two_rdm_bb: 4-index numpy array storing beta-beta spin
        sector of 2-electron reduced density matrix.

    Returns:
      one_rdm: 2-index numpy array storing 1-electron density matrix
        in full spin-orbital space.
      two_rdm: 4-index numpy array storing 2-electron density matrix
        in full spin-orbital space.

    """
    # Initialize RDMs.
    n_orbitals = one_rdm_a.shape[0]
    n_qubits = 2 * n_orbitals
    one_rdm = numpy.zeros((n_qubits, n_qubits))
    two_rdm = numpy.zeros((n_qubits, n_qubits,
                           n_qubits, n_qubits))

    # Unpack compact representation.
    for p in range(n_orbitals):
        for q in range(n_orbitals):

            # Populate 1-RDM.
            one_rdm[2 * p, 2 * q] = one_rdm_a[p, q]
            one_rdm[2 * p + 1, 2 * q + 1] = one_rdm_b[p, q]

            # Continue looping to unpack 2-RDM.
            for r in range(n_orbitals):
                for s in range(n_orbitals):

                    # Handle case of same spin.
                    two_rdm[2 * p, 2 * q, 2 * r, 2 * s] = (
                        two_rdm_aa[p, r, q, s])
                    two_rdm[2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1] = (
                        two_rdm_bb[p, r, q, s])

                    # Handle case of mixed spin.
                    two_rdm[2 * p, 2 * q + 1, 2 * r, 2 * s + 1] = (
                        two_rdm_ab[p, r, q, s])
                    two_rdm[2 * p, 2 * q + 1, 2 * r + 1, 2 * s] = (
                        -1. * two_rdm_ab[p, s, q, r])
                    two_rdm[2 * p + 1, 2 * q, 2 * r + 1, 2 * s] = (
                        two_rdm_ab[q, s, p, r])
                    two_rdm[2 * p + 1, 2 * q, 2 * r, 2 * s + 1] = (
                        -1. * two_rdm_ab[q, r, p, s])

    # Map to physicist notation and return.
    two_rdm = numpy.einsum('pqsr', two_rdm)
    return one_rdm, two_rdm


class InteractionRDM(InteractionTensor):
    """Class for storing 1- and 2-body reduced density matrices.

    Attributes:   one_body_tensor: The expectation values <a^\dagger_p
    a_q>.   two_body_tensor: The expectation values <a^\dagger_p
    a^\dagger_q a_r a_s>.   n_qubits: An int giving the number of
    qubits.

    """

    def __init__(self, one_body_tensor, two_body_tensor):
        """Initialize the InteractionRDM class.

        Args:   one_body_tensor: Expectation values <a^\dagger_p a_q>.
        two_body_tensor: Expectation values <a^\dagger_p a^\dagger_q a_r
        a_s>.

        """
        super(InteractionRDM, self).__init__(None, one_body_tensor,
                                             two_body_tensor)

    @classmethod
    def from_spatial_rdm(cls, one_rdm_a, one_rdm_b,
                         two_rdm_aa, two_rdm_ab, two_rdm_bb):
        one_rdm, two_rdm = unpack_spatial_rdm(one_rdm_a, one_rdm_b,
                                              two_rdm_aa, two_rdm_ab,
                                              two_rdm_bb)
        return cls(constant, one_rdm, two_rdm)

    def qubit_term_expectation(self, qubit_term):
        """Return expectation value of a QubitTerm with an InteractionRDM
        (self).

        Args:   qubit_term: QubitTerm instance to be evaluated on this
        InteractionRDM.

        """
        expectation = 0.
        reversed_fermion_operators = qubit_term.reverse_jordan_wigner(
            self.n_qubits)
        reversed_fermion_operators.normal_order()
        for fermion_term in reversed_fermion_operators:

            # Handle molecular terms.
            if fermion_term.is_molecular_term():
                if fermion_term.is_identity():
                    expectation += fermion_term.coefficient
                else:
                    indices = [operator[0] for operator in fermion_term]
                    rdm_element = self[indices]
                    expectation += rdm_element * fermion_term.coefficient

                # Handle non-molecular terms.
            elif len(fermion_term.operators) > 4:
                raise InteractionRDMError('Observable not contained '
                                          'in 1-RDM or 2-RDM.')
        return expectation

    def expectation(self, operator):
        """Return expectation value of an InteractionRDM with an operator.

        Args:
          operator: A QubitOperator or InteractionOperator.

        Returns:
          expectation: A float giving the expectation value.

        Raises:
          InteractionRDMError: Invalid operator provided.

        """
        # Import here to avoid circular dependency.
        from fermilib import interaction_operators, qubit_operators

        if isinstance(operator, qubit_operators.QubitOperator):
            expectation_value = 0.
            for qubit_term in operator:
                expectation += self.qubit_term_expectation(qubit_term)
        elif isinstance(operator, interaction_operators.InteractionOperator):
            expectation = operator.constant
            expectation += numpy.sum(self.one_body_tensor *
                                     operator.one_body_tensor)
            expectation += numpy.sum(self.two_body_tensor *
                                     operator.two_body_tensor)
        else:
            raise InteractionRDMError('Invalid operator type provided.')
        return expectation

    def get_qubit_expectations(self, qubit_operator):
        """Return expectations of qubit op as coefficients of new qubit op.

        Args:
          qubit_operator: QubitOperator instance to be evaluated on this
              InteractionRDM.

        Returns:
          qubit_operator_expectations: QubitOperator with coefficients
              corresponding to expectation values of those operators.

        Raises:
          InteractionRDMError: Observable not contained in 1-RDM or 2-RDM.

        """
        qubit_operator_expectations = copy.deepcopy(qubit_operator)
        for qubit_term in qubit_operator_expectations:
            if not qubit_term.is_identity():
                qubit_term.coefficient = 1.
                qubit_term.coefficient = self.qubit_term_expectation(
                    qubit_term)
        qubit_operator_expectations[[]] = 0.
        return qubit_operator_expectations
