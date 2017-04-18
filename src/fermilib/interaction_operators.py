"""Class and functions to store interaction operators."""
from __future__ import absolute_import

import itertools

from fermilib import qubit_operators
from fermilib.fermion_operators import FermionTerm, FermionOperator
from fermilib.interaction_tensors import InteractionTensor


class InteractionOperatorError(Exception):
    pass


class InteractionOperator(InteractionTensor):
    """Class for storing 'interaction operators' which are defined to be
    fermionic operators consisting of one-body and two-body terms which
    conserve particle number and spin. The most common examples of data that
    will use this structure are molecular Hamiltonians. In principle,
    everything stored in this class could also be represented using the more
    general FermionOperator class. However, this class is able to exploit
    specific properties of how fermions interact to enable more numerically
    efficient manipulation of the data. Note that the operators stored in this
    class take the form: constant + \sum_{p, q} h_[p, q] a^\dagger_p a_q +

        \sum_{p, q, r, s} h_[p, q, r, s] a^\dagger_p a^\dagger_q a_r a_s.

    Attributes:
      n_qubits: An int giving the number of qubits.
      constant: A constant term in the operator given as a float.
          For instance, the nuclear repulsion energy.
      one_body_tensor: The coefficients of the one-body terms (h[p, q]).
          This is an n_qubits x n_qubits numpy array of floats.
      two_body_tensor: The coefficients of the two-body terms
          (h[p, q, r, s]). This is an n_qubits x n_qubits x n_qubits x
          n_qubits numpy array of floats.

    """

    def __init__(self, constant, one_body_tensor, two_body_tensor):
        """Initialize the InteractionOperator class.

        Args:   constant: A constant term in the operator given as a
        float.       For instance, the nuclear repulsion energy.
        one_body_tensor: The coefficients of the one-body terms (h[p,
        q]).       This is an n_qubits x n_qubits numpy array of floats.
        two_body_tensor: The coefficients of the two-body terms
        (h[p, q, r, s]). This is an n_qubits x n_qubits x n_qubits x
        n_qubits numpy array of floats.

        """
        # Make sure nonzero elements are only for normal ordered terms.
        super(InteractionOperator, self).__init__(constant, one_body_tensor,
                                                  two_body_tensor)

    def unique_iter(self, complex_valued=False):
        """Iterate all terms that are not in the same symmetry group.
        Four point symmetry:
          1. pq = qp.
          2. pqrs = srqp = qpsr = rspq.
        Eight point symmetry:
          1. pq = qp.
          2. pqrs = rqps = psrq = srqp = qpsr = rspq = spqr = qrsp.

        Args:
          complex_valued: Bool, whether the operator has complex coefficients.
        """
        # Constant.
        if self.constant:
            yield []

        # One-body terms.
        for p in range(self.n_qubits):
            for q in range(p + 1):
                if self.one_body_tensor[p, q]:
                    yield [p, q]

        # Two-body terms.
        record_map = {}
        for p in range(self.n_qubits):
            for q in range(self.n_qubits):
                for r in range(self.n_qubits):
                    for s in range(self.n_qubits):
                        if self.two_body_tensor[p, q, r, s] and \
                           (p, q, r, s) not in record_map:
                            yield [p, q, r, s]
                            record_map[(p, q, r, s)] = []
                            record_map[(s, r, q, p)] = []
                            record_map[(q, p, s, r)] = []
                            record_map[(r, s, p, q)] = []
                            if not complex_valued:
                                record_map[(p, s, r, q)] = []
                                record_map[(s, p, q, r)] = []
                                record_map[(q, r, s, p)] = []
                                record_map[(r, q, p, s)] = []
