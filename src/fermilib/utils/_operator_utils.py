"""This module provides generic tools for classes in ops/"""
from __future__ import absolute_import

import numpy

from fermilib.ops import *
from projectqtemp.ops._qubit_operator import QubitOperator


def eigenspectrum(operator):
    """Compute the eigenspectrum of an operator.

    WARNING: This function has cubic runtime in dimension of
        Hilbert space operator, which might be exponential.

    Args:
        operator: QubitOperator, InteractionOperator, FermionOperator,
            InteractionTensor, or InteractionRDM.

    Returns:
        eigenspectrum: dense numpy array of floats giving eigenspectrum.
    """
    from fermilib.transforms import get_sparse_operator
    from fermilib.utils import sparse_eigenspectrum
    sparse_operator = get_sparse_operator(operator)
    eigenspectrum = sparse_eigenspectrum(sparse_operator)
    return eigenspectrum


def count_qubits(operator):
    """Compute the minimum number of qubits on which operator acts.

    Args:
        operator: QubitOperator, InteractionOperator, FermionOperator,
            InteractionTensor, or InteractionRDM.

    Returns:
        n_qubits (int): The minimum number of qubits on which operator acts.

    Raises:
       TypeError: Operator of invalid type.
    """
    # Handle FermionOperator.
    if isinstance(operator, FermionOperator):
        n_qubits = 0
        for term in operator.terms:
            for ladder_operator in term:
                if ladder_operator[0] + 1 > n_qubits:
                    n_qubits = ladder_operator[0] + 1
        return n_qubits

    # Handle QubitOperator.
    elif isinstance(operator, QubitOperator):
        n_qubits = 0
        for term in operator.terms:
            if term:
                if term[-1][0] + 1 > n_qubits:
                    n_qubits = term[-1][0] + 1
        return n_qubits

    # Handle InteractionOperator, InteractionRDM, InteractionTensor.
    elif isinstance(operator, (InteractionOperator,
                               InteractionRDM,
                               InteractionTensor)):
        return operator.n_qubits

    # Raise for other classes.
    else:
        raise TypeError('Operator of invalid type.')


def is_identity(operator):
    """Check whether QubitOperator of FermionOperator is identity.

    Args:
        operator: QubitOperator or FermionOperator.

    Raises:
       TypeError: Operator of invalid type.
    """
    if isinstance(operator, (QubitOperator, FermionOperator)):
        return operator.terms.keys() == [(),]
    else:
        raise TypeError('Operator of invalid type.')
