"""Jordan-Wigner transform on FermionOperator."""
from __future__ import absolute_import

from projectqtemp.ops._qubit_operator import QubitOperator, qubit_identity


def jordan_wigner_term(term, coeff=1.):
    """Jordan-Wigner transform term and return the resulting qubit operator.

    Returns:
      transformed_term: An instance of the QubitOperator class.

    Warning:
      The runtime of this method is exponential in the locality of the
      original FermionOperator.

    """
    # Initialize identity matrix.
    transformed_term = coeff * qubit_identity()

    # Loop through operators, transform and multiply.
    for operator in term:
        z_factors = tuple((index, 'Z') for index in range(0, operator[0]))

        # Handle identity.
        pauli_x_component = QubitOperator(z_factors +
                                          ((operator[0], 'X'),), 0.5)
        if operator[1]:
            pauli_y_component = QubitOperator(z_factors +
                                              ((operator[0], 'Y'),), -0.5j)
        else:
            pauli_y_component = QubitOperator(z_factors +
                                              ((operator[0], 'Y'),), 0.5j)

        transformed_term *= pauli_x_component + pauli_y_component
    return transformed_term
