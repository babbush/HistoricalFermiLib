"""Useful class to store and Jordan Wigner transform fermionic operators.
"""

import symbolic_fermions
import hamiltonian_utils


class ErrorJordanWigner(Exception):
  pass


class FermionicOperator(object):
  """Stores a fermionic operator.

  Attributes:
    fermionic_operator: List of integers (excluding 0). Stores a fermionic
                        operator, absolute value of integer is index to which
                        the operator is applied. A positive sign means it is a
                        creation operator, a negative sign means it is an
                        annihiliation operator. Example:
                        a_i^dagger a_p a_j^dagger a_q equals [i, -p, j, -q]
    coefficient: A real or complex number.

  Note:
    We use the same convention to store a fermionic operator as in
    symbolic_fermions.py in order to make the codes easily compatible.
  """

  def __init__(self, coefficient, operator):
    """Inits a fermionic operator.

    Args:
      coefficient: A real or complex number.
      operator: List of integers representing a fermionic operator.
    """
    self.fermionic_operator = operator
    self.coefficient = coefficient

  def get_hermitian_conjugate(self):
    """Calculate hermitian conjugate.

    Returns:
      New FermionicOperator object which is the hermitian conjugate.
    """
    conjugate_tensor_factor = [-1 * index
                               for index in reversed(self.fermionic_operator)]
    conjugate_coefficient = self.coefficient.real - 1j * self.coefficient.imag
    return FermionicOperator(conjugate_coefficient, conjugate_tensor_factor)


def jordan_wigner_transform(fermionic_operator, num_qubits):
  """Jordan Wigner transformation a Fermionic term.

  Note: FermionicOperator object numbers fermions from 1, 2, ...
        PauliString object numbers spins from 0, 1, ...

  Args:
    fermionic_operator: A FermionOperator object.
    num_qubits: Integer. Total number of qubits.

  Returns:
    List of PauliString objects. The PauliString objects summed together are
    the transformed fermionic_operator.

  Raises:
    ErrorJordanWigner: See detailed error.
  """
  fermionic_coefficient = fermionic_operator.coefficient
  fermionic_operator = fermionic_operator.fermionic_operator
  if max([abs(index) for index in fermionic_operator]) > num_qubits:
    raise ErrorJordanWigner("Not enough qubits to represent fermionic operator")
  jw_transform = symbolic_fermions.SymbolicTermJW(fermionic_coefficient,
                                                  fermionic_operator,
                                                  num_qubits)
  # Format of jw_transform from symbolic_fermions.SymbolicTermJW:
  # List of List. First list is list of coefficients and second is a list of
  # lists where each entry is a pauli string represented as e.g.
  # ['X', 'I', 'Y'].
  # Example: [[1.0, -2.0], [['X', 'I'], ['Y', 'Z']]]
  result = []
  if len(jw_transform) != 2:
    raise ErrorJordanWigner("Cannot convert this to PauliString objects.",
                            jw_transform)
  for ii in range(len(jw_transform[0])):
    # Convert to PauliString object
    if len(jw_transform[1][ii]) != num_qubits:
      raise ErrorJordanWigner("String has incorrect size")
    coefficient = jw_transform[0][ii]
    pauli_x = []
    pauli_y = []
    pauli_z = []
    for index in range(num_qubits):
      operator = jw_transform[1][ii][index]
      if operator == "I":
        pass
      elif operator == "X":
        pauli_x.append(index)
      elif operator == "Y":
        pauli_y.append(index)
      elif operator == "Z":
        pauli_z.append(index)
      else:
        raise ErrorJordanWigner("Unknown Pauli Operator ", operator)
    pauli_string = hamiltonian_utils.PauliString(num_qubits, coefficient,
                                                 pauli_x, pauli_y, pauli_z)
    result.append(pauli_string)
  return result
