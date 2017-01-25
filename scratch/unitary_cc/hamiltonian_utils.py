"""This files has useful utilities to read and store qubit hamiltonians.
"""

import re

class ErrorPauliString(Exception):
  pass


class ErrorQubitHamiltonian(Exception):
  pass


class PauliString(object):
  """Single term of a hamiltonian for a system of spin 1/2 particles or qubits.

  A hamiltonian of qubits can be written as a sum of PauliString objects.
  Suppose you have n=5 qubits (saved in num_qubits) a term of the hamiltonian
  could be coefficient * X1Z3 which we call a PauliString object. It means
  coefficient *(1 x PauliX x 1 x PauliZ x 1),
  where x is the tensor product, 1 the identity matrix, and the others are Pauli
  matrices. We only allow to apply one single Pauli Matrix to each qubit.
  (Multiplications of Pauli matrices must be simplied beforehand)

  Note: It is always assumed in this class that indices start from 0 to
        num_qubits-1

  Attributes:
    pauli_x: List. Stores to which indices a PauliX is applied to (sorted)
    pauli_y: List. Stores to which indices a PauliY is applied to (sorted)
    pauli_z: List. Stores to which indices a PauliZ is applied to (sorted)
    num_qubits: Total number of qubits this Hamiltonian acts on
    coefficient: a real or complex floating point number
  """

  def __init__(self, num_qubits, coefficient, pauli_x, pauli_y, pauli_z):
    """Inits PauliString.

    Specify to which qubits a Pauli X, Y, or Z is applied. To all not
    specified qubits (numbered 0, 1, ..., n-1) the identity is applied.
    Only one Pauli Matrix can be applied to each qubit.

    Example usage: PauliString(11, 0.5, [2,4], [], [10]) is 0.5 * X2X4Z10
                   assuming 11 qubits in total (numbered 0, 1,..., n-1)
                   PauliString(11, 1.0, [], [], []) initializes the identity

    Args:
      num_qubits: is the total number of qubits
      coefficient: numerical coefficient
      pauli_x: list of indices of qubits to which a pauli_x is applied
      pauli_y: list of indices of qubits to which a pauli_y is applied
      pauli_z: list of indices of qubits to which a pauli_z is applied

    Raises:
      ErrorPauliString: Wrong input.
    """
    self.coefficient = coefficient
    self.num_qubits = num_qubits
    self.pauli_x = sorted(pauli_x)
    self.pauli_y = sorted(pauli_y)
    self.pauli_z = sorted(pauli_z)
    # Check that num_qubits is an integer
    if not isinstance(num_qubits, int):
      raise ErrorPauliString("Number of qubits needs to be an integer")
    # Check that all indices are int
    all_indices = self.pauli_x + self.pauli_y + self.pauli_z
    if not all(isinstance(index, int) for index in all_indices):
      raise ErrorPauliString("Non integer index")
    # Check that no dublicate indices
    if len(set(all_indices)) != len(all_indices):
      raise ErrorPauliString("More than one Pauli matrix applied to one qubit")
    # Check that all indices between 0, ..., num_qubit -1
    if all_indices:
      if not sorted(all_indices)[-1] < num_qubits:
        raise ErrorPauliString("Qubit indices out of range")

  def is_identical_pauli(self, pauli_string):
    """Compare tensor factors with another PauliString object.

    Args:
      pauli_string: Another PauliString object

    Returns:
      bool. True if pauli matrices are equal (not necessarily the
             coefficients)

    Raises:
      ErrorPauliString: Not same number of qubits in each term
    """
    if self.num_qubits != pauli_string.num_qubits:
      raise ErrorPauliString("Comparing terms with different num_qubits")
    if (pauli_string.pauli_x == self.pauli_x and
        pauli_string.pauli_y == self.pauli_y and
        pauli_string.pauli_z == self.pauli_z):
      return True
    return False


def init_pauli_string_from_string(num_qubits, coefficient, pauli_string):
  """Inits PauliString from a String and disregard previous values stored.

  Example usage: init_PauliString_from_string(11, 0.5, "X2X4Y6Z10")
                 init_PauliString_from_string(5, 1.0, "I")

  Args:
    num_qubits: an integer. Number of qubits
    coefficient: a float
    pauli_string: string e.g. "X2X6Z12Z14", identity is "I"

  Returns:
    PauliString object

  Raises:
    ErrorPauliString: Wrong input.
  """
  pauli_x = []
  pauli_y = []
  pauli_z = []
  if pauli_string == "I":
    return PauliString(num_qubits, coefficient, [], [], [])
  splitted_string = re.split(r"(\d+)", pauli_string)  # Might have trailing ''
  if splitted_string[-1] == "":  # Remove trailing ''
    splitted_string = splitted_string[:-1]
  if len(splitted_string) % 2 != 0:
    raise ErrorPauliString("Invalid input String: ", pauli_string)
  for ii in range(0, len(splitted_string), 2):
    if splitted_string[ii] == "X":
      pauli_x.append(int(splitted_string[ii + 1]))
    elif splitted_string[ii] == "Y":
      pauli_y.append(int(splitted_string[ii + 1]))
    elif splitted_string[ii] == "Z":
      pauli_z.append(int(splitted_string[ii + 1]))
    else:
      raise ErrorPauliString("Unknown Pauli matrix: ", splitted_string[ii])
  return PauliString(num_qubits, coefficient, pauli_x, pauli_y, pauli_z)


def init_pauli_string_from_string_list(num_qubits, coefficient, pauli_list):
  """Converts output list element of symbolic_fermions to a PauliString object.

  symbolic_fermions.SymbolicTermJW outputs
  PauliString objects in the format pauli_list. This function is used to
  initialize a PauliString objects with this format.

  Args:
    num_qubits: Integer. Total number of qubits in the system.
    coefficient: A real or complex floating point number.
    pauli_list: List of strings with length num_qubits. E.g. ['X','I','Y'],
                which means Pauli X on qubit 0, identity of qubit 1 and
                Pauli Y on qubit 2.

  Returns:
    A PauliString object.

  Raises:
    ErrorPauliString: if it cannot convert the format to a PauliString.
  """
  if len(pauli_list) != num_qubits:
    raise ErrorPauliString("Not correct number of qubits")
  pauli_x = []
  pauli_y = []
  pauli_z = []
  for index in range(num_qubits):
    operator = pauli_list[index]
    if operator == "I":
      pass
    elif operator == "X":
      pauli_x.append(index)
    elif operator == "Y":
      pauli_y.append(index)
    elif operator == "Z":
      pauli_z.append(index)
    else:
      raise ErrorPauliString("Unknown Pauli Operator ", operator)
  return PauliString(num_qubits, coefficient, pauli_x, pauli_y, pauli_z)


class QubitHamiltonian(object):
  """A collection of PauliString objects acting on same number of qubits.

  In order to be a Hamiltonian which is a hermitian operator, the
  individual PauliString objects need to have only real valued coefficients.
  The Hamiltonian corresponds to the sum of all PauliString objects saved in
  hamiltonian_terms.

  Attributes:
    terms: List of PauliString objects with real coefficients
  """

  def __init__(self, pauli_string=None):
    """Inits QubitHamiltonian.

    Args:
      pauli_string: a PauliString object. The initial pauli string in the
                    Hamiltonian. Default none.
    Raises:
      ErrorQubitHamiltonian: Coefficients must be real.
    """
    self.terms = []
    if pauli_string is not None:
      if not pauli_string.coefficient.imag:
        self.terms.append(pauli_string)
      else:
        raise ErrorQubitHamiltonian("Coefficient must be real")

  def add_term(self, new_term):
    """Add another PauliString to hamiltonian.

    If hamiltonian already has this term, then the coefficients are merged.

    Args:
      new_term: PauliString object. It is added to the Hamiltonian.

    Raises:
      ErrorQubitHamiltonian: Not allowed to add this term.
    """
    if new_term.coefficient.imag:
      raise ErrorQubitHamiltonian("Coefficient must be real")
    if self.terms:
      if self.terms[0].num_qubits != new_term.num_qubits:
        raise ErrorQubitHamiltonian("Terms need to act on same num_qubits")
    for term in self.terms:
      if term.is_identical_pauli(new_term):
        term.coefficient += new_term.coefficient
        return
    self.terms.append(new_term)


def read_hamiltonian_file(inputfile, num_qubits, check_order=True):
  """Read a hamiltonian file and store it in class Hamiltonian Object.

  Note: Qubits need to be numbered 0,1,...,num_qubits-1

  Args:
    inputfile: a path to a text file.  Each line within the file specifies
               a PauliString.
    num_qubits: an integer. Number of total qubits this hamiltonian acts on.
    check_order: bool. Default is True. Checks that spin orbitals are order
                 from low to highest energy. I.e. qubit 0 encodes the spin
                 orbital with the lowest energy, qubit 1 the spin orbital with
                 the second lowest energy orbital, ...
                 If this check fails, a ErrorQubitHamiltonian is raised.

  Returns:
    QubitHamiltonian class object

  Raises:
    ErrorQubitHamiltonian: If check_order fails.
  """
  hamiltonian = QubitHamiltonian()
  with open(inputfile, "r") as f:
    for line in f:
      elements = line.split()
      if elements:  # ignores empty lines
        term = init_pauli_string_from_string(num_qubits, float(elements[0]),
                                             elements[1])
        hamiltonian.add_term(term)
  # Check that spin orbitals are ordered according to their energies.
  # Only terms with only pauli_z contribute to this energy.
  if check_order:
    diagonal_hamiltonian = QubitHamiltonian()
    for term in hamiltonian.terms:
      if not term.pauli_x and not term.pauli_y:
        diagonal_hamiltonian.add_term(term)
    previous_energy = -99999999
    for qubit in range(num_qubits):
      energy = 0.
      for term in diagonal_hamiltonian.terms:
        if qubit in term.pauli_z:
          energy -= term.coefficient
        else:
          energy += term.coefficient
      if energy - previous_energy < -0.000001:
        raise ErrorQubitHamiltonian("Qubits are not ordered according to spin"
                                    " orbital energies")
      previous_energy = energy

  return hamiltonian
