"""This file is used to produce a unitary coupled cluster ansatz.
"""

import itertools

import fermionic_utils
import hamiltonian_utils


class ErrorCoupledCluster(Exception):
  pass


class UccAnsatz(object):
  """Stores unitary coupled cluster ansatz.

  Unitary coupled clusters ansatz is defined as e^(T(theta)-T^dagger(theta)).
  Here we use only single and double excitations, i.e.
  T = T_1(theta) + T_2(theta), where
  T_1(theta) = Sum_{p,i} theta_{pi} a_p^dagger a_i
  T_2(theta) = Sum_{q,p,j,i} theta_{qpji} a_q^dagger a_p^dagger a_j a_i
  Indices p,q correspond to unoccupied spin orbitals in Hartree Fock ground
  state. Indices i,j correspond to occupied spin orbitals in Hartree Fock ground
  state.

  Note: In the fermionic operator picture, indices start from 1,..., number of
        spin orbitals, whereas in the spin operator picture the indices are
        numbered from 0, ... , number of spin orbitals - 1.

  Note: Spatial orbitals are degenerate with regard to spin. We define odd
        fermionic indices to be spin up and even fermionic indices to be spin
        down. (Hence, even spin indices are spin up, odd spin indices are
        spin down)

  Note: We restrict terms in T_1 and T_2 to those terms which conserve spin.

  The ucc ansatz is saved in the fermionic and spin operator picture. The
  ucc ansatz is given by

  e^(sum_{k} theta[k] (fermionic_operator[k] - fermionic_operator[k]^dagger))

  in the fermionic operator picture, where fermionic_operator[k] is one of the
  terms in T_1 or T_2. This will also be transformed into the spin operator
  pictures:

  e^(sum{k} -i spin_hamiltonian[k]),

  where spin_hamiltonian[k] is obtained by e.g. a Jordan Wigner transformation
  of the term fermionic_operator[k] - fermionic_operator[k]^dagger and dividing
  the result by -i.

  Here we use the Jordan Wigner transformation. If you want to use another
  transformation, then create a derived class and override
  _fermion_to_spin_transformation()

  Attributes:
    ucc_terms: List of Lists. Each sublist stores
               [theta[k], fermionic_operator[k], spin_hamiltonian[k]],
               where theta[k] is a real number, fermionic_operator[k] is a
               FermionicOperator object, and spin_hamiltonian[k] is a
               QubitHamiltonian object. Explicit form of ucc ansatz needs to
               be computed using the formulas above.
               The fermionic_operator[k] is stored in normal order/ Wick order.
               e.g. a term a_q^dagger a_p^dagger a_j a_i with indices
               q > p > j > i. It is saved as [q, p, -j, -i].
    num_electrons: Int. Number of electrons in the system.
    num_qubits: Int. Number of spin orbitals or equivalently number of qubits.
                We require an even number.
  """

  def __init__(self, num_qubits, num_electrons):
    """Creates all possible terms and sets coefficients to 0.

    Args:
      num_qubits: Integer. Number of qubits or spin orbitals
      num_electrons: Integer. Number of electrons.

    Raises:
      ErrorCoupledCluster: Only even number of qubits allowed

    Note: FermionicOperator needs to be in normal ordering.
    """
    if num_qubits % 2:
      raise ErrorCoupledCluster("Even number of qubits required.")
    self.num_qubits = num_qubits
    self.num_electrons = num_electrons
    self.ucc_terms = []
    occupied_spin_up_indices = range(1, num_electrons + 1, 2)
    occupied_spin_down_indices = range(2, num_electrons + 1, 2)
    occupied_indices = range(1, num_electrons + 1)
    last_filled_up = num_electrons % 2
    unoccupied_spin_up_indices = range(num_electrons + 1 + last_filled_up,
                                       num_qubits + 1, 2)
    unoccupied_spin_down_indices = range(num_electrons + 2 - last_filled_up,
                                         num_qubits + 1, 2)
    # Create single excitation operators from spin up to spin up
    for a_dagger in unoccupied_spin_up_indices:
      for a in occupied_spin_up_indices:
        fermionic_term = fermionic_utils.FermionicOperator(1, [a_dagger, -a])
        qubit_hamiltonian = self._get_ucc_spin_hamiltonian(fermionic_term)
        self.ucc_terms.append([0, fermionic_term, qubit_hamiltonian])
    # Create single excitation operators from spin down to spin down
    for a_dagger in unoccupied_spin_down_indices:
      for a in occupied_spin_down_indices:
        fermionic_term = fermionic_utils.FermionicOperator(1, [a_dagger, -a])
        qubit_hamiltonian = self._get_ucc_spin_hamiltonian(fermionic_term)
        self.ucc_terms.append([0, fermionic_term, qubit_hamiltonian])
    # Create double excitation operators all conserving spin
    possible_a_combinations = itertools.combinations(occupied_indices, 2)
    for a_1, a_2 in possible_a_combinations:
      # Both spin up
      if a_1 % 2 == 1 and a_2 % 2 == 1:
        for a_dagger_1, a_dagger_2 in itertools.combinations(
            unoccupied_spin_up_indices, 2):
          fermionic_term = fermionic_utils.FermionicOperator(
              1, [a_dagger_2, a_dagger_1, -a_2, -a_1])
          qubit_hamiltonian = self._get_ucc_spin_hamiltonian(fermionic_term)
          self.ucc_terms.append([0, fermionic_term, qubit_hamiltonian])
      # Both spin down
      elif a_1 % 2 == 0 and a_2 % 2 == 0:
        for a_dagger_1, a_dagger_2 in itertools.combinations(
            unoccupied_spin_down_indices, 2):
          fermionic_term = fermionic_utils.FermionicOperator(
              1, [a_dagger_2, a_dagger_1, -a_2, -a_1])
          qubit_hamiltonian = self._get_ucc_spin_hamiltonian(fermionic_term)
          self.ucc_terms.append([0, fermionic_term, qubit_hamiltonian])
      # One up and one down
      else:
        for a_dagger_1 in unoccupied_spin_up_indices:
          for a_dagger_2 in unoccupied_spin_down_indices:
            if a_dagger_1 > a_dagger_2:
              ind_1, ind_2 = a_dagger_1, a_dagger_2
            else:
              ind_1, ind_2 = a_dagger_2, a_dagger_1
            fermionic_term = fermionic_utils.FermionicOperator(
                1, [ind_1, ind_2, -a_2, -a_1])
            qubit_hamiltonian = self._get_ucc_spin_hamiltonian(fermionic_term)
            self.ucc_terms.append([0, fermionic_term, qubit_hamiltonian])

  def _transform_fermion_to_spin(self, fermionic_operator, num_qubits):
    """Transforms a Fermionic term to a sum of PauliString objects.

    Here we use the Jordan Wigner transformation.

    Args:
      fermionic_operator: A FermionOperator object.
      num_qubits: Integer. Total number of qubits.

    Returns:
      List of PauliString objects. The PauliString objects summed together are
      the transformed fermionic_operator.
    """
    return fermionic_utils.jordan_wigner_transform(fermionic_operator,
                                                   num_qubits)

  def _get_ucc_spin_hamiltonian(self, fermionic_operator):
    """Jordan Wigner transforms a FermionicOperator to a QubitHamiltonian.

    Calculates the Jordan Wigner transformation of
    (fermionic_operator - fermionic_operator^dagger), which is antihermitian.
    It returns the result divided by (-i) so the return value is hermitian
    and corresponds to the spin_hamiltonian[k] in the class description.

    Args:
      fermionic_operator: A FermionicOperator object.

    Returns:
      A QubitHamiltonian object.

    Raises:
      ErrorCoupledCluster: If spin_hamiltonian wouldn't be hermitian.
    """
    # Create - fermionic_operator^dagger
    dagger_operator = fermionic_operator.get_hermitian_conjugate()
    dagger_operator.coefficient = -1 * dagger_operator.coefficient
    # Transform fermionic_operator - fermionic_operator^dagger to spin operators
    spin_terms = self._transform_fermion_to_spin(fermionic_operator,
                                                 self.num_qubits)
    spin_terms += self._transform_fermion_to_spin(dagger_operator,
                                                  self.num_qubits)
    # Combine PauliString objects with identical pauli terms
    new_spin_terms = []
    for term in spin_terms:
      # Check if term is alread in new_spin_terms
      already_there = False
      for new_term in new_spin_terms:
        if new_term.is_identical_pauli(term):
          new_term.coefficient += term.coefficient
          already_there = True
          break
      if not already_there:
        new_spin_terms.append(term)
    # Remove terms which are too small and divide all terms by -i
    tolerance = 10e-6
    hamiltonian = hamiltonian_utils.QubitHamiltonian()
    for new_term in new_spin_terms:
      if abs(new_term.coefficient) > tolerance:
        if new_term.coefficient.real > tolerance:
          raise ErrorCoupledCluster("JW transformed term is not anti hermitian")
        else:
          # Dividing by (-i)
          new_term.coefficient = -1 * new_term.coefficient.imag
          hamiltonian.add_term(new_term)
    return hamiltonian

  def get_parameters(self):
    """Retuns parameters of each fermionic term as a list.

    Returns:
      List of parameters theta[k]
    """
    parameters = []
    for term in self.ucc_terms:
      parameters.append(term[0])
    return parameters

  def set_parameters(self, new_parameters):
    """Assign parameters of each fermionic term.

    Args:
      new_parameters: List of float. Will be assigned to fermionic terms

    Raises:
      ErrorCoupledCluster: If parameter list is longer than ucc_terms.
    """
    if len(new_parameters) != len(self.ucc_terms):
      raise ErrorCoupledCluster("Different number of terms then parameters")
    for ii in range(len(new_parameters)):
      if new_parameters[ii].imag:
        raise ErrorCoupledCluster("Parameters need to be real")
      self.ucc_terms[ii][0] = new_parameters[ii]

  def set_parameters_with_dict(self, fermionic_term_dict):
    """Set parameters according to value in input dictionary.

    ucc_terms which are not in dictionary are set to 0.

    Args:
      fermionic_term_dict: dictionary where keys are tuples of fermionic
                           operators. Value is the amplitude for this term.
                           Hence, if self.ucc_terms[k][1].fermionic_operator
                           is in the dictionary then theta[k] will be set
                           to value in dictionary.
    Raises:
      ErrorCoupledCluster: If fermionic_term_dict contains key which is
      not in ucc_terms.
    """
    terms_found = 0
    for term in self.ucc_terms:
      if tuple(term[1].fermionic_operator) in fermionic_term_dict:
        term[0] = fermionic_term_dict[tuple(term[1].fermionic_operator)]
        terms_found += 1
      else:
        term[0] = 0
    if terms_found != len(fermionic_term_dict):
      raise ErrorCoupledCluster(
          "Dictionary contains term not found in ucc_terms")

  def set_parameters_to_cc_amplitudes(self, filename, closed_shell):
    cc_amplitudes = read_cc_file(filename)
    ucc_amplitude_dict = convert_cc_amplitudes_to_ucc(cc_amplitudes,
                                                      self.num_electrons,
                                                      closed_shell)
    self.set_parameters_with_dict(ucc_amplitude_dict)


def read_cc_file(inputfile):
  """Reads new cc file from Jarrod.

  Args:
    inputfile: String. Path to inputfile.
  Returns:
    List of lists. [TIA Amplitudes, Tia Amplitudes, TIJAB Amplitudes,
    Tijab Amplitudes, TIjAb Amplitudes]
  Raises:
    ErrorCoupledCluster: Wrong file format.
  """
  t1_up_up = []
  t1_down_down = []
  t2_up_up_up_up = []
  t2_down_down_down_down = []
  t2_up_down_up_down = []
  with open(inputfile, "r") as f:
    # Reading TIA amplitudes
    if next(f).split()[0] != "TIA":
      raise ErrorCoupledCluster("Didn't find TIA section in inputfile")
    for line in f:
      if line == "\n":  # After empty line Tia amplitudes start
        break
      elements = line.split()
      if len(elements) != 3:
        raise ErrorCoupledCluster("TIA needs three elements per line")
      t1_up_up.append([int(elements[0]), int(elements[1]), float(elements[2])])
    # Reading Tia amplitudes
    if next(f).split()[0] != "Tia":
      raise ErrorCoupledCluster("Didn't find Tia section in inputfile")
    for line in f:
      if line == "\n":  # After empty line TIJAB amplitudes start
        break
      elements = line.split()
      if len(elements) != 3:
        raise ErrorCoupledCluster("Tia needs three elements per line")
      t1_down_down.append(
          [int(elements[0]), int(elements[1]), float(elements[2])])
    # Reading TIJAB amplitudes
    if next(f).split()[0] != "TIJAB":
      raise ErrorCoupledCluster("Didn't find TIJAB section in inputfile")
    for line in f:
      if line == "\n":  # After empty line Tijab amplitudes start
        break
      elements = line.split()
      if len(elements) != 5:
        raise ErrorCoupledCluster("TIJAB needs five elements per line")
      t2_up_up_up_up.append([int(elements[0]), int(elements[1]),
                             int(elements[2]), int(elements[3]),
                             float(elements[4])])
    # Reading Tijab amplitudes
    if next(f).split()[0] != "Tijab":
      raise ErrorCoupledCluster("Didn't find Tijab section in inputfile")
    for line in f:
      if line == "\n":  # After empty line TIjAb amplitudes start
        break
      elements = line.split()
      if len(elements) != 5:
        raise ErrorCoupledCluster("Tijab needs five elements per line")
      t2_down_down_down_down.append([int(elements[0]), int(elements[1]),
                                     int(elements[2]), int(elements[3]),
                                     float(elements[4])])
    # Reading TIjAb amplitudes
    if next(f).split()[0] != "TIjAb":
      raise ErrorCoupledCluster("Didn't find TIjAb section in inputfile")
    for line in f:
      if line == "\n":  # Finished reading the file
        break
      elements = line.split()
      if len(elements) != 5:
        raise ErrorCoupledCluster("TIjAb needs five elements per line")
      t2_up_down_up_down.append([int(elements[0]), int(elements[1]),
                                 int(elements[2]), int(elements[3]),
                                 float(elements[4])])
  return [t1_up_up, t1_down_down, t2_up_up_up_up, t2_down_down_down_down,
          t2_up_down_up_down]


def convert_cc_numbering_to_spin_orbital(spatial_orbital, spin, occupied,
                                         num_electrons):
  """Converts numbering of spin orbitals from cc file to our numbering.

  Numbering of spin orbitals in cc file:
  occupied spin up orbitals: 0, 1, 2, ...
  occupied spin down orbitals: 0, 1, 2, ...
  unoccupied spin up orbitals: 0, 1, 2, ...
  unoccupied_spin_down orbitals: 0, 1, 2, ...

  Our spin orbital indices are number from 1,2,... with odd numbers for spin up
  and even numbers for spin down.

  Args:
    spatial_orbital: integer from cc file.
    spin: bool. True of spin up and False for spin down
    occupied: bool. True if spatial_orbital number is from an occupied orbital
    num_electrons: integer.
  Returns:
    Integer representing the spin orbital index.
  """
  if occupied:
    if spin:
      spin_orbital_index = spatial_orbital * 2 + 1
    else:
      spin_orbital_index = spatial_orbital * 2 + 2
  if not occupied:
    if num_electrons % 2 == 1:  # Odd number of electrons
      highest_spin_up_occ_spin_orbital = (num_electrons // 2) * 2 + 1
      highest_spin_down_occ_spin_orbital = (num_electrons // 2 - 1) * 2 + 2
    else:  # Even number of electrons
      highest_spin_up_occ_spin_orbital = (num_electrons // 2 - 1) * 2 + 1
      highest_spin_down_occ_spin_orbital = (num_electrons // 2 - 1) * 2 + 2
    if spin:
      spin_orbital_index = highest_spin_up_occ_spin_orbital + (spatial_orbital +
                                                               1) * 2
    else:
      spin_orbital_index = highest_spin_down_occ_spin_orbital + (spatial_orbital
                                                                 + 1) * 2
  return spin_orbital_index


def convert_cc_amplitudes_to_ucc(cc_amplitudes, num_electrons, closed_shell):
  """Converts cc amplitudes to our unitary coupled cluster amplitudes.

  This function converts the coupled cluster amplitudes from Psi4 and
  converts them to amplitudes of unitary coupled clusters.

  Note: When tranforming spatial to spin orbitals, we only allow transitions
        which preserve the spin.

  Args:
    cc_amplitudes: cc_amplitudes from read_cc_file
    num_electrons: int. Used to determine the number of occupied spin orbitals.
    closed_shell: bool. True if system is closed shell

  Returns:
    dictionary containing ucc amplitudes. Key are tuples of fermionic operators,
    value is the corresponding amplitude.
  """
  # Closed shell
  # Psi4 outputs only TIjAb and TIA amplitudes for closed shell.
  # Here we create the other amplitudes:
  # We add all transition from TIjAb to TIJAB and Tijab which are possible. An
  # example of a TIjAb transition which is impossible add would be 0 0 1 1, as
  # this transition involves two electrons from the same spatial orbital and
  # therefore must have opposite spin. Note, that the amplitudes in TIJAB and
  # Tijab are half the strength of the TIjAb amplitudes.
  if closed_shell:
    t1_up_up = cc_amplitudes[0]
    t1_down_down = cc_amplitudes[0]
    t2_up_up_up_up = []
    t2_down_down_down_down = []
    for spatial_term in cc_amplitudes[4]:
      if (spatial_term[0] != spatial_term[1] and
          spatial_term[2] != spatial_term[3]):
        new_spatial_term = list(spatial_term)  # Makes a copy
        new_spatial_term[4] /= 2.
        t2_up_up_up_up.append(new_spatial_term)
        t2_down_down_down_down.append(new_spatial_term)
    t2_up_down_up_down = cc_amplitudes[4]
  # Open shell
  else:
    t1_up_up = cc_amplitudes[0]
    t1_down_down = cc_amplitudes[1]
    t2_up_up_up_up = cc_amplitudes[2]
    t2_down_down_down_down = cc_amplitudes[3]
    t2_up_down_up_down = cc_amplitudes[4]

  ucc_amplitudes = {}

  for occ_spin_up, unocc_spin_up, amplitude in t1_up_up:
    index_i = convert_cc_numbering_to_spin_orbital(occ_spin_up, True, True,
                                                   num_electrons)
    index_a = convert_cc_numbering_to_spin_orbital(unocc_spin_up, True, False,
                                                   num_electrons)
    ucc_amplitudes[(index_a, -index_i)] = amplitude

  for occ_spin_down, unocc_spin_down, amplitude in t1_down_down:
    index_i = convert_cc_numbering_to_spin_orbital(occ_spin_down, False, True,
                                                   num_electrons)
    index_a = convert_cc_numbering_to_spin_orbital(unocc_spin_down, False,
                                                   False, num_electrons)
    ucc_amplitudes[(index_a, -index_i)] = amplitude

  for occ_up_1, occ_up_2, unocc_up_1, unocc_up_2, amplitude in t2_up_up_up_up:
    index_i = convert_cc_numbering_to_spin_orbital(occ_up_1, True, True,
                                                   num_electrons)
    index_j = convert_cc_numbering_to_spin_orbital(occ_up_2, True, True,
                                                   num_electrons)
    index_a = convert_cc_numbering_to_spin_orbital(unocc_up_1, True, False,
                                                   num_electrons)
    index_b = convert_cc_numbering_to_spin_orbital(unocc_up_2, True, False,
                                                   num_electrons)
    # Normal ordering
    if index_i < index_j:
      index_i, index_j = index_j, index_i
      amplitude = -amplitude
    if index_a < index_b:
      index_a, index_b = index_b, index_a
      amplitude = -amplitude

    if (index_a, index_b, -index_i, -index_j) in ucc_amplitudes:
      ucc_amplitudes[(index_a, index_b, -index_i, -index_j)] += -amplitude
    else:
      ucc_amplitudes[(index_a, index_b, -index_i, -index_j)] = -amplitude

  for (occ_down_1, occ_down_2, unocc_down_1, unocc_down_2,
       amplitude) in t2_down_down_down_down:
    index_i = convert_cc_numbering_to_spin_orbital(occ_down_1, False, True,
                                                   num_electrons)
    index_j = convert_cc_numbering_to_spin_orbital(occ_down_2, False, True,
                                                   num_electrons)
    index_a = convert_cc_numbering_to_spin_orbital(unocc_down_1, False, False,
                                                   num_electrons)
    index_b = convert_cc_numbering_to_spin_orbital(unocc_down_2, False, False,
                                                   num_electrons)
    # Normal ordering
    if index_i < index_j:
      index_i, index_j = index_j, index_i
      amplitude = -amplitude
    if index_a < index_b:
      index_a, index_b = index_b, index_a
      amplitude = -amplitude

    if (index_a, index_b, -index_i, -index_j) in ucc_amplitudes:
      ucc_amplitudes[(index_a, index_b, -index_i, -index_j)] += -amplitude
    else:
      ucc_amplitudes[(index_a, index_b, -index_i, -index_j)] = -amplitude

  for occ_up, occ_down, unocc_up, unocc_down, amplitude in t2_up_down_up_down:
    index_i = convert_cc_numbering_to_spin_orbital(occ_up, True, True,
                                                   num_electrons)
    index_j = convert_cc_numbering_to_spin_orbital(occ_down, False, True,
                                                   num_electrons)
    index_a = convert_cc_numbering_to_spin_orbital(unocc_up, True, False,
                                                   num_electrons)
    index_b = convert_cc_numbering_to_spin_orbital(unocc_down, False, False,
                                                   num_electrons)
    # Normal ordering
    if index_i < index_j:
      index_i, index_j = index_j, index_i
      amplitude = -amplitude
    if index_a < index_b:
      index_a, index_b = index_b, index_a
      amplitude = -amplitude

    if (index_a, index_b, -index_i, -index_j) in ucc_amplitudes:
      ucc_amplitudes[(index_a, index_b, -index_i, -index_j)] += -amplitude
    else:
      ucc_amplitudes[(index_a, index_b, -index_i, -index_j)] = -amplitude

  return ucc_amplitudes
