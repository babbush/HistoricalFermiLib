import symbolic_fermions
import fermi_hubbard
import unittest

class SymbolicFermionsTest(unittest.TestCase):

  def testCheckEvaluatePauliProduct(self):
    self.assertEqual(
        symbolic_fermions.EvaluatePauliProduct('XXX'), (1.0, 'X'))
    self.assertEqual(
        symbolic_fermions.EvaluatePauliProduct('XXZZ'), (1.0, 'I'))
    self.assertEqual(
        symbolic_fermions.EvaluatePauliProduct('YXXZZ'), (1.0, 'Y'))

  def testSimplifyPauliString(self):
    test_example = ['XXX', 'XXZZ', 'YXXZZ']
    test_solution = (1.0, ['X', 'I', 'Y'])
    function_output = symbolic_fermions.SimplifyPauliString(test_example)
    self.assertEqual(test_solution[0], function_output[0])
    for tensor_factor in range(len(test_solution[1])):
      self.assertEqual(test_solution[1][tensor_factor],
                       function_output[1][tensor_factor])

  def testSymbolicOperatorJW(self):
    n_qubits = 20
    for index in range(-n_qubits, n_qubits + 1):
      if index:
        [x_output, y_output] = symbolic_fermions.SymbolicOperatorJW(
            index, n_qubits)[1]
        qubit = abs(index) - 1
        for tensor_factor in range(n_qubits):
          if tensor_factor < qubit:
            self.assertEqual(x_output[tensor_factor], 'Z')
            self.assertEqual(y_output[tensor_factor], 'Z')
          if tensor_factor == qubit:
            self.assertEqual(x_output[tensor_factor], 'X')
            self.assertEqual(y_output[tensor_factor], 'Y')
          if tensor_factor > qubit:
            self.assertEqual(x_output[tensor_factor], 'I')
            self.assertEqual(y_output[tensor_factor], 'I')

  def testSymbolicTermJW(self):
    n_qubits = 10
    fermion_coefficient = 1.
    for qubit in range(1, n_qubits):
      fermion_term = [qubit, -qubit]
      pauli_strings = symbolic_fermions.SymbolicTermJW(
          fermion_coefficient, fermion_term, n_qubits)[1]
      reduced_strings = [symbolic_fermions.SimplifyPauliString(
          pauli_string)[1] for pauli_string in pauli_strings]
      for reduced_string in reduced_strings:
        for tensor_factor in range(n_qubits):
          if tensor_factor != (qubit - 1):
            self.assertEqual(reduced_string[tensor_factor], 'I')
          elif reduced_string[tensor_factor] != 'I':
            self.assertEqual(reduced_string[tensor_factor], 'Z')

  def testSymbolicTransformationJW(self):
    x_dimension = 3
    y_dimension = 1
    tunneling = 2.
    coloumb = 0.
    chemical_potential = 0.
    magnetic_field = 0.
    verbose = 1
    periodic = 0
    spinless = 1

    # Get fermionic coefficients and terms..
    coefficients, terms = fermi_hubbard.FermiHubbardSymbolic(x_dimension,
                                                             y_dimension,
                                                             tunneling, coloumb,
                                                             chemical_potential,
                                                             magnetic_field,
                                                             periodic, spinless,
                                                             verbose)

    # Get unique spin operator representation.
    unique_terms = symbolic_fermions.SymbolicTransformationJW(coefficients,
                                                              terms)

    # Test.
    self.assertTrue(('X', 'X', 'I') in unique_terms)
    self.assertTrue(('I', 'X', 'X') in unique_terms)
    self.assertTrue(('Y', 'Y', 'I') in unique_terms)
    self.assertTrue(('I', 'Y', 'Y') in unique_terms)
    self.assertEqual(unique_terms[('X', 'X', 'I')],
                     unique_terms[('Y', 'Y', 'I')])
    self.assertEqual(unique_terms[('I', 'X', 'X')],
                     unique_terms[('I', 'Y', 'Y')])
    self.assertFalse(('Z', 'Z', 'I') in unique_terms)
    self.assertFalse(('I', 'Z', 'Z') in unique_terms)
    self.assertFalse(('X', 'I', 'X') in unique_terms)
    self.assertFalse(('Y', 'I', 'Y') in unique_terms)


# Run test.
if __name__ == '__main__':
  unittest.main()
