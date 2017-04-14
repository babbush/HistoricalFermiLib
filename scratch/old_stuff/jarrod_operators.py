"""
Operators.py -- Module for symbolic and numerical manipulation of operators
common to quantum chemistry and quantum computing.
"""

__author__ = "Jarrod R. McClean"
__email__ = "jarrod.mcc@gmail.com"

import numpy as np
import scipy as sp
import scipy.linalg
import cPickle as pickle
import itertools
import re

#A few of the classes below use sympy for special cases, but it's not required for most operations
use_sympy = True
try:
    import sympy
    import sympy.physics
    import sympy.physics.secondquant
    from sympy import KroneckerDelta
except:
    use_sympy = False
    pass

class PauliOperatorFactory(object):
    """Class to process Pauli operator strings and their variants, both
	symbolically and numerically in a Kronecker product representation.  Also
	converts to Gate sequences from Pauli operators strings using standard circuit
	synthesis recipies."""
    def __init__(self, n_qubits=None, encoding="JW"):
        """Initializes a Pauli operator factory with n_qubits in the specified encoding
        
        A fundamental data structure of this class is the Pauli operator string, which
        takes the form of "[XYZ][0-9]+" or "I", which represents a series of Pauli
        X, Y, Z operators followed by the qubit number it acts on.
        
        Args:
        n_qubits (int): Number of qubits the operators may act on
        encoding (str): 'JW' for Jordan-Wigner or 'BK' for Bravyi-Kitaev
        """
        self.n_qubits = n_qubits
        self.encoding = encoding
        self.one_body_int = None
        self.two_body_int = None
        self.e_nuc = None
		#Numerical representations of Pauli operators for building Kronecker products
        self.op_dict = {
            '':np.eye(2),
            'I':np.eye(2),
            'X':np.array([[0.0, 1.0],[1.0, 0.0]], dtype = complex),
            'Y':np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype = complex),
            'Z':np.array([[1.0, 0.0],[0.0, -1.0]], dtype = complex),
            'P0':np.array([[1.0, 0.0], [0.0, 0.0]], dtype = complex),
            'P1':np.array([[0.0, 0.0], [0.0, 1.0]], dtype = complex)}

        #Define sets for BK transformation
        self.bk_update_set = None
        self.bk_parity_set = None
        self.bk_remainder_set = None
        self.bk_rho_set = None

        #Define dictionary for symbolic products of single Pauli operators, letting 'I'=''
        self.prod_dict_op={'ZZ':'','YY':'','XX':'','II':'',
                           'XY':'Z','XZ':'Y', 'YX':'Z','YZ':'X','ZX':'Y',
                           'ZY':'X','IX':'X','IY':'Y','IZ':'Z',
                           'ZI':'Z','YI':'Y','XI':'X', 
                           'X':'X', 'Y':'Y', 'Z':'Z', '':'', 'I':''}
        self.prod_dict_coeff={'ZZ':1.0,'YY':1.0,'XX':1.0,'II':1.0,
                              'XY':1.0j,'XZ':-1.0j,'YX':-1.0j,'YZ':1.0j,'ZX':1.0j,
                              'ZY':-1.0j,'IX':1.0,'IY':1.0,'IZ':1.0,'ZI':1.0,
                              'YI':1.0,'XI':1.0, 
                              'X':1.0, 'Y':1.0, 'Z':1.0, '':1.0, 'I':1.0}

        #Define dictionary for QASM translation to Kronecker representation
        self.qasm_dict = {
            'G': (lambda theta: np.exp(-1.0j * float(theta)) * np.eye(2**self.n_qubits)),
            'CNOT': (lambda control, target: self.embed_cnot(int(control), int(target))),
            'H': (lambda q: self.embed_op(1./np.sqrt(2) * np.array([[1.0, 1.0], [1.0, -1.0]]), 
                                          int(q))),
            'X': (lambda q: self.embed_op(self.op_dict['X'], int(q))),
            'Yb':(lambda q: self.embed_op(1./np.sqrt(2) * np.array([[1.0, 1.0j], [1.0j, 1.0]]), 
                                          int(q))),
            'Ybd':(lambda q: self.embed_op(1./np.sqrt(2) * np.array([[1.0, -1.0j], [-1.0j, 1.0]]), 
                                          int(q))),
            'Rx': (lambda theta, q: self.embed_op(sp.linalg.expm(-1.0j * float(theta)/2.0 * self.op_dict['X']),
                                                  int(q))),
            'Rz': (lambda theta, q: self.embed_op(sp.linalg.expm(-1.0j * float(theta)/2.0 * self.op_dict['Z']),
                                                  int(q)))}

    def tokenize_op(self, op):
        """Take an operator string op, and return tokenized list of operators"""
        tmp_list = [''] * self.n_qubits

        #Check for trivial list
        if (op == 'I' or op == ''): return tmp_list

        #Otherwise split based on string pattern and number of qubits
        tok_list = re.findall('[XYZ][0-9]+', op)
        for val in tok_list:
            label, ind = val[0], int(val[1:])
            tmp_list[ind] = label
        return tmp_list
    
    def untokenize_op(self, op):
        """Return the operator string form of a tokenized operator"""
        return ''.join(['{}{}'.format(val, i) for (i,val) in enumerate(op) if val != ''])
        
    def product(self, op1, op2):
        """Return the operator string resulting from op1*op2"""
        coeff_factor = 1.0
        result_list = [''] * self.n_qubits
        tok_op1 = self.tokenize_op(op1)
        tok_op2 = self.tokenize_op(op2)
        for i in range(self.n_qubits):
            result_list[i] = self.prod_dict_op[tok_op1[i]+tok_op2[i]]
            coeff_factor *= self.prod_dict_coeff[tok_op1[i]+tok_op2[i]]

        result_op = self.untokenize_op(result_list)
        if (result_op == ''): result_op = 'I'

        return coeff_factor, result_op
                
    def list_product(self, coeff_list1, op_list1, coeff_list2, op_list2):
        """Return the product of two lists of operator strings"""
        result_coeffs = []
        result_ops = []
        for i, val1 in enumerate(coeff_list1):
            for j, val2 in enumerate(coeff_list2):
                coeff_factor, result_op = self.product(op_list1[i], op_list2[j])
                result_coeffs.append(val1 * val2 * coeff_factor)
                result_ops.append(result_op)
        return result_coeffs, result_ops
    
    def embed_op(self, op, q):
        """Embed a 1 qubit operator op in the Kronecker product space at q"""
        assert (q < self.n_qubits)

        base = np.array([1.0], dtype=complex)
        for i in range(self.n_qubits):
            base = np.kron(base, op) if (i==q) else np.kron(base, np.eye(2))
        return base

    def embed_cnot(self, control, target):
        """Embed a CNOT operator in the kronecker product space"""
        assert (control != target)
        #CNOT = (I_t P_{0c} + NOT_t P_{1c}

        #I_t P_{0c}
        base1 = np.array([1.0], dtype=complex)
        for i in range(self.n_qubits):
            base1 = np.kron(base1, self.op_dict['P0']) if (i==control) else np.kron(base1, np.eye(2))

        #X_t P_{1c}
        base2 = np.array([1.0],dtype=complex)
        for i in range(self.n_qubits):
            if (i == control):
                base2 = np.kron(base2, self.op_dict['P1'])
            elif (i == target):
                base2 = np.kron(base2, self.op_dict['X'])
            else:
                base2 = np.kron(base2, np.eye(2))

        return base1 + base2

    def kron_rep(self, coeff, op):
        """Return the Kronecker product representation of an operator string"""
        tok_op = self.tokenize_op(op)
        base = np.array([1.0], dtype=complex)
        for label in tok_op:
            base = np.kron(base, self.op_dict[label])

        return coeff * base

    def kron_rep_list(self, coeff_list, op_list):
        """Build a matrix representation with the Kronecker product of coeff_list, op_list"""
        O = np.zeros((2**self.n_qubits, 2**self.n_qubits), dtype=complex)
        
        for i, v in enumerate(coeff_list):
            O += self.kron_rep(v, op_list[i])

        return O

    def kron_element_list(self, coeff_list, op_list):
        """Return a list of Kron matrix representations of all the operators in op_list"""
        kr_op_list = []
        for i, v in enumerate(coeff_list):
            kr_op_list.append(self.kron_rep(1.0, op_list[i]))
        return coeff_list, kr_op_list

    def reduce_list(self, coeff_list, op_list, simplify=False, threshold=1e-12):
        """Combine commen elements in op_list, adjusting coeff_list accordingly
		
        Args:
        coeff_list - List of complex numbers representing coefficients of operators in op_list,
        may also be sympy coefficients for symbolic manipulation
        op_list - List of Pauli operators strings
        simplify (bool) - Use sympy simplificaton.  Most useful when coeff_list is symbolic.
        threshold (float) - Magnitude threshold for considering an element 0 and removing it.
        
        Returns:
        reduced_coeffs - Coeff list of reduced operator
        reduced_ops - Operator list of reduced operator
        """
        #Catch empty lists
        if (len(coeff_list) == 0): return coeff_list, op_list

        #Sort by op name
        sorted_coeffs, sorted_ops = zip(*sorted(zip(coeff_list, op_list), 
                                           key=lambda pair: pair[1]))
        #Iterate through and combine elements
        tmp_coeffs = []
        tmp_ops = []
        for i, val in enumerate(sorted_coeffs):
            if (len(tmp_coeffs) > 0 and tmp_ops[-1] == sorted_ops[i]):
                tmp_coeffs[-1] += sorted_coeffs[i]
            else:
                tmp_coeffs.append(sorted_coeffs[i])
                tmp_ops.append(sorted_ops[i])

        reduced_coeffs = []
        reduced_ops = []

        #Remove elements with 0 coeff
        for i, val in enumerate(tmp_coeffs):
            if (((not use_sympy) and np.abs(val) > threshold) or
                (use_sympy and sympy.S(val).is_number and (np.abs(val) > threshold))):
                reduced_coeffs.append(val)
                reduced_ops.append(tmp_ops[i])
            elif ((use_sympy) and (not sympy.S(val).is_number) and sympy.simplify(val) != 0):
                reduced_coeffs.append(sympy.simplify(val))
                reduced_ops.append(tmp_ops[i])

        return reduced_coeffs, reduced_ops

    def op_exp_qasm(self, coeff, op_str):
        """Return the QASM gate sequence of the exponential of an operator string,
        namely exp[-1.0j * coeff * op_str]"""

        #Special case for identity element to global phase
        if (op_str == 'I' or op_str == ''): return ['G {}'.format(np.real(coeff))]

        #Process the rest
        gate_seq = []
        tok_op = self.tokenize_op(op_str)

        #Change basis to all Z's
        for i, op in enumerate(tok_op):
            if ('X' in op): gate_seq.append('H {}'.format(i))
            elif ('Y' in op): gate_seq.append('Yb {}'.format(i))

        #CNOT Ladder
        cnot_seq = [] 
        qubits = re.findall('[0-9]+',op_str)
        for i, q in enumerate(qubits[:-1]):
            cnot_seq.append('CNOT {} {}'.format(int(q), qubits[i+1]))
        gate_seq.extend(cnot_seq)        
        
        #Rotation
        gate_seq.append('Rz {} {}'.format(2.0 * np.real(coeff), qubits[-1]))
        
        #Reverse CNOT Ladder
        gate_seq.extend(cnot_seq[::-1])

        #Change basis back to original
        for i, op in enumerate(tok_op):
            if ('X' in op): gate_seq.append('H {}'.format(i))
            elif ('Y' in op): gate_seq.append('Ybd {}'.format(i))
        
        return gate_seq
        
    def op_list_exp_qasm(self, coeff_list, op_list):
        """Return the QASM gate sequence from exponentiating a sequence of operators,
		 namely exp[-1.0j * coeff * op_str] for each coeff and op_str in the list"""
        gate_seq = []

        for i, v in enumerate(coeff_list):
            gate_seq.extend(self.op_exp_qasm(v, op_list[i]))

        return gate_seq

    def op_exp_kron(self, coeff, op_str):
        """Return the Kronecker product representation of the exponential of an op string,
        specifically exp[-1.0j * coeff * op_str]"""
        kron_op = self.kron_rep(coeff, op_str)
        return sp.linalg.expm(-1.0j * kron_op)
    
    def op_list_exp_kron(self, coeff_list, op_list):
        """Return the Kronecker product rep of the exponentials of a series of op_strings,
        in particular Prod_i(exp[-1.0j * coeffs[i] * op_strs[i])"""

        U = np.eye(2**self.n_qubits)

        for i, coeff in enumerate(coeff_list):
            U = np.dot(self.op_exp_kron(coeff, op_list[i]), U)
        
        return U

    def qasm_to_kron(self, qasm_seq):
        """Return a kronecker product representation of the unitary given by a QASM sequence"""
        U = np.eye(2**self.n_qubits, dtype=complex)

        for op in qasm_seq:
            gate = op.split()[0]
            args = op.split()[1:]
            U_op = self.qasm_dict[gate](*args)
            U = np.dot(U_op, U)
        
        return U
    
    def fermi_to_pauli(self, coeff, indices, conjugate, bos=False, no_reduce=False):
        """Convert a list of site indices and coefficient to a Pauli op 
        string list, with creation operators marked by -1 from [-1,1] in conjugate. The
        representation  (JW vs BK) is determined by the class' encoding.
        
        Args:
        coeff (complex) - Coefficient of the term
        indices - List of ints specifying the site the fermionic operator acts on, e.g. [0,2,4,6]
        conjugate - List of -1, 1 specifying which of the indices are creation operators (-1) and which
        are annihilation operators (1).  e.g. [-1,-1,1,1]
        bos (bool) - Use quasi-bosonization transformations that ignore parity tracking
        """

        if (self.encoding == "JW"):
            return self.JW_transform(coeff, indices, conjugate, bos=bos, no_reduce=no_reduce)
        elif (self.encoding == "BK"):
            return self.BK_transform(coeff, indices, conjugate, bos=bos, no_reduce=no_reduce)
        elif (self.encoding == "P"):
            return self.P_transform(coeff, indices, conjugate, bos=bos, no_reduce=no_reduce)

    def JW_transform(self, coeff, indices, conjugate, bos=False, no_reduce=False):
        """Convert a list of site indices and coefficients to a Pauli Op String
        list with the Jordan-Wigner (JW) transformation
		
        Args:
        coeff (complex) - Coefficient of the term
        indices - List of ints specifying the site the fermionic operator acts on, e.g. [0,2,4,6]
        conjugate - List of -1, 1 specifying which of the indices are creation operators (-1) and which
        are annihilation operators (1).  e.g. [-1,-1,1,1]
        bos (bool) - Use quasi-bosonization transformations that ignore parity tracking
        """
        
        N = self.n_qubits
        
        coeff_list = [1.0]
        op_list = ['I']
        
        for pos, i in enumerate(indices):
            tmp_coeff_list = []
            tmp_op_list = []
            
            tmp_coeff_list.append(0.5)
            tmp_op_list.append('X{}'.format(i) 
                                + (''.join(['Z{}'.format(n) for n in range(i+1, N)]) if (not bos) else ''))

            y_factor = conjugate[pos]
            tmp_coeff_list.append(0.5 * y_factor * 1.0j)
            tmp_op_list.append('Y{}'.format(i) 
                                + (''.join(['Z{}'.format(n) for n in range(i+1, N)]) if (not bos) else ''))
            coeff_list, op_list = self.list_product(coeff_list, op_list,
                                                    tmp_coeff_list, tmp_op_list)

        coeff_list = [coeff * x for x in coeff_list]

        if (no_reduce):
            return coeff_list, op_list
        else:
            return self.reduce_list(coeff_list, op_list)
        
    def P_transform(self, coeff, indices, conjugate, bos=False, no_reduce=False):
        """Convert a list of site indices and coefficients to a Pauli Op String
        list with the Parity transformation
		
        Args:
        coeff (complex) - Coefficient of the term
        indices - List of ints specifying the site the fermionic operator acts on, e.g. [0,2,4,6]
        conjugate - List of -1, 1 specifying which of the indices are creation operators (-1) and which
        are annihilation operators (1).  e.g. [-1,-1,1,1]
        bos (bool) - Use quasi-bosonization transformations that ignore parity tracking, note that 
          this hasn't been fully tested for the parity transform
        """
        
        N = self.n_qubits
        
        coeff_list = [1.0]
        op_list = ['I']
        
        for pos, i in enumerate(indices):
            tmp_coeff_list = []
            tmp_op_list = []
            
            tmp_coeff_list.append(0.5)
            tmp_op_list.append(('Z{}'.format(i-1) if (i > 0 and (not bos)) else '')
                               + 'X{}'.format(i)
                               + ''.join(['X{}'.format(n) for n in range(i+1, N)]))

            y_factor = conjugate[pos]
            tmp_coeff_list.append(0.5 * y_factor * 1.0j)
            tmp_op_list.append('Y{}'.format(i)
                               + ''.join(['X{}'.format(n) for n in range(i+1, N)]))
            coeff_list, op_list = self.list_product(coeff_list, op_list,
                                                    tmp_coeff_list, tmp_op_list)

        coeff_list = [coeff * x for x in coeff_list]

        if (no_reduce):
            return coeff_list, op_list
        else:
            return self.reduce_list(coeff_list, op_list)

    def BK_transform(self, coeff, indices, conjugate, bos=False, no_reduce=False):
        """Convert a list of site indices and coefficients to a Pauli Op String list
        with Bravyi-Kitaev transformation
        
        Args:
        coeff (complex) - Coefficient of the term
        indices - List of ints specifying the site the fermionic operator acts on, e.g. [0,2,4,6]
        conjugate - List of -1, 1 specifying which of the indices are creation operators (-1) and which
        are annihilation operators (1).  e.g. [-1,-1,1,1]		
        bos (bool) - Use quasi-bosonization transformations that ignore parity tracking
        """
        N = self.n_qubits

        coeff_list = [1.0]
        op_list = ['I']

        if (self.bk_update_set == None):
            self.BK_compute_sets()

        for pos, i in enumerate(indices):
            tmp_coeff_list = []
            tmp_op_list = []
            
            tmp_coeff_list.append(0.5)
            tmp_op_list.append(''.join(['X{}'.format(n) for n in self.bk_update_set[i]])
                                + 'X{}'.format(i)
                                + (''.join(['Z{}'.format(n) for n in self.bk_parity_set[i]]) if (not bos) else ''))

            y_factor = conjugate[pos]
            tmp_coeff_list.append(0.5 * y_factor * 1.0j)
            tmp_op_list.append(''.join(['X{}'.format(n) for n in self.bk_update_set[i]])
                               + 'Y{}'.format(i)
                               + (''.join(['Z{}'.format(n) for n in self.bk_rho_set[i]]) if (True) else ''))

            coeff_list, op_list = self.list_product(coeff_list, op_list,
                                                    tmp_coeff_list, tmp_op_list)

        coeff_list = [coeff * x for x in coeff_list]
        if (no_reduce):
            return coeff_list, op_list
        else:
            return self.reduce_list(coeff_list, op_list)
        

    def BK_compute_sets(self):
        """Compute the index sets required for the Bravyi-Kitaev(BK) Transformation"""
        N = self.n_qubits

        #We compute the BK sets via the recursive methods outlined by 
        #Tranter et. al, IJQC 115, 1431-1441 (2015)

        #Number of recursion levels
        l_max = int(np.ceil(np.log2(self.n_qubits)))

        #Define base cases
        update_set_old = [[]]
        parity_set_old = [[]]
        flip_set_old = [[]]

        for level in range(1, l_max+1):
            n = 2**level
            update_set = [ [] for i in range(n)] 
            parity_set = [ [] for i in range(n)]
            flip_set = [ [] for i in range(n)]
 
            for j in range(n):
                if (j < n/2):
                    update_set[j] = update_set_old[j] + [n-1]
                    parity_set[j] = parity_set_old[j]
                    flip_set[j] = flip_set_old[j]
                else:
                    update_set[j] = [x + n/2 for x in update_set_old[j-n/2]]
                    parity_set[j] = [x + n/2 for x in parity_set_old[j-n/2]] + [n/2 - 1]
                    if (j == n-1):
                        flip_set[j] = [x + n/2 for x in flip_set_old[j-n/2]] + [n/2 - 1]
                    else:
                        flip_set[j] = [x + n/2 for x in flip_set_old[j-n/2]]

            update_set_old = update_set
            parity_set_old = parity_set
            flip_set_old = flip_set

        remainder_set = [ list( set(parity_set[i])-set(flip_set[i]) ) for i, v in enumerate(parity_set) ]
        rho_set = [parity_set[j] if (j%2==0) else remainder_set[j] for j in range(2**(l_max))]

        #Record and remove elements referencing excess qubits
        self.bk_update_set = [ [x for x in q if (x < N)] for q in update_set[:N]]
        self.bk_parity_set = [ [x for x in q if (x < N)] for q in parity_set[:N]]
        self.bk_rho_set = [ [x for x in q if (x < N)] for q in rho_set[:N]]

    def JW_inverse_transform(self,coeff,op):
        """Restricted inverse Jordan-Wigner transform from Pauli Operator to weighted sum over fermionic operators,
        currently customized to work only for particle conserving like terms in the Hamiltonian.  Use transforms
        as defined in Michael Nielsens notes on the JW transformation.

        Args:
        coeff - complex coefficient of the Pauli operator
        op - Pauli op string to be transformed

        Returns:
        coeff_list  - list of complex coefficients of fermionic operators
        op_list - List of fermionic op_strings 
        """
        
        tok_op = self.tokenize_op(op)
        fermi_coeff_list = []
        fermi_op_list = []

        #Add operators 1 at a time
        for i, v in reversed(list(enumerate(tok_op))):
            tmp_coeff_list, tmp_op_list = [], []

            if (tok_op[i] == 'X'):
                tmp_coeff, tmp_op = np.array([-1.,-1.]),['A{}'.format(i),'B{}'.format(i)]
                #Z-Transform lower operators
                for j in reversed(range(i)):
                    tok_op[j] = self.prod_dict_op[tok_op[j] + 'Z']
                    tmp_coeff = self.prod_dict_coeff[tok_op[j] + 'Z'] * tmp_coeff
            elif (tok_op[i] == 'Y'):
                tmp_coeff, tmp_op = np.array([1.j,-1.j]),['A{}'.format(i),'B{}'.format(i)]
                #Z-Transform lower operators
                for j in reversed(range(i)):
                    tok_op[j] = self.prod_dict_op[tok_op[j] + 'Z']
                    tmp_coeff = self.prod_dict_coeff[tok_op[j] + 'Z'] * tmp_coeff
            elif (tok_op[i] == 'Z'):
                #tmp_coeff, tmp_op = [-2.,1],['A{}B{}'.format(i,i),'']
                tmp_coeff, tmp_op = np.array([-1.,1.]),['A{}B{}'.format(i,i), 'B{}A{}'.format(i,i)]
            elif (tok_op[i] == '' or tok_op[i] == 'I'):
                continue

            if (not fermi_coeff_list):
                tmp_coeff_list.extend(tmp_coeff)
                tmp_op_list.extend(tmp_op)
            else:
                for j, v2 in enumerate(fermi_coeff_list):
                    for k, v3 in enumerate(tmp_coeff):
                        tmp_coeff_list.append(v3*v2)
                        tmp_op_list.append(tmp_op[k] + fermi_op_list[j])

            fermi_coeff_list, fermi_op_list = tmp_coeff_list[:], tmp_op_list[:]

        #Apply initial coefficient
        fermi_coeff_list = [coeff * x for x in fermi_coeff_list]

        return fermi_coeff_list, fermi_op_list

    def fermi_to_index_lists(self, fermi_op):
        """Covert a fermionic operator string e.g. A1B2... to indices it acts on, and [1,-1,...] used in other routines"""
        sign_map = {'A':-1, 'B':1}
        ind_list = []
        sign_list = []

        tok_list = re.findall('[AB][0-9]+', fermi_op)
        for val in tok_list:
            label, ind = val[0], int(val[1:])
            ind_list.append(ind)
            sign_list.append(sign_map[label])

        return ind_list, sign_list

    def restrict_to_active_space(self, OEI, TEI, active_space):
        """Restrict the molecule at a spatial orbital level to the active space 
        defined by active_space=[start,stop].  Assume TEI in chemists ordering,
        and active space is spin orbital indexing.  Note that OEI and TEI must
        be defined in an orthonormal basis set, such as the CMOs or NMOs here,
        which is typically the case when defining an active space.
        
        Args:
        OEI - (N,N) numpy array containing the one-electron spatial integrals for a molecule
        TEI - (N,N,N,N) numpy array containing the two-electron spatial integrals 
        in chemists ordering for a molecule
        active_space - [start,stop] - spin-orbital indices defining an active space
        
        Returns:
        coreConst - Adjustment to constant shift in Hamiltonian from integrating out core orbitals
        OEI_new - New adjusted one-electron integrals over active space
        TEI_new - New adjusted two-electron integrals over active space
        """
        
        #All orbitals are spatial here
        start, stop = active_space[0]/2, active_space[1]/2

        #Determine core constant
        coreConst = 0.0

        for i in range(start):
            coreConst += 2 * OEI[i,i]
            for j in range(start):
                coreConst += 2*TEI[i,i,j,j] - TEI[i,j,j,i]

        #Modified one electron integrals
        OEI_new = np.copy(OEI)
        for u in range(start, stop):
            for v in range(start, stop):
                for i in range(start):
                    OEI_new[u,v] += 2*TEI[i,i,u,v] - TEI[i,v,u,i]

        #Restrict integral ranges and change M appropriately
        return coreConst, OEI_new[start:stop,start:stop], TEI[start:stop,start:stop,start:stop,start:stop]

    def read_mol_ham(self, filename, 
                     transform_type = "CANONICAL_MO", 
                     active_space=None, load_active_space=False,
                     store_integrals=False):
        """Read a molecular pickle file and return an operator string list representing the Hamiltonian
        of that Molecule
        
        Args:
        transform_type - Determines the basis set transformation that is performed.
        active_space - [start,stop] spin-orbital indices defining active space
        load_active_space (bool) - load the active space used in the original molecular calculation
        store_integrals (bool) - Store integrals in this class
        
        Returns:
        coeff_list - List of coefficients of the molecular Hamiltonian
        op_list - List of Pauli operator strings representing molecular Hamiltonian
        """

        assert(transform_type == "CANONICAL_MO")
        molecule = pickle.load( open(filename, 'rb') )
        if (store_integrals): self.e_nuc = molecule.nuclearRepulsion_
        M = molecule.getM()
        print 'Molecule SCF', molecule.SCFEnergy_
        print 'Molecule FCI', molecule.FCIEnergy_

        if (load_active_space):
            nFrozen = molecule.nFrozen_ if molecule.nFrozen_ is not None else 0
            active_space = [2*nFrozen, 2*M]
            print 'Loading Active Space: {}'.format(active_space)
        try:
            if (molecule.nFrozen_ is not None and molecule.nFrozen_ > 0):
                if (active_space is None or
                    active_space != [2*molecule.nFrozen_, 2*M]):
                    print 'Warning: Correlated calculation active space differs, may change indexing'
        except:
            pass

        #Select integral transformation
        X = molecule.getCanonicalOrbitals()

        #Grab the integrals from the molecule file
        if (molecule.hasECP_):
            coreIntegrals = molecule.getTotalCoreECP(X)
        else:
            coreIntegrals = molecule.getTotalCore(X)

        twoElectronIntegrals = molecule.getTEI(X)
        nucRep = molecule.getNucRep()
        print "Nuc Rep", nucRep
        #Restrict to active space if desired
        if (active_space is not None):
            coreConst, coreIntegrals, twoElectronIntegrals = \
                self.restrict_to_active_space(coreIntegrals, twoElectronIntegrals, active_space)
            M = (active_space[1] - active_space[0])/2
            #Add core constant to nuclear rep for now
            nucRep += coreConst

        self.n_qubits = 2*M

        if (store_integrals):
            self.one_body_int = np.zeros((2*M,2*M))
            self.two_body_int = np.zeros((2*M,2*M,2*M,2*M))

        coeff_list = []
        op_str_list = []

        #Nuclear Repulsion Energy
        #Add as diagonal shift for now, may want to remove it later
        coeff_list.append(nucRep)
        op_str_list.append('I')

        #One-electron Integrals (Should optimize for uniques only later)
        for i in range(2*M):
            for j in range(2*M):
                spinIntegral = (i % 2 and j % 2) or ((not i % 2) and (not j % 2))
                spatialIntegral = coreIntegrals[i/2,j/2]
                totalIntegral = spinIntegral * spatialIntegral
                if (store_integrals): self.one_body_int[i,j] = totalIntegral
                if (np.abs(totalIntegral) > 0.0):

                    tmp_coeff_list, tmp_op_list = self.fermi_to_pauli(totalIntegral,
                                                                      [i,j], [-1,1])
                    coeff_list.extend(tmp_coeff_list)
                    op_str_list.extend(tmp_op_list)

        #Two-electron Integrals (Should optimize for uniques only later)
        for i in range(2*M):
            for j in range(2*M):
                for k in range(2*M):
                    for l in range(2*M):
                        #Going to print physicist's integral <ij|kl>, be careful with indexes
                        #For spin, need i and k to be the same spin, and j and l to be the same spin
                        spinIntegral = (((i % 2 and k % 2) or ((not i % 2) and (not k % 2))) and
                                        ((j % 2 and l % 2) or ((not j % 2) and (not l % 2))))
                        #Don't allow the creation or destruction of two particles in the same orbital
                        fermionFactor = not (i==j or k==l)
                        #Change k and j in integral as we store chemist notation integrals (ij|kl)
                        spatialIntegral = -twoElectronIntegrals[i/2,k/2,j/2,l/2]

                        totalIntegral = spinIntegral * fermionFactor * spatialIntegral
                        if (store_integrals): self.two_body_int[i,j,k,l] = totalIntegral
                        if (np.abs(totalIntegral) > 0.0):
                            tmp_coeff_list, tmp_op_list = self.fermi_to_pauli(0.5 * totalIntegral,
                                                                              [i,j,k,l],
                                                                              [-1,-1,1,1])
                            coeff_list.extend(tmp_coeff_list)
                            op_str_list.extend(tmp_op_list)

        #Reduce the final list
        coeff_list, op_str_list = self.reduce_list(coeff_list, op_str_list)

        return coeff_list, op_str_list

    def tex_print_list(self, coeff_list, op_list, re_only=False, im_only=False, no_id=False):
        """Return a string LaTex representation of a list of Pauli operators given in coeff_list, op_list form"""
        assert (not (re_only and im_only))
        tex_ops = ""
        for i, op in enumerate(op_list):
            if (op == 'I' and no_id): continue
            if (re_only and np.abs(np.real(coeff_list[i])) > 0):
                tex_ops += '{}'.format(np.real(coeff_list[i]))
            elif (im_only and np.abs(np.imag(coeff_list[i])) > 0):
                tex_ops += '{}'.format(np.imag(coeff_list[i]))
            elif ((not re_only) and (not im_only) and (np.abs(coeff_list[i]) > 0)):
                tex_ops += '{}'.format(coeff_list[i])
            else:
                continue
            tok_list = re.findall('[XYZ][0-9]+', op)
            for subop in tok_list:
                tex_ops += subop[0] + '_{{{}}}'.format(subop[1:])
            tex_ops += '+'
        #Remove final +
        tex_ops = tex_ops[:-1]
        return tex_ops

    def print_list(self, coeff_list, op_list, re_only=False, im_only=False, no_id=False):
        """Return a string representation of a list of Pauli operators given in coeff_list, op_list form"""
        assert (not (re_only and im_only))
        text_ops = ""
        for i, op in enumerate(op_list):
            if (op == 'I' and no_id): continue
            if (re_only and np.abs(np.real(coeff_list[i])) > 0):
                text_ops += '({})'.format(np.real(coeff_list[i])) + '{}'.format(op_list[i]) + "+"
            elif (im_only and np.abs(np.imag(coeff_list[i])) > 0):
                text_ops += '({})'.format(np.imag(coeff_list[i])) + '{}'.format(op_list[i]) + "+"
            elif ((not re_only) and (not im_only) and (np.abs(coeff_list[i]) > 0)):
                text_ops += '{}'.format(coeff_list[i]) + '{}'.format(op_list[i]) + "+"
            else:
                continue

        #Remove final +
        text_ops = text_ops[:-1]
        return text_ops        

class UCCFactory(object):
    """Class that deals with Unitary Coupled Cluster fermionic operators and the corresponding state
    ansatz."""
    def __init__(self, op_factory, n_electrons, 
                 amplitude_threshold=1e-12, bos=False):
        """Initialize unitary coupled cluster factory, built upon a Pauli operator factory.
        The fermionic operator encodings are handled by the Pauli operator factory.
        
        Args:
        op_factory - Pauli operator factory initialized to the correct number of qubits
        n_fermions - Number of electrons UCC ansatz will be built upon
        amplitude_threshold - Magnitude cutoff for discarding elements
        bos (bool) - Use quasi-bosonization in translation to Pauli operators
        """
        self.op_factory = op_factory
        self.encoding = op_factory.encoding
        self.N = op_factory.n_qubits #Number of total spin orbitals
        self.n_electrons = n_electrons #Number of electrons
        self.bos = bos
        self.lattice_edges = None #Defines lattice for lucc
        self.lattice_passes = None #Number of repeats in lucc
        assert (self.N >= self.n_electrons)
		
        self.PsiRef = None
        self.PsiRefString = None
        self.uccsd_coeff_cache = None
        self.gucc_coeff_cache = None
        self.amplitude_threshold = amplitude_threshold

    def ref_qasm(self):
        """Return qasm seqeuence for initial state preparation, assuming HF Ref State
          Note: There is a general fermion operator way of doing this, but without better gate compilation
          it adds a lot of uneeded garbage, do it manually for now each encoding, assuming starting from
          the comp reference |000...00>
          """

        if (self.PsiRefString is None):
            self.ref_string()

        gate_seq = ['X {}'.format(i) for i, v in enumerate(self.PsiRefString) if (v == '1')]
        return gate_seq

    def ref_string(self):
        """Return a string of bits representing the reference in the given encoding,
        with 0'th bit on the left"""

        if (self.encoding == 'JW'):
            self.PsiRefString = ''.join(['1']*self.n_electrons+['0']*(self.N-self.n_electrons))

        elif (self.encoding == 'BK'):
            #Translate n-electron occupation into qubit flips
            qubit_occupation = np.zeros(self.N, dtype=int)
            for i in range(self.n_electrons):
                qubit_occupation[i] += 1
                qubit_occupation[self.op_factory.bk_update_set[i]] += 1
            #Modulo 2 on occupation sum
            qubit_occupation = np.remainder(qubit_occupation, 2)
            self.PsiRefString = ''.join([str(i) for i in qubit_occupation])

        return self.PsiRefString

    def init_reference(self):
        """Initialze Kronecker representation of reference state for numerical UCC action"""
        #Populate n_occ lowest orbitals
        self.PsiRef = np.array([[1.0]], dtype = complex)
        Psi0 = np.array([[1.0],[0.0]], dtype = complex)
    
        for i in range(self.N):
            self.PsiRef = np.kron(self.PsiRef,Psi0) 
        coeff_list, op_list = self.op_factory.fermi_to_pauli(1.0,
                                                             list(reversed(range(self.n_electrons))),
                                                             [-1]*self.n_electrons)
        kron_prep = self.op_factory.kron_rep_list(coeff_list, op_list)
        self.PsiRef = np.dot(kron_prep, self.PsiRef)

        #Normalize
        self.PsiRef /= np.sqrt(np.dot(np.conj(self.PsiRef.T), self.PsiRef))
        
        return self.PsiRef


    def cc_amps_to_vec(self, t1_amps, t2_amps, correction=0):
        """Convert lists of CC amplitudes from PSI4 to the format taken by uccsd_unitary,
        correct occupied space with occ_corr.  This routine assumes the same active space is
        being assumed as the original CC calculation, otherwise correction arg must be used.
        
        Args:
        t1_amps - List of amplitudes of one-electron excitations, in format [i,p,amplitude],
        where i indexes a virtual orbital starting from 0 as the first virtual, and p 
        indexes an occupied orbital starting from 0 as the first occupied orbital.
        t2_amps - Amplitudes of two-electron excitations in format [i,p,j,q, amplitude],
        where i,j indexes virtual orbitals starting from 0 as the first virtual, and p,q 
        indexes occupied orbitals starting from 0 as the first occupied orbital.
        correction (int) - Shift in occupied space indexing.  Required if active space or
        number of electrons is being adjusted from PSI4 calculation.
        """
        
        #Be careful in accounting for active space relative to PSI4's active space
        N = self.N #Number of Spin-Orbitals / Qubits
        assert (N % 2 == 0) #Only support even numbers of spin orbitals

        n_occ = int(np.ceil(self.n_electrons/2.)) #Occupied Spatial Oribtals (with at least single occ)
        n_virt = N/2 - n_occ

        n_t1 = n_occ * n_virt
        n_t2 = n_t1**2
        
        t_result = np.zeros(n_t1 + n_t2)

        t1_ind = lambda i, j: i*n_occ + j
        t2_ind = lambda i, j, k, l: i*n_occ*n_virt*n_occ \
            + j * n_virt*n_occ \
            + k * n_occ \
            + l + n_t1

        for amp in t1_amps:
            virt1 = amp[1]
            occ1 = amp[0] - correction
            val = amp[2]
            #Don't count orbitals beyond truncated active space
            if (virt1 >= n_virt or occ1 >= n_occ or occ1 < 0): continue
            t_result[t1_ind(virt1, occ1)] = val

        for amp in t2_amps:
            virt1, virt2 = amp[3], amp[2]
            occ1, occ2 = amp[1]-correction, amp[0]-correction
            #print 'Occ Virt: {} {} {} {}'.format(occ1, occ2, virt1, virt2)
            val = amp[4]/2.0 #magnitude convention
            #Don't count orbitals beyond truncated active space
            if (virt1 >= n_virt or virt2 >= n_virt or
                occ1 >= n_occ or occ2 >= n_occ or
                occ1 < 0 or occ2 < 0): continue
            t_result[t2_ind(virt1, occ1, virt2, occ2)] = val 

        return t_result

    def uccsd_unitary(self, t, kron=True, cache=False):
        """Build spin conserving UCCSD sequence from a list of amplitudes, 
        spin singlet excitations from spin-eigenstate only.
        
        Args:
        t - Amplitude list, t1 followed by t2
        kron (bool) - Build a kronecker representation of the unitary, 
        return operator string representaiton of generators if False.
        
        Returns:
        Numpy array representing unitary operator if kron is True, otherwise coeff_list, op_list of generators
        """
        #For shorthand below
        #The active space will have already been imposed on the factory at creation
        N = self.N #Number of Spin-Orbitals / Qubits
        assert (N % 2 == 0) #Only support even numbers of spin orbitals

        n_occ = int(np.ceil(self.n_electrons/2.)) #Occupied Spatial Oribtals (with at least single occ)
        n_virt = N/2 - n_occ #Virtual (totally unoccupied) Spatial Orbitals

        self.coeff_list = []
        self.op_list = []

        n_t1 = n_occ * n_virt
        n_t2 = n_t1**2

        if (t is None):
            t1 = sympy.symbols('s0:{}'.format(n_t1), real=True)
            t2 = sympy.symbols('d0:{}'.format(n_t2), real=True)
        else:
            t1 = t[:n_t1]
            t2 = t[n_t1:]

        t1_ind = lambda i, j: i*n_occ + j
        t2_ind = lambda i, j, k, l: i*n_occ*n_virt*n_occ \
            + j * n_virt*n_occ \
            + k * n_occ \
            + l 

        #Singles
        for i in range(n_virt):
            for j in range(n_occ):
                for s1 in range(2):
                    #Alpha
                    tmp_coeff_list, tmp_op_list = \
                        self.op_factory.fermi_to_pauli(1.0j * t1[t1_ind(i,j)], 
                                                       [2*(i+n_occ)+s1,2*j+s1], 
                                                       [-1,1], bos=self.bos)
                    self.coeff_list.extend([x + np.conj(x) for x in tmp_coeff_list])
                    self.op_list.extend(tmp_op_list)

        #Doubles
        for i in range(n_virt):
            for j in range(n_occ):
                for s1 in range(2):
                    for k in range(n_virt):
                        for l in range(n_occ):
                            for s2 in range(2):
                                tmp_coeff_list1, tmp_op_list1 = \
                                    self.op_factory.fermi_to_pauli(1.0j * t2[t2_ind(i,j,k,l)],
                                                                   [2*(i+n_occ)+s1,2*j+s1], 
                                                                   [-1,1], bos=self.bos)

                                tmp_coeff_list2, tmp_op_list2 = \
                                    self.op_factory.fermi_to_pauli(1.0, 
                                                                   [2*(k+n_occ)+s2,2*l+s2], 
                                                                   [-1,1], bos=self.bos)

                                tmp_coeff_list, tmp_op_list = \
                                    self.op_factory.list_product(tmp_coeff_list1, tmp_op_list1,
                                                                 tmp_coeff_list2, tmp_op_list2)

                                self.coeff_list.extend([x + np.conj(x) for x in tmp_coeff_list])
                                self.op_list.extend(tmp_op_list)

        self.coeff_list, self.op_list = self.op_factory.reduce_list(self.coeff_list, 
                                                                    self.op_list, 
                                                                    simplify = (t is None),
                                                                    threshold=self.amplitude_threshold)
        if (t is None and cache):
            return self.op_factory.kron_element_list(self.coeff_list, self.op_list)
        elif (t is None or kron is False):
            return self.coeff_list, self.op_list
        else:
            self.uccsd_kron_exp = self.op_factory.op_list_exp_kron(self.coeff_list, self.op_list)
            #print 'Difference: {}'.format(sp.linalg.norm(kron_exp-qasm_exp))
            return self.uccsd_kron_exp

    def gucc_unitary(self, t, kron=True, cache=False):
        """Build generalized UCC unitary
        
        Args:
        t - Amplitude list, all possible particle creations, followed by all possible pair excitations
        kron (bool) - Build a kronecker representation of the unitary 
        return operator string representaiton of generators if False.
        
        Returns:
        Numpy array representing unitary operator if kron is True, otherwise coeff_list, op_list of generators
        """
        #For shorthand below
        #The active space will have already been imposed on the factory at creation
        N = self.N #Number of Spin-Orbitals / Qubits
        M = N / 2 #Number of Spatial Orbitals / Sites

        self.coeff_list = []
        self.op_list = []

        n_t0 = N
        n_t1 = M**2
        n_t2 = M**4

        if (t is None):
            t0 = sympy.symbols('p0:{}'.format(n_t0), real=True)
            t1 = sympy.symbols('s0:{}'.format(n_t1), real=True)
            t2 = sympy.symbols('d0:{}'.format(n_t2), real=True)
        else:
            t0 = t[:n_t0]
            t1 = t[n_t0:n_t0+n_t1]
            t2 = t[n_t0+n_t1:]

        t0_ind = lambda i: i
        t1_ind = lambda i, j: i * M + j
        t2_ind = lambda i, j, k, l: i*M**3 \
            + j * M**2 \
            + k * M \
            + l 

        #Particle Creation (Abbreviated 0'th Excitations Here)
        for i in range(N):
            tmp_coeff_list, tmp_op_list = \
                self.op_factory.fermi_to_pauli(1.0j * t0[t0_ind(i)], 
                                               [i], 
                                               [-1], bos=self.bos)
            self.coeff_list.extend([x + np.conj(x) for x in tmp_coeff_list])
            self.op_list.extend(tmp_op_list)

        #Single Excitations - Plays the role of mean-field here also. Spin conserving excitations only
        for i in range(M):
            for j in range(M):
                for s1 in range(2):
                    tmp_coeff_list, tmp_op_list = \
                        self.op_factory.fermi_to_pauli(1.0j * t1[t1_ind(i,j)],
                                                       [2*i+s1,2*j+s1], 
                                                       [-1,1], bos=self.bos)                    
                    self.coeff_list.extend([x + np.conj(x) for x in tmp_coeff_list])
                    self.op_list.extend(tmp_op_list)

        #Double Excitations - Spin Conserving Double Excitations, as particle creation can alter spin arbitrarily
        for i in range(M):
            for j in range(M):
                for k in range(M):
                    for l in range(M):
                        for s1 in range(2):
                            for s2 in range(2):
                                tmp_coeff_list1, tmp_op_list1 = \
                                    self.op_factory.fermi_to_pauli(1.0j * t2[t2_ind(i,j,k,l)],
                                                                   [2*i+s1,2*j+s1], 
                                                                   [-1,1], bos=self.bos)

                                tmp_coeff_list2, tmp_op_list2 = \
                                    self.op_factory.fermi_to_pauli(1.0, 
                                                                   [2*k+s2,2*l+s2], 
                                                                   [-1,1], bos=self.bos)

                                tmp_coeff_list, tmp_op_list = \
                                    self.op_factory.list_product(tmp_coeff_list1, tmp_op_list1,
                                                                 tmp_coeff_list2, tmp_op_list2)

                                self.coeff_list.extend([x + np.conj(x) for x in tmp_coeff_list])
                                self.op_list.extend(tmp_op_list)

        self.coeff_list, self.op_list = self.op_factory.reduce_list(self.coeff_list, 
                                                                    self.op_list, 
                                                                    simplify = (t is None),
                                                                    threshold=self.amplitude_threshold)
        if (t is None and cache):
            return self.op_factory.kron_element_list(self.coeff_list, self.op_list)
        elif (t is None or kron is False):
            return self.coeff_list, self.op_list
        else:
            self.gucc_kron_exp = self.op_factory.op_list_exp_kron(self.coeff_list, self.op_list)
            #print 'Difference: {}'.format(sp.linalg.norm(kron_exp-qasm_exp))
            return self.gucc_kron_exp

    def lucc_unitary(self, t, kron=True, cache=False):
        """Build Lattice UCC unitary based on factories edge structure
        
        Args:
        t - Amplitude list, all possible particle creations, followed by all possible pair excitations
        kron (bool) - Build a kronecker representation of the unitary 
        return operator string representaiton of generators if False.
        
        Returns:
        Numpy array representing unitary operator if kron is True, otherwise coeff_list, op_list of generators
        """

        assert ((self.lattice_edges is not None) and (self.lattice_passes is not None))
        assert (self.lattice_passes < 2) #Not really implemented yet for multiple passes
        #For shorthand below
        #The active space will have already been imposed on the factory at creation
        N = self.N #Number of Spin-Orbitals / Qubits
        M = N / 2 #Number of Spatial Orbitals / Sites

        n_t0 = 0 #Assume we will know electron number and spin for now
        n_t1 = len(self.lattice_edges) * self.lattice_passes
        n_t2 = ((len(self.lattice_edges)**2 - len(self.lattice_edges))/2) * self.lattice_passes

        if (t is None):
            t0 = sympy.symbols('p0:{}'.format(n_t0), real=True)
            t1 = sympy.symbols('s0:{}'.format(n_t1), real=True)
            t2 = sympy.symbols('d0:{}'.format(n_t2), real=True)
        else:
            t0 = t[:n_t0]
            t1 = t[n_t0:n_t0+n_t1]
            t2 = t[n_t0+n_t1:]

        #t0_ind = lambda i: i
        t1_ind = lambda i, j: i * len(self.lattice_edges) + j
        t2_ind = lambda n, i, j: n * ((len(self.lattice_edges)**2 - len(self.lattice_edges))/2) \
            + (i*(i-1)/2 + j if i > j else j * (j-1)/2 + i)

        self.lucc_kron_exp = np.eye(2**self.N)
        total_coeff_list, total_op_list = [], []

        for n in range(self.lattice_passes):
            coeff_list = []
            op_list = []
            
            #Single Excitations
            for m, edge in enumerate(self.lattice_edges):
                i, j = edge
                tmp_coeff_list, tmp_op_list = \
                    self.op_factory.fermi_to_pauli(1.0j * t1[t1_ind(n,m)],
                                                   [i, j], 
                                                   [-1,1], bos=self.bos)                    
                coeff_list.extend([x + np.conj(x) for x in tmp_coeff_list])
                op_list.extend(tmp_op_list)
            
            #Double Excitations - 
            for m1, edge1 in enumerate(self.lattice_edges):
                i, j = edge1
                for m2, edge2 in enumerate(self.lattice_edges[:m1]):
                    k,l = edge2
                    tmp_coeff_list, tmp_op_list = \
                        self.op_factory.fermi_to_pauli(1.0j * t2[t2_ind(n,m1,m2)],
                                                       [i, j, k, l], 
                                                       [-1,1,-1,1], bos=self.bos)                    
                    coeff_list.extend([x + np.conj(x) for x in tmp_coeff_list])
                    op_list.extend(tmp_op_list)
            
            
            coeff_list, op_list = self.op_factory.reduce_list(coeff_list, 
                                                              op_list, 
                                                              simplify = (t is None),
                                                              threshold=self.amplitude_threshold)
            self.lucc_kron_exp = np.dot(self.op_factory.op_list_exp_kron(coeff_list, op_list),
                                        self.lucc_kron_exp)
            total_coeff_list.extend(coeff_list)
            total_op_list.extend(op_list)
  
        if (t is None or kron is False):
            return total_coeff_list, total_op_list
        else:
            return self.lucc_kron_exp

    def qasm_uccsd(self, t=None):
        """Return the qasm sequence corresponding to t, or the last computation if None"""
        if (t is None):
            qasm_seq = self.op_factory.op_list_exp_qasm(self.coeff_list, self.op_list)
        elif (len(t) < 1):
            #Empty
            return []
        else:
            #Populates values, need to refactor the way this is done
            self.uccsd_unitary(t, kron=False)
            qasm_seq = self.op_factory.op_list_exp_qasm(self.coeff_list, self.op_list)
        return qasm_seq

    def uccsd_state(self, t):
        """Compute the Kronecker representation of the UCCSD state resulting 
		  from acting upon the reference"""
        if (self.PsiRef is None):
            self.init_reference()
        if (len(t) < 1): return np.copy(self.PsiRef)
        #Optimize with SymPy by doing symbolic coefficient replacement instead of operator rebuilds
        if (False and use_sympy):
            if (self.uccsd_coeff_cache is None):
                self.uccsd_coeff_cache, self.uccsd_op_cache = self.uccsd_unitary(None, cache=True)
            self.uccsd_kron_exp = self.use_uccsd_cache(t)
        else:
            self.uccsd_kron_exp = self.uccsd_unitary(t)
        Psi = np.dot(self.uccsd_kron_exp, self.PsiRef)
        return Psi

    def gucc_state(self, t):
        """Compute the Kronecker representation of the NCUCCD state resulting 
		  from acting upon the reference"""
        if (self.PsiRef is None):
            self.init_reference()
        if (len(t) < 1): return np.copy(self.PsiRef)
        #Optimize with SymPy by doing symbolic coefficient replacement instead of operator rebuilds
        if (use_sympy):
            if (self.gucc_coeff_cache is None):
                self.gucc_coeff_cache, self.gucc_op_cache = self.gucc_unitary(None, cache=True)
            self.gucc_kron_exp = self.use_gucc_cache(t)
        else:
            self.gucc_kron_exp = self.gucc_unitary(t)
        Psi = np.dot(self.gucc_kron_exp, self.PsiRef)
        return Psi

    def lucc_state(self, t):
        """Compute the Kronecker representation of the NCUCCD state resulting 
		  from acting upon the reference"""
        if (self.PsiRef is None):
            self.init_reference()
        if (len(t) < 1): return np.copy(self.PsiRef)
        self.lucc_kron_exp = self.lucc_unitary(t)
        Psi = np.dot(self.lucc_kron_exp, self.PsiRef)
        return Psi
        
    def use_gucc_cache(self, t):
        """Build the unitary operator corresponding to given cluster amplitudes with cached operators,
        note that due to the slowness of sympy this is actually not faster at the moment"""
        N = self.N #Number of Spin-Orbitals / Qubits
        M = N / 2 #Number of Spatial Orbitals / Sites

        n_t0 = N
        n_t1 = M**2
        n_t2 = M**4

        t0_symb = sympy.symbols('p0:{}'.format(n_t0), real=True)
        t1_symb = sympy.symbols('s0:{}'.format(n_t1), real=True)
        t2_symb = sympy.symbols('d0:{}'.format(n_t2), real=True)

        t0 = t[:n_t0]
        t1 = t[n_t0:n_t0+n_t1]
        t2 = t[n_t0+n_t1:]

        coeff_vals = []

        for coeff in self.gucc_coeff_cache:
            tmp_coeff = coeff.subs(zip(t0_symb, t0))
            tmp_coeff = tmp_coeff.subs(zip(t1_symb, t1))
            tmp_coeff = tmp_coeff.subs(zip(t2_symb, t2))
            coeff_vals.append(float(tmp_coeff))

        #Compute the result with stored Kronecker operators, it would be faster to sum then exponentiate,
        #but this would not be true to the operator splitting in the quantum algorithm
        result_op = np.eye(2**N)
        for i, coeff in enumerate(coeff_vals):
            result_op = np.dot(sp.linalg.expm(-1.0j * coeff * self.gucc_op_cache[i]),result_op)

        return result_op

    def use_uccsd_cache(self, t):
        """Build the unitary operator corresponding to given cluster amplitudes with cached operators,
        note that due to the slowness of sympy, this is actually not faster at the moment, nor
        is this particular function implemented..."""        
        assert False #Not implemented yet
        
class FermiRDMFactory(object):
    """Class to handle fermionic reduced density matrices (RDMs) and their manipulations"""
    def __init__(self, op_factory):
        """Initialize attached to Pauli operator factory op_factory"""
        self.op_factory = op_factory

    def perm_parity(self, perm):
        """Given a permutation of the digits 0..N in order as a list, 
        returns its parity (or sign): +1 for even parity; -1 for odd."""
        lst = list(perm)
        parity = 1
        for i in range(0,len(lst)-1):
            if lst[i] != i:
                parity *= -1
                mn = min(range(i,len(lst)), key=lst.__getitem__)
                lst[i],lst[mn] = lst[mn],lst[i]
        return parity    

    def kRDM(self, rho, k):
        """Return the Fermionic k-RDM of the full state density matrix rho
        
        Args:
        rho - (2**N, 2**N) numpy array of full quantum density matrix
        k - order of the RDM requested
        
        Return:
        density_matrix - 2k index numpy array representing fermionic reduced density matrix
        """
        n_qubits = self.op_factory.n_qubits

        density_matrix = np.zeros((n_qubits,)*(2*k), dtype=complex)

        full_iter = itertools.product(range(n_qubits), repeat=2*k)

        for index_set in full_iter: 
            #Check for easy 0's in fermi density matrix
            creat_inds = index_set[:k]
            anni_inds = index_set[k:]
            if (len(set(creat_inds)) != len(creat_inds) or 
                len(set(anni_inds)) != len(anni_inds)): continue
            #Need to reverse annihilation ops to be consistent w/ Mazziotti's Convention
            index_set_swap = index_set[:k] + index_set[k:][::-1]
            ex_coeff, ex_op_list = self.op_factory.fermi_to_pauli(1.0, index_set_swap, [-1]*k + [1]*k)
            ex_kron = self.op_factory.kron_rep_list(ex_coeff, ex_op_list)
            density_matrix[index_set] = np.trace(np.dot(rho, ex_kron))
    
        density_matrix /= sp.math.factorial(k)

        return density_matrix

    def partial_trace(self, rdm, index=0):
        """Return the partial trace over the given index, default is last index"""

        assert(len(rdm.shape) % 2 == 0)
        k = len(rdm.shape)/2
        
        return np.trace(rdm, axis1 = k-index-1, axis2 = 2*k-index-1)

    def expected_number(self, rho):
        """Compute expected number of particles on the density matrix rho
        
        rho - Kronecker representation of the full density matrix of a quantum system
        """
        symm_op_factory = SymmetryOperatorsFactory(self.op_factory)
        coeff_list, op_list = symm_op_factory.number_op()
        num_op = self.op_factory.kron_rep_list(coeff_list, op_list)
        num = np.real(np.trace(np.dot(rho, num_op)))
        return num

    def wedge_product(self, a, b):
        """Compute the Grassmann Wedge Product between tensors a and b"""
        
        #Get number of upper and lower indices
        assert(len(a.shape) % 2 == 0 and len(b.shape) % 2 == 0)
        ka, kb = len(a.shape)/2, len(b.shape)/2
        N = ka + kb

        #Form initial tensor product
        ab = np.kron(a, b)
        ab = np.reshape(ab, a.shape[:ka] + b.shape[:kb] + a.shape[ka:] + b.shape[kb:])

        #Make buffer and sum in permutations using numpy transpose
        result = np.zeros_like(ab)

        for perm1 in itertools.permutations(range(N)):
            for perm2 in itertools.permutations(range(N)):
                parity = self.perm_parity(perm1) * self.perm_parity(perm2)
                trans_list = [i for i in perm1] + [i+N for i in perm2]
                result += parity * np.transpose(ab, trans_list)
        
        return (1.0/sp.math.factorial(N))**2 * result

    def approx_4_RDM(self, rho, D3=None, approx_method="zero"):
        """Build a 4 RDM reconstruction based on a specified approximation. Defined by
        D4[i,j,k,l,p,q,r,s] = <\Psi| a_i^\dagger a_j^\dagger a_k^\dagger a_l^\dagger a_s a_r a_q a_p | \Psi>
        
        Args:
        rho - (2**N, 2**N) Kronecker representation of the full density matrix of the quantum system
        D3 - Provide the 3RDM from another method if desired, if none, compute from zero approximation
        approx_method (string) - Approximation method from cumulant expansions to use
        "zero" is the only one implemented at the moment.
        
        Returns:
        rdm4 - 8 index numpy array representing the 4-electron reduced density matrix
        """
        assert (approx_method == "zero")

        #Use "zero" approximation for now that sets connected 3 and 4 cumulants to zero
        rdm2 = self.kRDM(rho, 2)
        rdm1 = self.kRDM(rho, 1)

        #Construct cumulant factors
        c1 = rdm1
        c2 = rdm2 - self.wedge_product(c1, c1)

        #Build 4-RDM
        if (D3 is not None):
            c3 = D3 - self.wedge_product(c1, self.wedge_product(c1, c1)) - 3. * self.wedge_product(c2, c1)
            rdm4 = 3.0 * self.wedge_product(c2, c2) + \
                6 * self.wedge_product(c2, self.wedge_product(c1, c1)) + \
                4. * self.wedge_product(c3, c1) + \
                self.wedge_product(c1, self.wedge_product(c1, self.wedge_product(c1, c1)))
        else:
            rdm4 = 3.0 * self.wedge_product(c2, c2) + \
                6 * self.wedge_product(c2, self.wedge_product(c1, c1)) + \
                self.wedge_product(c1, self.wedge_product(c1, self.wedge_product(c1, c1)))

        return rdm4

    def approx_3_RDM(self, rho, approx_method="zero"):
        """Build a 3-RDM reconstruction based on a specific approximation. Defined by
        D3[i,j,k,p,q,r] = <\Psi| a_i^\dagger a_j^\dagger a_k^\dagger a_r a_q a_p | \Psi>
        
        Args:
        rho - (2**N, 2**N) Kronecker representation of the full density matrix of the quantum system
        approx_method (string) - Approximation method from cumulant expansions to use
        "zero" is the only one implemented at the moment.
        
        Returns:
        rdm3 - 6 index numpy array representing the 3 RDM
        """
        assert (approx_method == "zero")
        
        #Use "zero" approximation for now that sets connected 3 and 4 cumulants to zero
        rdm2 = self.kRDM(rho, 2)
        rdm1 = self.kRDM(rho, 1)

        #Construct cumulant factors
        c1 = rdm1
        c2 = rdm2 - self.wedge_product(c1, c1)

        #Build 3-RDM
        rdm3 = self.wedge_product( c1, self.wedge_product(c1, c1)) + 3.0 * self.wedge_product(c2, c1)

        return rdm3
    
    def approx_2_RDM(self, rho, approx_method="zero"):
        """Build an approximate 2-RDM from only the seperable components. Defined by
        D2[i,j,p,q] = <\Psi| a_i^\dagger a_j^\dagger a_q a_p | \Psi>
        
        Args:
        rho - (2**N, 2**N) Kronecker representation of the full density matrix of the quantum system
        approx_method (string) - Approximation method from cumulant expansions to use
        "zero" is the only one implemented at the moment.
        
        Returns:
        rdm2 - 4 index numpy array representing the 3 RDM
        """
        assert (approx_method == "zero")
        c1 = self.kRDM(rho, 1)
        rdm2 = self.wedge_product(c1, c1)
        return rdm2

class SymmetryOperatorsFactory(object):
    """Class for building an manipulating number and spin fermionic operators"""
    def __init__(self, op_factory):
        """Initialize the symmetry operator factory which depends on a generic Pauli operator factory"""
        self.op_factory = op_factory

    def number_op(self):
        """Return a Pauli operator string representation of the number operator acting on Fermions"""
        n_qubits = self.op_factory.n_qubits

        coeff_list = []
        op_list = []

        for i in range(n_qubits):
            coeff, op = self.op_factory.fermi_to_pauli(1.0, [i, i], [-1, 1])
            coeff_list.extend(coeff)
            op_list.extend(op)

        coeff_list, op_list = self.op_factory.reduce_list(coeff_list, op_list)

        return coeff_list, op_list

    def S2_op(self):
        """Return a Pauli operator string representation of the S^2 operator acting on Fermions"""
        n_qubits = self.op_factory.n_qubits

        X_coeff_list = []
        X_op_str_list = []

        Y_coeff_list = []
        Y_op_str_list = []

        Z_coeff_list = []
        Z_op_str_list = []

        #Get individual operators Sx Sy Sz
        for i in range(n_qubits/2):
            coeff, op = self.op_factory.fermi_to_pauli(0.5, [2*i,2*i+1], [-1,1])
            X_coeff_list.extend(coeff)
            X_op_str_list.extend(op)

            coeff, op = self.op_factory.fermi_to_pauli(0.5, [2*i+1, 2*i], [-1,1])
            X_coeff_list.extend(coeff)
            X_op_str_list.extend(op)        

            coeff, op = self.op_factory.fermi_to_pauli(-0.5j, [2*i,2*i+1], [-1,1])
            Y_coeff_list.extend(coeff)
            Y_op_str_list.extend(op)

            coeff, op = self.op_factory.fermi_to_pauli(0.5j, [2*i+1, 2*i], [-1,1])
            Y_coeff_list.extend(coeff)
            Y_op_str_list.extend(op)

            coeff, op = self.op_factory.fermi_to_pauli(0.5, [2*i,2*i], [-1,1])
            Z_coeff_list.extend(coeff)
            Z_op_str_list.extend(op)

            coeff, op = self.op_factory.fermi_to_pauli(-0.5, [2*i+1, 2*i+1], [-1,1])
            Z_coeff_list.extend(coeff)
            Z_op_str_list.extend(op)

        #Square the components Sx^2 Sy^2 Sz^2
        X_coeff_list, X_op_str_list = self.op_factory.list_product(X_coeff_list, X_op_str_list,
                                            X_coeff_list, X_op_str_list)
        Y_coeff_list, Y_op_str_list = self.op_factory.list_product(Y_coeff_list, Y_op_str_list,
                                            Y_coeff_list, Y_op_str_list)
        Z_coeff_list, Z_op_str_list = self.op_factory.list_product(Z_coeff_list, Z_op_str_list,
                                            Z_coeff_list, Z_op_str_list)

        #Add them and reduce
        coeff_list = []
        op_list = []
        coeff_list.extend(X_coeff_list)
        op_list.extend(X_op_str_list)
        coeff_list.extend(Y_coeff_list)
        op_list.extend(Y_op_str_list)
        coeff_list.extend(Z_coeff_list)
        op_list.extend(Z_op_str_list)

        coeff_list, op_list = self.op_factory.reduce_list(coeff_list, op_list)
        
        return coeff_list, op_list

    def SZ_op(self):
        """Return a Pauli string representation of the S_z Operator acting on Fermions"""
        n_qubits = self.op_factory.n_qubits

        coeff_list = []
        op_list = []

        for i in range(n_qubits/2):
            coeff, op = self.op_factory.fermi_to_pauli(0.5, [2*i,2*i], [-1,1])
            coeff_list.extend(coeff)
            op_list.extend(op)

            coeff, op = self.op_factory.fermi_to_pauli(-0.5, [2*i+1, 2*i+1], [-1,1])
            coeff_list.extend(coeff)
            op_list.extend(op)

        coeff_list, op_list = self.op_factory.reduce_list(coeff_list, op_list)

        return coeff_list, op_list

class LinearResponseFactory(object):
    """Class to manipulate and build fermionic linear response(LR) operators"""
    def __init__(self, op_factory, rdm_factory):
        self.op_factory = op_factory
        self.rdm_factory = rdm_factory
    def build_operator(self, op1, op2, c, D1, D2, D3, D4, couple_ref_excited=True):
        """Build Linear Response Operator from one-body (op1) and two-body (op2) integrals,
        and the reduced k-RDMs D1-D4. Specifically assume operator we want to build is of the form
        O = \sum_{pq} op1[p,q] a_p^\dag a_q + (1/2) \sum_{pqrs} op2[p,q,r,s] a_p^\dag a_q^\dag a_r a_s + c"""
        #return build_operator_(self, op1, op2, c, D1, D2, D3, D4, couple_ref_excited)

        #Expressions below are machine generated by WicksCalculator Class
        M = self.op_factory.n_qubits
        result_op = np.zeros((M**2+1, M**2+1), dtype=complex)
        result_s = np.zeros((M**2+1, M**2+1), dtype=complex)

        #Overlap
        result_s[0,0] = 1.0
        for i in range(M):
            for j in range(M):
                result_s[i*M+j+1, 0] = D1[j,i]
                result_s[0, i*M+j+1] = np.conj(result_s[i*M+j+1, 0])
                for k in range(M):
                    for l in range(M):
                        result_s[i*M+j+1, k*M+l+1] = KroneckerDelta(i, k)*D1[j,l] - 2*D2[j,k,l,i]
        #Ref-Ref
        for p in range(M):
            for r in range(M):
                result_op[0,0] +=  op1[p,r] * (D1[p,r])
                for q in range(M):
                    for s in range(M):
                        #Ref - Ref
                        result_op[0,0] += op2[p,q,r,s] * (D2[p,q,s,r])

        #Ref - LR
        for i in range(M):
            for j in range(M):
                for p in range(M):
                    for r in range(M):
                        tmp = op1[p,r] * (KroneckerDelta(i, p)*1*D1[j,r] - 2*D2[j,p,r,i])
                        result_op[i*M+j+1, 0] += tmp
                        result_op[0, i*M+j+1] += np.conj(tmp)
                        for q in range(M):
                            for s in range(M):
                                #Ref - LR
                                tmp = 0.5*op2[p,q,r,s] * (KroneckerDelta(i, p)*2*D2[j,q,s,r] - \
                                                                KroneckerDelta(i, q)*2*D2[j,p,s,r] + \
                                                                6*D3[j,p,q,s,r,i])

                                result_op[i*M+j+1, 0] += tmp
                                result_op[0, i*M+j+1] += np.conj(tmp)
                            
        #LR - LR
        for i in range(M):
            for j in range(M):
                for k in range(M):
                    for l in range(M):
                        for p in range(M):
                            for r in range(M):
                                result_op[i*M+j+1, k*M+l+1] += op1[p,r] * (-KroneckerDelta(i, k)*2*D2[j,p,r,l] + \
                                                                KroneckerDelta(i, p)*KroneckerDelta(k, r)*1*D1[j,l] + \
                                                                KroneckerDelta(i, p)*2*D2[j,k,r,l] - \
                                                                KroneckerDelta(k, r)*2*D2[j,p,l,i] - \
                                                                6*D3[j,k,p,r,l,i])
                                for q in range(M):
                                    for s in range(M):
                                        result_op[i*M+j+1, k*M+l+1] += 0.5 * op2[p,q,r,s] * (KroneckerDelta(i, k)*6*D3[j,p,q,s,r,l] + \
                                                                         KroneckerDelta(i, p)*KroneckerDelta(k, r)*2*D2[j,q,s,l] - \
                                                                         KroneckerDelta(i, p)*KroneckerDelta(k, s)*2*D2[j,q,r,l] - \
                                                                         KroneckerDelta(i, p)*6*D3[j,k,q,s,r,l] - \
                                                                         KroneckerDelta(i, q)*KroneckerDelta(k, r)*2*D2[j,p,s,l] + \
                                                                         KroneckerDelta(i, q)*KroneckerDelta(k, s)*2*D2[j,p,r,l] + \
                                                                         KroneckerDelta(i, q)*6*D3[j,k,p,s,r,l] + \
                                                                         KroneckerDelta(k, r)*6*D3[j,p,q,s,l,i] - \
                                                                         KroneckerDelta(k, s)*6*D3[j,p,q,r,l,i] - \
                                                                         24*D4[j,k,p,q,s,r,l,i])
                                        
        #Add constant shift (accounting for non-orthogonal basis)
        result_op += c * result_s

        #Remove coupling between ground and excited states if desired
        if (not couple_ref_excited):
            result_s_pinv = sp.linalg.pinv(result_s)
            selector = np.zeros_like(result_op)
            selector[1:,0] = selector[0,1:] = 1.0
            H_cp = result_op * selector
            result_op -= np.dot(result_s_pinv, H_cp)

        return result_op, result_s

    def build_comm_approx_operator(self, op1, op2, c, D1, D2, D3, couple_ref_excited=True):
        """Build linear response operator based on commutator reduction that assumes
        reference density is an exact eignstate of the Hamiltonian,
        and the reduced k-RDMs D1-D3. Specifically assume operator we want to build is of the form
        O = \sum_{pq} op1[p,q] a_p^\dag a_q + (1/2) \sum_{pqrs} op2[p,q,r,s] a_p^\dag a_q^\dag a_r a_s + c"""
        #Expressions below are machine generated by WicksCalculator Class
        M = self.op_factory.n_qubits
        result_op = np.zeros((M**2+1, M**2+1), dtype=complex)
        result_s = np.zeros((M**2+1, M**2+1), dtype=complex)

        #Overlap
        result_s[0,0] = 1.0
        for i in range(M):
            for j in range(M):
                result_s[i*M+j+1, 0] = D1[j,i]
                result_s[0, i*M+j+1] = np.conj(result_s[i*M+j+1, 0])
                for k in range(M):
                    for l in range(M):
                        result_s[i*M+j+1, k*M+l+1] = KroneckerDelta(i, k)*D1[j,l] - 2*D2[j,k,l,i]
        #Ref-Ref
        for p in range(M):
            for r in range(M):
                result_op[0,0] +=  op1[p,r] * (D1[p,r])
                for q in range(M):
                    for s in range(M):
                        #Ref - Ref
                        result_op[0,0] += op2[p,q,r,s] * (D2[p,q,s,r])

        #LR - LR
        for i in range(M):
            for j in range(M):
                for k in range(M):
                    for l in range(M):
                        for p in range(M):
                            for r in range(M):
                                result_op[i*M+j+1, k*M+l+1] += op1[p,r] * (-KroneckerDelta(i, k)*KroneckerDelta(l, p)*1*D1[j,r] + \
                                                                           KroneckerDelta(i, p)*KroneckerDelta(k, r)*1*D1[j,l] - \
                                                                           KroneckerDelta(k, r)*2*D2[j,p,l,i] + \
                                                                           KroneckerDelta(l, p)*2*D2[j,k,r,i])
                                for q in range(M):
                                    for s in range(M):
                                        result_op[i*M+j+1, k*M+l+1] += \
                                            0.5 * op2[p,q,r,s] * (-KroneckerDelta(i, k)*KroneckerDelta(l, p)*2*D2[j,q,s,r] + \
                                                                       KroneckerDelta(i, k)*KroneckerDelta(l, q)*2*D2[j,p,s,r] + \
                                                                       KroneckerDelta(i, p)*KroneckerDelta(k, r)*2*D2[j,q,s,l] - \
                                                                       KroneckerDelta(i, p)*KroneckerDelta(k, s)*2*D2[j,q,r,l] - \
                                                                       KroneckerDelta(i, p)*KroneckerDelta(l, q)*2*D2[j,k,s,r] - \
                                                                       KroneckerDelta(i, q)*KroneckerDelta(k, r)*2*D2[j,p,s,l] + \
                                                                       KroneckerDelta(i, q)*KroneckerDelta(k, s)*2*D2[j,p,r,l] + \
                                                                       KroneckerDelta(i, q)*KroneckerDelta(l, p)*2*D2[j,k,s,r] + \
                                                                       KroneckerDelta(k, r)*6*D3[j,p,q,s,l,i] - \
                                                                       KroneckerDelta(k, s)*6*D3[j,p,q,r,l,i] - \
                                                                       KroneckerDelta(l, p)*6*D3[j,k,q,s,r,i] + \
                                                                       KroneckerDelta(l, q)*6*D3[j,k,p,s,r,i])

        #Add E_g term(s)
        Eg = result_op[0,0]
        result_op += Eg * result_s
        result_op[0,0] -= Eg #Remove double counting
        
        #Add constant shift (accounting for non-orthogonal basis)
        result_op += c * result_s

        #Remove coupling between ground and excited states if desired
        if (not couple_ref_excited):
            result_s_pinv = sp.linalg.pinv(result_s)
            selector = np.zeros_like(result_op)
            selector[1:,0] = selector[0,1:] = 1.0
            H_cp = result_op * selector
            result_op -= np.dot(result_s_pinv, H_cp)
            
        return result_op, result_s

    def build_operator_kron(self, rho, O):
        """Build Linear Response Operator from Kron Representation"""
        n_qubits = self.op_factory.n_qubits

        #Find exact ground state of Channel Hamiltonian
        e_vals, e_vecs = sp.linalg.eigh(O)
        ref_state = np.dot(e_vecs[:,0][:,np.newaxis], np.conj(e_vecs[:,0])[np.newaxis, :])

        n_basis = (n_qubits)**2 + 1
        H_LR = np.zeros((n_basis, n_basis), dtype=complex)
        S_LR = np.zeros((n_basis, n_basis), dtype=complex)

        #Build QSE-LR Matrix Reps
        H_LR[0,0] = np.trace(np.dot(ref_state, O), dtype=complex)
        S_LR[0,0] = np.trace(ref_state, dtype=complex)

        for k in range(n_qubits):
            for l in range(n_qubits):
                    ex_coeff, ex_op_list = self.op_factory.fermi_to_pauli(1.0, [k, l], [-1, 1])
                    ex_kron1 = self.op_factory.kron_rep_list(ex_coeff, ex_op_list)

                    H_LR[0, k*n_qubits+l+1] = np.trace(np.dot(ref_state, np.dot(O, ex_kron1)))
                    S_LR[0, k*n_qubits+l+1] = np.trace(np.dot(ref_state, ex_kron1))

                    H_LR[k*n_qubits+l+1, 0] = np.conj(H_LR[0, k*n_qubits+l+1])
                    S_LR[k*n_qubits+l+1, 0] = np.conj(S_LR[0, k*n_qubits+l+1])

        for i in range(n_qubits):
            for j in range(n_qubits):
                for k in range(n_qubits):
                    for l in range(n_qubits):
                        ex_coeff, ex_op_list = self.op_factory.fermi_to_pauli(1.0, [k, l], [-1, 1])
                        ex_kron1 = self.op_factory.kron_rep_list(ex_coeff, ex_op_list)

                        ex_coeff, ex_op_list = self.op_factory.fermi_to_pauli(1.0, [i, j], [-1, 1])
                        ex_kron2 = np.conj(self.op_factory.kron_rep_list(ex_coeff, ex_op_list)).T

                        #Weight avg done w.r.t. exact Hamiltonian coeffs
                        H_LR[i*n_qubits+j+1,k*n_qubits+l+1] = np.trace(np.dot(ref_state, np.dot(ex_kron2, np.dot(O, ex_kron1))))
                        S_LR[i*n_qubits+j+1,k*n_qubits+l+1] = np.trace(np.dot(ref_state, np.dot(ex_kron2, ex_kron1)))

        return H_LR, S_LR

class WicksCalculator(object):
    """Convenience wrapper for calculating Wick's normal-ordered products of creation
    and annililation built off sympy's technology.  This class doesn't do much directly at the moment,
    but is useful for deriving expressions and coverting them to code for other uses."""
    def __init__(self):
        """No initialization needed at the moment, mostly just a utility wrapper at the moment."""

    def NO_to_D(self, no_match):
        """Convert normal ordered string to density matrix string in python syntax. Useful for
        automated code generation, such as that used in the LinearResponse code.
        
        Args:
        no_match - re module match object resulting from a search on normal-ordered strings
        """
        no_string = no_match.group(0)
        matches = re.findall('Fermion\(([a-z]+)\)', no_string)
        D_order = int(len(matches)/2)
        #Remember to reverse index on annihilation operators in accordance with Mazziotti's convention
        return '{}*D{}[{}]'.format(sp.math.factorial(D_order), D_order, 
                                  ','.join(matches[:D_order] + matches[D_order:][::-1]))

    def NO_to_tex(self, no_match, as_density=False):
        """Convert normal ordered string to TeX string in density matrix format
        
        Args:
        no_match - re module match object resulting from a search on normal-ordered strings
        as_density(bool) - if True, return result as density matrices, if False return result
        as creation and annihiliation operators.
        """
        no_string = no_match.group(0)
        creation_matches = re.findall('CreateFermion\(([a-z]+)\)', no_string)
        annihilation_matches = re.findall('AnnihilateFermion\(([a-z]+)\)', no_string)
        if (as_density):
            rdm_order = len(creation_matches)
            #Remember to reverse index on annihilation operators in accordance with Mazziotti's convention
            result_string = '{}{{}}^{{{}}}D^{{{}}}_{{{}}}'.format(sp.math.factorial(rdm_order),rdm_order,
                                                                ''.join(creation_matches), 
                                                                ''.join(annihilation_matches[::-1]))
        else:
            result_string = '{}{}'.format(''.join(['a_{}^{{\dagger}} '.format(i) for i in creation_matches]),
                                          ''.join(['a_{} '.format(i) for i in annihilation_matches])).rstrip(' ')
        return result_string

    def kron_to_tex(self, match):
        """Convert Kronecker delta function from sympy to a latex string
        
        Args:
        match - regular expression (re module) match object
        """
        kron_string = match.group(0)
        matches = re.findall('KroneckerDelta\(([a-z,\s]+)\)', kron_string)
        return '\delta_{{{}}} '.format(re.sub(',','',''.join(matches)))

    def sym_to_python(self, sym_expr):
        """Convert a symbolic string made by sympy to python syntax for use"""
        sym_string = str(sympy.simplify(sympy.physics.secondquant.evaluate_deltas(
                    sympy.physics.secondquant.wicks(sym_expr))))
        new_string = re.sub('NO\(((CreateFermion\([a-z]+\)\*?)|(AnnihilateFermion\([a-z]+\)\*?))*\)', 
                            self.NO_to_D, sym_string)
        new_string = re.sub(':((CreateFermion\([a-z]+\)\*?)|(AnnihilateFermion\([a-z]+\)\*?))*:', 
                            self.NO_to_D, new_string)
        return new_string

    def sym_to_latex(self, sym_expr, as_density=False):
        """Convert a symbolic string made by sympy to LaTeX reasonable for a paper,
        as_density controls if creation/anniliation operators are output or
        reduced density matrices"""
        sym_string = str(sympy.simplify(sympy.physics.secondquant.evaluate_deltas(
                    sympy.physics.secondquant.wicks(sym_expr))))
        new_string = re.sub('NO\(((CreateFermion\([a-z]+\)\*?)|(AnnihilateFermion\([a-z]+\)\*?))*\)', 
                            lambda x: self.NO_to_tex(x, as_density), sym_string)        
        new_string = re.sub('(KroneckerDelta\([a-z, ]+\))', 
                            self.kron_to_tex, new_string)
        new_string = re.sub('\*', '', new_string)
        return new_string
        
    def linear_response(self, output_type='python', as_density=False):
        """Symbolic output for exact linear response matrix, convenient for writing
        python code above and paper output.  This is only used currently by hand for convenience,
        and doesn't have explicit other uses."""

        if (output_type == 'python'):
            expr_transform = self.sym_to_python
        elif (output_type == 'latex'):
            expr_transform = lambda x: self.sym_to_latex(x, as_density)

        F = sympy.physics.secondquant.F
        Fd = sympy.physics.secondquant.Fd
        j,i,k,l = sympy.symbols('j,i,k,l', above_fermi=True)
        p,q,r,s = sympy.symbols('p,q,r,s', above_fermi=True)
        
        print 'One Body Linear Response Space: Sum over p, r'
        fermi_expr = Fd(j)*F(i)*Fd(p)*F(r)*Fd(k)*F(l)
        print expr_transform(fermi_expr)

        print 'One Body Linear Response to Reference Space:'
        fermi_expr = Fd(j)*F(i)*Fd(p)*F(r)
        print expr_transform(fermi_expr)

        print 'One Body Reference Space Only:'
        fermi_expr = Fd(p)*F(r)
        print expr_transform(fermi_expr)

        print
        print 'Two Body Linear Response: Sum over p,q,r,s'
        fermi_expr = Fd(j)*F(i)*Fd(p)*Fd(q)*F(r)*F(s)*Fd(k)*F(l)
        print expr_transform(fermi_expr)

        print 'Two Body Linear Response Space to Reference Space: Sum over p, q, r, s'
        fermi_expr = Fd(j)*F(i)*Fd(p)*Fd(q)*F(r)*F(s)
        print expr_transform(fermi_expr)
        
        print 'Two Body Reference Space Only:'
        fermi_expr = Fd(p)*Fd(q)*F(r)*F(s)
        print expr_transform(fermi_expr)

        print
        print 'Overlap - Linear Response to Reference'
        fermi_expr = Fd(j)*F(i)
        print expr_transform(fermi_expr)

        print 'Overlap - Linear Response to Linear Reponse'
        fermi_expr = Fd(j)*F(i)*Fd(k)*F(l)
        print expr_transform(fermi_expr)

    def commutator_linear_response(self, output_type='python', as_density=False):
        """Symbolic output for commutator linear response space.  Convenience functions
        for recent paper evaluation.  No other uses at the moment."""
        
        if (output_type == 'python'):
            expr_transform = self.sym_to_python
        elif (output_type == 'latex'):
            expr_transform = lambda x: self.sym_to_latex(x, as_density)

        F = sympy.physics.secondquant.F
        Fd = sympy.physics.secondquant.Fd
        Commutator = sympy.physics.secondquant.Commutator
        j,i,k,l = sympy.symbols('j,i,k,l', above_fermi=True)
        p,q,r,s = sympy.symbols('p,q,r,s', above_fermi=True)
        
        print 'One Body Linear Response Space: Sum over p, r'
        fermi_expr = Fd(j)*F(i)*Commutator(Fd(p)*F(r),Fd(k)*F(l))
        print expr_transform(fermi_expr)

        print
        print 'Two Body Linear Response: Sum over p,q,r,s'
        fermi_expr = Fd(j)*F(i)*Commutator(Fd(p)*Fd(q)*F(r)*F(s),Fd(k)*F(l))
        print expr_transform(fermi_expr)


        print 'Overlap - Linear Response to Linear Reponse'
        fermi_expr = Fd(j)*F(i)*Fd(k)*F(l)
        print expr_transform(fermi_expr)


    def particle_hole_matrix_relations(self, output_type='python', as_density=True):
        """Symbolic output for commutator linear response space.  Convenience functions
        for recent paper evaluation.  No other uses at the moment."""
        
        if (output_type == 'python'):
            expr_transform = self.sym_to_python
        elif (output_type == 'latex'):
            expr_transform = lambda x: self.sym_to_latex(x, as_density)

        F = sympy.physics.secondquant.F
        Fd = sympy.physics.secondquant.Fd
        Commutator = sympy.physics.secondquant.Commutator
        j,i,k,l = sympy.symbols('j,i,k,l', above_fermi=True)
        p,q,r,s = sympy.symbols('p,q,r,s', above_fermi=True)

        print 'Q^{i}_{j} ([1, -1])'
        fermi_expr = F(i)*Fd(j)
        print expr_transform(fermi_expr)
        print

        print 'Q^{ij}_{kl}'
        fermi_expr = F(i)*F(j)*Fd(l)*Fd(k)
        print expr_transform(fermi_expr)
        print 

        print 'G^{ij}_{kl}'
        fermi_expr = Fd(i)*F(j)*Fd(l)*F(k)
        print expr_transform(fermi_expr)
        print 

        print '([-1,-1,1,1])'
        fermi_expr = Fd(i)*Fd(j)*F(k)*F(l)
        print expr_transform(fermi_expr)
        print 

        print '([1,1,-1,-1])'
        fermi_expr = F(i)*F(j)*Fd(k)*Fd(l)
        print expr_transform(fermi_expr)
        print 

        print '([-1,1,-1,1])'
        fermi_expr = Fd(i)*F(j)*Fd(k)*F(l)
        print expr_transform(fermi_expr)
        print 

        print '([1,-1,1,-1])'
        fermi_expr = F(i)*Fd(j)*F(k)*Fd(l)
        print expr_transform(fermi_expr)
        print

        print '([1,-1,-1,1])'
        fermi_expr = F(i)*Fd(j)*Fd(k)*F(l)
        print expr_transform(fermi_expr)
        print 

        print '([-1,1,1,-1])'
        fermi_expr = Fd(i)*F(j)*F(k)*Fd(l)
        print expr_transform(fermi_expr)
        print 

        print '([-1,-1,-1,1])'
        fermi_expr = Fd(i)*Fd(j)*Fd(k)*F(l)
        print expr_transform(fermi_expr)
        print 

class ConstraintCalculator(object):
    """Calculate the Low-Order Positivity Constraints on the 2-Electron Reduced Density Matrix"""

    def __init__(self, op_factory, n_electrons):
        self.op_factory = op_factory
        self.n_electrons = n_electrons
        
    def constraints_1RDM(self, exclude_number_cons=False):
        """Return a list of constraints in string form on the Pauli operators related to 
        positivity conditions on the 1RDM"""
        total_constraints = []

        #One RDM Trace Condition
        if (not exclude_number_cons):
            coeff_list, op_list = [], []
            for i in range(self.op_factory.n_qubits):
                tmp_coeff_list, tmp_op_list = self.op_factory.fermi_to_pauli(1.0, [i, i], [-1, 1])
                coeff_list.extend(tmp_coeff_list)
                op_list.extend(tmp_op_list)
            coeff_list, op_list = self.op_factory.reduce_list(coeff_list, op_list)
            i_coeff = self.I_Coeff(coeff_list, op_list)
            #total_constraints.append(self.op_factory.print_list(coeff_list, op_list, re_only=True, no_id = True) 
            #                         + '={}'.format(self.n_fermions - np.real(i_coeff)))
            #total_constraints.append(self.op_factory.print_list(coeff_list, op_list, im_only=True, no_id = True) 
            #                         + '={}'.format(-np.imag(i_coeff)))
            c = self.op_factory.print_list(coeff_list, op_list, re_only=True, no_id = True) \
                                     + '={}'.format(self.n_electrons - np.real(i_coeff))
            if ('X' in c or 'Y' in c or 'Z' in c): yield c
            c = self.op_factory.print_list(coeff_list, op_list, im_only=True, no_id = True) \
                                     + '={}'.format(-np.imag(i_coeff))
            if ('X' in c or 'Y' in c or 'Z' in c): yield c

        #Diagonal 1-RDM particle-hole condition
        for i in range(self.op_factory.n_qubits):
            coeff_list, op_list = [], []
            #{}^1D
            tmp_coeff_list, tmp_op_list = self.op_factory.fermi_to_pauli(1.0, [i,i], [-1, 1])
            coeff_list.extend(tmp_coeff_list)
            op_list.extend(tmp_op_list)
            #{}^1Q
            tmp_coeff_list, tmp_op_list = self.op_factory.fermi_to_pauli(1.0, [i,i], [1, -1])
            coeff_list.extend(tmp_coeff_list)
            op_list.extend(tmp_op_list)
            coeff_list, op_list = self.op_factory.reduce_list(coeff_list, op_list)
            i_coeff = self.I_Coeff(coeff_list, op_list)
            #total_constraints.append(self.op_factory.print_list(coeff_list, op_list, re_only=True, no_id=True) + '={}'.format(1 - np.real(i_coeff)))
            #total_constraints.append(self.op_factory.print_list(coeff_list, op_list, im_only=True, no_id=True) + '={}'.format(-np.imag(i_coeff)))
            c = self.op_factory.print_list(coeff_list, op_list, re_only=True, no_id=True) + '={}'.format(1 - np.real(i_coeff))
            if ('X' in c or 'Y' in c or 'Z' in c): yield c
            c = self.op_factory.print_list(coeff_list, op_list, im_only=True, no_id=True) + '={}'.format(-np.imag(i_coeff))
            if ('X' in c or 'Y' in c or 'Z' in c): yield c

        #Off-Diagonal 1-RDM Particle Hole Condition
        for i in range(self.op_factory.n_qubits):
            for j in range(i+1, self.op_factory.n_qubits):
                coeff_list, op_list = [], []
                #{}^1D
                tmp_coeff_list, tmp_op_list = self.op_factory.fermi_to_pauli(1.0, [i,j], [-1, 1])
                coeff_list.extend(tmp_coeff_list)
                op_list.extend(tmp_op_list)
                #{}^1Q
                tmp_coeff_list, tmp_op_list = self.op_factory.fermi_to_pauli(1.0, [i,j], [1, -1])
                coeff_list.extend(tmp_coeff_list)
                op_list.extend(tmp_op_list)

                coeff_list, op_list = self.op_factory.reduce_list(coeff_list, op_list)
                i_coeff = self.I_Coeff(coeff_list, op_list)
                #total_constraints.append(self.op_factory.print_list(coeff_list, op_list, re_only=True, no_id=True) + '={}'.format(-np.real(i_coeff)))
                #total_constraints.append(self.op_factory.print_list(coeff_list, op_list, im_only=True, no_id=True) + '={}'.format(-np.imag(i_coeff)))
                c = self.op_factory.print_list(coeff_list, op_list, re_only=True, no_id=True) + '={}'.format(-np.real(i_coeff))
                if ('X' in c or 'Y' in c or 'Z' in c): yield c
                c = self.op_factory.print_list(coeff_list, op_list, im_only=True, no_id=True) + '={}'.format(-np.imag(i_coeff))
                if ('X' in c or 'Y' in c or 'Z' in c): yield c

        #1-RDM Hermiticity Condition
        for i in range(self.op_factory.n_qubits):
            for j in range(i+1, self.op_factory.n_qubits):
                coeff_list, op_list = [], []

                tmp_coeff_list, tmp_op_list = self.op_factory.fermi_to_pauli(1.0, [i,j], [-1, 1])
                coeff_list.extend(tmp_coeff_list)
                op_list.extend(tmp_op_list)

                tmp_coeff_list, tmp_op_list = self.op_factory.fermi_to_pauli(-1.0, [j,i], [-1, 1])
                coeff_list.extend(tmp_coeff_list)
                op_list.extend(tmp_op_list)

                coeff_list, op_list = self.op_factory.reduce_list(coeff_list, op_list)
                i_coeff = self.I_Coeff(coeff_list, op_list)
                #total_constraints.append(self.op_factory.print_list(coeff_list, op_list, re_only=True, no_id=True) + '={}'.format(-np.real(i_coeff)))
                #total_constraints.append(self.op_factory.print_list(coeff_list, op_list, im_only=True, no_id=True) + '={}'.format(-np.imag(i_coeff)))
                c = self.op_factory.print_list(coeff_list, op_list, re_only=True, no_id=True) + '={}'.format(-np.real(i_coeff))
                if ('X' in c or 'Y' in c or 'Z' in c): yield c
                c = self.op_factory.print_list(coeff_list, op_list, im_only=True, no_id=True) + '={}'.format(-np.imag(i_coeff))
                if ('X' in c or 'Y' in c or 'Z' in c): yield c

        #Filter out trivial constraints that don't act on any Pauli operators
        #total_constraints = [cons for cons in total_constraints 
        #                     if (('X' in cons) or ('Y' in cons) or ('Z' in cons))]
        #
        #return total_constraints

    def symbolic_2RDM(self):
        """Return a list with entries of the 2-RDM Represented as Pauli Operators"""
        index_list = []
        element_list = []
        
        for i in range(self.op_factory.n_qubits):
            for j in range(self.op_factory.n_qubits):
                for k in range(self.op_factory.n_qubits):
                    for l in range(self.op_factory.n_qubits):
                        tmp_coeff_list, tmp_op_list = self.op_factory.fermi_to_pauli(1.0, [i,j,l,k], [-1, -1, 1, 1])
                        tmp_coeff_list, tmp_op_list = self.op_factory.reduce_list(tmp_coeff_list, tmp_op_list)
                        element = self.op_factory.print_list(tmp_coeff_list, tmp_op_list)
                        if (len(element) > 0):
                            yield [[i,j,k,l], element]
                            #index_list.append([i,j,k,l])
                            #element_list.append(element)
        #return zip(*[index_list, element_list])

    def RDM_to_spin_orbitals(self, fermi_1rdm_aa, fermi_1rdm_bb, fermi_2rdm_aa, fermi_2rdm_ab, fermi_2rdm_bb, active_space=None):
        """Covert from spatial orbital format to spin-orbital format for 1- and 2-RDM, then restrict to given active space"""
        #Note: Psi4's Internal 2-RDM is stored as 
        #D[pqrs] = <|a_{p\alpha}^\dagger a_{r\alpha}^\dagger a_{s\alpha} a_{q\alpha}|> for "AA"/"BB"
        #D[pqrs] = <|a_{p\alpha}^\dagger a_{r\beta}^\dagger a_{s\beta} a_{q\alpha}|> for "AB"
        M = fermi_1rdm_aa.shape[0]
        D1 = np.zeros((2*M, 2*M)) #Spin-Orbital 1-RDM
        D2 = np.zeros((2*M, 2*M, 2*M, 2*M)) #Spin-Orbital 2-RDM

        for i in range(M):
            for j in range(M):
                D1[2*i, 2*j] = fermi_1rdm_aa[i,j]
                D1[2*i+1, 2*j+1] = fermi_1rdm_bb[i,j]
                for k in range(M):
                    for l in range(M):
                        #print '{} {} {} {}: {} {} {}'.format(i,j,k,l, fermi_2rdm_aa[i,j,k,l], fermi_2rdm_ab[i,j,k,l], fermi_2rdm_bb[i,j,k,l])
                        D2[2*i, 2*j, 2*k, 2*l] = 0.5 * fermi_2rdm_aa[i,k,j,l]
                        D2[2*i, 2*j+1, 2*k, 2*l+1] = 0.5 * fermi_2rdm_ab[i,k,j,l] ##
                        D2[2*i, 2*j+1, 2*k+1, 2*l] = -0.5 * fermi_2rdm_ab[i,l,j,k] ##
                        D2[2*i+1, 2*j, 2*k+1, 2*l] = 0.5 * fermi_2rdm_ab[j,l,i,k] 
                        D2[2*i+1, 2*j, 2*k, 2*l+1] = -0.5 * fermi_2rdm_ab[j,k,i,l] 
                        D2[2*i+1, 2*j+1, 2*k+1, 2*l+1] = 0.5 * fermi_2rdm_bb[i,k,j,l]

        #Restrict density matrices to active space
        if (active_space is not None):
            inds = np.array(range(*active_space))
            D1 = D1[np.ix_(inds, inds)]
            D2 = D2[np.ix_(inds, inds, inds, inds)]

        return D1, D2

    def guess_pauli_operator(self, D1, D2, op):
        """Guess associated Pauli expectation value of op from spinful 1- and 2-RDM"""
        assert (self.op_factory.encoding == 'JW') #Inverse Transform only implemented for JW at the moment

        if (op == 'I'): return 1.0

        expectation_value = 0.0

        fermi_coeff_list, fermi_op_list = self.op_factory.JW_inverse_transform(1.0, op)
        if not fermi_coeff_list: return 1.0

        for i, v in enumerate(fermi_coeff_list):
            ind_list, sign_list = self.op_factory.fermi_to_index_lists(fermi_op_list[i])
            val = self.fermi_expt_value(D1, D2, ind_list, sign_list)
            expectation_value += v * val

        return expectation_value

    def fermi_expt_value(self, D1, D2, ind_list, sign_list):
        """Convert value for a_i^\dagger a_j"""

        #Identity
        if (not ind_list): 
            return 1.

        #Particle conservation
        if (np.sum(sign_list) != 0): return 0

        if len(ind_list) == 2:
            i,j = ind_list
        elif len(ind_list) == 4:
            i,j,k,l = ind_list
        else:
            print 'Unexpected index list size'
            exit()

        if (sign_list == [-1,1]):
            val = D1[i,j]
        elif (sign_list == [1,-1]):
            val = KroneckerDelta(i, j) - 1*D1[j,i]
        elif (sign_list == [-1,-1, 1,1]):
            val = 2*D2[i,j,l,k]
        elif(sign_list == [1,1,-1,-1]):
            val = -KroneckerDelta(i, k)*KroneckerDelta(j, l) + KroneckerDelta(i, k)*1*D1[l,j] \
                + KroneckerDelta(i, l)*KroneckerDelta(j, k) - KroneckerDelta(i, l)*1*D1[k,j] \
                - KroneckerDelta(j, k)*1*D1[l,i] + KroneckerDelta(j, l)*1*D1[k,i] + 2*D2[k,l,j,i]
        elif (sign_list == [-1, 1, -1, 1]):
            val = KroneckerDelta(j, k)*1*D1[i,l] - 2*D2[i,k,l,j]
        elif (sign_list == [1, -1, -1, 1]):
            val = KroneckerDelta(i, j)*1*D1[k,l] - KroneckerDelta(i, k)*1*D1[j,l] + 2*D2[j,k,l,i]
        elif (sign_list == [1, -1, 1, -1]):
            val = KroneckerDelta(i, j)*KroneckerDelta(k, l) - KroneckerDelta(i, j)*1*D1[l,k] \
                + KroneckerDelta(i, l)*1*D1[j,k] - KroneckerDelta(k, l)*1*D1[j,i] - 2*D2[j,l,k,i]
        elif (sign_list == [-1, 1, 1, -1]):
            val = -KroneckerDelta(j, l)*1*D1[i,k] + KroneckerDelta(k, l)*1*D1[i,j] + 2*D2[i,l,k,j]
        else:
            print 'Unknown Entry Value Error'
            exit()

        return val

    def constraints_2RDM(self, exclude_number_cons=False):
        """Return a list of constraints in string form on the Pauli operators related to 
        positivity conditions on the 2-RDM"""        
        total_constraints = []

        #2-RDM Trace Condition
        if (not exclude_number_cons):
            coeff_list, op_list = [], []
            for i in range(self.op_factory.n_qubits):
                for j in range(self.op_factory.n_qubits):
                    tmp_coeff_list, tmp_op_list = self.op_factory.fermi_to_pauli(1.0, [i,j,j,i], [-1, -1, 1, 1])
                    coeff_list.extend(tmp_coeff_list)
                    op_list.extend(tmp_op_list)
            coeff_list, op_list = self.op_factory.reduce_list(coeff_list, op_list)
            i_coeff = self.I_Coeff(coeff_list, op_list)
            #total_constraints.append(self.op_factory.print_list(coeff_list, op_list, re_only=True, no_id=True) 
            #                         + '={}'.format(self.n_fermions*(self.n_fermions-1)-np.real(i_coeff)))
            #total_constraints.append(self.op_factory.print_list(coeff_list, op_list, im_only=True, no_id=True) + '={}'.format(-np.imag(i_coeff)))
            #
            c = self.op_factory.print_list(coeff_list, op_list, re_only=True, no_id=True) \
                + '={}'.format(self.n_electrons*(self.n_electrons-1)-np.real(i_coeff))
            if ('X' in c or 'Y' in c or 'Z' in c): yield c
            c = self.op_factory.print_list(coeff_list, op_list, im_only=True, no_id=True) + '={}'.format(-np.imag(i_coeff))
            if ('X' in c or 'Y' in c or 'Z' in c): yield c

        #2-RDM Hermiticity Conditions
        for ij in range(self.op_factory.n_qubits**2):
            i, j = (ij / self.op_factory.n_qubits), (ij % self.op_factory.n_qubits)
            for kl in range(ij+1, self.op_factory.n_qubits**2):
                k, l = (kl / self.op_factory.n_qubits), (kl % self.op_factory.n_qubits)
                coeff_list, op_list = [], []

                tmp_coeff_list, tmp_op_list = self.op_factory.fermi_to_pauli(1.0, [i,j,l,k], [-1, -1, 1, 1])
                coeff_list.extend(tmp_coeff_list)
                op_list.extend(tmp_op_list)

                tmp_coeff_list, tmp_op_list = self.op_factory.fermi_to_pauli(-1.0, [k,l,j,i], [-1, -1, 1, 1])
                coeff_list.extend(tmp_coeff_list)
                op_list.extend(tmp_op_list)

                coeff_list, op_list = self.op_factory.reduce_list(coeff_list, op_list)
                i_coeff = self.I_Coeff(coeff_list, op_list)
                #total_constraints.append(self.op_factory.print_list(coeff_list, op_list, re_only=True, no_id=True) 
                #                         + '={}'.format(-np.real(i_coeff)))        
                #total_constraints.append(self.op_factory.print_list(coeff_list, op_list, im_only=True, no_id=True) 
                #                         + '={}'.format(-np.imag(i_coeff)))
                c = self.op_factory.print_list(coeff_list, op_list, re_only=True, no_id=True) \
                                         + '={}'.format(-np.real(i_coeff))
                if ('X' in c or 'Y' in c or 'Z' in c): yield c
                c = self.op_factory.print_list(coeff_list, op_list, im_only=True, no_id=True) \
                                         + '={}'.format(-np.imag(i_coeff))
                if ('X' in c or 'Y' in c or 'Z' in c): yield c
        
        #2-RDM Anti-Symmetry Conditions
        for ij in range(self.op_factory.n_qubits**2):
            i, j = (ij / self.op_factory.n_qubits), (ij % self.op_factory.n_qubits)
            for kl in range(ij+1, self.op_factory.n_qubits**2):
                k, l = (kl / self.op_factory.n_qubits), (kl % self.op_factory.n_qubits)
                coeff_list, op_list = [], []

                tmp_coeff_list, tmp_op_list = self.op_factory.fermi_to_pauli(1.0, [i,j,l,k], [-1, -1, 1, 1])
                coeff_list.extend(tmp_coeff_list)
                op_list.extend(tmp_op_list)

                tmp_coeff_list, tmp_op_list = self.op_factory.fermi_to_pauli(1.0, [j,i,l,k], [-1, -1, 1, 1])
                coeff_list.extend(tmp_coeff_list)
                op_list.extend(tmp_op_list)

                coeff_list, op_list = self.op_factory.reduce_list(coeff_list, op_list)
                i_coeff = self.I_Coeff(coeff_list, op_list)
                #total_constraints.append(self.op_factory.print_list(coeff_list, op_list, re_only=True, no_id=True) 
                #                         + '={}'.format(-np.real(i_coeff)))        
                #total_constraints.append(self.op_factory.print_list(coeff_list, op_list, im_only=True, no_id=True) 
                #                         + '={}'.format(-np.imag(i_coeff)))        
                c = self.op_factory.print_list(coeff_list, op_list, re_only=True, no_id=True) \
                                         + '={}'.format(-np.real(i_coeff))
                if ('X' in c or 'Y' in c or 'Z' in c): yield c                
                c = self.op_factory.print_list(coeff_list, op_list, im_only=True, no_id=True) \
                                         + '={}'.format(-np.imag(i_coeff))        
                if ('X' in c or 'Y' in c or 'Z' in c): yield c

                coeff_list, op_list = [], []

                tmp_coeff_list, tmp_op_list = self.op_factory.fermi_to_pauli(1.0, [i,j,l,k], [-1, -1, 1, 1])
                coeff_list.extend(tmp_coeff_list)
                op_list.extend(tmp_op_list)

                tmp_coeff_list, tmp_op_list = self.op_factory.fermi_to_pauli(1.0, [i,j,k,l], [-1, -1, 1, 1])
                coeff_list.extend(tmp_coeff_list)
                op_list.extend(tmp_op_list)

                coeff_list, op_list = self.op_factory.reduce_list(coeff_list, op_list)
                i_coeff = self.I_Coeff(coeff_list, op_list)
                #total_constraints.append(self.op_factory.print_list(coeff_list, op_list, re_only=True, no_id=True) 
                #                         + '={}'.format(-np.real(i_coeff)))        
                #total_constraints.append(self.op_factory.print_list(coeff_list, op_list, im_only=True, no_id=True) 
                #                         + '={}'.format(-np.imag(i_coeff)))        
                c = self.op_factory.print_list(coeff_list, op_list, re_only=True, no_id=True) \
                    + '={}'.format(-np.real(i_coeff))        
                if ('X' in c or 'Y' in c or 'Z' in c): yield c
                c = self.op_factory.print_list(coeff_list, op_list, im_only=True, no_id=True) \
                    + '={}'.format(-np.imag(i_coeff))        
                if ('X' in c or 'Y' in c or 'Z' in c): yield c

                coeff_list, op_list = [], []

                tmp_coeff_list, tmp_op_list = self.op_factory.fermi_to_pauli(1.0, [i,j,l,k], [-1, -1, 1, 1])
                coeff_list.extend(tmp_coeff_list)
                op_list.extend(tmp_op_list)

                tmp_coeff_list, tmp_op_list = self.op_factory.fermi_to_pauli(-1.0, [j,i,k,l], [-1, -1, 1, 1])
                coeff_list.extend(tmp_coeff_list)
                op_list.extend(tmp_op_list)

                coeff_list, op_list = self.op_factory.reduce_list(coeff_list, op_list)
                i_coeff = self.I_Coeff(coeff_list, op_list)
                #total_constraints.append(self.op_factory.print_list(coeff_list, op_list, re_only=True, no_id=True) 
                #                         + '={}'.format(-np.real(i_coeff)))
                #total_constraints.append(self.op_factory.print_list(coeff_list, op_list, im_only=True, no_id=True) 
                #                         + '={}'.format(-np.imag(i_coeff)))
                c = self.op_factory.print_list(coeff_list, op_list, re_only=True, no_id=True) \
                    + '={}'.format(-np.real(i_coeff))
                if ('X' in c or 'Y' in c or 'Z' in c): yield c
                c = self.op_factory.print_list(coeff_list, op_list, im_only=True, no_id=True) \
                    + '={}'.format(-np.imag(i_coeff))
                if ('X' in c or 'Y' in c or 'Z' in c): yield c

        #2-RDM Contraction to 1-RDM Condition
        if (not exclude_number_cons):
            for i in range(self.op_factory.n_qubits):
                for j in range(self.op_factory.n_qubits):
                    coeff_list, op_list = [], []
                    for p in range(self.op_factory.n_qubits):
                        tmp_coeff_list, tmp_op_list = self.op_factory.fermi_to_pauli(1.0, [i,p,p,j], [-1, -1, 1, 1])
                        coeff_list.extend(tmp_coeff_list)
                        op_list.extend(tmp_op_list)

                    tmp_coeff_list, tmp_op_list = self.op_factory.fermi_to_pauli(-(self.n_electrons - 1), 
                                                                                      [i,j], [-1, 1])
                    coeff_list.extend(tmp_coeff_list)
                    op_list.extend(tmp_op_list)

                    coeff_list, op_list = self.op_factory.reduce_list(coeff_list, op_list)
                    i_coeff = self.I_Coeff(coeff_list, op_list)
                    #total_constraints.append(self.op_factory.print_list(coeff_list, op_list, re_only=True, no_id=True) 
                    #                         + '={}'.format(-np.real(i_coeff)))
                    #total_constraints.append(self.op_factory.print_list(coeff_list, op_list, im_only=True, no_id=True) 
                    #                         + '={}'.format(-np.imag(i_coeff)))
                    c = self.op_factory.print_list(coeff_list, op_list, re_only=True, no_id=True) \
                        + '={}'.format(-np.real(i_coeff))
                    if ('X' in c or 'Y' in c or 'Z' in c): yield c
                    c = self.op_factory.print_list(coeff_list, op_list, im_only=True, no_id=True) \
                        + '={}'.format(-np.imag(i_coeff))
                    if ('X' in c or 'Y' in c or 'Z' in c): yield c

        #Linear Relations between 2-particle matrices
        for ij in range(self.op_factory.n_qubits**2):
            i, j = (ij / self.op_factory.n_qubits), (ij % self.op_factory.n_qubits)
            for kl in range(ij, self.op_factory.n_qubits**2):
                k, l = (kl / self.op_factory.n_qubits), (kl % self.op_factory.n_qubits)
                #Q Matrix
                coeff_list, op_list = [], []

                tmp_coeff_list, tmp_op_list = self.op_factory.fermi_to_pauli(-1.0 * (j==l), [i,k], [-1, 1])
                coeff_list.extend(tmp_coeff_list)
                op_list.extend(tmp_op_list)

                tmp_coeff_list, tmp_op_list = self.op_factory.fermi_to_pauli(1.0 * (i==l), [j,k], [-1, 1])
                coeff_list.extend(tmp_coeff_list)
                op_list.extend(tmp_op_list)

                tmp_coeff_list, tmp_op_list = self.op_factory.fermi_to_pauli(1.0 * (j==k), [i,l], [-1, 1])
                coeff_list.extend(tmp_coeff_list)
                op_list.extend(tmp_op_list)

                tmp_coeff_list, tmp_op_list = self.op_factory.fermi_to_pauli(-1.0 * (i==k), [j,l], [-1, 1])
                coeff_list.extend(tmp_coeff_list)
                op_list.extend(tmp_op_list)

                tmp_coeff_list, tmp_op_list = self.op_factory.fermi_to_pauli(1.0, [i,j,l,k], [-1, -1, 1, 1])
                coeff_list.extend(tmp_coeff_list)
                op_list.extend(tmp_op_list)

                tmp_coeff_list, tmp_op_list = self.op_factory.fermi_to_pauli(-1.0, [i,j,l,k], [1, 1, -1, -1])
                coeff_list.extend(tmp_coeff_list)
                op_list.extend(tmp_op_list)

                coeff_list, op_list = self.op_factory.reduce_list(coeff_list, op_list)

                I_term = -2*(i==k and j==l) if (i==k and j==l) else 0
                i_coeff = self.I_Coeff(coeff_list, op_list)
                #total_constraints.append(self.op_factory.print_list(coeff_list, op_list, re_only=True, no_id=True) 
                #                         + '={}'.format(I_term-np.real(i_coeff)))        
                #total_constraints.append(self.op_factory.print_list(coeff_list, op_list, im_only=True, no_id=True) 
                #                         + '={}'.format(-np.imag(i_coeff)))        
                c = self.op_factory.print_list(coeff_list, op_list, re_only=True, no_id=True) \
                                          + '={}'.format(I_term-np.real(i_coeff))         
                if ('X' in c or 'Y' in c or 'Z' in c): yield c
                c = self.op_factory.print_list(coeff_list, op_list, im_only=True, no_id=True) \
                                         + '={}'.format(-np.imag(i_coeff))         
                if ('X' in c or 'Y' in c or 'Z' in c): yield c

                #G Matrix
                coeff_list, op_list = [], []

                tmp_coeff_list, tmp_op_list = self.op_factory.fermi_to_pauli(1.0 * (j==l), [i,k], [-1, 1])
                coeff_list.extend(tmp_coeff_list)
                op_list.extend(tmp_op_list)

                tmp_coeff_list, tmp_op_list = self.op_factory.fermi_to_pauli(1.0, [i,l,k,j], [-1, -1, 1, 1])
                coeff_list.extend(tmp_coeff_list)
                op_list.extend(tmp_op_list)

                tmp_coeff_list, tmp_op_list = self.op_factory.fermi_to_pauli(-1.0, [i,j,l,k], [-1, 1, -1, 1])
                coeff_list.extend(tmp_coeff_list)
                op_list.extend(tmp_op_list)

                coeff_list, op_list = self.op_factory.reduce_list(coeff_list, op_list)
                i_coeff = self.I_Coeff(coeff_list, op_list)
                #total_constraints.append(self.op_factory.print_list(coeff_list, op_list, re_only=True, no_id=True) 
                #                         + '={}'.format(-np.real(i_coeff)))        
                #total_constraints.append(self.op_factory.print_list(coeff_list, op_list, im_only=True, no_id=True) 
                #                         + '={}'.format(-np.imag(i_coeff)))        
                c = self.op_factory.print_list(coeff_list, op_list, re_only=True, no_id=True) \
                    + '={}'.format(-np.real(i_coeff))
                if ('X' in c or 'Y' in c or 'Z' in c): yield c
                c = self.op_factory.print_list(coeff_list, op_list, im_only=True, no_id=True) \
                    + '={}'.format(-np.imag(i_coeff))        
                if ('X' in c or 'Y' in c or 'Z' in c): yield c

        #Filter out trivial constraints that don't act on any Pauli operators
        #total_constraints = [cons for cons in total_constraints 
        # if (('X' in cons) or ('Y' in cons) or ('Z' in cons))]
        #
        #return total_constraints

    @staticmethod
    def tokenize_constraint(cons):
        """Takes a string constraint of the form (coeff1)OP1+...=eqv_value and returns
        a list of the coeffs, operators, and eq_value to facilitate checking of constraints"""
        op_tok_list = re.findall('\([0-9\.\-]+\)(?:[XYZ][0-9]+)+', cons)
        coeff_list, op_list = [], []
        for val in op_tok_list:
            coeff, op = float(re.sub('\(|\)','', re.findall('\([0-9\.\-]+\)', val)[0])), re.findall('(?:[XYZ][0-9]+)+', val)[0]
            
            coeff_list.append(coeff)
            op_list.append(op)

        eq_value = float(cons.split('=')[1])

        return coeff_list, op_list, eq_value

    def test_constraints(self, cons_list, state):
        """Test a list of string constraints on a known quantum state in a kronecker product representation"""
        for cons in cons_list:
            coeff_list, op_list, eq_value = self.tokenize_constraint(cons)
            kron_op = self.op_factory.kron_rep_list(coeff_list, op_list)
            expected_value = np.dot(np.conj(state.T), np.dot(kron_op, state))[0,0]
            #print '{} = {}'.format(expected_value, eq_value)
            if (np.abs(eq_value - expected_value) > 1e-7):
                print 'Constraint Violation on {}'.format(cons)
                print '{} != {}'.format(eq_value, expected_value)

    @staticmethod
    def I_Coeff(coeff_list, op_list):
        """Return the coefficient of the identity operator in an operator list representation"""
        i_coeff = 0
        for i, op in enumerate(op_list):
            if (op == 'I'):
                i_coeff = coeff_list[i]

        return i_coeff
