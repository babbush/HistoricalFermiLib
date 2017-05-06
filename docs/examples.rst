.. _examples:

Examples
========

All of these example codes (and many more!) can be found on `GitHub <https://github.com/babbush/fermilib/tree/master/examples/>`_.

.. toctree::
   :maxdepth: 2	

Fermionic Operators
-------------------

Fermionic systems are often treated in second quantization, where arbitrary operators can be expressed using the fermionic creation and annihilation operators, :math:`a_k^\dagger` and :math:`a_k`. Any weighted sum of products of these operators can be represented with the FermionOperator data structure in FermiLib.

.. code-block:: python

	from fermilib.ops import FermionOperator
	
	my_term = FermionOperator(((3, 1), (1, 0)))
	print(my_term)
	
	my_term = FermionOperator('3^ 1')
	print(my_term)

These two examples yield the same fermionic operator, :math:`a_3^\dagger a_1`.

The preferred way to specify the coefficient in FermiLib is to provide an optional coefficient argument. If not provided, the coefficient defaults to 1. In the code below, the first method is preferred. The multiplication in the last method actually creates a copy of the term, which introduces some additional cost. All inplace operands (such as +=) modify classes whereas binary operands such as + create copies. Important caveats are that the empty tuple FermionOperator(()) and the empty string FermionOperator('') initialize identity. The empty initializer FermionOperator() initializes the zero operator. We demonstrate some of these below.

.. code-block:: python

	from fermilib.ops import FermionOperator
	
	good_way_to_initialize = FermionOperator('3^ 1', -1.7)
	print(good_way_to_initialize)
	
	bad_way_to_initialize = -1.7 * FermionOperator('3^ 1')
	print(bad_way_to_initialize)
	
	identity = FermionOperator('')
	print(identity)
	
	zero_operator = FermionOperator()
	print(zero_operator)

This creates the previous FermionOperator with a coefficient -1.7, as well as the identity and zero operators.

FermionOperator has only one attribute: .terms. This attribute is the dictionary which stores the term tuples.

.. code-block:: python

	from fermilib.ops import FermionOperator
	
	my_operator = FermionOperator('4^ 1^ 3 9', 1. + 2.j)
	print(my_operator)
	print(my_operator.terms)

FermionOperator supports a wide range of builtins including str(), repr(), =, , /, /=, +, +=, -, -=, - and **. Note that instead of supporting != and ==, we have the method .isclose(), since FermionOperators involve floats.

Qubit Operators
---------------

The QubitOperator data structure is another essential part of FermiLib. While the QubitOperator was originally developed for FermiLib, it is now part of the core ProjectQ library so that it can be interpreted by the ProjectQ compiler using the TimeEvolution gate. As the name suggests, QubitOperator is used to store qubit operators in almost exactly the same way that FermionOperator is used to store fermion operators. For instance :math:`X_0 Z_3 Y_4` is a QubitOperator. The internal representation of this as a terms tuple would be :math:`((0,"X"),(3,"Z"),(4,"Y"))((0,"X"),(3,"Z"),(4,"Y"))`. Note that one important difference between QubitOperator and FermionOperator is that the terms in QubitOperator are always sorted in order of tensor factor. In some cases, this enables faster manipulation. We initialize some QubitOperators below.

.. code-block:: python

	from projectqtemp.ops import QubitOperator
	
	my_first_qubit_operator = QubitOperator('X1 Y2 Z3')
	print(my_first_qubit_operator)
	print(my_first_qubit_operator.terms)
	
	operator_2 = QubitOperator('X3 Z4', 3.17)
	operator_2 -= 77. * my_first_qubit_operator
	print('')
	print(operator_2)

Transformations
---------------

FermiLib also provides functions for mapping FermionOperators to QubitOperators, including the Jordan-Wigner and Bravyi-Kitaev transforms.

.. code-block:: python

	from fermilib.ops import FermionOperator, hermitian_conjugated
	from fermilib.transforms import jordan_wigner, bravyi_kitaev
	from fermilib.utils import eigenspectrum
	
	# Initialize an operator.
	fermion_operator = FermionOperator('2^ 0', 3.17)
	fermion_operator += hermitian_conjugated(fermion_operator)
	print(fermion_operator)
		
	# Transform to qubits under the Jordan-Wigner transformation and print its spectrum.
	jw_operator = jordan_wigner(fermion_operator)
	print('')
	print(jw_operator)
	jw_spectrum = eigenspectrum(jw_operator)
	print(jw_spectrum)
	
	# Transform to qubits under the Bravyi-Kitaev transformation and print its spectrum.
	bk_operator = bravyi_kitaev(fermion_operator)
	print('')
	print(bk_operator)
	bk_spectrum = eigenspectrum(bk_operator)
	print(bk_spectrum)

We see that despite the different representation, these operators are iso-spectral. We can also apply the Jordan-Wigner transform in reverse to map arbitrary QubitOperators to FermionOperators. Note that we also demonstrate the .compress() method (a method on both FermionOperators and QubitOperators) which removes zero entries.

.. code-block:: python

	from projectqtemp.ops import QubitOperator
	from fermilib.transforms import jordan_wigner, reverse_jordan_wigner
	
	# Initialize QubitOperator.
	my_operator = QubitOperator('X0 Y1 Z2', 88.)
	my_operator += QubitOperator('Z1 Z4', 3.17)
	print(my_operator)
	
	# Map QubitOperator to a FermionOperator.
	mapped_operator = reverse_jordan_wigner(my_operator)
	print('')
	print(mapped_operator)
	
	# Map the operator back to qubits and make sure it is the same.
	back_to_normal = jordan_wigner(mapped_operator)
	back_to_normal.compress()
	print('')
	print(back_to_normal)

Sparse matrices and the Hubbard model
-------------------------------------

Often, one would like to obtain a sparse matrix representation of an operator which can be analyzed numerically. There is code in both fermilib.transforms and fermilib.utils which facilitates this. The function get_sparse_operator converts either a FermionOperator, a QubitOperator or other more advanced classes such as InteractionOperator to a scipy.sparse.csc matrix. There are numerous functions in fermilib.utils which one can call on the sparse operators such as "get_gap", "get_hartree_fock_state", "get_ground_state", ect. We show this off by computing the ground state energy of the Hubbard model. To do that, we use code from the fermilib.utils module which constructs lattice models of fermions such as Hubbard models.

.. code-block:: python

	from fermilib.transforms import get_sparse_operator, jordan_wigner
	from fermilib.utils import fermi_hubbard, get_ground_state
	
	# Set model.
	x_dimension = 2
	y_dimension = 2
	tunneling = 2.
	coulomb = 1.
	magnetic_field = 0.5
	chemical_potential = 0.25
	periodic = 1
	spinless = 1
	
	# Get fermion operator.
	hubbard_model = fermi_hubbard(
	    x_dimension, y_dimension, tunneling, coulomb, chemical_potential,
	    magnetic_field, periodic, spinless)
	print(hubbard_model)
	
	# Get qubit operator under Jordan-Wigner.
	jw_hamiltonian = jordan_wigner(hubbard_model)
	jw_hamiltonian.compress()
	print('')
	print(jw_hamiltonian)
	
	# Get scipy.sparse.csc representation.
	sparse_operator = get_sparse_operator(hubbard_model)
	print('')
	print(sparse_operator)
	print('\nEnergy of the model is {} in units of T and J.'.format(
	    get_ground_state(sparse_operator)[0]))

Hamiltonians in the plane wave basis
------------------------------------

FermiLib uses a third-party electronic structure package to compute molecular orbitals, Hamiltonians, energies, reduced density matrices, coupled cluster amplitudes, etc using Gaussian basis sets. However, this third-party electronic structure package has a restrictive GPL license. Accordingly, we cannot even mention it by name in this tutorial. While we provide scripts which interface between that package and FermiLib, we cannot discuss it here.

When using simpler basis sets such as plane waves, these packages are not needed. FermiLib comes with code which computes Hamiltonians in the plane wave basis. Note that when using plane waves, one is working with the periodized Coulomb operator, best suited for condensed phase calculations such as studying the electronic structure of a solid. To obtain these Hamiltonians one must choose to study the system without a spin degree of freedom (spinless), one must the specify dimension in which the calculation is performed (n_dimensions, usually 3), one must specify how many plane waves are in each dimension (grid_length) and one must specify the length scale of the plane wave harmonics in each dimension (length_scale) and also the locations and charges of the nuclei. One can generate these models with plane_wave_hamiltonian() found in fermilib.utils. For simplicity, below we compute the Hamiltonian in the case of zero external charge (corresponding to the uniform electron gas, aka jellium). We also demonstrate that one can transform the plane wave Hamiltonian using a Fourier transform without effecting the spectrum of the operator.

.. code-block:: python

	from fermilib.utils import eigenspectrum, fourier_transform, jellium_model
	from fermilib.transforms import jordan_wigner
	
	# Let's look at a very small model of jellium in 1D.
	n_dimensions = 1
	grid_length = 3
	length_scale = 1.
	spinless = True
	
	# Get the momentum Hamiltonian.
	momentum_hamiltonian = jellium_model(n_dimensions, grid_length, length_scale, spinless)
	momentum_qubit_operator = jordan_wigner(momentum_hamiltonian)
	momentum_qubit_operator.compress()
	print(momentum_qubit_operator)
	
	# Fourier transform the Hamiltonian to the position basis.
	position_hamiltonian = fourier_transform(momentum_hamiltonian, n_dimensions, grid_length, length_scale, spinless)
	position_qubit_operator = jordan_wigner(position_hamiltonian)
	position_qubit_operator.compress()
	print('')
	print (position_qubit_operator)
	
	# Check the spectra to make sure these representations are iso-spectral.
	spectral_difference = eigenspectrum(momentum_qubit_operator) -  eigenspectrum(position_qubit_operator)
	print('')
	print(spectral_difference)

Basics of MolecularData class
-----------------------------

Perhaps the most useful features in FermiLib concern its interaction with open source electronic structure packages. Once again, we provide scripts to interact with one such package but cannot legally refer to it by name here due to its GPL license.

Data from electronic structure calculations is generated using scripts which perform the calculations and then populate a FermiLib data structure called MolecularData. Often, one would like to analyze a chemical series or look at many different Hamiltonians and sometimes the electronic structure calculations are either expensive to compute or difficult to converge (e.g. one needs to mess around with different types of SCF routines to make things converge). Accordingly, we anticipate that users will want some way to automatically database the results of their electronic structure calculations so that important data (such as the SCF intergrals) can be looked up on-the-fly if the user has computed them in the past. FermiLib supports a data provenance strategy which saves key results of the electronic structure calculation (including pointers to files containing large amounts of data, such as the molecular integrals) in an HDF5 container.

The MolecularData class stores information about molecules. One initializes a MolecularData object by specifying parameters of a molecule such as its geometry, basis, multiplicity, charge and an optional string describing it. One can also initialize MolecularData simply by providing a string giving a filename where a previous MolecularData object was saved in an HDF5 container. One can save a MolecularData instance by calling the class's .save() method. This automatically saves the instance in a data folder specified during FermiLib installation. The name of the file is generated automatically from the instance attributes and optionally provided description. Alternatively, a filename can also be provided as an optional input if one wishes to manually name the file.

When electronic structure calculations are run using our scripts, the data files for the molecule are automatically updated. For instance, once one runs an FCI calculation on the molecule, the attribute MolecularData.fci_energy will be saved and set equal the FCI energy. If one wishes to later use that data they either initialize MolecularData with the instance filename or initialize the instance and then later call the .load() method.

Basis functions are provided to initialization using a string such as "6-31g". Geometries can be specified using a simple txt input file (see geometry_from_file function in molecular_data.py) or can be passed using a simple python list format demonstrated below. Atoms are specified using a string for their atomic symbol. Distances should be provided in atomic units (Bohr). Below we initialize a simple instance of MolecularData without performing any electronic structure calculations.

FermiLib supports a wide range of models of fermions beyond what is shown in this basic tutorial. See the `GitHub examples <https://github.com/babbush/fermilib/tree/master/examples>`_ for more.