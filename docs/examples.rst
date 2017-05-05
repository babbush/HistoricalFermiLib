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

These two examples yield the fermionic operator :math:`a_3^\dagger a_1`.

The preferred way to specify the coefficient in FermiLib is to provide an optional coefficient argument. If not provided, the coefficient defaults to 1. The first method is preferred. The multiplication in the last method actually creates a copy of the term, which introduces some additional cost. All inplace operands (such as +=) modify classes where as operands such as + create copies. Important caveats are that the empty tuple FermionOperator(()) or the empty string FermionOperator('') initializes identity. Whereas an empty initializer FermionOperator() initializes the zero operator. We demonstrate some of these things below.

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

The QubitOperator data structure is another essential part of FermiLib. While the QubitOperator was originally developed for FermiLib, it is now part of the core ProjectQ library so that it can be interpreted by the ProjectQ compiler using the TimeEvolution gate. As the name suggests, QubitOperator is used to store qubit operators in almost exactly the same way that FermionOperator is used to store fermion operators. For instance :math:`X_0 Z_3 Y_4` is a QubitOperator. The internal representation of this as a terms tuple would be :math:`((0,X),(3,Z),(4,Y))((0,X),(3,Z),(4,Y))`. Note that one important difference between QubitOperator and FermionOperator is that the terms in QubitOperator are always sorted in order of tensor factor. We initialize some QubitOperators below.

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

FermiLib also provides functions for mapping FermionOperators to QubitOperators.

.. code-block:: python

	from projectqtemp.ops import QubitOperator
	from fermilib.ops import FermionOperator, hermitian_conjugated
	from fermilib.transforms import jordan_wigner, bravyi_kitaev
	from fermilib.utils import eigenspectrum
	
	fermion_operator = FermionOperator('2^ 0', 3.17)
	fermion_operator += hermitian_conjugated(fermion_operator)
	print(fermion_operator)
	
	jw_operator = jordan_wigner(fermion_operator)
	print('')
	print(jw_operator)
	
	bk_operator = bravyi_kitaev(fermion_operator)
	print('')
	print(bk_operator)
	
	jw_spectrum = eigenspectrum(jw_operator)
	bk_spectrum = eigenspectrum(bk_operator)
	print('')
	print(jw_spectrum)
	print(bk_spectrum)

We can also apply the Jordan-Wigner transform in reverse to map arbitrary QubitOperators to FermionOperators. Note that we also demonstrate the .compress() method (a method on both FermionOperators and QubitOperators) which removes zero entries.

.. code-block:: python

	from projectqtemp.ops import QubitOperator
	from fermilib.transforms import jordan_wigner, reverse_jordan_wigner
	
	my_operator = QubitOperator('X0 Y1 Z2', 88.)
	my_operator += QubitOperator('Z1 Z4', 3.17)
	print(my_operator)
	
	mapped_operator = reverse_jordan_wigner(my_operator)
	print('')
	print(mapped_operator)
	
	back_to_normal = jordan_wigner(mapped_operator)
	back_to_normal.compress()
	print('')
	print(back_to_normal)

FermiLib supports a wide range of models of fermions beyond what is shown in this basic tutorial. See the `GitHub examples <https://github.com/babbush/fermilib/tree/master/examples>`_ for more.