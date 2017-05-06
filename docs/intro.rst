.. _intro:

Tutorial
========

.. toctree::
   :maxdepth: 2	

Getting started with FermiLib
-----------------------------

To install FermiLib, first install its dependencies. This can be done with

.. code-block:: bash
	
	pip install future
	pip install h5py
	pip install scipy
	pip install matplotlib

Then, install FermiLib, by running

.. code-block:: bash

	python -m pip install --user fermilib

Alternatively, clone/download `this repo <https://github.com/babbush/fermilib>`_ (e.g., to your /home directory) and run

.. code-block:: bash

	cd /home/fermilib
	python -m pip install --user .

This will install both FermiLib and `ProjectQ <projectq.ch>`_ as well as all dependencies. FermiLib is compatible with both Python 2 and 3.


Basic FermiLib example
----------------------

To see a basic example with both fermionic and qubit operators as well as whether the installation worked, try to run the following code.

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


This code creates the fermionic operator :math:`a^\dagger_2 a_0` and adds its Hermitian conjugate :math:`a^\dagger_0 a_2` to it. It then maps the resulting fermionic operator to qubit operators using two transforms included in FermiLib, the Jordan-Wigner and Bravyi-Kitaev transforms. Despite the different representations, these operators are iso-spectral. The example also shows some of the intuitive string methods included in FermiLib.

Further examples can be found in the docs (`Examples` in the panel on the left) and in the FermiLib examples folder on `GitHub <https://github.com/babbush/fermilib/tree/master/examples>`_.