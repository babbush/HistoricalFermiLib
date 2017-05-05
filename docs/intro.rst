.. _intro:

Tutorial
========

.. toctree::
   :maxdepth: 2	

Getting started with FermiLib
-----------------------------

To start using FermiLib, simply run

.. code-block:: bash

	python -m pip install --user fermilib

or, alternatively, clone/download `this repo <https://github.com/babbush/fermilib>`_ (e.g., to your /home directory) and run

.. code-block:: bash

	cd /home/fermilib
	python -m pip install --user .

This will install both FermiLib and `ProjectQ <projectq.ch>`_ as well as all dependencies. FermiLib is compatible with both Python 2 and 3.


Basic FermiLib example
----------------------

To see a basic example with fermionic operators as well as whether the installation worked, try to run the following code.

.. code-block:: python

	from fermilib.ops import FermionOperator  # import fermionic operator class

	term_1 = FermionOperator('3^ 1', -1.7)
	term_2 = FermionOperator('4^ 3^ 9 1', 1. + 2.j)
	
	my_operator = term_1 + term_2
	print(my_operator)
	
	my_operator = FermionOperator('4^ 3^ 9 1', 1. + 2.j)
	term_2 = FermionOperator('3^ 1', -1.7)
	my_operator += term_2
	print('')
	print(my_operator)


This code creates two fermionic operators, adds them, and shows some of the intuitive string methods in FermiLib.

Further examples can be found in the docs (`Examples` in the panel on the left) and in the FermiLib examples folder on `GitHub <https://github.com/babbush/fermilib/tree/master/examples>`_.