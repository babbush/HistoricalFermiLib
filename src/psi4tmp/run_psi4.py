"""Functions to prepare psi4 input and run calculations."""
from __future__ import absolute_import

import os
import re
import subprocess

from fermilib.config import *


def create_geometry_string(geometry):
    """This function converts MolecularData geometry to psi4 geometry.

    Args:
      geometry: A list of tuples giving the coordinates of each atom.
        example is [('H', (0, 0, 0)), ('H', (0, 0, 0.7414))]. Distances in
        atomic units. Use atomic symbols to specify atoms.

    Returns:
      geo_string: A string giving the geometry for each atom on a line, e.g.:
        H 0. 0. 0.
        H 0. 0. 0.7414

    """
    geo_string = ''
    for item in geometry:
        atom = item[0]
        coordinates = item[1]
        line = '{} {} {} {}'.format(atom,
                                    coordinates[0],
                                    coordinates[1],
                                    coordinates[2])
        if len(geo_string) > 0:
            geo_string += '\n'
        geo_string += line
    return geo_string


def generate_psi4_input(molecule,
                        run_scf,
                        run_mp2,
                        run_cisd,
                        run_ccsd,
                        run_fci,
                        verbose,
                        tolerate_error,
                        memory):
    """This function creates and saves a psi4 input file.

    Args:
      molecule: An instance of the MolecularData class.
      run_scf: Boolean to run SCF calculation.
      run_mp2: Boolean to run MP2 calculation.
      run_cisd: Boolean to run CISD calculation.
      run_ccsd: Boolean to run CCSD calculation.
      run_fci: Boolean to FCI calculation.
      verbose: Boolean whether to print calculation results to screen.
      tolerate_error: Whether to fail or merely warn when Psi4 fails.
      memory: Int giving amount of memory to allocate in MB.

    Returns:
      input_file: A string giving the name of the saved input file.

    """
    # Create Psi4 geometry string.
    geo_string = create_geometry_string(molecule.geometry)

    # Parse input template.
    template_file = PSI4_DIRECTORY + '/psi4_template'
    input_template = []
    with open(template_file, 'r') as stream:
        for line in stream:
            input_template += [line]

    # Populate contents of input file based on automatic parameters.
    input_content = [re.sub('&THIS_DIRECTORY',
                            PSI4_DIRECTORY, line)
                     for line in input_template]

    # Populate contents of input file based on MolecularData parameters.
    input_content = [re.sub('&geometry', str(molecule.geometry), line)
                     for line in input_content]
    input_content = [re.sub('&basis', molecule.basis, line)
                     for line in input_content]
    input_content = [re.sub('&charge', str(molecule.charge), line)
                     for line in input_content]
    input_content = [re.sub('&multiplicity', str(molecule.multiplicity), line)
                     for line in input_content]
    input_content = [re.sub('&description', str(molecule.description), line)
                     for line in input_content]
    input_content = [re.sub('&geo_string', geo_string, line)
                     for line in input_content]

    # Populate contents of input file based on provided calculation parameters.
    input_content = [re.sub('&run_scf', str(run_scf), line)
                     for line in input_content]
    input_content = [re.sub('&run_mp2', str(run_mp2), line)
                     for line in input_content]
    input_content = [re.sub('&run_cisd', str(run_cisd), line)
                     for line in input_content]
    input_content = [re.sub('&run_ccsd', str(run_ccsd), line)
                     for line in input_content]
    input_content = [re.sub('&run_fci', str(run_fci), line)
                     for line in input_content]
    input_content = [re.sub('&tolerate_error', str(tolerate_error), line)
                     for line in input_content]
    input_content = [re.sub('&verbose', str(verbose), line)
                     for line in input_content]
    input_content = [re.sub('&memory', str(memory), line)
                     for line in input_content]

    # Write input file and return handle.
    input_file = molecule.data_handle() + '.inp'
    with open(input_file, 'w') as stream:
        stream.write(''.join(input_content))
    return input_file


def clean_up(molecule, delete_input=True, delete_output=False):
    input_file = molecule.data_handle() + '.inp'
    output_file = molecule.data_handle() + '.out'
    run_directory = os.getcwd()
    for local_file in os.listdir(run_directory):
        if local_file.endswith('.clean'):
            os.remove(run_directory + '/' + local_file)
    try:
        os.remove('timer.dat')
    except:
        pass
    if delete_input:
        os.remove(input_file)
    if delete_output:
        os.remove(output_file)


def run_psi4(molecule,
             run_scf=True,
             run_mp2=False,
             run_cisd=False,
             run_ccsd=False,
             run_fci=False,
             verbose=False,
             tolerate_error=False,
             delete_input=True,
             delete_output=False,
             memory=8000):
    """This function runs a Psi4 calculation.

    Args:
      molecule: An instance of the MolecularData class.
      run_scf: Optional boolean to run SCF calculation.
      run_mp2: Optional boolean to run MP2 calculation.
      run_cisd: Optional boolean to run CISD calculation.
      run_ccsd: Optional boolean to run CCSD calculation.
      run_fci: Optional boolean to FCI calculation.
      verbose: Boolean whether to print calculation results to screen.
      tolerate_error: Optional boolean to warn or raise when Psi4 fails.
      delete_input: Optional boolean to delete psi4 input file.
      delete_output: Optional boolean to delete psi4 output file.
      memory: Optional int giving amount of memory to allocate in MB.

    Returns:
      molecule: The updated MolecularData object.

    Raises:
      psi4 errors: An error from psi4.

    """
    # Prepare input.
    input_file = generate_psi4_input(molecule,
                                     run_scf,
                                     run_mp2,
                                     run_cisd,
                                     run_ccsd,
                                     run_fci,
                                     verbose,
                                     tolerate_error,
                                     memory)

    # Run psi4.
    output_file = molecule.data_handle() + '.out'
    try:
        process = subprocess.Popen(['psi4', input_file, output_file])
        process.wait()
    except:
        print('Psi4 calculation for {} has failed.'.format(molecule.name))
        process.kill()
        clean_up(molecule, delete_input, delete_output)
        if not tolerate_error:
            raise
    else:
        clean_up(molecule, delete_input, delete_output)

    # Return updated molecule instance.
    molecule.refresh()
    return molecule
