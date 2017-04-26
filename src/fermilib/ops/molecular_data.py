"""Class and functions to store quantum chemistry data."""
from __future__ import absolute_import

import os
import pickle
import sys

import numpy

from fermilib import interaction_operators
from fermilib import interaction_rdms
from fermilib.config import *


"""NOTE ON PQRS CONVENTION:
  The data structures which hold fermionic operators / integrals /
  coefficients assume a particular convention which depends on how integrals
  are labeled:
  h[p,q]=\int \phi_p(x)* (T + V_{ext}) \phi_q(x) dx
  h[p,q,r,s]=\int \phi_p(x)* \phi_q(y)* V_{elec-elec} \phi_r(y) \phi_s(x) dxdy
  With this convention, the molecular Hamiltonian becomes
  H =\sum_{p,q} h[p,q] a_p^\dagger a_q
    + 0.5 * \sum_{p,q,r,s} h[p,q,r,s] a_p^\dagger a_q^\dagger a_r a_s
"""


# Define error objects which inherit from Exception.
class MoleculeNameError(Exception):
    pass


class MissingCalculationError(Exception):
    pass


# Functions to change from Bohr to Angstroms and back.
def bohr_to_angstroms(distance):
    return 0.529177 * distance


def angstroms_to_bohr(distance):
    return 1.889726 * distance


# The Periodic Table as a python list and dictionary.
_PERIODIC_TABLE = [
    '?',
    'H', 'He',
    'Li', 'Be',
    'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg',
    'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',
    'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr',
    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
    'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    'Cs', 'Ba',
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
    'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',
    'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
    'Fr', 'Ra',
    'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',
    'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']
_PERIODIC_HASH_TABLE = {}
for atomic_number, atom in enumerate(_PERIODIC_TABLE):
    _PERIODIC_HASH_TABLE[atom] = atomic_number


# Spin polarization of atoms on period table.
_PERIODIC_POLARIZATION = [-1,
                          1, 0,
                          1, 0, 1, 2, 3, 2, 1, 0,
                          1, 0, 1, 2, 3, 2, 1, 0,
                          1, 0, 1, 2, 3, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0,
                          1, 0, 1, 2, 5, 6, 5, 8, 9, 0, 1, 0, 1, 2, 3, 2, 1, 0]


def name_molecule(geometry,
                  basis,
                  multiplicity,
                  charge,
                  description):
    """Function to name molecules.

    Args:
        geometry: A list of tuples giving the coordinates of each atom.
            example is [('H', (0, 0, 0)), ('H', (0, 0, 0.7414))].
            Distances in atomic units. Use atomic symbols to specify atoms.
        basis: A string giving the basis set. An example is 'cc-pvtz'.
        multiplicity: An integer giving the spin multiplicity.
        charge: An integer giving the total molecular charge.
        description: A string giving a description. As an example,
            for dimers a likely description is the bond length (e.g. 0.7414).

    Returns:
        name: A string giving the name of the instance.

    Raises:
        MoleculeNameError: If spin multiplicity is not valid.

    """
    # Get sorted atom vector.
    atoms = [item[0] for item in geometry]
    atom_charge_info = [(atom, atoms.count(atom)) for atom in set(atoms)]
    sorted_info = sorted(atom_charge_info,
                         key=lambda atom: _PERIODIC_HASH_TABLE[atom[0]])

    # Name molecule.
    name = '{}{}'.format(sorted_info[0][0], sorted_info[0][1])
    for info in sorted_info[1::]:
        name += '-{}{}'.format(info[0], info[1])

    # Add basis.
    name += '_{}'.format(basis)

    # Add multiplicity.
    multiplicity_dict = {1: 'singlet',
                         2: 'doublet',
                         3: 'triplet',
                         4: 'quartet',
                         5: 'quintet',
                         6: 'sextet',
                         7: 'septet',
                         8: 'octet',
                         9: 'nonet',
                         10: 'dectet',
                         11: 'undectet',
                         12: 'duodectet'}
    if (multiplicity not in multiplicity_dict):
        raise MoleculeNameError('Invalid spin multiplicity provided.')
    else:
        name += '_{}'.format(multiplicity_dict[multiplicity])

    # Add charge.
    if charge > 0:
        name += '{}+'.format(charge)
    elif charge < 0:
        name += '{}-'.format(charge)

    # Optionally add descriptive tag and return.
    if description:
        name += '_{}'.format(description)
    return name


def geometry_from_file(file_name):
    """Function to create molecular geometry from text file.

    Args:
        file_name: a string giving the location of the geometry file.
            It is assumed that the geometry is given for each atom on a line,
            e.g.:
            H 0. 0. 0.
            H 0. 0. 0.7414

    Returns:
        geometry: A list of tuples giving the coordinates of each atom.
            example is [('H', (0, 0, 0)), ('H', (0, 0, 0.7414))].
            Distances in atomic units. Use atomic symbols to specify atoms.
    """
    geometry = []
    with open(file_name, 'r') as stream:
        for line in stream:
            data = line.split()
            if len(data) == 4:
                atom = data[0]
                coordinates = (float(data[1]), float(data[2]), float(data[3]))
                geometry += [(atom, coordinates)]
    return geometry


class MolecularData(object):

    """Class for storing molecule data from a fixed basis set at a fixed
    geometry that is obtained from classical electronic structure packages. Not
    every field is filled in every calculation. All data that can (for some
    instance) exceed 10Mb should be saved seperately. Intention is to pickle
    objects to database with unique name.

    Attributes:
        geometry: A list of tuples giving the coordinates of each atom. An
            example is [('H', (0, 0, 0)), ('H', (0, 0, 0.7414))].
            Distances in atomic units. Use atomic symbols to specify
            atoms.
        basis: A string giving the basis set. An example is 'cc-pvtz'.
        charge: An integer giving the total molecular charge. Defaults to 0.
        multiplicity: An integer giving the spin multiplicity.
        description: An optional string giving a description. As an example,
            for dimers a likely description is the bond length (e.g. 0.7414).
        name: A string that identifies the instance.
        n_atoms: Integer giving the number of atoms in the molecule.
        n_electrons: Integer giving the number of electrons in the molecule.
        atoms: List of the atoms in molecule sorted by atomic number.
        protons: List of the atomic charges in molecule sorted by atomic
            number.
        hf_energy: Energy from open or closed shell Hartree-Fock.
        nuclear_repulsion: Energy from nuclei-nuclei interaction.
        canonical_orbitals: numpy array giving canonical orbital coefficients.
        n_orbitals: Integer giving total number of spatial orbitals.
        n_qubits: Integer giving total number of qubits that would be needed.
        orbital_energies: Numpy array giving the canonical orbital energies.
        fock_matrix: Numpy array giving the Fock matrix.
        orbital_overlaps: Numpy array giving the orbital overlap coefficients.
        kinetic_integrals: Numpy array giving 1-body kinetic energy integrals.
        potential_integrals: Numpy array giving 1-body potential energy
            integrals.
        mp2_energy: Energy from MP2 perturbation theory.
        cisd_energy: Energy from configuration interaction singles and
            doubles.
        cisd_one_rdm: Numpy array giving 1-RDM from CISD calculation.
        fci_energy: Exact energy of molecule within given basis.
        fci_one_rdm: Numpy array giving 1-RDM from FCI calculation.
        ccsd_energy: Energy from coupled cluster singles and doubles.
        ccsd_amplitudes: Molecular operator holding coupled cluster
            amplitudes.
    """

    def __init__(self,
                 geometry,
                 basis,
                 multiplicity,
                 charge=0,
                 description=None,
                 autosave=True):
        """Initialize molecular metadata which defines class.

        Args:
            geometry: A list of tuples giving the coordinates of each atom.
                An example is [('H', (0, 0, 0)), ('H', (0, 0, 0.7414))].
                Distances in atomic units. Use atomic symbols to
                specify atoms.
            basis: A string giving the basis set. An example is 'cc-pvtz'.
            charge: An integer giving the total molecular charge. Defaults
                to 0.
            multiplicity: An integer giving the spin multiplicity.
            description: A optional string giving a description. As an
                example, for dimers a likely description is the bond length
                (e.g. 0.7414).
            autosave: Whether to save molecular data automatically.
        """
        # Metadata fields which must be provided.
        self.geometry = geometry
        self.basis = basis
        self.multiplicity = multiplicity

        # Metadata fields with default values.
        self.charge = charge
        self.description = description

        # Name molecule and load any fields that have been previously computed.
        self.name = name_molecule(geometry, basis, multiplicity,
                                  charge, description)
        if os.path.isfile(self.data_handle() + '.pkl'):
            self.refresh()
            return

        # Attributes generated automatically by class.
        self.n_atoms = len(geometry)
        self.atoms = sorted([row[0] for row in geometry],
                            key=lambda atom: _PERIODIC_HASH_TABLE[atom])
        self.protons = [_PERIODIC_HASH_TABLE[atom] for atom in self.atoms]
        self.n_electrons = sum(self.protons) - charge

        # Generic attributes from calculations.
        self.n_orbitals = None
        self.n_qubits = None
        self.nuclear_repulsion = None

        # Attributes generated from SCF calculation.
        self.hf_energy = None
        self.canonical_orbitals = None
        self.orbital_energies = None

        # Attributes generated from integrals.
        self.orbital_overlaps = None
        self.one_body_integrals = None

        # Attributes generated from MP2 calculation.
        self.mp2_energy = None

        # Attributes generated from CISD calculation.
        self.cisd_energy = None
        self.cisd_one_rdm = None

        # Attributes generated from exact diagonalization.
        self.fci_energy = None
        self.fci_one_rdm = None

        # Attributes generated from CCSD calculation.
        self.ccsd_energy = None
        self.ccsd_amplitudes = None

        # Save the new molecule.
        if autosave:
            self.save()

    def data_handle(self):
        """Method to automatically give file name of molecule."""
        return DATA_DIRECTORY + '/' + self.name

    def save(self):
        """Method to automatically pickle the class under systematic name."""
        with open(self.data_handle() + '.pkl', 'wb') as stream:
            pickle.dump(self, stream)

    def refresh(self):
        """Method to automatically unPickle the class under systematic name."""
        with open(self.data_handle() + '.pkl', 'rb') as stream:
            sys.path.append(THIS_DIRECTORY)
            updated_molecular_data = pickle.load(stream)
            self.__dict__ = updated_molecular_data.__dict__

    def get_n_alpha_electrons(self):
        """Return number of alpha electrons."""
        return self.n_electrons / 2 + (self.multiplicity - 1)

    def get_n_beta_electrons(self):
        """Return number of beta electrons."""
        return self.n_electrons / 2 - (self.multiplicity - 1)

    def get_integrals(self):
        """Method to return 1-electron and 2-electron integrals in MO basis.

        Returns:
          one_body_integrals: An array of the one-electron integrals having
            shape of (n_orbitals, n_orbitals).
          two_body_integrals: An array of the two-electron integrals having
            shape of (n_orbitals, n_orbitals, n_orbitals, n_orbitals).

        Raises:
          MisissingCalculationError: If the SCF calculation has not been
            performed.
        """
        # Make sure integrals have been computed.
        if self.hf_energy is None:
            raise MissingCalculationError(
                'Missing file {}. Run SCF before loading integrals.'.format(
                    self.data_handle() + '_eri.npy'))

        # Get integrals and return.
        two_body_integrals = numpy.load(self.data_handle() + '_eri.npy')
        return self.one_body_integrals, two_body_integrals

    def get_active_space_integrals(self, active_space_start,
                                   active_space_stop=None):
        """Restricts a molecule at a spatial orbital level to the active space
        defined by active_space=[start,stop]. Note that one_body_integrals and
        two_body_integrals must be defined in an orthonormal basis set, which
        is typically the case when defining an active space.

        Args:
            active_space_start(int): spatial orbital index defining active
                space start.
            active_space_stop(int): spatial orbital index defining active
                space stop.

        Returns:
            core_constant: Adjustment to constant shift in Hamiltonian from
                integrating out core orbitals
            one_body_integrals_new: New one-electron integrals over active
                space.
            two_body_integrals_new: New two-electron integrals over active
                space.
        """
        # Get integrals.
        one_body_integrals, two_body_integrals = self.get_integrals()
        n_orbitals = one_body_integrals.shape[0]
        if active_space_stop is None:
            active_space_stop = n_orbitals

        # Determine core constant
        core_constant = 0.0
        for i in range(active_space_start):
            core_constant += 2 * one_body_integrals[i, i]
            for j in range(active_space_start):
                core_constant += (2 * two_body_integrals[i, j, j, i] -
                                  two_body_integrals[i, j, i, j])

        # Modified one electron integrals
        one_body_integrals_new = numpy.copy(one_body_integrals)
        for u in range(active_space_start, active_space_stop):
            for v in range(active_space_start, active_space_stop):
                for i in range(active_space_start):
                    one_body_integrals_new[u, v] += (
                        2 * two_body_integrals[i, u, v, i] -
                        two_body_integrals[i, u, i, v])

        # Restrict integral ranges and change M appropriately
        return (core_constant,
                one_body_integrals_new[active_space_start: active_space_stop,
                                       active_space_start: active_space_stop],
                two_body_integrals[active_space_start: active_space_stop,
                                   active_space_start: active_space_stop,
                                   active_space_start: active_space_stop,
                                   active_space_start: active_space_stop])

    def get_molecular_hamiltonian(self,
                                  active_space_start=None,
                                  active_space_stop=None):
        """Output arrays of the second quantized Hamiltonian coefficients.

        Args:
            rotation_matrix: A square numpy array or matrix having dimensions
                of n_orbitals by n_orbitals. Assumed to be real and
                invertible.
            active_space_start: An optional int giving the first orbital
                in the active space.
            active_space stop: An optional int giving the last orbital
                in the active space.

        Returns:
            molecular_hamiltonian: An instance of the MolecularOperator class.

        """
        # Get active space integrals.
        if active_space_start is None:
            one_body_integrals, two_body_integrals = self.get_integrals()
            constant = self.nuclear_repulsion
        else:
            core_adjustment, one_body_integrals, two_body_integrals = self.\
                get_active_space_integrals(
                    active_space_start, active_space_stop)
            constant = self.nuclear_repulsion + core_adjustment
        n_qubits = 2 * one_body_integrals.shape[0]

        # Initialize Hamiltonian coefficients.
        one_body_coefficients = numpy.zeros((n_qubits, n_qubits))
        two_body_coefficients = numpy.zeros((n_qubits, n_qubits,
                                             n_qubits, n_qubits))

        # Loop through integrals.
        for p in range(n_qubits / 2):
            for q in range(n_qubits / 2):

                # Populate 1-body coefficients. Require p and q have same spin.
                one_body_coefficients[2 * p, 2 * q] = one_body_integrals[p, q]
                one_body_coefficients[2 * p + 1, 2 *
                                      q + 1] = one_body_integrals[p, q]

                # Continue looping to prepare 2-body coefficients.
                for r in range(n_qubits / 2):
                    for s in range(n_qubits / 2):

                        # Require p,s and q,r to have same spin. Handle mixed
                        # spins.
                        two_body_coefficients[2 * p, 2 * q + 1, 2 * r + 1,
                                              2 * s] = (
                            two_body_integrals[p, q, r, s] / 2.)
                        two_body_coefficients[2 * p + 1, 2 * q, 2 * r,
                                              2 * s + 1] = (
                            two_body_integrals[p, q, r, s] / 2.)

                        # Avoid having two electrons in same orbital. Handle
                        # same spins.
                        if p != q and r != s:
                            two_body_coefficients[2 * p, 2 * q, 2 * r,
                                                  2 * s] = (
                                two_body_integrals[p, q, r, s] / 2.)
                            two_body_coefficients[2 * p + 1, 2 * q + 1,
                                                  2 * r + 1, 2 * s + 1] = (
                                two_body_integrals[p, q, r, s] / 2.)

        # Truncate.
        one_body_coefficients[
            numpy.absolute(one_body_coefficients) < EQ_TOLERANCE] = 0.
        two_body_coefficients[
            numpy.absolute(two_body_coefficients) < EQ_TOLERANCE] = 0.

        # Cast to InteractionOperator class and return.
        molecular_hamiltonian = interaction_operators.InteractionOperator(
            constant, one_body_coefficients, two_body_coefficients)
        return molecular_hamiltonian

    def get_molecular_rdm(self, use_fci=False):
        """Method to return 1-RDM and 2-RDMs from CISD or FCI.

        Args:
            use_fci: Boolean indicating whether to use RDM from FCI
                calculation.

        Returns:
            rdm: An instance of the MolecularRDM class.

        Raises:
            MisissingCalculationError: If the CI calculation has not been
                performed.
        """
        # Make sure requested RDM has been computed and load.
        if use_fci:
            if self.fci_energy is None:
                raise MissingCalculationError(
                    'Missing {}. '.format(
                        self.data_handle() + '_fci_rdm.npy') +
                    'Run FCI calculation before loading FCI RDMs.')
            else:
                rdm_name = self.data_handle() + '_fci_rdm.npy'
                one_rdm = self.fci_one_rdm
        else:
            if self.cisd_energy is None:
                raise MissingCalculationError(
                    'Missing {}. '.format(
                        self.data_handle() + '_cisd_rdm.npy') +
                    'Run CISD calculation before loading CISD RDMs.')
            else:
                rdm_name = self.data_handle() + '_cisd_rdm.npy'
                one_rdm = self.cisd_one_rdm
        two_rdm = numpy.load(rdm_name)

        # Truncate.
        one_rdm[numpy.absolute(one_rdm) < EQ_TOLERANCE] = 0.
        two_rdm[numpy.absolute(two_rdm) < EQ_TOLERANCE] = 0.

        # Cast to InteractionRDM class.
        rdm = interaction_rdms.InteractionRDM(one_rdm, two_rdm)
        return rdm
