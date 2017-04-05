"""Tests for molecular_data and run_psi4."""
from __future__ import absolute_import

import os
import unittest

import numpy

from fermilib import chemical_series


class ChemicalSeries(unittest.TestCase):

  def test_make_atomic_ring(self):
    spacing = 1.
    basis = 'sto-3g'
    for n_atoms in range(2, 10):
      molecule = chemical_series.make_atomic_ring(n_atoms, spacing, basis,
                                                  autosave=False)

      # Check that ring is centered.
      vector_that_should_sum_to_zero = 0.
      for atom in molecule.geometry:
        for coordinate in atom[1]:
          vector_that_should_sum_to_zero += coordinate
      self.assertAlmostEqual(vector_that_should_sum_to_zero, 0.)

      # Check that the spacing between the atoms is correct.
      for atom_index in range(n_atoms):
        if atom_index:
          atom_b = molecule.geometry[atom_index]
          coords_b = atom_b[1]
          atom_a = molecule.geometry[atom_index - 1]
          coords_a = atom_a[1]
          observed_spacing = numpy.sqrt(numpy.square(
              coords_b[0] - coords_a[0]) + numpy.square(
              coords_b[1] - coords_a[1]) + numpy.square(
              coords_b[2] - coords_a[2]))
          self.assertAlmostEqual(observed_spacing, spacing)


if __name__ == '__main__':
  unittest.main()
