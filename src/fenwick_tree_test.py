from fenwick_tree import (FenwickTree, FenwickNode)
import unittest

class FenwickTreeTest(unittest.TestCase):
  def setUp(self):
      pass;

  def test_fenwick_tree_structure(self):
    """ A lookup test on 5-qubit fenwick tree """ 
        
    f = FenwickTree(5)
    self.assertEqual(f.root.children[0].value, 2)
    self.assertEqual(f.root.children[1].value, 3)
    self.assertEqual(f.root.children[0].children[0].children[0].value, 0)

if __name__ == '__main__':
  unittest.main()       
