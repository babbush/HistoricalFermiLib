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

  def test_fenwick_tree_ancestors(self):
    """ Ancestor test.
    Check validity of the get_U(j) method on 8 qubit register. 
    Note that root is the last node.
    """

    f = FenwickTree(8)
    self.assertEqual(len(f.get_U(7)), 0)
    self.assertEqual(f.get_U(3)[0], f.root)           # Is the parent of the middle child the root?
    self.assertEqual(f.get_U(0)[0], f.get_node(1))    # Are the ancestors chained correctly?
    self.assertEqual(f.get_U(0)[1], f.get_node(3))

if __name__ == '__main__':
  unittest.main()       
