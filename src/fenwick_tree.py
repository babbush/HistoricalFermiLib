class FenwickNode:
  """ Fenwick Tree node """
  parent = None    
  children = None 
  value = None
  
  def __init__(self, parent = None, children=[], value = None):
    """ Fenwick Tree node

    Args: 
        parent: FenwickTree. A parent node.
        children: FenwickTree, list. A list of children nodes.
        value: Int. Currently for debugging purposes only. 
    """

    self.children = children
    self.parent = parent
    self.value = value


class FenwickTree: 
  """ Recursive implementation of the Fenwick tree. 
     
  Please see Subsection B.2. of Operator Locality in Quantum Simulation of Fermionic Models by
  Havlicek, Troyer and Whitfiled for a reference to U, P and F sets of the Fenwick. 
  """

  root = None # Root node
  
  def __init__(self, n_qubits):
    """ Builds a Fenwick tree on n_qubits qubits 
    
    Args: 
        n_qubits: Int, the number of qubits in the system 
    """

    self.root = FenwickNode()
    self.root.value = n_qubits-1 # For debug atm

    def fenwick(left, right, parent): 
      """ This inner function is used to define the Fenwick 
      tree recursivelly. See Algorithm 1 in the paper. 
      
      Args:
          left: Int. Left boundary of the range.
          right: Int. Right boundary of the range.
          parent: Parent node
      """
      
      if left == right:
        return
      else:
        pivot = (left + right) >> 1 # Split the register into two parts
        child = FenwickNode(parent, [], pivot) # Prepare the left child
        parent.children.append(child) # Link the parent to the child
        child.parent = parent # And the child to the parent

        fenwick(left, pivot, child) # Proceed recursivelly to the left part of the tree
        fenwick(pivot+1, right, parent) # and to the right part         

    fenwick(0, n_qubits-1, self.root) # Build the structure
    
  def get_U(j):
    """ The set of all ancestors of j, or the update set """
    pass

  def get_F(j):
    """ The set of children of j-th site """

    pass

  def get_C(j):
    """ Return the set of children with indices less than j od all ancestors of j"""
    pass

    
