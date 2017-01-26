class FenwickNode:
  """ Fenwick Tree node """
  parent = None
  children = None
  value = None
  
  def __init__(self, parent = None, children=[], value = None):
    self.children = children
    self.parent = parent
    self.value = value


class FenwickTree: 
  """ Recursive implementation of the Fenwick tree. 
     
  Please see Subsection B.2. of Operator Locality in Quantum Simulation of Fermionic Models by
  Havlicek, Troyer and Whitfiled for a reference to U, P and F sets of the Fenwick. 
  
  TOOD: this implementation is legible, but greedy. 
  """

  root = None # Root node
  
  def __init__(self, n_qubits):
    """ Builds a Fenwick tree on n_qubits qubits 
    
    Args: 
        n_qubits: Int, the number of qubits in the system 
    """

    self.root = FenwickNode()
    self.root.value = n_qubits-1 # For debug atm

    def F(L, R, parent): 
      """ This inner function defines the Fenwick tree recursivelly.
      See Algorithm 1 in the paper. """
      
      if L == R:
        return
      else:
        pivot = (L+R)>>1
      
        lchild = FenwickNode(parent, [])        
        lchild.value = pivot # For debug at the moment
        
        parent.children.append(lchild)
        lchild.parent = parent

        F(L, pivot, lchild) # build the left part of the tree
        F(pivot + 1, R,  parent) # build the right part         

    F(0, n_qubits-1, self.root) # Build the structure
    
  def U(j):
    """ The set of all ancestors of j, or the update set """
    pass

  def F(j):
    """ The set of children of j-th site """

    pass

  def C(j):
    """ Return the set of children with indices less than j od all ancestors of j"""
    pass

    
