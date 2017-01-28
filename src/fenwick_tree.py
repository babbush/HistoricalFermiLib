class FenwickNode:
  """ Fenwick Tree node """
  parent = None    
  children = None 
  value = None
  
  def __init__(self, parent, children, value = None):
    """ Fenwick Tree node

    Args: 
        parent: FenwickTree. A parent node.
        children: FenwickTree, list. A list of children nodes.
        value: Int. Currently for debugging purposes only. 
    """

    self.children = children
    self.parent = parent
    self.value = value        

  def get_ancestors(self):
    """ Returns a list of ancestors of the node. Ordered from the earliest 
    
    Returns: 
        ancestor_list: A list of FenwickNodes.
    """

    node = self
    ancestor_list = []
    while node.parent != None :
        ancestor_list.append(node.parent)
        node = node.parent

    return ancestor_list


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

    self.nodes = [FenwickNode(None, []) for _ in range(n_qubits)] 
    self.root = self.nodes[n_qubits-1]
    self.root.value = n_qubits-1  # For debug atm

    def fenwick(left, right, parent): 
      """ This inner function is used to build the Fenwick 
      tree on nodes recursivelly. See Algorithm 1 in the paper. 
      
      Args:
          left: Int. Left boundary of the range.
          right: Int. Right boundary of the range.
          parent: Parent node
      """
      
      if left == right:
        return
      else:
        pivot = (left + right) >> 1      
        child = self.nodes[pivot] 
        
        child.value = pivot  # For debug atm
        parent.children.append(child)    # Parent -> child
        child.parent = parent            # Child -> parent

        fenwick(left, pivot, child)      # Recursion on the left part of the tree
        fenwick(pivot+1, right, parent)  # and to the right part

    fenwick(0, n_qubits-1, self.root)    # Builds the structure on nodes


  def get_node(self, j):
    """ Returns the node at j in the qubit register. Wrapper.
    
    Args: 
        j: Int. Fermionic site index.

    Returns: 
        FenwickNode: the node at j. 
    """
    
    return self.nodes[j]


  def get_U(self, j):
    """ The set of all ancestors of j, or the update set from the paper 
    
    Args:
        j: Int. Fermionic site index.

    Returns: 
        List of ancestors of j, ordered from earliest.
    """
    
    node = self.get_node(j)
    return node.get_ancestors()


  def get_F(self, j):
    """ The set of children of j-th site """
    # TODO: Not yet implemented
    return NotImplemented

  def get_C(self, j):
    """ Return the set of children with indices less than j od all ancestors of j"""
    # TODO: Not yet implemented
    return NotImplemented
    
