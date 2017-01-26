class FenwickNode:
    """ Fenwick Tree node """

    parent = None
    children = None
    value = None
    
    def __init__(self, parent = None, children=[], value = None):
        self.children = children
        self.parent = parent
        self.value = value


# Recursive construction of the Fenwick. 
# TODO: move it outside the global scone, into FenwickTree? 
# TODO: ensure that nodes are not copied by value. 

def fenwick(L, R, parent): 
        if L == R:
            return
        else:
            pivot = (L+R)>>1
            
            lchild = FenwickNode(parent, [])        
            lchild.value = pivot # For debug at the moment
            parent.children.append(lchild)
            print(parent.children)
            lchild.parent = parent

            fenwick(L, pivot, lchild) # build the left part of the tree
            fenwick(pivot + 1, R,  parent) # build the right part


class FenwickTree: 
    """
    Recursive implementation of the Fenwick tree. 
   
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
        fenwick(0, n_qubits-1, self.root) # Build the structure
        


    def U(j):
        """ The set of all ancestors of j, or the update set """
        pass

    def F(j):
        """ The set of children of j-th site """

        pass

    def C(j):
        """ Return the set of children with indices less than j od all ancestors of j"""
        pass

    
