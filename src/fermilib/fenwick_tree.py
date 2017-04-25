"""Class to represent a Fenwick tree."""


class FenwickNode:
    """Fenwick Tree node."""
    parent = None
    children = None
    index = None

    def __init__(self, parent, children, index=None):
        """Fenwick Tree node. Single parent and multiple children.

        Args:     parent: FenwickNode. A parent node.     
                  children: List. A list of children nodes (FenwickNode).     
                  index: Int. Node label. 
        """

        self.children = children
        self.parent = parent
        self.index = index

    def get_ancestors(self):
        """Returns a list of ancestors of the node. Ordered from the earliest.

        Returns:     
            ancestor_list: A list of FenwickNodes.

        """

        node = self
        ancestor_list = []
        while node.parent is not None:
            ancestor_list.append(node.parent)
            node = node.parent

        return ancestor_list


class FenwickTree:
    """Recursive implementation of the Fenwick tree.

    Please see Subsection B.2. of Operator Locality in Quantum
    Simulation of Fermionic Models (arXiv:1701.07072) for
    a reference to the update set (U), the parity set (P) and the 
    children set (F) sets of the Fenwick.

    """

    root = None  # Root node

    def __init__(self, n_qubits):
        """Builds a Fenwick tree on n_qubits qubits.

        Args:     n_qubits: Int, the number of qubits in the system

        """

        self.nodes = [FenwickNode(None, []) for _ in range(n_qubits)]
        self.root = self.nodes[n_qubits - 1]
        self.root.index = n_qubits - 1  # For debug atm

        def fenwick(left, right, parent):
            """This inner function is used to build the Fenwick tree on nodes
            recursivelly. See Algorithm 1 in the paper.

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

                child.index = pivot  # For debug atm
                parent.children.append(child)    # Parent -> child
                child.parent = parent            # Child -> parent

                fenwick(left, pivot, child)      # Recursion on the left part
                fenwick(pivot + 1, right, parent)  # and to the right part

        fenwick(0, n_qubits - 1, self.root)    # Builds the structure on nodes

    def get_node(self, j):
        """Returns the node at j in the qubit register. Wrapper.

        Args:
            j: Int. Fermionic site index.

        Returns:
            FenwickNode: the node at j.

        """

        return self.nodes[j]

    def get_update_set(self, j):
        """The set of all ancestors of j, (the update set U from the paper).

        Args:
            j: Int. Fermionic site index.

        Returns:
            List of ancestors of j, ordered from earliest.

        """

        node = self.get_node(j)
        return node.get_ancestors()

    def get_children_set(self, j):
        """Returns the set of children of j-th site.

        Args:
            j: Int. Fermionic site index.

        Returns:
            A list of children of j. ordered from lowest index.

        """

        node = self.get_node(j)
        return node.children

    def get_remainder_set(self, j):
        """Return the set of children with indices less than j of all ancestors
        of j. The set C from (arXiv:1701.07072).   

        Args:
            j: Int. Fermionic site index.

        Returns:
            A list of children of j-ancestors with index less than j.

        """

        result = []
        ancestors = self.get_update_set(j)

        # This runs in O(log(N)log(N)) where N is the number of qubits. 
        for a in ancestors:
            for c in a.children:
                if c.index < j:
                    result.append(c)

        return result

    def get_parity_set(self, j):
        """Returns the union of the remainder set with children set. Coincides
        with the parity set of Tranter et al.

        Args:
            j: Int. Fermionic site index.

        Returns:
            A C union F

        """

        return self.get_remainder_set(j) + self.get_children_set(j)
