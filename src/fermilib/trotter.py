import projectq

def error_operator(terms):
    """Determine the operator giving the error in the ground state
    energy for the second order Trotter-Suzuki expansion and the triangle
    inequality.

    Args:
      terms: a list of QubitTerms in the Hamiltonian to be simulated.

    Returns:
      A float upper bound on the norm of the error in the ground state
      energy."""

    raise NotImplementedError

def error_bound(terms):
    """Numerically upper bound the error in the ground state energy
    for the second order Trotter-Suzuki expansion.

    Args:
      terms: a list of QubitTerms in the Hamiltonian to be simulated.

    Returns:
      A float upper bound on the norm of the error in the ground state
      energy.

    Notes: follows Poulin et al.'s work in "The Trotter Step Size
    Required for Accurate Quantum Simulation of Quantum Chemistry".
    See Equation 16 in particular.
    """
    
    raise NotImplementedError