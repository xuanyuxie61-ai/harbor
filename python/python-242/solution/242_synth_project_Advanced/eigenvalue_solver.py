"""
eigenvalue_solver.py  --  Legacy entry (redirects to sparse_hamiltonian)
=======================================================================
"""
from sparse_hamiltonian import (  # noqa: F401
    R8STOMatrix,
    lanczos_eigen,
    jacobi_eigenvalues,
)
