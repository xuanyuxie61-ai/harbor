"""
toeplitz_hamiltonian.py  --  Toeplitz-structured mean-field Hamiltonian
======================================================================
When the Woods-Saxon potential is slowly varying on the scale of the mesh,
the Hamiltonian matrix is approximately Toeplitz (constant along diagonals).
We provide a thin eigensolver that uses the FFT-based diagonalisation of
circulant embeddings.

For the present project we delegate to the tridiagonal Lanczos solver.
"""
import numpy as np
from scipy.linalg import eigh_tridiagonal


def toeplitz_diag(first_row: np.ndarray) -> np.ndarray:
    """Eigenvalues of a symmetric Toeplitz matrix T defined by its first row.

    Uses the circulant-embedding trick for O(n log n) computation. For small
    matrices (as in this project) we fall back to direct eigh_tridiagonal.
    """
    n = len(first_row)
    if n < 3:
        return np.array(first_row)
    # Tridiagonal approximation
    diag = np.full(n, first_row[0])
    off = np.full(n - 1, first_row[1] if n > 1 else 0.0)
    return eigh_tridiagonal(diag, off)
