"""
sparse_matrix_utils.py
Sparse matrix format conversions for large-scale ecological computations.

Adapted from:
  - 783_msm_to_st: Sparse matrix to triplet format conversion
  - 459_ge_to_st: Dense to sparse triplet conversion

Role in synthesis:
  Provides sparse matrix I/O and format conversion for Jacobian matrices
  arising from spatial discretization of the eco-epidemiological PDE system.
"""

import numpy as np
from scipy import sparse


def dense_to_triplet(A: np.ndarray, base: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a dense matrix to sparse triplet (COO) format.

    Parameters
    ----------
    A : ndarray
        Dense matrix.
    base : int
        Index base (0 or 1).

    Returns
    -------
    rows, cols, vals : arrays of non-zero entries.
    """
    A = np.asarray(A)
    rows, cols = np.nonzero(A)
    vals = A[rows, cols]
    if base != 0:
        rows = rows + base
        cols = cols + base
    return rows, cols, vals


def sparse_to_triplet(M: sparse.spmatrix, base: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a scipy sparse matrix to triplet format.
    """
    M = M.tocoo()
    rows = M.row.copy()
    cols = M.col.copy()
    vals = M.data.copy()
    if base != 0:
        rows = rows + base
        cols = cols + base
    return rows, cols, vals


def triplet_to_file(rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, filename: str):
    """
    Write triplet data to a text file.
    """
    with open(filename, 'w') as f:
        f.write(f"{len(rows)}\n")
        for r, c, v in zip(rows, cols, vals):
            f.write(f"{r} {c} {v:.16e}\n")


def build_coupled_jacobian_sparsity(
    nx: int,
    ny: int,
    stencil: str = '5point'
) -> sparse.csr_matrix:
    """
    Build the sparsity pattern of the Jacobian for the coupled 6-field system
    on an nx x ny grid with a 5-point Laplacian stencil.

    The Jacobian has block structure: 6x6 blocks per grid point,
    with each block connecting to nearest neighbors.
    """
    n_total = nx * ny
    n_vars = 6
    N = n_total * n_vars

    row_indices = []
    col_indices = []

    def idx(var, i, j):
        return var * n_total + i * ny + j

    for var in range(n_vars):
        for i in range(nx):
            for j in range(ny):
                r = idx(var, i, j)
                # Diagonal block: all 6 variables at same point
                for v2 in range(n_vars):
                    row_indices.append(r)
                    col_indices.append(idx(v2, i, j))
                # Off-diagonal: spatial neighbors for same variable
                if stencil == '5point':
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < nx and 0 <= nj < ny:
                            row_indices.append(r)
                            col_indices.append(idx(var, ni, nj))

    data = np.ones(len(row_indices))
    J = sparse.coo_matrix((data, (row_indices, col_indices)), shape=(N, N)).tocsr()
    return J


def rebase_indices(indices: np.ndarray, base_from: int, base_to: int) -> np.ndarray:
    """
    Rebase index array from one base to another.
    """
    return indices - base_from + base_to
