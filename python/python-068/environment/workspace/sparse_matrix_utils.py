
import numpy as np
from scipy import sparse


def dense_to_triplet(A: np.ndarray, base: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    A = np.asarray(A)
    rows, cols = np.nonzero(A)
    vals = A[rows, cols]
    if base != 0:
        rows = rows + base
        cols = cols + base
    return rows, cols, vals


def sparse_to_triplet(M: sparse.spmatrix, base: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    M = M.tocoo()
    rows = M.row.copy()
    cols = M.col.copy()
    vals = M.data.copy()
    if base != 0:
        rows = rows + base
        cols = cols + base
    return rows, cols, vals


def triplet_to_file(rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, filename: str):
    with open(filename, 'w') as f:
        f.write(f"{len(rows)}\n")
        for r, c, v in zip(rows, cols, vals):
            f.write(f"{r} {c} {v:.16e}\n")


def build_coupled_jacobian_sparsity(
    nx: int,
    ny: int,
    stencil: str = '5point'
) -> sparse.csr_matrix:
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

                for v2 in range(n_vars):
                    row_indices.append(r)
                    col_indices.append(idx(v2, i, j))

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
    return indices - base_from + base_to
