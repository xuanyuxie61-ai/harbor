
import numpy as np
from typing import Tuple, List


def ge_to_crs(A: np.ndarray) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2D array")

    n = A.shape[0]
    nz = np.count_nonzero(A)
    row_ptr = np.zeros(n + 1, dtype=int)
    col = np.zeros(nz, dtype=int)
    val = np.zeros(nz, dtype=float)

    row_ptr[0] = 0
    k = 0
    for i in range(n):
        for j in range(n):
            if A[i, j] != 0.0:
                col[k] = j
                val[k] = A[i, j]
                k += 1
        row_ptr[i + 1] = k

    return n, nz, row_ptr, col, val


def ge_to_st(A: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    if A.ndim != 2:
        raise ValueError("A must be a 2D array")

    m, n = A.shape
    nz = np.count_nonzero(A)
    ist = np.zeros(nz, dtype=int)
    jst = np.zeros(nz, dtype=int)
    Ast = np.zeros(nz, dtype=float)

    k = 0
    for j in range(n):
        for i in range(m):
            if A[i, j] != 0.0:
                ist[k] = i
                jst[k] = j
                Ast[k] = A[i, j]
                k += 1

    return nz, ist, jst, Ast


def crs_matvec(n: int, row_ptr: np.ndarray, col: np.ndarray,
               val: np.ndarray, x: np.ndarray) -> np.ndarray:
    if x.shape[0] != n:
        raise ValueError("Dimension mismatch between matrix and vector")
    y = np.zeros(n, dtype=float)
    for i in range(n):
        for idx in range(row_ptr[i], row_ptr[i + 1]):
            y[i] += val[idx] * x[col[idx]]
    return y


def build_laplacian_2d_crs(nx: int, ny: int, dx: float, dy: float,
                           boundary: str = "dirichlet") -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must be at least 2")
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("dx and dy must be positive")

    n = nx * ny

    nz_max = 5 * n
    row_ptr = np.zeros(n + 1, dtype=int)
    col = np.zeros(nz_max, dtype=int)
    val = np.zeros(nz_max, dtype=float)

    idx = 0
    for j in range(ny):
        for i in range(nx):
            row = j * nx + i
            start = idx

            cx = 1.0 / (dx * dx)
            cy = 1.0 / (dy * dy)


            diag = 0.0


            if i > 0:
                col[idx] = row - 1
                val[idx] = cx
                idx += 1
                diag -= cx
            elif boundary == "dirichlet":
                diag -= cx


            if i < nx - 1:
                col[idx] = row + 1
                val[idx] = cx
                idx += 1
                diag -= cx
            elif boundary == "dirichlet":
                diag -= cx


            if j > 0:
                col[idx] = row - nx
                val[idx] = cy
                idx += 1
                diag -= cy
            elif boundary == "dirichlet":
                diag -= cy


            if j < ny - 1:
                col[idx] = row + nx
                val[idx] = cy
                idx += 1
                diag -= cy
            elif boundary == "dirichlet":
                diag -= cy


            col[idx] = row
            val[idx] = diag
            idx += 1

            row_ptr[row] = start

    row_ptr[n] = idx
    nz = idx
    col = col[:nz]
    val = val[:nz]
    return n, nz, row_ptr, col, val


def build_sverdrup_matrix_crs(nx: int, ny: int, dx: float,
                              beta: float = 2.28e-11) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must be at least 2")
    if dx <= 0.0:
        raise ValueError("dx must be positive")

    n = nx * ny
    nz_max = 3 * n
    row_ptr = np.zeros(n + 1, dtype=int)
    col = np.zeros(nz_max, dtype=int)
    val = np.zeros(nz_max, dtype=float)

    idx = 0
    for j in range(ny):
        for i in range(nx):
            row = j * nx + i
            start = idx

            if i > 0:
                col[idx] = row - 1
                val[idx] = -beta / dx
                idx += 1

            col[idx] = row
            val[idx] = beta / dx
            idx += 1

            row_ptr[row] = start

    row_ptr[n] = idx
    nz = idx
    col = col[:nz]
    val = val[:nz]
    return n, nz, row_ptr, col, val


def crs_to_dense(n: int, row_ptr: np.ndarray, col: np.ndarray,
                 val: np.ndarray) -> np.ndarray:
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        for k in range(row_ptr[i], row_ptr[i + 1]):
            A[i, col[k]] += val[k]
    return A
