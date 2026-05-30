
import numpy as np
from utils_numerical import safe_divide


def condition_hager(n: int, a: np.ndarray) -> float:
    if n <= 0:
        return 1.0

    i1 = -1
    c1 = 0.0
    b = np.ones((n, 1)) / n

    max_iter = 10
    it = 0

    while it < max_iter:
        it += 1
        try:
            b = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:

            a_reg = a + 1e-10 * np.eye(n)
            b = np.linalg.solve(a_reg, b)

        c2 = np.sum(np.abs(b))
        b = np.sign(b)


        b[b == 0.0] = 1.0

        try:
            b = np.linalg.solve(a.T, b)
        except np.linalg.LinAlgError:
            a_reg = a.T + 1e-10 * np.eye(n)
            b = np.linalg.solve(a_reg, b)

        i2 = int(np.argmax(np.abs(b)))

        if i1 >= 0:
            if i1 == i2 or c2 <= c1:
                break

        i1 = i2
        c1 = c2
        b = np.zeros((n, 1))
        b[i1] = 1.0

    norm_a = np.linalg.norm(a, 1)
    cond = float(c2 * norm_a)


    if cond < 1.0:
        cond = 1.0
    if not np.isfinite(cond):
        cond = 1e16

    return cond


def lu_decomposition_with_pivot(a: np.ndarray) -> tuple:
    n = a.shape[0]
    if a.shape[0] != a.shape[1]:
        return None, None, None, False

    a_copy = a.astype(float).copy()
    L = np.eye(n)
    U = np.zeros((n, n))
    P = np.eye(n)

    for k in range(n):

        max_idx = np.argmax(np.abs(a_copy[k:, k])) + k
        if abs(a_copy[max_idx, k]) < 1e-15:

            a_copy[k, k] += 1e-12
            max_idx = k

        if max_idx != k:
            a_copy[[k, max_idx], :] = a_copy[[max_idx, k], :]
            P[[k, max_idx], :] = P[[max_idx, k], :]

        for i in range(k + 1, n):
            a_copy[i, k] /= a_copy[k, k]
            for j in range(k + 1, n):
                a_copy[i, j] -= a_copy[i, k] * a_copy[k, j]


    for i in range(n):
        for j in range(i + 1):
            if i == j:
                L[i, j] = 1.0
                U[j, i] = a_copy[j, i]
            elif j < i:
                L[i, j] = a_copy[i, j]
                U[j, i] = a_copy[j, i]
            else:
                U[j, i] = a_copy[j, i]

    return L, U, P, True


def solve_lu(L: np.ndarray, U: np.ndarray, P: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = L.shape[0]
    pb = P @ b


    y = np.zeros(n)
    for i in range(n):
        y[i] = pb[i] - np.dot(L[i, :i], y[:i])


    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x


def elementary_row_scale(A: np.ndarray, row: int, s: float) -> np.ndarray:
    if abs(s) < 1e-15:
        raise ValueError("Scale factor must be non-zero")
    if row < 0 or row >= A.shape[0]:
        raise ValueError("Row index out of bounds")
    A = A.copy()
    A[row, :] *= s
    return A


def elementary_row_swap(A: np.ndarray, row1: int, row2: int) -> np.ndarray:
    if row1 == row2:
        return A.copy()
    A = A.copy()
    A[[row1, row2], :] = A[[row2, row1], :]
    return A


def elementary_row_axpy(A: np.ndarray, target: int, source: int, s: float) -> np.ndarray:
    if target == source:
        raise ValueError("Target and source rows must differ")
    A = A.copy()
    A[target, :] += s * A[source, :]
    return A


def jacobi_preconditioner(A: np.ndarray) -> np.ndarray:
    diag = np.diag(A).copy()
    diag[np.abs(diag) < 1e-14] = 1e-14
    M_inv = np.diag(1.0 / diag)
    return M_inv


def apply_jacobi_precond(A: np.ndarray, b: np.ndarray, max_iter: int = 50, tol: float = 1e-10) -> np.ndarray:
    n = len(b)
    x = np.zeros(n)
    diag = np.diag(A).copy()
    diag[np.abs(diag) < 1e-14] = 1e-14

    for _ in range(max_iter):
        x_new = (b - (A @ x - diag * x)) / diag
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new

    return x
