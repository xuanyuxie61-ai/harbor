
import numpy as np


class LinearSolverError(Exception):
    pass


def plu_decomposition(A):
    A = np.array(A, dtype=float)
    m, n = A.shape
    L = np.eye(m)
    P = np.eye(m)
    U = A.copy()

    for j in range(min(m - 1, n)):
        pivot_value = 0.0
        pivot_row = -1
        for i in range(j, m):
            if abs(U[i, j]) > pivot_value:
                pivot_value = abs(U[i, j])
                pivot_row = i

        if pivot_row == -1 or pivot_value < 1e-15:
            continue


        U[[j, pivot_row], :] = U[[pivot_row, j], :]
        if j > 0:
            L[[j, pivot_row], :j] = L[[pivot_row, j], :j]
        P[[j, pivot_row], :] = P[[pivot_row, j], :]

        for i in range(j + 1, m):
            if abs(U[i, j]) > 1e-15:
                L[i, j] = U[i, j] / U[j, j]
                U[i, j] = 0.0
                U[i, j + 1:n] -= L[i, j] * U[j, j + 1:n]

    return P, L, U


def solve_plu(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    m, n = A.shape
    if m != n:
        raise LinearSolverError("PLU solve requires square matrix")
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    if b.shape[0] != m:
        raise LinearSolverError("Dimension mismatch in PLU solve")

    P, L, U = plu_decomposition(A)

    y = forward_substitution(L, P @ b)

    x = backward_substitution(U, y)
    return x.flatten()


def forward_substitution(L, b):
    L = np.array(L, dtype=float)
    b = np.array(b, dtype=float)
    n = L.shape[0]
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    x = b.copy()

    for j in range(n):
        if abs(L[j, j]) < 1e-15:
            raise LinearSolverError("Zero diagonal in forward substitution")
        x[j] = x[j] / L[j, j]
        if j + 1 < n:
            x[j + 1:n] -= L[j + 1:n, j].reshape(-1, 1) * x[j]
    return x


def backward_substitution(U, b):
    U = np.array(U, dtype=float)
    b = np.array(b, dtype=float)
    n = U.shape[0]
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    x = b.copy()

    for j in range(n - 1, -1, -1):
        if abs(U[j, j]) < 1e-15:
            raise LinearSolverError("Zero diagonal in backward substitution")
        x[j] = x[j] / U[j, j]
        if j > 0:
            x[0:j] -= U[0:j, j].reshape(-1, 1) * x[j]
    return x


def conjugate_gradient(A, b, x0=None, tol=1e-10, max_iter=None):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = A.shape[0]
    if max_iter is None:
        max_iter = n

    if x0 is None:
        x = np.zeros(n)
    else:
        x = np.array(x0, dtype=float)


    if A.shape[0] != A.shape[1]:
        raise LinearSolverError("CG requires square matrix")
    if b.shape[0] != n:
        raise LinearSolverError("Dimension mismatch in CG")


    diag_min = np.min(np.diag(A))
    if diag_min <= 0:

        A = 0.5 * (A + A.T)

    ap = A @ x
    r = b - ap
    p = r.copy()

    for it in range(max_iter):
        ap = A @ p
        pap = float(np.dot(p, ap))
        pr = float(np.dot(p, r))

        if abs(pap) < 1e-15:
            break

        alpha = pr / pap
        x = x + alpha * p
        r = r - alpha * ap


        if np.linalg.norm(r) < tol:
            break

        rap = float(np.dot(r, ap))
        beta = -rap / pap
        p = r + beta * p

    return x


def matrix_vector_product(A, x):
    A = np.array(A, dtype=float)
    x = np.array(x, dtype=float)
    return A @ x


def matrix_matrix_product(A, B):
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    return A @ B
