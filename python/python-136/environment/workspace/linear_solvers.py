
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg as scipy_cg


class LinearSolverError(Exception):
    pass


def solve_tridiagonal(a, b, c, rhs):
    n = a.size
    if b.size != n - 1 or c.size != n - 1 or rhs.size != n:
        raise LinearSolverError("三对角矩阵维度不匹配")

    a = a.astype(float, copy=True)
    b = b.astype(float, copy=True)
    c = c.astype(float, copy=True)
    rhs = rhs.astype(float, copy=True)
    u = np.zeros(n, dtype=float)


    for i in range(1, n):
        w = b[i - 1] / a[i - 1]
        a[i] -= w * c[i - 1]
        rhs[i] -= w * rhs[i - 1]


    u[-1] = rhs[-1] / a[-1]
    for i in range(n - 2, -1, -1):
        u[i] = (rhs[i] - c[i] * u[i + 1]) / a[i]

    return u


def conjugate_gradient_rc(n, b_vec, matvec, precon_solve,
                          max_iter=None, tol=1e-10):
    if max_iter is None:
        max_iter = n

    b_vec = np.asarray(b_vec, dtype=float)
    x = np.zeros(n, dtype=float)
    r = b_vec.copy()
    z = precon_solve(r)
    p = z.copy()

    rho_old = None
    for k in range(max_iter):
        q = matvec(p)
        pdotq = np.dot(p, q)
        if abs(pdotq) < np.finfo(float).eps:
            raise LinearSolverError("p^T A p 接近零，矩阵可能不正定")

        rho = np.dot(r, z)
        alpha = rho / pdotq
        x += alpha * p
        r -= alpha * q

        resid_norm = np.linalg.norm(r)
        if resid_norm < tol:
            return x, {"iter": k + 1, "resid": resid_norm}

        z = precon_solve(r)
        rho_new = np.dot(r, z)
        if rho_old is not None:
            beta = rho_new / rho_old
            p = z + beta * p
        else:
            p = z.copy()
        rho_old = rho_new

    return x, {"iter": max_iter, "resid": resid_norm}


def solve_sparse_system(A, b, tol=1e-10, max_iter=None):
    A = csr_matrix(A)
    n = A.shape[0]
    if max_iter is None:
        max_iter = n * 2

    x, info = scipy_cg(A, b, rtol=tol, maxiter=max_iter)
    if info < 0:
        raise LinearSolverError("CG 非法输入")
    if info > 0:

        pass
    return x


def jacobi_preconditioner(A):
    diag = np.diag(A)
    if np.any(np.abs(diag) < np.finfo(float).eps):
        raise LinearSolverError("矩阵对角线包含零元，无法使用 Jacobi 预处理")
    inv_diag = 1.0 / diag

    def precon_solve(r):
        return inv_diag * r

    return precon_solve
