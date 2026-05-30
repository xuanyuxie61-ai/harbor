
import numpy as np


class SparseCOO:
    def __init__(self, rows, cols, vals, shape):
        self.rows = np.asarray(rows, dtype=int)
        self.cols = np.asarray(cols, dtype=int)
        self.vals = np.asarray(vals, dtype=float)
        self.shape = shape
        self.n = shape[0]
        self.m = shape[1]
        self.nnz = len(vals)

    def mv(self, x):
        x = np.asarray(x, dtype=float)
        y = np.zeros(self.n, dtype=float)
        for i in range(self.nnz):
            y[self.rows[i]] += self.vals[i] * x[self.cols[i]]
        return y

    def mtv(self, x):
        x = np.asarray(x, dtype=float)
        y = np.zeros(self.m, dtype=float)
        for i in range(self.nnz):
            y[self.cols[i]] += self.vals[i] * x[self.rows[i]]
        return y

    def to_dense(self):
        A = np.zeros(self.shape, dtype=float)
        for i in range(self.nnz):
            A[self.rows[i], self.cols[i]] += self.vals[i]
        return A

    def residual(self, x, b):
        return b - self.mv(x)


def conjugate_gradient(A_sparse, b, x0=None, tol=1e-10, max_iter=None):
    b = np.asarray(b, dtype=float)
    n = len(b)
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).copy()
    if max_iter is None:
        max_iter = n

    r = b - A_sparse.mv(x)
    p = r.copy()
    rs_old = np.dot(r, r)

    for k in range(max_iter):
        Ap = A_sparse.mv(p)
        pAp = np.dot(p, Ap)
        if abs(pAp) < 1e-30:
            break
        alpha = rs_old / pAp
        x += alpha * p
        r -= alpha * Ap
        rs_new = np.dot(r, r)
        if np.sqrt(rs_new) < tol * np.linalg.norm(b):
            break
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x


def jacobi_iteration(A_sparse, b, x0=None, tol=1e-10, max_iter=1000):
    b = np.asarray(b, dtype=float)
    n = len(b)
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).copy()


    diag = np.zeros(n, dtype=float)
    for i in range(A_sparse.nnz):
        if A_sparse.rows[i] == A_sparse.cols[i]:
            diag[A_sparse.rows[i]] += A_sparse.vals[i]
    diag = np.where(np.abs(diag) < 1e-14, 1.0, diag)

    for _ in range(max_iter):
        x_new = b.copy()
        for i in range(A_sparse.nnz):
            if A_sparse.rows[i] != A_sparse.cols[i]:
                x_new[A_sparse.rows[i]] -= A_sparse.vals[i] * x[A_sparse.cols[i]]
        x_new /= diag
        if np.linalg.norm(x_new - x) < tol:
            return x_new
        x = x_new
    return x


def r8po_fa(n, a):
    a = np.asarray(a, dtype=float).copy()
    r = np.zeros((n, n), dtype=float)
    for j in range(n):
        s = 0.0
        for k in range(j):
            s += r[k, j] ** 2

        if a[j, j] - s <= 0.0:
            raise ValueError(f"Matrix is not positive definite at row {j}")
        r[j, j] = np.sqrt(a[j, j] - s)
        for i in range(j + 1, n):
            s = 0.0
            for k in range(j):
                s += r[k, i] * r[k, j]
            r[j, i] = (a[j, i] - s) / r[j, j]
    return r


def r8po_sl(n, r, b):
    b = np.asarray(b, dtype=float).copy()
    x = b.copy()

    for j in range(n):
        x[j] /= r[j, j]
        for i in range(j + 1, n):
            x[i] -= r[j, i] * x[j]

    for j in range(n - 1, -1, -1):
        x[j] /= r[j, j]
        for i in range(j):
            x[i] -= r[j, i] * x[j]
    return x


def r8po_det(n, r):
    det_r = np.prod(np.diag(r))
    return det_r ** 2


def r8po_inverse(n, r):

    r_inv = np.zeros((n, n), dtype=float)
    for i in range(n):
        r_inv[i, i] = 1.0 / r[i, i]
        for j in range(i + 1, n):
            s = 0.0
            for k in range(i, j):
                s += r[k, j] * r_inv[i, k]
            r_inv[i, j] = -s / r[j, j]

    return r_inv @ r_inv.T


def assemble_sparse_from_triplets(rows, cols, vals, n):

    accum = {}
    for i, j, v in zip(rows, cols, vals):
        key = (int(i), int(j))
        accum[key] = accum.get(key, 0.0) + v
    rows_out = np.array([k[0] for k in accum.keys()], dtype=int)
    cols_out = np.array([k[1] for k in accum.keys()], dtype=int)
    vals_out = np.array(list(accum.values()), dtype=float)
    return SparseCOO(rows_out, cols_out, vals_out, (n, n))
