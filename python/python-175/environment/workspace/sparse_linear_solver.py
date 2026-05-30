
import numpy as np


class SparseMatrixCOO:

    def __init__(self, n_rows, n_cols):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.rows = []
        self.cols = []
        self.vals = []

    def add_entry(self, i, j, v):
        if not (0 <= i < self.n_rows and 0 <= j < self.n_cols):
            raise IndexError("Matrix index out of bounds")
        if abs(v) > 0.0:
            self.rows.append(int(i))
            self.cols.append(int(j))
            self.vals.append(float(v))

    def to_dense(self):
        A = np.zeros((self.n_rows, self.n_cols))
        for i, j, v in zip(self.rows, self.cols, self.vals):
            A[i, j] += v
        return A

    def mv(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape[0] != self.n_cols:
            raise ValueError("Dimension mismatch in matrix-vector product")
        y = np.zeros(self.n_rows, dtype=float)
        for i, j, v in zip(self.rows, self.cols, self.vals):
            y[i] += v * x[j]
        return y

    def mtv(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape[0] != self.n_rows:
            raise ValueError("Dimension mismatch in transpose matrix-vector product")
        y = np.zeros(self.n_cols, dtype=float)
        for i, j, v in zip(self.rows, self.cols, self.vals):
            y[j] += v * x[i]
        return y

    def residual_norm(self, x, b):
        return np.linalg.norm(b - self.mv(x))

    def nnz(self):
        return len(self.vals)


def sparse_from_dense(A, threshold=1e-15):
    A = np.asarray(A, dtype=float)
    n_rows, n_cols = A.shape
    S = SparseMatrixCOO(n_rows, n_cols)
    for i in range(n_rows):
        for j in range(n_cols):
            if abs(A[i, j]) > threshold:
                S.add_entry(i, j, A[i, j])
    return S


def conjugate_gradient_sparse(A_sparse, b, x0=None, max_iter=None, tol=1e-12, atol=1e-12):
    b = np.asarray(b, dtype=float)
    N = A_sparse.n_rows
    if max_iter is None:
        max_iter = N
    if x0 is None:
        x = np.zeros(N, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).copy()

    r = b - A_sparse.mv(x)
    p = r.copy()
    rs_old = np.dot(r, r)
    b_norm = np.linalg.norm(b)
    threshold = max(atol, tol * b_norm)

    for k in range(max_iter):
        Ap = A_sparse.mv(p)
        pAp = np.dot(p, Ap)
        if abs(pAp) < 1e-30:
            raise RuntimeError("CG breakdown: p^T A p is numerically zero")
        alpha = rs_old / pAp
        x += alpha * p
        r -= alpha * Ap
        rs_new = np.dot(r, r)
        if np.sqrt(rs_new) < threshold:
            return x, {'iterations': k + 1, 'residual_norm': np.sqrt(rs_new), 'converged': True}
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x, {'iterations': max_iter, 'residual_norm': np.sqrt(rs_old), 'converged': False}


def jacobi_preconditioned_cg(A_sparse, b, x0=None, max_iter=None, tol=1e-12, atol=1e-12):
    b = np.asarray(b, dtype=float)
    N = A_sparse.n_rows
    if max_iter is None:
        max_iter = N
    if x0 is None:
        x = np.zeros(N, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).copy()


    diag = np.zeros(N, dtype=float)
    for i, j, v in zip(A_sparse.rows, A_sparse.cols, A_sparse.vals):
        if i == j:
            diag[i] += v
    diag = np.where(np.abs(diag) > 1e-15, diag, 1.0)
    Minv = 1.0 / diag

    r = b - A_sparse.mv(x)
    z = Minv * r
    p = z.copy()
    rz_old = np.dot(r, z)
    b_norm = np.linalg.norm(b)
    threshold = max(atol, tol * b_norm)

    for k in range(max_iter):
        Ap = A_sparse.mv(p)
        pAp = np.dot(p, Ap)
        if abs(pAp) < 1e-30:
            raise RuntimeError("PCG breakdown")
        alpha = rz_old / pAp
        x += alpha * p
        r -= alpha * Ap
        z = Minv * r
        rz_new = np.dot(r, z)
        if np.linalg.norm(r) < threshold:
            return x, {'iterations': k + 1, 'residual_norm': np.linalg.norm(r), 'converged': True}
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    return x, {'iterations': max_iter, 'residual_norm': np.linalg.norm(r), 'converged': False}


def test_sparse_linear_solver():
    N = 50
    A = SparseMatrixCOO(N, N)
    for i in range(N):
        A.add_entry(i, i, 2.0)
        if i > 0:
            A.add_entry(i, i - 1, -1.0)
            A.add_entry(i - 1, i, -1.0)
    b = np.ones(N)
    x, info = conjugate_gradient_sparse(A, b, tol=1e-10)
    assert info['converged'], "CG did not converge"
    assert info['residual_norm'] < 1e-8

    x_dense = np.linalg.solve(A.to_dense(), b)
    assert np.allclose(x, x_dense, atol=1e-8)
    print("sparse_linear_solver: all self-tests passed")


if __name__ == "__main__":
    test_sparse_linear_solver()
