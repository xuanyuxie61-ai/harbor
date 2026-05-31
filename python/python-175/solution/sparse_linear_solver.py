"""
sparse_linear_solver.py
=======================
Sparse matrix storage and iterative solvers for large linear systems arising
from discretized stochastic PDE Galerkin projections.

Fused from seed project:
- 998_r8st : Sparse Triplet (COO) format storage, sparse matrix-vector product,
             conjugate gradient (CG) solver

Mathematical foundation
-----------------------
Given a symmetric positive-definite (SPD) matrix A \in \mathbb{R}^{N\times N},
the linear system A x = b is solved by the Conjugate Gradient method.

CG Algorithm (Hestenes & Stiefel, 1952):
    r_0 = b - A x_0,   p_0 = r_0
    for k = 0, 1, 2, ...
        alpha_k = (r_k^T r_k) / (p_k^T A p_k)
        x_{k+1} = x_k + alpha_k p_k
        r_{k+1} = r_k - alpha_k A p_k
        beta_k  = (r_{k+1}^T r_{k+1}) / (r_k^T r_k)
        p_{k+1} = r_{k+1} + beta_k p_k

Convergence bound:
    ||e_k||_A / ||e_0||_A \le 2 ( (\sqrt{\kappa}-1) / (\sqrt{\kappa}+1) )^k
where \kappa = cond_2(A) is the spectral condition number.

Sparse storage (COO / triplet):
    A is stored as three arrays: row indices I, column indices J, values V.
    Only non-zero elements are stored.  Matrix-vector product runs in O(nnz).
"""

import numpy as np


class SparseMatrixCOO:
    """Sparse matrix in Coordinate (COO) format."""

    def __init__(self, n_rows, n_cols):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.rows = []
        self.cols = []
        self.vals = []

    def add_entry(self, i, j, v):
        """Add a non-zero entry A[i,j] = v."""
        if not (0 <= i < self.n_rows and 0 <= j < self.n_cols):
            raise IndexError("Matrix index out of bounds")
        if abs(v) > 0.0:
            self.rows.append(int(i))
            self.cols.append(int(j))
            self.vals.append(float(v))

    def to_dense(self):
        """Convert to dense numpy array (for small matrices only)."""
        A = np.zeros((self.n_rows, self.n_cols))
        for i, j, v in zip(self.rows, self.cols, self.vals):
            A[i, j] += v
        return A

    def mv(self, x):
        """Sparse matrix-vector product y = A @ x."""
        x = np.asarray(x, dtype=float)
        if x.shape[0] != self.n_cols:
            raise ValueError("Dimension mismatch in matrix-vector product")
        y = np.zeros(self.n_rows, dtype=float)
        for i, j, v in zip(self.rows, self.cols, self.vals):
            y[i] += v * x[j]
        return y

    def mtv(self, x):
        """Sparse transpose matrix-vector product y = A^T @ x."""
        x = np.asarray(x, dtype=float)
        if x.shape[0] != self.n_rows:
            raise ValueError("Dimension mismatch in transpose matrix-vector product")
        y = np.zeros(self.n_cols, dtype=float)
        for i, j, v in zip(self.rows, self.cols, self.vals):
            y[j] += v * x[i]
        return y

    def residual_norm(self, x, b):
        """Compute ||b - A x||_2."""
        return np.linalg.norm(b - self.mv(x))

    def nnz(self):
        return len(self.vals)


def sparse_from_dense(A, threshold=1e-15):
    """Create a SparseMatrixCOO from a dense numpy array."""
    A = np.asarray(A, dtype=float)
    n_rows, n_cols = A.shape
    S = SparseMatrixCOO(n_rows, n_cols)
    for i in range(n_rows):
        for j in range(n_cols):
            if abs(A[i, j]) > threshold:
                S.add_entry(i, j, A[i, j])
    return S


def conjugate_gradient_sparse(A_sparse, b, x0=None, max_iter=None, tol=1e-12, atol=1e-12):
    """
    Solve A x = b using the Conjugate Gradient method with a sparse matrix.

    Parameters
    ----------
    A_sparse : SparseMatrixCOO
        Symmetric positive-definite sparse matrix.
    b : ndarray
        Right-hand side vector.
    x0 : ndarray, optional
        Initial guess.
    max_iter : int, optional
        Maximum iterations (default N).
    tol : float
        Relative tolerance on residual.
    atol : float
        Absolute tolerance on residual.

    Returns
    -------
    x : ndarray
        Approximate solution.
    info : dict
        Contains 'iterations', 'residual_norm', 'converged'.
    """
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
    """
    Jacobi-preconditioned CG: M = diag(A)^{-1}.
    Algorithm:
        r_0 = b - A x_0,  z_0 = M^{-1} r_0,  p_0 = z_0
        alpha_k = (r_k^T z_k) / (p_k^T A p_k)
        x_{k+1} = x_k + alpha_k p_k
        r_{k+1} = r_k - alpha_k A p_k
        z_{k+1} = M^{-1} r_{k+1}
        beta_k  = (r_{k+1}^T z_{k+1}) / (r_k^T z_k)
        p_{k+1} = z_{k+1} + beta_k p_k
    """
    b = np.asarray(b, dtype=float)
    N = A_sparse.n_rows
    if max_iter is None:
        max_iter = N
    if x0 is None:
        x = np.zeros(N, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).copy()

    # Extract diagonal
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
    """Self-test with a small SPD tridiagonal system."""
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
    # Compare with dense solve
    x_dense = np.linalg.solve(A.to_dense(), b)
    assert np.allclose(x, x_dense, atol=1e-8)
    print("sparse_linear_solver: all self-tests passed")


if __name__ == "__main__":
    test_sparse_linear_solver()
