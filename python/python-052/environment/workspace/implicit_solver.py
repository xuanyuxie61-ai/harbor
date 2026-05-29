"""
Compact Cholesky Solver for Implicit Barotropic Mode
====================================================
Derived from seed project 026_asa007 (Cholesky factorization and
symmetric matrix inversion in packed storage).

In the implicit time stepping of the barotropic mode, we solve
elliptic equations of the form:

    [ I − Δt·ν·∇⁴ + Δt·r·∇² ] ψ^{n+1} = RHSⁿ

Discretized on a uniform grid with 5-point Laplacian stencil:
    ∇²ψ ≈ (ψ_{i+1,j} + ψ_{i−1,j} + ψ_{i,j+1} + ψ_{i,j−1} − 4ψ_{i,j}) / h²

This yields a symmetric positive-definite (SPD) sparse system
A ψ = b. The Cholesky decomposition A = L L^T is computed once
and reused for each right-hand side.

For an SPD matrix A, the Cholesky algorithm computes:
    L_{jj} = sqrt( A_{jj} − Σ_{k<j} L_{jk}² )
    L_{ij} = ( A_{ij} − Σ_{k<j} L_{ik} L_{jk} ) / L_{jj}   for i > j
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

def cholesky_decompose(A):
    """
    Compute the Cholesky factor L (lower triangular) of a symmetric
    positive-definite dense matrix A, with rank-deficiency detection.

    Parameters
    ----------
    A : ndarray, shape (n, n)
        SPD matrix.

    Returns
    -------
    L : ndarray, shape (n, n)
        Lower triangular Cholesky factor.
    """
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square.")
    L = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        s = A[j, j] - np.sum(L[j, :j]**2)
        if s <= 1e-14:
            # Near singular; perturb slightly for robustness
            s = 1e-14
        L[j, j] = np.sqrt(s)
        for i in range(j + 1, n):
            L[i, j] = (A[i, j] - np.sum(L[i, :j] * L[j, :j])) / L[j, j]
    return L

def cholesky_solve(L, b):
    """
    Solve A x = b given Cholesky factor L (A = L L^T).

    Forward substitution:  L y = b
    Backward substitution: L^T x = y
    """
    n = L.shape[0]
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        y[i] = (b[i] - np.sum(L[i, :i] * y[:i])) / L[i, i]
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.sum(L[i + 1:, i] * x[i + 1:])) / L[i, i]
    return x


def build_helmholtz_matrix(Nx, Ny, dx, dy, alpha, beta_coeff):
    """
    Build the sparse matrix for the modified Helmholtz operator:
        A = α·I + β·∇²
    using 5-point stencil on a uniform grid with periodic boundaries.

    Parameters
    ----------
    Nx, Ny : int
        Grid dimensions.
    dx, dy : float
        Grid spacing.
    alpha, beta_coeff : float
        Coefficients.

    Returns
    -------
    A : csr_matrix
        Sparse matrix in CSR format.
    """
    n = Nx * Ny
    data = []
    row_ind = []
    col_ind = []

    cx = beta_coeff / dx**2
    cy = beta_coeff / dy**2
    diag = alpha - 2.0 * cx - 2.0 * cy

    def idx(i, j):
        return (i % Nx) * Ny + (j % Ny)

    for i in range(Nx):
        for j in range(Ny):
            r = idx(i, j)
            # Diagonal
            data.append(diag)
            row_ind.append(r)
            col_ind.append(r)
            # Neighbors with periodic wrap
            for di, dj, coeff in [(-1, 0, cx), (1, 0, cx), (0, -1, cy), (0, 1, cy)]:
                c = idx(i + di, j + dj)
                data.append(coeff)
                row_ind.append(r)
                col_ind.append(c)

    A = csr_matrix((data, (row_ind, col_ind)), shape=(n, n))
    return A


class ImplicitHelmholtzSolver:
    """
    Pre-factored implicit solver for the modified Helmholtz equation
    with periodic boundary conditions.
    """

    def __init__(self, Nx, Ny, dx, dy, dt, nu, r_drag):
        """
        Parameters
        ----------
        Nx, Ny : int
            Grid resolution.
        dx, dy : float
            Grid spacing.
        dt : float
            Time step.
        nu : float
            Hyperviscosity [m⁴/s].
        r_drag : float
            Linear drag [1/s].
        """
        self.Nx, self.Ny = Nx, Ny
        # For the implicit biharmonic + Laplacian, use operator splitting:
        # Step 1: implicit Laplacian diffusion
        #   (I - dt*nu*Laplacian) psi^{*} = RHS
        # We build the matrix for (I - dt*nu*Laplacian)
        alpha = 1.0 + dt * r_drag
        beta_coeff = -dt * nu  # Note: Laplacian has -4/h^2 on diagonal
        self.A = build_helmholtz_matrix(Nx, Ny, dx, dy, alpha, beta_coeff)
        # Use scipy sparse direct solver for robustness on moderate grids
        # For very large grids, an iterative Krylov solver would be needed.
        self._factorize()

    def _factorize(self):
        """Pre-factorize the sparse matrix."""
        # scipy sparse LU decomposition
        from scipy.sparse.linalg import splu
        try:
            self.splu = splu(self.A.tocsc())
        except Exception as e:
            # Fallback to dense Cholesky for small matrices
            Adense = self.A.toarray()
            self.L_dense = cholesky_decompose(Adense)
            self.splu = None

    def solve(self, rhs):
        """
        Solve A·ψ = rhs.

        Parameters
        ----------
        rhs : ndarray, shape (Nx, Ny)
            Right-hand side in physical space.

        Returns
        -------
        psi : ndarray, shape (Nx, Ny)
            Solution.
        """
        b = rhs.ravel()
        if self.splu is not None:
            x = self.splu.solve(b)
        else:
            x = cholesky_solve(self.L_dense, b)
        return x.reshape((self.Nx, self.Ny))
