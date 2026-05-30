
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

def cholesky_decompose(A):
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square.")
    L = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        s = A[j, j] - np.sum(L[j, :j]**2)
        if s <= 1e-14:

            s = 1e-14
        L[j, j] = np.sqrt(s)
        for i in range(j + 1, n):
            L[i, j] = (A[i, j] - np.sum(L[i, :j] * L[j, :j])) / L[j, j]
    return L

def cholesky_solve(L, b):
    n = L.shape[0]
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        y[i] = (b[i] - np.sum(L[i, :i] * y[:i])) / L[i, i]
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.sum(L[i + 1:, i] * x[i + 1:])) / L[i, i]
    return x


def build_helmholtz_matrix(Nx, Ny, dx, dy, alpha, beta_coeff):
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

            data.append(diag)
            row_ind.append(r)
            col_ind.append(r)

            for di, dj, coeff in [(-1, 0, cx), (1, 0, cx), (0, -1, cy), (0, 1, cy)]:
                c = idx(i + di, j + dj)
                data.append(coeff)
                row_ind.append(r)
                col_ind.append(c)

    A = csr_matrix((data, (row_ind, col_ind)), shape=(n, n))
    return A


class ImplicitHelmholtzSolver:

    def __init__(self, Nx, Ny, dx, dy, dt, nu, r_drag):
        self.Nx, self.Ny = Nx, Ny




        alpha = 1.0 + dt * r_drag
        beta_coeff = -dt * nu
        self.A = build_helmholtz_matrix(Nx, Ny, dx, dy, alpha, beta_coeff)


        self._factorize()

    def _factorize(self):

        from scipy.sparse.linalg import splu
        try:
            self.splu = splu(self.A.tocsc())
        except Exception as e:

            Adense = self.A.toarray()
            self.L_dense = cholesky_decompose(Adense)
            self.splu = None

    def solve(self, rhs):
        b = rhs.ravel()
        if self.splu is not None:
            x = self.splu.solve(b)
        else:
            x = cholesky_solve(self.L_dense, b)
        return x.reshape((self.Nx, self.Ny))
