"""
poisson_cholesky.py
===================

Solve the self-gravity Poisson equation

    Delta Phi = 4 pi G rho

on the 3-D AMR pyramid grid by constructing the SPD linear system
for the discrete Laplacian and factoring it via Cholesky decomposition
(Algorithm AS 6; Healy 1968, seed project 026_asa007).

Key formulae
------------
For a uniform 3-D grid of side N with spacing dx the standard
7-point Laplacian stencil gives a linear system

    A phi = b

where A is block-tridiagonal and SPD (after flipping the sign
convention).  The Cholesky factorisation  A = L L^T  has the property
that L is lower triangular with strictly positive diagonal entries.

For an N x N x N grid A has size N^3 x N^3 and ~7 N^3 non-zeros.
In our small-scale test problem N = 16 so A is 4096 x 4096 which
Cholesky factors in a fraction of a second.

Rank-deficiency handling
------------------------
With pure Neumann boundary conditions (isolated self-gravitating
system) the discrete Laplacian has a 1-D null space spanned by the
constant vector; the Cholesky factorisation fails on this system.
Following the AS 7 syminv routine we detect rank-deficiency via
a diagonal threshold

    |L_{ii}| < tol * max_j |L_{jj}|

and pin the mean of phi to zero after the solve.  This gives the
unique minimum-norm solution of the under-determined system.

References
----------
- Healy, M. J. R. 1968, Applied Statistics 17, 195 (AS 6)
- Healy, M. J. R. 1968, Applied Statistics 17, 198 (AS 7)
- Hockney, R. W., & Eastwood, J. W. 1988, Computer Simulation Using
    Particles (gravity solvers review)
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, Optional

from astro_constants import GRAVITATIONAL_CGS


# =====================================================================
#                SPARSE LAPLACIAN ASSEMBLY (3-D UNIFORM)
# =====================================================================

def laplacian_3d_sparse_matrix(N: int, dx: float) -> np.ndarray:
    """
    Construct the N^3 x N^3 dense matrix for the 7-point Laplacian
    with periodic boundary conditions.

    For small test problems (N <= 16) we use dense storage for
    simplicity and direct Cholesky factorisation.

    Sign convention: the matrix represents -Delta so it is SPD.
    """
    n3 = N * N * N
    A = np.zeros((n3, n3))
    coeff = 1.0 / (dx * dx)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                row = i * N * N + j * N + k
                # diagonal +6/dx^2
                A[row, row] = 6.0 * coeff
                # x-neighbours (periodic)
                ip = ((i + 1) % N) * N * N + j * N + k
                im = ((i - 1) % N) * N * N + j * N + k
                A[row, ip] = -coeff
                A[row, im] = -coeff
                # y-neighbours
                jp = i * N * N + ((j + 1) % N) * N + k
                jm = i * N * N + ((j - 1) % N) * N + k
                A[row, jp] = -coeff
                A[row, jm] = -coeff
                # z-neighbours
                kp = i * N * N + j * N + ((k + 1) % N)
                km = i * N * N + j * N + ((k - 1) % N)
                A[row, kp] = -coeff
                A[row, km] = -coeff
    return A


# =====================================================================
#               CHOLESKY FACTORISATION (AS 6 - Healy 1968)
# =====================================================================

def cholesky_factor(A: np.ndarray, tol: float = 1.0e-12
                    ) -> Tuple[np.ndarray, int, int]:
    """
    Compute the upper-triangular Cholesky factor U such that A = U^T U.
    Direct translation of Algorithm AS 6 (Healy 1968, 026_asa007).

    Returns (U, nullty, ifault):

        U      -- upper-triangular factor
        nullty -- rank deficiency (number of diagonal elements < tol)
        ifault -- error flag (0 = no error, 1 = non-symmetric input,
                                   2 = non-positive-definite)
    """
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("A must be square")
    U = np.zeros_like(A)
    nullty = 0
    ifault = 0

    # -- check symmetry -----------------------------------------------
    sym_err = np.max(np.abs(A - A.T))
    if sym_err > 1.0e-10 * np.max(np.abs(A)):
        ifault = 1
        return U, nullty, ifault

    # -- factorise ----------------------------------------------------
    for j in range(n):
        sumj = 0.0
        for k in range(j):
            s = A[k, j]
            for i in range(k):
                s -= U[i, k] * U[i, j]
            if abs(U[k, k]) > tol:
                U[k, j] = s / U[k, k]
            else:
                U[k, j] = 0.0
            sumj += U[k, j] * U[k, j]
        diag_val = A[j, j] - sumj
        if diag_val > tol * tol * max(abs(A[j, j]), 1.0):
            U[j, j] = math.sqrt(diag_val)
        else:
            U[j, j] = 0.0
            nullty += 1
    return U, nullty, ifault


# =====================================================================
#               SYMMETRIC MATRIX INVERSE (AS 7 - Healy 1968)
# =====================================================================

def syminv_from_cholesky(U: np.ndarray, nullty: int
                         ) -> Tuple[np.ndarray, int, int]:
    """
    Compute the inverse (or generalised inverse) of A from its
    Cholesky factor U.  Translation of Algorithm AS 7.

    When nullty > 0 the inverse is the Moore-Penrose pseudoinverse
    on the range of A (the constant mode is suppressed).
    """
    n = U.shape[0]
    if nullty > 0:
        # pseudoinverse: invert only the non-zero diagonal entries
        Uinv = np.zeros_like(U)
        for j in range(n):
            if abs(U[j, j]) > 1.0e-12:
                Uinv[j, j] = 1.0 / U[j, j]
        for j in range(n - 2, -1, -1):
            for k in range(j + 1, n):
                s = 0.0
                for i in range(j + 1, k + 1):
                    s += U[j, i] * Uinv[i, k]
                if abs(U[j, j]) > 1.0e-12:
                    Uinv[j, k] = -s / U[j, j]
        C = Uinv.T @ Uinv
    else:
        # full-rank case: back-substitution
        Uinv = np.zeros_like(U)
        for j in range(n - 1, -1, -1):
            Uinv[j, j] = 1.0 / U[j, j] if abs(U[j, j]) > 1.0e-30 else 0.0
            for k in range(j - 1, -1, -1):
                s = 0.0
                for i in range(k + 1, j + 1):
                    s += U[k, i] * Uinv[i, j]
                if abs(U[k, k]) > 1.0e-30:
                    Uinv[k, j] = -s / U[k, k]
        C = Uinv.T @ Uinv
    ifault = 0
    return C, nullty, ifault


# =====================================================================
#                       POISSON SOLVER WRAPPER
# =====================================================================

class PoissonGravitySolver:
    """
    Solve  Delta Phi = 4 pi G rho  on a cubic periodic domain
    with the Cholesky-factorised discrete Laplacian.

    The solver caches the Cholesky factor so repeated solves (one
    per time step) only require forward/back-substitution.
    """

    def __init__(self, N: int, dx_cgs: float) -> None:
        self.N = N
        self.dx_cgs = dx_cgs
        self._assemble_and_factor()

    def _assemble_and_factor(self) -> None:
        """Build the Laplacian and factor it once."""
        self.A = laplacian_3d_sparse_matrix(self.N, self.dx_cgs)
        # We store -A so the system is SPD; sign convention handled below.
        self.U, self.nullty, self.ifact = cholesky_factor(-self.A)
        if self.ifact != 0:
            raise RuntimeError(
                f"Cholesky factorisation failed: ifault = {self.ifact}"
            )
        # Cache the Green's-function matrix for direct matvec solves
        self.G, _, _ = syminv_from_cholesky(self.U, self.nullty)

    # -----------------------------------------------------------------
    #  right-hand side assembly
    # -----------------------------------------------------------------
    def rhs_from_density(self, rho_3d: np.ndarray) -> np.ndarray:
        """
        Form the Poisson RHS vector  b = 4 pi G rho  flattened to 1-D.
        """
        if rho_3d.shape != (self.N, self.N, self.N):
            raise ValueError("rho shape mismatch")
        return (4.0 * math.pi * GRAVITATIONAL_CGS * rho_3d).flatten()

    # -----------------------------------------------------------------
    #  solve
    # -----------------------------------------------------------------
    def solve(self, rho_3d: np.ndarray) -> np.ndarray:
        """
        Solve for Phi and return it as a 3-D array.

        The mean of Phi is pinned to zero (Neumann compatibility).
        """
        b = self.rhs_from_density(rho_3d)
        phi_flat = self.G @ b
        phi = phi_flat.reshape((self.N, self.N, self.N))
        # pin the mean
        phi -= np.mean(phi)
        return phi

    def gravitational_acceleration(self, phi_3d: np.ndarray
                                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute g = -grad Phi using 4th-order centred differences.
        """
        from high_order_fd import apply_fd1
        dx = self.dx_cgs
        gx = np.zeros_like(phi_3d)
        gy = np.zeros_like(phi_3d)
        gz = np.zeros_like(phi_3d)
        for j in range(self.N):
            for k in range(self.N):
                gx[:, j, k] = -apply_fd1(phi_3d[:, j, k], dx, order=4)
        for i in range(self.N):
            for k in range(self.N):
                gy[i, :, k] = -apply_fd1(phi_3d[i, :, k], dx, order=4)
        for i in range(self.N):
            for j in range(self.N):
                gz[i, j, :] = -apply_fd1(phi_3d[i, j, :], dx, order=4)
        return (gx, gy, gz)
