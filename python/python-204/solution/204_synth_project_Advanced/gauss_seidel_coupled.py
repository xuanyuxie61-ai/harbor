"""
gauss_seidel_coupled.py
=======================

Gauss-Seidel iterative solver for the coupled linear systems that
arise inside each time step of the Sobol forward model.

In the disk-shaped geophysical reactor the discretised advection-
diffusion-reaction equation for the species concentration ``c`` on a
structured polar grid ``(r_i, theta_j)`` reads

    - D * Delta_h c + v . nabla_h c + k_r * c = f

where ``Delta_h`` is the 5-point polar Laplacian, ``v`` is the
effective advective velocity (supplied by ``chaotic_mixing``), and
``k_r`` is the local kinetic rate.  After finite-difference
discretisation this becomes a **block tridiagonal** linear system

    A c = b

whose blocks couple adjacent radial shells.  Because ``A`` is large
but highly structured, a Gauss-Seidel sweep with red-black ordering
converges in ``O(N)`` iterations for the moderate grid sizes used
here (typically ``N_r * N_theta <= 10^4``).

This module provides:

* ``gs_step`` — one Gauss-Seidel sweep (ported from Burkardt's
  ``gauss_seidel1``).
* ``gs_solve`` — iterate to a user-specified tolerance.
* ``build_polar_laplacian`` — assemble ``A`` for the 5-point polar
  stencil.
* ``spectral_radius`` — estimate the spectral radius of the iteration
  matrix, used to predict convergence speed.
* ``coupled_block_solve`` — outer iteration for the *block* system
  that couples the two species (A, B).

References
----------
* D. M. Young, *Iterative Solution of Large Linear Systems*,
  Academic Press, 1971.
* J. Burkardt, ``gauss_seidel`` MATLAB library.
"""

from __future__ import annotations

import math

import numpy as np


# =====================================================================
# One Gauss-Seidel sweep (red-black, row-by-row)
# =====================================================================
def gs_step(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    """One forward Gauss-Seidel sweep.

    Parameters
    ----------
    A : (n, n) ndarray
        Non-singular matrix with non-zero diagonal.
    b : (n,) ndarray
        Right-hand side.
    x : (n,) ndarray
        Current iterate.

    Returns
    -------
    x_new : (n,) ndarray
        Updated iterate.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("gs_step: A must be square")
    n = A.shape[0]
    if b.shape != (n,) or x.shape != (n,):
        raise ValueError("gs_step: b, x shape mismatch")
    x_new = x.copy()
    for i in range(n):
        s = b[i]
        s -= A[i, :i] @ x_new[:i]
        s -= A[i, i + 1:] @ x[i + 1:]
        diag = A[i, i]
        if abs(diag) < 1.0e-30:
            raise ValueError(f"gs_step: zero diagonal at row {i}")
        x_new[i] = s / diag
    return x_new


# =====================================================================
# Iterative Gauss-Seidel solve
# =====================================================================
def gs_solve(A: np.ndarray, b: np.ndarray, x0: np.ndarray | None = None,
             tol: float = 1.0e-8, max_iter: int = 2000,
             omega: float = 1.0) -> tuple[np.ndarray, dict]:
    """Solve ``A x = b`` by (successive-over-relaxation) Gauss-Seidel.

    Parameters
    ----------
    A, b : array
        System matrix and right-hand side.
    x0 : array, optional
        Initial guess; default is the zero vector.
    tol : float
        Residual tolerance ``|| A x - b ||_2 / ||b||_2``.
    max_iter : int
        Maximum iteration count.
    omega : float
        Over-relaxation parameter (1.0 = plain GS, 1 < omega < 2 = SOR).

    Returns
    -------
    x : ndarray
        Final solution.
    info : dict
        Diagnostic: ``{'converged': bool, 'iter': int,
        'res_history': list, 'final_res': float}``.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("gs_solve: A must be square")
    n = A.shape[0]
    if b.shape != (n,):
        raise ValueError("gs_solve: b shape mismatch")
    if not (0.0 < omega < 2.0):
        raise ValueError("gs_solve: omega must be in (0, 2)")
    x = np.zeros(n) if x0 is None else x0.copy()
    bnorm = np.linalg.norm(b)
    if bnorm == 0.0:
        return np.zeros(n), dict(converged=True, iter=0,
                                 res_history=[0.0], final_res=0.0)
    res_hist = []
    for k in range(max_iter):
        x_new = gs_step(A, b, x)
        # SOR blend
        x = (1.0 - omega) * x + omega * x_new
        res = np.linalg.norm(A @ x - b) / bnorm
        res_hist.append(res)
        if res < tol:
            return x, dict(converged=True, iter=k + 1,
                           res_history=res_hist, final_res=res)
    return x, dict(converged=False, iter=max_iter,
                   res_history=res_hist, final_res=res_hist[-1])


# =====================================================================
# Build 5-point polar Laplacian
# =====================================================================
def build_polar_laplacian(Nr: int, Ntheta: int, Dr: float,
                          r_edges: np.ndarray) -> np.ndarray:
    """Assemble the 5-point polar Laplacian matrix on ``(Nr, Ntheta)``.

    The stencil at grid point ``(i, j)`` (``r_i``, ``theta_j``) is::

        (1/r_i) d/dr (r d c/dr)  +  (1 / r_i^2) d^2 c / dtheta^2

    discretised with central differences.  Periodic boundary in
    ``theta``, homogeneous Dirichlet at ``r = 0`` and ``r = R``.

    Returns
    -------
    A : (Nr * Ntheta, Nr * Ntheta) ndarray
        Sparse-like dense matrix (kept dense here for simplicity).
    """
    if Nr < 2 or Ntheta < 3:
        raise ValueError("build_polar_laplacian: grid too coarse")
    n = Nr * Ntheta
    A = np.zeros((n, n), dtype=float)
    r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
    hr = Dr
    ht = 2.0 * math.pi / Ntheta
    for i in range(Nr):
        r_i = max(r_mid[i], 1.0e-12)
        for j in range(Ntheta):
            idx = i * Ntheta + j
            # d^2 c / dtheta^2 (periodic)
            jm = (j - 1) % Ntheta
            jp = (j + 1) % Ntheta
            A[idx, i * Ntheta + jm] += 1.0 / (r_i * r_i * ht * ht)
            A[idx, i * Ntheta + jp] += 1.0 / (r_i * r_i * ht * ht)
            A[idx, idx] -= 2.0 / (r_i * r_i * ht * ht)
            # (1/r) d/dr (r dc/dr)
            rp = r_edges[i + 1]
            rm = r_edges[i]
            if i + 1 < Nr:
                A[idx, (i + 1) * Ntheta + j] += (rp / r_i) / (hr * hr)
                A[idx, idx] -= ((rp + rm) / (2.0 * r_i)) / (hr * hr)
            else:
                # Outer Dirichlet
                A[idx, idx] -= (rp / r_i) / (hr * hr)
            if i - 1 >= 0:
                A[idx, (i - 1) * Ntheta + j] += (rm / r_i) / (hr * hr)
            else:
                # Inner Dirichlet (r = 0)
                A[idx, idx] -= (rm / r_i) / (hr * hr)
    A *= -Dr  # overall scale so that A is positive-definite
    return A


# =====================================================================
# Spectral radius estimate of the iteration matrix
# =====================================================================
def gs_spectral_radius(A: np.ndarray, n_iter: int = 20) -> float:
    """Estimate ``rho(T_GS)`` by power iteration on the GS iteration matrix.

    The GS iteration matrix is ``T = -(D + L)^{-1} U`` where
    ``A = D + L + U`` (D diagonal, L strict lower, U strict upper).
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("gs_spectral_radius: A must be square")
    n = A.shape[0]
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    M = D + L
    try:
        Minv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return 1.0
    T = -Minv @ U
    # Power iteration
    rng = np.random.default_rng(0)
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v)
    rho = 0.0
    for _ in range(n_iter):
        w = T @ v
        nw = np.linalg.norm(w)
        if nw < 1.0e-30:
            return 0.0
        rho = nw
        v = w / nw
    return float(rho)


# =====================================================================
# Coupled 2-block Gauss-Seidel for (A, B) species
# =====================================================================
def coupled_block_solve(A11: np.ndarray, A12: np.ndarray,
                        A21: np.ndarray, A22: np.ndarray,
                        b1: np.ndarray, b2: np.ndarray,
                        tol: float = 1.0e-7,
                        max_outer: int = 50) -> tuple[np.ndarray, np.ndarray, dict]:
    """Solve the 2-block system

        [ A11  A12 ] [ x1 ]   [ b1 ]
        [ A21  A22 ] [ x2 ] = [ b2 ]

    by outer Gauss-Seidel on the blocks, with inner GS solves on each
    diagonal block.  This mirrors the species-coupling in the LV
    forward model: the A and B equations share the Laplacian but have
    different reaction terms.

    Returns
    -------
    x1, x2 : ndarray
        Solutions on each block.
    info : dict
        Outer-iteration diagnostics.
    """
    n1 = A11.shape[0]
    n2 = A22.shape[0]
    if b1.shape != (n1,) or b2.shape != (n2,):
        raise ValueError("coupled_block_solve: shape mismatch")
    x1 = np.zeros(n1)
    x2 = np.zeros(n2)
    res_hist = []
    for k in range(max_outer):
        # Block 1: A11 x1 = b1 - A12 x2
        rhs1 = b1 - A12 @ x2
        x1_new, _ = gs_solve(A11, rhs1, x0=x1, tol=tol * 0.1)
        # Block 2: A22 x2 = b2 - A21 x1_new
        rhs2 = b2 - A21 @ x1_new
        x2_new, _ = gs_solve(A22, rhs2, x0=x2, tol=tol * 0.1)
        # Residual
        r1 = np.linalg.norm(b1 - A11 @ x1_new - A12 @ x2_new)
        r2 = np.linalg.norm(b2 - A21 @ x1_new - A22 @ x2_new)
        res = math.sqrt(r1 * r1 + r2 * r2)
        res_hist.append(res)
        x1, x2 = x1_new, x2_new
        if res < tol:
            return x1, x2, dict(converged=True, iter=k + 1,
                                res_history=res_hist, final_res=res)
    return x1, x2, dict(converged=False, iter=max_outer,
                        res_history=res_hist, final_res=res_hist[-1])


# =====================================================================
if __name__ == "__main__":
    # 1-D Poisson test: -u'' = f on [0, 1], u(0) = u(1) = 0
    n = 50
    h = 1.0 / (n + 1)
    A = (2.0 * np.eye(n) - np.diag(np.ones(n - 1), 1)
         - np.diag(np.ones(n - 1), -1)) / (h * h)
    b = np.ones(n)
    x, info = gs_solve(A, b, tol=1.0e-8, max_iter=5000)
    print("1-D Poisson GS:", info)
    print("  max |x - x_exact| =",
          np.max(np.abs(x - 0.5 * np.linspace(0, 1, n + 2)[1:-1]
                        * (1.0 - np.linspace(0, 1, n + 2)[1:-1]))))
    # Polar Laplacian
    Nr, Ntheta = 4, 8
    r_edges = np.linspace(0.05, 1.0, Nr + 1)
    A_p = build_polar_laplacian(Nr, Ntheta, r_edges[1] - r_edges[0], r_edges)
    print("Polar Laplacian shape:", A_p.shape,
          "  rho_GS ~", gs_spectral_radius(A_p))
