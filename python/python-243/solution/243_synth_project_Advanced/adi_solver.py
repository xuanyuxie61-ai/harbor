"""
adi_solver.py - Alternating Direction Implicit solver for 2D (T, rho) evolution.

Adapted from 1121_katyushapolye_PyADI-Fluid-Sim ADI patterns.  Solves the
heat/diffusion equation on a 2D grid with mixed boundary conditions:

    dU/dt = alpha_x d^2 U / dx^2 + alpha_y d^2 U / dy^2 + S(x, y, t)

using Peaceman-Rachford ADI:
    (I - 0.5 dt L_x) U^{n+1/2} = (I + 0.5 dt L_y) U^n  + 0.5 dt S^{n+1/2}
    (I - 0.5 dt L_y) U^{n+1}   = (I + 0.5 dt L_x) U^{n+1/2} + 0.5 dt S^{n+1/2}
"""
from __future__ import annotations
import numpy as np
from typing import Tuple

def build_tridiag(n: int, diag: float, off: float) -> np.ndarray:
    """Return an n x n tridiagonal matrix with diag on diagonal, off on off-diag."""
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        A[i, i] = diag
        if i > 0:
            A[i, i - 1] = off
        if i < n - 1:
            A[i, i + 1] = off
    return A


def solve_tridiag(a: np.ndarray, b: np.ndarray, c: np.ndarray,
                  d: np.ndarray) -> np.ndarray:
    """
    Thomas algorithm for tridiagonal system  a x_{i-1} + b x_i + c x_{i+1} = d.
    a, b, c are arrays of length n (a[0] and c[n-1] unused).
    """
    n = len(b)
    cp = np.zeros(n, dtype=np.float64)
    dp = np.zeros(n, dtype=np.float64)
    x = np.zeros(n, dtype=np.float64)
    if abs(b[0]) < 1e-300:
        raise ValueError("Zero pivot at i=0")
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        if abs(m) < 1e-300:
            m = 1e-300
        cp[i] = c[i] / m if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def peaceman_rachford_step(U: np.ndarray,
                           alpha_x: float, alpha_y: float,
                           dx: float, dy: float, dt: float,
                           S: np.ndarray = None) -> np.ndarray:
    """
    One Peaceman-Rachford ADI step on a 2D grid U of shape (Nx, Ny).
    Returns U^{n+1}.  Zero-Neumann BCs.
    """
    Nx, Ny = U.shape
    rx = 0.5 * dt * alpha_x / (dx * dx)
    ry = 0.5 * dt * alpha_y / (dy * dy)

    # Half-step: solve in x-direction for each y
    U_half = np.zeros_like(U)
    for j in range(Ny):
        a_vec = np.full(Nx, -rx)
        b_vec = np.full(Nx, 1.0 + 2.0 * rx)
        c_vec = np.full(Nx, -rx)
        a_vec[0] = 0.0
        c_vec[-1] = 0.0
        # RHS: (I + ry L_y) U^n + 0.5 dt S
        rhs = np.zeros(Nx)
        for i in range(Nx):
            L_y_U = 0.0
            if j > 0:
                L_y_U += U[i, j - 1]
            if j < Ny - 1:
                L_y_U += U[i, j + 1]
            L_y_U -= 2.0 * U[i, j]
            src = 0.5 * dt * S[i, j] if S is not None else 0.0
            rhs[i] = U[i, j] + ry * L_y_U + src
        U_half[:, j] = solve_tridiag(a_vec, b_vec, c_vec, rhs)

    # Full step: solve in y-direction for each x
    U_new = np.zeros_like(U)
    for i in range(Nx):
        a_vec = np.full(Ny, -ry)
        b_vec = np.full(Ny, 1.0 + 2.0 * ry)
        c_vec = np.full(Ny, -ry)
        a_vec[0] = 0.0
        c_vec[-1] = 0.0
        rhs = np.zeros(Ny)
        for j in range(Ny):
            L_x_Uh = 0.0
            if i > 0:
                L_x_Uh += U_half[i - 1, j]
            if i < Nx - 1:
                L_x_Uh += U_half[i + 1, j]
            L_x_Uh -= 2.0 * U_half[i, j]
            src = 0.5 * dt * S[i, j] if S is not None else 0.0
            rhs[j] = U_half[i, j] + rx * L_x_Uh + src
        U_new[i, :] = solve_tridiag(a_vec, b_vec, c_vec, rhs)

    return U_new


def adi_evolve(U0: np.ndarray,
               alpha_x: float, alpha_y: float,
               dx: float, dy: float, dt: float, nsteps: int,
               S: np.ndarray = None) -> np.ndarray:
    """Evolve U for nsteps ADI steps."""
    U = U0.copy()
    for _ in range(nsteps):
        U = peaceman_rachford_step(U, alpha_x, alpha_y, dx, dy, dt, S)
    return U


__all__ = [
    "build_tridiag", "solve_tridiag",
    "peaceman_rachford_step", "adi_evolve",
]
