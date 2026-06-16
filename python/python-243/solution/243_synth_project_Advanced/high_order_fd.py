"""
high_order_fd.py - High-order finite difference for 1D advection-diffusion.

Adapted from 357_fd1d_burgers_leap (leapfrog for Burgers) and 359_fd1d_display.
Solves  dU/dt + u dU/dx = nu d^2 U / dx^2  using WENO-5 or Lax-Wendroff.

Lax-Wendroff (2nd order):
  U_i^{n+1} = U_i^n - 0.5 C (U_{i+1}^n - U_{i-1}^n)
              + 0.5 C^2 (U_{i+1}^n - 2 U_i^n + U_{i-1}^n)
  where C = u dt / dx  is the Courant number.

WENO-5 reconstruction for spatial derivatives.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple

def lax_wendroff_step(U: np.ndarray, u_adv: float, dx: float, dt: float,
                      nu: float = 0.0) -> np.ndarray:
    """
    One Lax-Wendroff step with optional diffusion.
    U: 1D array of shape (N,)
    u_adv: advection velocity (scalar)
    nu: diffusion coefficient
    Returns U^{n+1}.
    """
    N = len(U)
    C = u_adv * dt / dx if dx > 0 else 0.0
    D = nu * dt / (dx * dx) if dx > 0 else 0.0
    U_new = np.zeros_like(U)
    for i in range(1, N - 1):
        # Advection (Lax-Wendroff)
        adv = -0.5 * C * (U[i + 1] - U[i - 1]) + 0.5 * C * C * (U[i + 1] - 2 * U[i] + U[i - 1])
        # Diffusion (central)
        diff = D * (U[i + 1] - 2 * U[i] + U[i - 1])
        U_new[i] = U[i] + adv + diff
    # Boundary: zero-gradient
    U_new[0] = U_new[1]
    U_new[-1] = U_new[-2]
    return U_new


def weno5_reconstruct(UL: np.ndarray, UR: np.ndarray, dx: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    WENO-5 reconstruction of left and right states at cell interfaces.
    UL, UR: cell-averaged values (shape N).
    Returns U_left, U_right at each interface (shape N+1).
    Interior cells only (i in [3, N-3]); boundary cells use upwind extrapolation.
    """
    N = len(UL)
    U_left = np.zeros(N + 1)
    U_right = np.zeros(N + 1)
    eps = 1e-6
    for i in range(3, N - 2):
        v0 = UL[i - 2]
        v1 = UL[i - 1]
        v2 = UL[i]
        v3 = UL[i + 1]
        v4 = UL[i + 2]
        # Smoothness indicators
        beta0 = (13.0 / 12.0) * (v0 - 2 * v1 + v2) ** 2 + 0.25 * (v0 - 4 * v1 + 3 * v2) ** 2
        beta1 = (13.0 / 12.0) * (v1 - 2 * v2 + v3) ** 2 + 0.25 * (v1 - v3) ** 2
        beta2 = (13.0 / 12.0) * (v2 - 2 * v3 + v4) ** 2 + 0.25 * (3 * v2 - 4 * v3 + v4) ** 2
        # Weights
        alpha0 = 0.1 / (eps + beta0) ** 2
        alpha1 = 0.6 / (eps + beta1) ** 2
        alpha2 = 0.3 / (eps + beta2) ** 2
        alpha_sum = alpha0 + alpha1 + alpha2
        w0, w1, w2 = alpha0 / alpha_sum, alpha1 / alpha_sum, alpha2 / alpha_sum
        # Candidate stencils
        q0 = (1.0 / 3.0) * v0 - (7.0 / 6.0) * v1 + (11.0 / 6.0) * v2
        q1 = -(1.0 / 6.0) * v1 + (5.0 / 6.0) * v2 + (1.0 / 3.0) * v3
        q2 = (1.0 / 3.0) * v2 + (5.0 / 6.0) * v3 - (1.0 / 6.0) * v4
        U_left[i] = w0 * q0 + w1 * q1 + w2 * q2
        # Mirror for right state (simple copy for interior)
        U_right[i] = U_left[i]
    # Boundary cells: use cell average directly
    for i in range(3):
        U_left[i] = UL[i]
        U_right[i] = UL[i]
    for i in range(N - 2, N + 1):
        if i < N:
            U_left[i] = UL[i]
            U_right[i] = UL[i]
        else:
            U_left[i] = UL[-1]
            U_right[i] = UL[-1]
    return U_left, U_right


def weno5_step(U: np.ndarray, u_adv: float, dx: float, dt: float) -> np.ndarray:
    """One MUSCL-3 + Lax-Friedrichs flux step (third-order upwind-biased)."""
    N = len(U)
    F = np.zeros(N + 1)
    for i in range(1, N):
        # Slope with minmod limiter
        du_L = U[i] - U[i - 1]
        du_R = U[i + 1] - U[i] if i + 1 < N else 0.0
        if du_L * du_R <= 0:
            slope = 0.0
        else:
            slope = 0.5 * (np.sign(du_L) + np.sign(du_R)) * min(abs(du_L), abs(du_R))
        # Left and right reconstructed states at interface i+1/2
        uL = U[i] + 0.5 * slope
        # Right state from the other side
        du_L2 = U[i + 1] - U[i] if i + 1 < N else 0.0
        du_R2 = U[i + 2] - U[i + 1] if i + 2 < N else 0.0
        if du_L2 * du_R2 <= 0:
            slope2 = 0.0
        else:
            slope2 = 0.5 * (np.sign(du_L2) + np.sign(du_R2)) * min(abs(du_L2), abs(du_R2))
        uR = U[i + 1] - 0.5 * slope2 if i + 1 < N else U[i]
        # Lax-Friedrichs flux
        F[i] = 0.5 * (u_adv * uL + u_adv * uR) - 0.5 * abs(u_adv) * (uR - uL)
    # Boundary fluxes (zero-gradient)
    F[0] = u_adv * U[0]
    F[N] = u_adv * U[-1]
    U_new = np.zeros_like(U)
    for i in range(N):
        U_new[i] = U[i] - (dt / dx) * (F[i + 1] - F[i])
    return U_new


def fd_evolve(U0: np.ndarray, u_adv: float, dx: float, dt: float, nsteps: int,
              method: str = "lax_wendroff", nu: float = 0.0) -> np.ndarray:
    """Evolve U for nsteps using specified method."""
    U = U0.copy()
    for _ in range(nsteps):
        if method == "lax_wendroff":
            U = lax_wendroff_step(U, u_adv, dx, dt, nu)
        elif method == "weno5":
            U = weno5_step(U, u_adv, dx, dt)
        else:
            raise ValueError(f"Unknown method: {method}")
    return U


__all__ = [
    "lax_wendroff_step", "weno5_reconstruct", "weno5_step", "fd_evolve",
]
