# -*- coding: utf-8 -*-
"""
linear_eigenvalue.py
====================

Linear stability analysis of the slab ITG mode.

We solve the linearised gyrokinetic equation in the slab geometry for
a single Fourier component in  y  (k_y) and  z  (k_z = pi / (q R0) for
the fundamental ballooning harmonic):

    - i omega delta h  +  v_parallel b0 . grad || delta h
        - i omega_T* (1 + eta_i (v^2 / v_th^2 - 3/2)) J_0(k_perp rho) (e phi / T_i) F0
        + i omega_d (v_parallel) delta h
        = - i omega delta h

where  delta h = delta f - (e <phi>_R / T_i) (1 + eta_i (v^2 / v_th^2 - 3/2)) F0
is the non-adiabatic part of the distribution function.

In the 1-D radial reduction the parallel streaming  v_parallel b0 . grad ||
becomes  i k_z v_parallel  plus the magnetic-drift coupling, yielding a
spectrum of "ballooning harmonics" coupled through the curvature drift.
The discrete eigenvalue problem is

    A(k_y, k_z, omega) xi  =  0

where  A  is an  (N_radial x N_vpar x N_balloon) matrix and  omega  is the
complex eigenvalue.  We solve for  omega  by inverse iteration / shift-and-
invert Arnoldi using scipy's sparse eigensolver (when available) and
otherwise by dense QR on the companion matrix.

Growth rate and real frequency
------------------------------
    omega = omega_r + i gamma
    gamma > 0  : unstable ITG mode
    omega_r    : diamagnetic propagation frequency ~ omega*_i

Threshold search
----------------
The critical  R/L_Ti  at which  gamma  crosses zero is the ITG threshold.
We locate it by bisection in  R/L_Ti  on the sign change of  gamma.

References:
    [1] Romanelli, Phys. Fluids B 1, 1018 (1989)
    [2] Lee & Tang, Phys. Fluids 31, 999 (1988)
    [3] Candy & Waltz, J. Comp. Phys. 186, 545 (2003)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from gyroaverage import gamma0
from physics_constants import PI
from velocity_space import jn_eval


@dataclass
class SlabITGConfig:
    """Configuration for the slab ITG eigenvalue problem."""
    N_radial: int = 40
    N_vpar: int = 24
    N_balloon: int = 3
    k_y_rho: float = 0.3
    k_z_qR: float = 0.3          # k_z q R0
    L_Ti_over_L_n: float = 3.0   # eta_i = L_n / L_Ti
    tau_e: float = 1.0           # T_e / T_i
    q_safety: float = 2.0
    epsilon: float = 0.3
    v_par_max: float = 5.0
    L_s: float = 10.0            # magnetic shear length


# ============================================================================
# Matrix assembly
# ============================================================================
def _build_vpar_grid(N: int, v_max: float) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes / weights on [-v_max, v_max]."""
    nodes, weights = np.polynomial.legendre.leggauss(N)
    nodes = v_max * nodes
    weights = v_max * weights
    return nodes, weights


def _ballooning_harmonics(N: int, theta: np.ndarray) -> np.ndarray:
    """First N ballooning harmonics  exp(i m theta)  at given poloidal angles."""
    m = np.arange(N)
    return np.exp(1j * np.outer(theta, m))


def build_slab_itg_matrix(cfg: SlabITGConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Assemble the slab ITG matrix  A  and the norm matrix  B  such that

        A xi = omega B xi

    The discretisation uses:
        - N_radial radial grid points  x_j
        - N_vpar Gauss-Legendre nodes in v_parallel
        - N_balloon ballooning harmonics  m = 0, ..., N_balloon - 1

    The matrix acts on the vector
        xi = (delta h_0, delta h_1, ..., delta h_{N_radial-1})
    where each  delta h_j  is a (N_vpar x N_balloon) block.
    """
    Nr = cfg.N_radial
    Nv = cfg.N_vpar
    Nb = cfg.N_balloon
    x = np.linspace(0.0, 1.0, Nr)
    vp, wv = _build_vpar_grid(Nv, cfg.v_par_max)

    # equilibrium gradients (radial)
    L_n = 1.0 / max(cfg.L_Ti_over_L_n * 0.1, 1.0e-3)
    L_Ti = L_n / max(cfg.L_Ti_over_L_n, 1.0e-3)
    omega_star = cfg.k_y_rho * L_n           # normalised
    eta_i = L_n / L_Ti

    total = Nr * Nv * Nb
    A = np.zeros((total, total), dtype=np.complex128)
    B = np.zeros((total, total), dtype=np.complex128)

    for j in range(Nr):
        for iv in range(Nv):
            v = vp[iv]
            for mb in range(Nb):
                row = ((j * Nv + iv) * Nb) + mb
                # parallel streaming  -> i k_z v
                A[row, row] += 1j * cfg.k_z_qR * v
                # magnetic drift  ~ k_y v_par^2 / L_s
                omega_d = cfg.k_y_rho * v * v / max(cfg.L_s, 1.0)
                # coupling to m +/- 1 via curvature
                if mb + 1 < Nb:
                    col = ((j * Nv + iv) * Nb) + (mb + 1)
                    A[row, col] += 1j * 0.5 * omega_d
                if mb - 1 >= 0:
                    col = ((j * Nv + iv) * Nb) + (mb - 1)
                    A[row, col] += 1j * 0.5 * omega_d
                # drive term (diagonal) -- ITG curvature drive
                # - i omega* (1 + eta_i (v^2 - 3/2)) * Gamma_0(b)
                b = 0.5 * cfg.k_y_rho ** 2 * (1.0 + v * v)
                g0 = float(gamma0(np.array([b]))[0])
                drive = -1j * omega_star * (1.0 + eta_i * (v * v - 1.5)) * g0
                A[row, row] += drive
                # adiabatic response (B)
                B[row, row] = 1.0 + cfg.tau_e * (1.0 - g0)
    # boundary rows: zero out to enforce Dirichlet
    for j in (0, Nr - 1):
        for iv in range(Nv):
            for mb in range(Nb):
                row = ((j * Nv + iv) * Nb) + mb
                A[row, :] = 0.0
                B[row, :] = 0.0
                A[row, row] = 1.0
                B[row, row] = 1.0
    return A, B


# ============================================================================
# Eigenvalue solve
# ============================================================================
def solve_slab_itg(cfg: SlabITGConfig, sigma: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Solve  A xi = omega B xi  via dense QZ decomposition.

    Returns
    -------
    omega : (M,) complex array of eigenvalues (sorted by descending growth rate)
    growth : (M,) real array of growth rates  gamma = Im(omega)
    """
    A, B_mat = build_slab_itg_matrix(cfg)
    try:
        from scipy.linalg import eig
        omega, _ = eig(A, b=B_mat)
    except ImportError:
        # fallback: naive QZ via np.linalg.eig of B^{-1} A
        B_inv = np.linalg.pinv(B_mat)
        omega = np.linalg.eigvals(B_inv @ A)
    # sort by descending Im
    order = np.argsort(-np.imag(omega))
    omega = omega[order]
    return omega, np.imag(omega)


# ============================================================================
# Threshold search in R/L_Ti
# ============================================================================
def find_itg_threshold(
    cfg_base: SlabITGConfig,
    rl_ti_lo: float = 1.0,
    rl_ti_hi: float = 12.0,
    tol: float = 1.0e-2,
    max_iter: int = 30,
) -> Tuple[float, dict]:
    """Bisect in  R/L_Ti  to locate the ITG threshold  gamma(rl_ti) = 0.

    Returns the critical value and a small dict with diagnostic info.
    """
    lo, hi = rl_ti_lo, rl_ti_hi
    info = {"lo_hist": [lo], "hi_hist": [hi], "gamma_lo": 0.0, "gamma_hi": 0.0}
    cfg = SlabITGConfig(**{**cfg_base.__dict__})

    def _gamma(rlti: float) -> float:
        cfg.L_Ti_over_L_n = rlti
        omega, growth = solve_slab_itg(cfg)
        return float(np.max(growth)) if growth.size else 0.0

    g_lo = _gamma(lo)
    g_hi = _gamma(hi)
    info["gamma_lo"] = g_lo
    info["gamma_hi"] = g_hi
    if g_lo * g_hi > 0:
        # threshold is outside the interval
        return (hi if g_hi > 0 else lo), info
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        g_mid = _gamma(mid)
        if abs(g_mid) < tol or (hi - lo) < tol:
            return mid, info
        if g_lo * g_mid < 0:
            hi = mid
            g_hi = g_mid
        else:
            lo = mid
            g_lo = g_mid
        info["lo_hist"].append(lo)
        info["hi_hist"].append(hi)
    return 0.5 * (lo + hi), info


# ============================================================================
# Sanity self-check
# ============================================================================
if __name__ == "__main__":
    cfg = SlabITGConfig()
    omega, growth = solve_slab_itg(cfg)
    print("top 5 eigenvalues (Im, Re):")
    for w in omega[:5]:
        print(f"  omega = {w.real:+.4f} + {w.imag:+.4f} j")
    # threshold search on a coarse grid
    thr, info = find_itg_threshold(cfg, rl_ti_lo=1.0, rl_ti_hi=10.0, tol=0.1, max_iter=6)
    print("approximate ITG threshold R/L_Ti =", thr)
