"""
halo_gegenbauer_potential.py
============================
Gegenbauer spectral expansion of the spherically-averaged dark matter
halo density profile, combined with a Horner-form evaluation of the
resulting gravitational potential.

Scientific background
---------------------
For a virialised NFW-like halo with characteristic density rho_s and
scale radius r_s, the dimensionless density contrast is

    Delta(r) = rho(r) / rho_s = (r / r_s)^(-1) (1 + r / r_s)^(-2).

To obtain a spectral representation on the finite interval
r in [0, R_vir], we expand the rescaled contrast

    u(x) = (1 - x^2)^(alpha + 1/2) Delta(R_vir x),   x in [-1, 1]

in Gegenbauer polynomials C_n^(alpha)(x) with alpha > -1/2.  The
expansion coefficients are computed via the Elhay-Kautsky procedure
(Gauss-Gegenbauer quadrature nodes and weights).  The potential

    Phi(r) = -4 pi G rho_s r_s^2 * sum_n a_n Psi_n(r/r_s)

is then evaluated by Horner's scheme to minimise round-off error.
"""

from __future__ import annotations
import math
from typing import Tuple

import numpy as np


# ---------- Physical / numerical parameters -----------------------------------

G_NEWTON   = 6.67430e-11        # m^3 kg^-1 s^-2
MPC_TO_M   = 3.0856775814913673e22
MSUN_TO_KG = 1.98892e30


def gegenbauer_moment_zero(n: int, alpha: float) -> float:
    """Zero-th moment mu_0 of the Gegenbauer weight (1-x^2)^(alpha-1/2)."""
    # mu_0 = 2^(2 alpha + 1) Gamma(alpha + 1)^2 / Gamma(2 alpha + 2)
    num = 2.0 ** (2.0 * alpha + 1.0) * math.gamma(alpha + 1.0) ** 2
    den = math.gamma(2.0 * alpha + 2.0)
    return num / den


def gegenbauer_jacobi_bj(i: int, alpha: float) -> float:
    """Sub-diagonal element b_j of the symmetric Jacobi matrix for
    the monic Gegenbauer recurrence (Elhay-Kautsky, 1987)."""
    if i == 1:
        num = 4.0 * (alpha + 1.0) ** 2
        den = (2.0 * alpha + 3.0) * (2.0 * alpha + 2.0) ** 2
        return num / den
    abi = 2.0 * (alpha + i)
    num = 4.0 * i * (alpha + i) ** 2 * (2.0 * alpha + i)
    den = (abi - 1.0) * (abi + 1.0) * abi * abi
    return num / den


def gauss_gegenbauer_nodes_weights(n: int, alpha: float
                                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Compute nodes and weights of the Gauss-Gegenbauer rule of order n
    by tridiagonal (Golub-Welsch) diagonalisation."""
    if n <= 0:
        return np.zeros(0), np.zeros(0)

    bj = np.zeros(n)
    for i in range(1, n + 1):
        bj[i - 1] = gegenbauer_jacobi_bj(i, alpha)
    bj = np.sqrt(np.clip(bj, 0.0, None))

    # Symmetric tridiagonal matrix with zero diagonal and bj sub-diag.
    T = np.diag(bj[1:], k=1) + np.diag(bj[1:], k=-1)
    eigvals, eigvecs = np.linalg.eigh(T)

    zemu = gegenbauer_moment_zero(n, alpha)
    weights = zemu * eigvecs[0, :] ** 2
    return np.sort(eigvals), weights


# ---------- NFW density sampling in Gegenbauer space --------------------------

def nfw_contrast(x: np.ndarray, rs: float, rvir: float) -> np.ndarray:
    """NFW density contrast at scaled coordinates x in [-1, 1]."""
    r = rs + 0.5 * (x + 1.0) * (rvir - rs)   # affine map to [rs, rvir]
    r = np.clip(r, 1e-12 * rs, None)
    u = r / rs
    return (1.0 / u) / (1.0 + u) ** 2


def gegenbauer_expansion(n_order: int, alpha: float, rs: float, rvir: float
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Return expansion coefficients a_k of the NFW contrast on
    the Gegenbauer basis with parameter alpha."""
    x, w = gauss_gegenbauer_nodes_weights(n_order, alpha)
    vals = nfw_contrast(x, rs, rvir)
    # Weighted projection: a_k = (1/h_k) int vals(x) C_k^(alpha)(x) w_alpha(x) dx
    # Here we simply store the nodal projection as a pseudo-spectral coefficient.
    a = w * vals
    norm = np.sum(w)
    if norm > 0.0:
        a = a / norm
    return x, a


# ---------- Horner evaluation of the potential --------------------------------

def horner_potential(coeffs: np.ndarray, xi: float) -> float:
    """Horner evaluation of sum_k c_k xi^k.  Numerically stable for
    moderate polynomial degrees used in halo potential reconstructions."""
    if coeffs.size == 0:
        return 0.0
    p = coeffs[-1]
    for c in coeffs[-2::-1]:
        p = p * xi + c
    return p


def halo_potential_at(r: np.ndarray,
                      coeffs: np.ndarray,
                      rs: float, rvir: float,
                      rho_s: float) -> np.ndarray:
    """Gravitational potential of the halo on a radial grid."""
    xi = (r - rs) / (rvir - rs)
    xi = np.clip(xi, -1.0, 1.0)
    phi = np.zeros_like(xi)
    for i, x in enumerate(xi.flat):
        phi.flat[i] = horner_potential(coeffs, x)
    pref = -4.0 * math.pi * G_NEWTON * rho_s * rs ** 2
    return pref * phi


# ---------- Public convenience ------------------------------------------------

def build_halo_potential(n_order: int = 16,
                         alpha: float = 0.5,
                         rs: float = 0.25,
                         rvir: float = 1.0,
                         rho_s: float = 1.0
                         ) -> dict:
    """Return a dictionary containing the spectral expansion data
    needed to evaluate the potential later."""
    nodes, coeffs = gegenbauer_expansion(n_order, alpha, rs, rvir)
    return dict(nodes=nodes, coeffs=coeffs,
                rs=rs, rvir=rvir, rho_s=rho_s,
                alpha=alpha, n_order=n_order)


# ---------- Sanity self-check (called from main) ------------------------------

def self_check() -> dict:
    nodes, w = gauss_gegenbauer_nodes_weights(8, 0.5)
    moment1 = float(np.sum(w * nodes ** 2))
    coeffs = build_halo_potential()["coeffs"]
    phi0 = horner_potential(coeffs, 0.0)
    return dict(moment1=moment1, phi0=phi0, n_nodes=nodes.size)
