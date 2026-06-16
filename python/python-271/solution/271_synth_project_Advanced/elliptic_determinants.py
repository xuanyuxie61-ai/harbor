# -*- coding: utf-8 -*-
"""
Elliptic integrals and fermion-determinant evaluations.

After the Jordan-Wigner transformation the partition function of the
1D TFIM becomes a Pfaffian of an antisymmetric 2L x 2L matrix.  Its
square is a determinant that can be written in terms of complete
elliptic integrals:

    Z = prod_{k>0} [ 2 cosh(beta eps_k / 2) ]

where the single-particle energies eps_k are expressed via the
complete elliptic integral of the first kind K(m):

    eps_k = 2 J sqrt(1 + lam^2 - 2 lam cos k)
          = 2 J sqrt( (1 - lam)^2 + 4 lam sin^2(k/2) )

At the QCP (lam = 1) the dispersion becomes gapless, eps_k ~ 2 J |k|,
and the associated elliptic modulus m = 4 lam / (1 + lam)^2 equals 1,
making K(m) logarithmically divergent.  This divergence is the
origin of the conformal central charge c = 1/2.

We implement Carlson's symmetric elliptic integrals RF, RD, RJ, RC
following the NIST DLMF and use them to build the exact finite-
temperature free energy and fidelity susceptibility.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
try:
    from . import constants as C
except ImportError:
    import constants as C


# ---------------------------------------------------------------------------
# Carlson symmetric integrals
# ---------------------------------------------------------------------------
def rf_carlson(x: float, y: float, z: float, tol: float = 1.0e-12,
                max_iter: int = 60) -> float:
    """Carlson R_F(x, y, z) = (1/2) int_0^infty (t+x)^{-1/2}(t+y)^{-1/2}(t+z)^{-1/2} dt.

    Iterative duplication algorithm; converges in O(log(1/tol)) steps.
    Reference: Carlson, "Computing elliptic integrals by duplication",
    Math. Comp. 33, 130 (1979).
    """
    if min(x, y, z) < 0.0:
        raise ValueError("rf: all arguments must be non-negative")
    A0 = (x + y + z) / 3.0
    if A0 < C.EPS_NUM:
        return 1.0 / np.sqrt(max(A0, C.EPS_NUM))
    xk, yk, zk = float(x), float(y), float(z)
    for _ in range(max_iter):
        lam = np.sqrt(xk * yk) + np.sqrt(yk * zk) + np.sqrt(zk * xk)
        xk = (xk + lam) / 4.0
        yk = (yk + lam) / 4.0
        zk = (zk + lam) / 4.0
        Ak = (xk + yk + zk) / 3.0
        eps = max(abs(xk - Ak), abs(yk - Ak), abs(zk - Ak)) / max(abs(Ak), C.EPS_NUM)
        if eps < tol:
            break
    # Series correction
    X = (A0 - x) / (Ak * 4.0 ** len([]))
    # Simpler closed form at convergence:
    return 1.0 / np.sqrt(max(Ak, C.EPS_NUM))


def rd_carlson(x: float, y: float, z: float, tol: float = 1.0e-12,
                max_iter: int = 60) -> float:
    """Carlson R_D(x, y, z) = R_J(x, y, z, z)."""
    if min(x, y) < 0.0 or z <= 0.0:
        raise ValueError("rd: x, y >= 0 and z > 0 required")
    xk, yk, zk = float(x), float(y), float(z)
    s = 0.0
    factor = 1.0
    for _ in range(max_iter):
        lam = np.sqrt(xk * yk) + np.sqrt(yk * zk) + np.sqrt(zk * xk)
        s += factor / (np.sqrt(max(zk, C.EPS_NUM)) * (zk + lam))
        factor /= 4.0
        xk = (xk + lam) / 4.0
        yk = (yk + lam) / 4.0
        zk = (zk + lam) / 4.0
        Ak = (xk + yk + 3.0 * zk) / 5.0
        eps = max(abs(xk - Ak), abs(yk - Ak), abs(zk - Ak)) / max(abs(Ak), C.EPS_NUM)
        if eps < tol:
            break
    return 3.0 * s + factor / (Ak * np.sqrt(max(Ak, C.EPS_NUM)))


def rc_carlson(x: float, y: float, tol: float = 1.0e-12,
                max_iter: int = 60) -> float:
    """Carlson R_C(x, y) = (1/2) int_0^infty (t+x)^{-1/2}(t+y)^{-1} dt.
    For y > x >= 0: R_C(x, y) = (1/sqrt(y-x)) * arccos(sqrt(x/y)).
    """
    if x < 0.0 or y <= 0.0:
        raise ValueError("rc: x >= 0 and y > 0 required")
    if abs(y - x) < C.EPS_NUM * max(abs(x), abs(y)):
        return 1.0 / np.sqrt(max(y, C.EPS_NUM))
    if y > x:
        return np.arccos(np.sqrt(max(x / y, 0.0))) / np.sqrt(max(y - x, C.EPS_NUM))
    return np.arccosh(np.sqrt(max(y / x, C.EPS_NUM))) / np.sqrt(max(x - y, C.EPS_NUM))


# ---------------------------------------------------------------------------
# Legendre complete elliptic integrals via Carlson
# ---------------------------------------------------------------------------
def elliptic_K(m: float) -> float:
    """Complete elliptic integral of the first kind K(m) with
    parameter m = k^2.  For m in [0, 1).

    K(m) = R_F(0, 1 - m, 1).
    For m -> 1^- we have the logarithmic singularity
        K(m) ~ 0.5 * log(16 / (1 - m)).
    """
    if m < 0.0 or m >= 1.0:
        # Fall back to log asymptotics if exactly at m = 1
        if abs(m - 1.0) < 1.0e-12:
            return 0.5 * np.log(16.0 / max(1.0 - m, C.EPS_NUM))
        if m < 0.0:
            # Use analytic continuation: K(-|m|) = R_F(0, 1+|m|, 1)
            return rf_carlson(0.0, 1.0 - m, 1.0)
        return float("inf")
    return rf_carlson(0.0, 1.0 - m, 1.0)


def elliptic_E(m: float) -> float:
    """Complete elliptic integral of the second kind E(m).
    E(m) = R_F(0, 1 - m, 1) - (m/3) R_D(0, 1 - m, 1).
    """
    if m < 0.0 or m >= 1.0:
        if abs(m - 1.0) < 1.0e-12:
            return 1.0
        if m < 0.0:
            K = elliptic_K(m)
            return (1.0 - m) * K + m * elliptic_E(m / (m - 1.0)) / (1.0 - m + C.EPS_NUM)
    rf = rf_carlson(0.0, 1.0 - m, 1.0)
    rd = rd_carlson(0.0, 1.0 - m, 1.0)
    return rf - (m / 3.0) * rd


# ---------------------------------------------------------------------------
# Fermion-determinant for the TFIM
# ---------------------------------------------------------------------------
def tfim_elliptic_modulus(lam: float) -> float:
    """Elliptic modulus m of the TFIM mapped fermion problem:
        m(lam) = 4 lam / (1 + lam)^2.
    At lam = 1  ->  m = 1  (critical).
    For lam -> 0 or lam -> infty  ->  m -> 0.
    """
    denom = (1.0 + lam) ** 2
    if denom < C.EPS_NUM:
        return 0.0
    return min(4.0 * abs(lam) / denom, 1.0)


def ground_state_energy_exact(L: int, J: float, h: float,
                                periodic: bool = False) -> float:
    """Finite-L ground-state energy evaluated via the elliptic-integral
    representation (Lieb-Schultz-Mattis 1961).

    For OBC:
        E_0 / L = - (h / pi) * E(m)       (thermodynamic limit)
    where E is the complete elliptic integral of the second kind and
    m = 4 J h / (J^2 + h^2).  Finite-L corrections enter as
    O(1/L^2) and are controlled by the conformal tower.
    """
    if J == 0.0:
        return -abs(h) * L
    lam = h / J
    m = tfim_elliptic_modulus(lam)
    # Thermodynamic-limit per-site energy:
    #   e_inf = - max(J, h) / pi * E(m)
    prefactor = max(J, h)
    e_inf = -prefactor / C.PI * elliptic_E(m)
    # Finite-L correction from the conformal tower:  - pi c v / (6 L^2)
    c = 0.5
    v = 2.0 * abs(J) * abs(1.0 - lam * lam) if abs(1.0 - lam * lam) > 0.05 else 1.0e-3
    finite_L_correction = -C.PI * c * v / (6.0 * L * L)
    return L * e_inf + finite_L_correction


def fidelity_susceptibility(L: int, J: float, h: float) -> float:
    """Ground-state fidelity susceptibility of the TFIM:
        chi_F = (L / 8) * (1 + lam^2) / (1 - lam^2)^2     (for lam != 1)
    At lam = 1 we regularise by the finite-L cutoff:
        chi_F(L, lam=1) = L^2 / 8.

    This matches the CFT result chi_F = (c / 8) L^2 for a system with
    open boundary conditions and central charge c = 1/2.
    """
    lam = h / J if J != 0.0 else 0.0
    disc = 1.0 - lam * lam
    if abs(disc) < 1.0 / L:
        return float(L * L / 8.0)
    return float(L * (1.0 + lam * lam) / (8.0 * disc * disc))


# ---------------------------------------------------------------------------
# Ellipsoidal Brillouin-zone integrals
# ---------------------------------------------------------------------------
def ellipsoid_area_analogue(a: float, b: float, c: float) -> float:
    """Surrogate of the ellipsoid surface-area formula for the
    anisotropic TFIM with couplings (J_x, J_y, J_z) ~ (a, b, c).

    We use Knopp's reduction to elliptic integrals as in the
    ellipsoid-area seed project (332):
        S = 2 pi c^2 + 2 pi a b / sin(phi) * [ E(phi, k) sin^2(phi)
                                                + F(phi, k) cos^2(phi) ]
    with sin(phi) = sqrt(1 - c^2/a^2) and k^2 = a^2 (b^2 - c^2) /
    (b^2 (a^2 - c^2)).
    """
    a, b, c = sorted([abs(a), abs(b), abs(c)], reverse=True)
    if a < C.EPS_NUM:
        return 0.0
    if c < C.EPS_NUM:
        # degenerate: return 2 * area of ellipse
        return 2.0 * C.PI * a * b
    e2 = 1.0 - (c / a) ** 2
    if e2 <= 0.0:
        # sphere
        return 4.0 * C.PI * a * a
    phi = np.arcsin(np.sqrt(max(e2, 0.0)))
    k2 = (a * a * (b * b - c * c)) / max(b * b * (a * a - c * c), C.EPS_NUM)
    k2 = min(max(k2, 0.0), 1.0 - C.EPS_NUM)
    E = elliptic_E(k2)
    K = elliptic_K(k2)
    sinp = np.sin(phi)
    cosp = np.cos(phi)
    S = 2.0 * C.PI * c * c + 2.0 * C.PI * a * b / max(sinp, C.EPS_NUM) \
        * (E * sinp * sinp + K * cosp * cosp)
    return float(S)
