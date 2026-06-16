"""
conformal_mapping.py  --  Complex-energy mapping for nuclear resonances
=====================================================================
Fused seeds:
    611_joukowsky_transform  -- Joukowsky conformal map z -> z + c^2 / z
    184_circle_segment       -- circular-arc geometry

Nuclear resonances (quasi-bound states) correspond to poles of the S-matrix
at complex energies

    E_R = E_0 - i Gamma / 2

where E_0 is the resonance position and Gamma is the width. Direct numerical
solution on the real axis cannot access these poles, but conformal mapping
of the complex momentum plane rotates the bound-state cut onto the resonance
pole.

We implement three mappings:

1. Joukowsky map (complex z-plane):
       w = z + (k_0 / k)^2 * (1 / z)
   which maps the unit circle |z| = 1 onto a segment of the real axis
   and the exterior onto the cut plane.

2. Momentum-plane exponential map (for Berggren / Gamow states):
       k = kappa e^{i theta},   theta in [0, pi/2]
   rotates the real-k axis onto a complex contour that captures the
   Gamow-Siegert resonance poles.

3. Berggren completeness relation:
       sum_n u_n(r) u_n(r') / (2 k_n)  +  int_L u(k, r) u(k, r') / (2 k) dk
       = delta(r - r')
   where L is the rotated contour in the complex-k plane.

We also compute the Gamow penetration factor
    P_L(eta) = k R / (F_L^2(eta, kR) + G_L^2(eta, kR))
and the resonance width
    Gamma = 2 P_L W_sp

References:
    Berggren, Nucl. Phys. A 109 (1968) 265
    Gamow, Z. Phys. 51 (1928) 204
    Siegert, Phys. Rev. 56 (1939) 750
"""

from __future__ import annotations
import math
import cmath
import numpy as np
from nuclear_constants import HBAR_C, R0_FM, PI, R_EPSILON


# ======================================================================
#  Joukowsky map and inverse
# ======================================================================
def joukowsky_forward(z: complex, c: float = 1.0) -> complex:
    """w = z + c^2 / z (Joukowsky conformal map).

    Maps |z| = c onto the segment [-2c, 2c] of the real axis.
    """
    if abs(z) < R_EPSILON:
        return complex(float("inf"), 0.0)
    return z + (c * c) / z


def joukowsky_inverse(w: complex, c: float = 1.0, exterior: bool = True) -> complex:
    """Inverse Joukowsky: z = (w +/- sqrt(w^2 - 4 c^2)) / 2.

    exterior=True returns the root with |z| >= c, else |z| <= c.
    """
    disc = cmath.sqrt(w * w - 4.0 * c * c)
    z_plus = (w + disc) / 2.0
    z_minus = (w - disc) / 2.0
    if exterior:
        return z_plus if abs(z_plus) >= abs(z_minus) else z_minus
    return z_minus if abs(z_minus) <= abs(z_plus) else z_plus


def joukowsky_derivative(z: complex, c: float = 1.0) -> complex:
    """dw / dz = 1 - c^2 / z^2."""
    if abs(z) < R_EPSILON:
        return complex(float("inf"), 0.0)
    return 1.0 - (c * c) / (z * z)


def joukowsky_map_energy(E_complex: complex, E_threshold: float = 0.0, m_red: float = 500.0) -> complex:
    """Map a complex energy E to the Joukowsky w-plane.

    Uses the momentum k = sqrt(2 m (E - E_th)) / hbar and the map
        w = k / k_0 + (k_0 / k) / 2  (symmetric Joukowsky on k/k_0)
    where k_0 is a reference momentum.
    """
    if E_complex == E_threshold:
        return complex(0.0, 0.0)
    k_sq = 2.0 * m_red * (E_complex - E_threshold) / (HBAR_C ** 2)
    k = cmath.sqrt(k_sq)
    k_0 = 1.0  # reference scale in fm^{-1}
    z = k / k_0
    return joukowsky_forward(z, c=1.0)


# ======================================================================
#  Complex-momentum contour (Berggren / Gamow rotation)
# ======================================================================
def complex_momentum_contour(
    kappa: float, theta_max: float, n_points: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Generate a contour k = kappa e^{i theta} for theta in [0, theta_max].

    Used in the Berggren completeness relation to capture Gamow poles.
    Returns (k_array, dk_array) where dk is the differential along the contour.
    """
    theta = np.linspace(0.0, theta_max, n_points)
    k = kappa * np.exp(1j * theta)
    dk = 1j * kappa * np.exp(1j * theta) * (theta[1] - theta[0])
    return k, dk


def resonant_energy_from_k(k_complex: complex, m_red: float, E_threshold: float = 0.0) -> complex:
    """Convert complex momentum to complex energy: E = hbar^2 k^2 / (2 m) + E_th."""
    return (HBAR_C ** 2) * (k_complex ** 2) / (2.0 * m_red) + E_threshold


def gamow_penetration_factor(L: int, eta: complex, rho: complex) -> complex:
    r"""Coulomb penetration factor P_L for complex arguments.

    P_L(eta, rho) = rho / (F_L^2 + G_L^2)

    For the small-scale problem we use the asymptotic form:
        F_L + i G_L ~ exp(i (rho - eta ln(2 rho) - L pi/2 + sigma_L))
    where sigma_L is the Coulomb phase.
    """
    # Coulomb phase sigma_L = arg Gamma(L+1 + i eta)
    sigma_L = 0.0
    try:
        from scipy.special import gamma as gamma_complex
        g = gamma_complex(L + 1.0 + 1j * eta)
        sigma_L = cmath.phase(g)
    except (OverflowError, ValueError, ImportError):
        sigma_L = 0.0
    phase = rho - eta * cmath.log(2.0 * rho + R_EPSILON) - L * PI / 2.0 + sigma_L
    fg = cmath.exp(1j * phase)
    F_L = fg.real
    G_L = fg.imag
    denom = F_L * F_L + G_L * G_L
    if abs(denom) < R_EPSILON:
        return complex(0.0, 0.0)
    return rho / denom


def resonance_width(
    L: int, eta: complex, rho: complex, W_sp: float, m_red: float,
) -> complex:
    """Single-particle resonance width Gamma = 2 P_L W_sp.

    W_sp is the spectroscopic factor (dimensionless, 0 < W_sp <= 1).
    """
    P_L = gamow_penetration_factor(L, eta, rho)
    return 2.0 * P_L * W_sp


# ======================================================================
#  Berggren completeness check
# ======================================================================
def berggren_sum_rule(
    k_bound: list, k_resonant: list, k_contour: np.ndarray,
    r_test: float, m_red: float,
) -> dict:
    """Evaluate the Berggren completeness sum at a test radius.

    sum_n u_n(r)^2 / (2 k_n)  +  integral over contour
    should approximate delta(r - r') at r = r'.

    For a real test we evaluate a proxy: sum of 1 / (2 k_n) contributions.
    """
    sum_bound = 0.0 + 0j
    for k in k_bound:
        if abs(k) > R_EPSILON:
            sum_bound += 1.0 / (2.0 * k)
    sum_res = 0.0 + 0j
    for k in k_resonant:
        if abs(k) > R_EPSILON:
            sum_res += 1.0 / (2.0 * k)
    sum_cont = 0.0 + 0j
    for k in k_contour:
        if abs(k) > R_EPSILON:
            sum_cont += 1.0 / (2.0 * k)
    total = sum_bound + sum_res + sum_cont
    return {
        "bound_contribution": complex(sum_bound),
        "resonant_contribution": complex(sum_res),
        "contour_contribution": complex(sum_cont),
        "total": complex(total),
        "abs_total": abs(total),
    }


# ======================================================================
#  S-matrix pole finder (toy)
# ======================================================================
def s_matrix_pole_scan(
    k_min: float, k_max: float, n_points: int,
    phase_shift_fn: callable,
) -> list:
    """Scan the real-k axis for rapid phase-shift rises indicating a resonance.

    A resonance manifests as delta_L passing through pi/2 (mod pi).
    Returns list of (k_res, delta, d_delta_dk) at candidates.
    """
    ks = np.linspace(k_min, k_max, n_points)
    deltas = np.array([phase_shift_fn(k) for k in ks])
    dk = ks[1] - ks[0]
    candidates = []
    for i in range(1, len(deltas) - 1):
        d_ddelta = (deltas[i + 1] - deltas[i - 1]) / (2.0 * dk)
        # Pass through pi/2 with positive derivative
        if (deltas[i - 1] < PI / 2.0 <= deltas[i] or deltas[i - 1] > PI / 2.0 > deltas[i]) and d_ddelta > 0:
            candidates.append({
                "k_res": float(ks[i]),
                "delta_rad": float(deltas[i]),
                "d_delta_dk": float(d_ddelta),
            })
    return candidates
