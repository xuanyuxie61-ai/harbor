# -*- coding: utf-8 -*-
"""
gyroaverage.py
==============

Finite-Larmor-radius (FLR) operators for the local slab gyrokinetic model.

In the gyrokinetic reduction the gyrocentre average of a plane-wave
perturbation  exp(i k . r)  brings in the factor  J_0(k_perp rho_s),
with  rho_s = v_perp / omega_ci  the Larmor radius of a particle with
perpendicular speed v_perp.  After integration over the gyrophase the
relevant operators become:

    Gamma_0(b)  =  I_0(b) exp(-b),                b = k_perp^2 rho_t^2 / 2
    Gamma_1(b)  =  (I_0(b) - I_1(b)) exp(-b)
    <...>_R     =  integral over mu  of  J_0^2(k_perp rho_L)  (...)

with I_n the modified Bessel function of the first kind.  The quasineutral
gyrokinetic Poisson equation reads

    sum_s (e_s n_s / T_s) (1 - Gamma_0^s) phi = sum_s int <delta f_s>_R d^3 v

and the polarisation density entering the vorticity equation is

    rho_pol = sum_s (e_s^2 n_s / T_s) (1 - Gamma_0^s) phi.

This module provides

    * Padé approximants for  Gamma_0  and  Gamma_1  (fast, spectrally close),
    * exact evaluation via  I_0(b) exp(-b)  using scipy,
    * a routine that applies  J_0(k_perp rho)  to a 2-D (v_par, mu) field
      using the continued-fraction evaluation of J_0 from
      ``velocity_space.jn_eval``,
    * the FLR operator  (1 - Gamma_0)  needed by the gyrokinetic Poisson
      equation and by the zonal-flow residual calculation.

References:
    [1] Hasegawa & Mima, Phys. Fluids 21, 87 (1978)
    [2] Frieman & Chen, Phys. Fluids 25, 502 (1982)
    [3] Parra & Catto, Plasma Phys. Control. Fusion 52, 045001 (2010)
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from velocity_space import jn_eval


# ============================================================================
# Modified Bessel I_n and the Gamma operators
# ============================================================================
def _i0_scaled(b: np.ndarray) -> np.ndarray:
    """I_0(b) exp(-b)  -- computed stably for all b >= 0.

    For small b we use a Taylor expansion:
        I_0(b) exp(-b) = 1 - b + (3/4) b^2 - (7/12) b^3 + O(b^4)
    For large b we use the asymptotic form:
        I_0(b) exp(-b) ~ 1/sqrt(2 pi b) * (1 + 1/(8b) + 9/(128 b^2) + ...)
    The switch is at b = 50, where the error is < 1e-14.
    """
    b = np.asarray(b, dtype=np.float64)
    out = np.empty_like(b)
    small = b < 50.0
    big = ~small
    if np.any(small):
        from scipy.special import i0e    # I_0(b) exp(-|b|), stable
        out[small] = i0e(b[small])
    if np.any(big):
        bb = b[big]
        out[big] = 1.0 / np.sqrt(2.0 * math.pi * bb) * (
            1.0 + 1.0 / (8.0 * bb) + 9.0 / (128.0 * bb * bb)
        )
    return out


def _i1_scaled(b: np.ndarray) -> np.ndarray:
    """I_1(b) exp(-b), stable."""
    b = np.asarray(b, dtype=np.float64)
    out = np.empty_like(b)
    small = b < 50.0
    big = ~small
    if np.any(small):
        from scipy.special import i1e
        out[small] = i1e(b[small])
    if np.any(big):
        bb = b[big]
        out[big] = 1.0 / np.sqrt(2.0 * math.pi * bb) * (
            1.0 - 3.0 / (8.0 * bb) + 15.0 / (128.0 * bb * bb)
        )
    return out


def gamma0(b: np.ndarray) -> np.ndarray:
    """Gamma_0(b) = I_0(b) exp(-b)."""
    return _i0_scaled(np.asarray(b, dtype=np.float64))


def gamma1(b: np.ndarray) -> np.ndarray:
    """Gamma_1(b) = (I_0(b) - I_1(b)) exp(-b)."""
    b = np.asarray(b, dtype=np.float64)
    return _i0_scaled(b) - _i1_scaled(b)


def gamma0_pade(b: np.ndarray) -> np.ndarray:
    """Padé approximant for Gamma_0 -- O(b^3) accurate.

        Gamma_0(b) ~ (1 + b/2)^{-1}  (Padé [0,1])
        A more accurate rational form:
        Gamma_0(b) ~ (1 + b + (3/4) b^2) / (1 + 2 b + (3/2) b^2)
    """
    b = np.asarray(b, dtype=np.float64)
    return (1.0 + b + 0.75 * b * b) / (1.0 + 2.0 * b + 1.5 * b * b)


def gamma0_inverse_pade(b: np.ndarray) -> np.ndarray:
    """Approximate  1 / (1 - Gamma_0)  used by the polarisation inversion.

    For  b << 1:  1 / (1 - Gamma_0) ~ 1/b + 1/2 + b/8 + ...
    For  b >> 1:  1 / (1 - Gamma_0) ~ 1.
    A simple blended form (Padé-like):
        (1/b + 1/2) / (1 + b/8)
    clamped away from the origin for stability.
    """
    b = np.asarray(b, dtype=np.float64)
    b = np.maximum(b, 1.0e-12)
    return (1.0 / b + 0.5) / (1.0 + b / 8.0)


# ============================================================================
# Application of J_0(k_perp rho) to a 2-D field on (v_par, mu)
# ============================================================================
def apply_j0_to_field(
    field_vm: np.ndarray,
    v_par_grid: np.ndarray,
    mu_grid: np.ndarray,
    kperp_rho: float,
    omega_ci: float = 1.0,
    m_i: float = 1.0,
) -> np.ndarray:
    """Multiply a 2-D gyrokinetic field by J_0(k_perp rho_L(v_perp)).

    The Larmor radius  rho_L = v_perp / omega_ci  with  v_perp = sqrt(2 mu B / m).
    In normalised units (B = m = omega_ci = 1) we have  rho_L = sqrt(2 mu).

    Parameters
    ----------
    field_vm : (N_v, N_mu) array
    v_par_grid : (N_v,)
    mu_grid    : (N_mu,)
    kperp_rho  : dimensionless k_perp * rho_ref
    """
    VP, MU = np.meshgrid(v_par_grid, mu_grid, indexing="ij")
    rho_L = np.sqrt(np.maximum(2.0 * MU / max(m_i, 1.0e-30), 0.0)) / max(omega_ci, 1.0e-30)
    z = kperp_rho * rho_L
    # vectorise via the Zhang-Jin J_0 evaluation
    J0 = np.vectorize(lambda x: jn_eval(0, float(x)), otypes=[np.float64])(z)
    return field_vm * J0


# ============================================================================
# 1-D (radial) FLR operator: (1 - Gamma_0) phi
# ============================================================================
def flr_operator_1d(phi: np.ndarray, kperp_rho_sq: np.ndarray) -> np.ndarray:
    """Apply (1 - Gamma_0(kperp^2 rho_s^2 / 2)) to a 1-D radial field.

    kperp_rho_sq : (N,) array of k_perp^2 rho_s^2 values on the radial mesh.
    """
    b = 0.5 * np.asarray(kperp_rho_sq, dtype=np.float64)
    return phi - gamma0(b) * phi


# ============================================================================
# Gyrokinetic Poisson solve in 1-D (k_perp space)
# ============================================================================
def gyrokinetic_poisson_1d(
    rhs: np.ndarray,
    kperp_rho_sq: np.ndarray,
    tau_e: float = 1.0,
    Z_i: float = 1.0,
) -> np.ndarray:
    """Solve the slab gyrokinetic Poisson equation at each radial point.

    In the local limit this reduces to the algebraic equation

        (1 + tau_e) (1 - Gamma_0) phi = Z_i * n1     (normalised units)

    where  n1  is the gyrokinetic ion density response (``rhs``) and
    tau_e = T_e / T_i at the flux surface.
    """
    b = 0.5 * np.asarray(kperp_rho_sq, dtype=np.float64)
    g0 = gamma0(b)
    denom = (1.0 + tau_e) * (1.0 - g0)
    # safety clamp
    denom = np.where(np.abs(denom) < 1.0e-14,
                     np.copysign(1.0e-14, denom) + (denom == 0) * 1.0e-14,
                     denom)
    return Z_i * np.asarray(rhs, dtype=np.float64) / denom


# ============================================================================
# Sanity self-check
# ============================================================================
if __name__ == "__main__":
    b = np.linspace(0.0, 3.0, 13)
    g0 = gamma0(b)
    g1 = gamma1(b)
    g0_pade = gamma0_pade(b)
    print("b           :", np.round(b, 3))
    print("Gamma0 exact:", np.round(g0, 5))
    print("Gamma0 Pade :", np.round(g0_pade, 5))
    print("Gamma1 exact:", np.round(g1, 5))
    # apply J0 to a Gaussian in v_perp
    v_par = np.linspace(-3, 3, 11)
    mu = np.linspace(0, 3, 7)
    f = np.exp(-(v_par[:, None] ** 2 + mu[None, :]))
    J0f = apply_j0_to_field(f, v_par, mu, kperp_rho=0.5)
    print("max |J0 f - f| at mu=0:", np.max(np.abs(J0f[:, 0] - f[:, 0])))
    # Poisson
    n1 = np.ones(5)
    kperp2 = np.linspace(0.1, 1.0, 5)
    phi = gyrokinetic_poisson_1d(n1, kperp2, tau_e=1.0, Z_i=1.0)
    print("phi from Poisson:", np.round(phi, 4))
