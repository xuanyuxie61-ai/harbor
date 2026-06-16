"""
trigonometric_orbit_interpolator.py
===================================
Trigonometric Lagrange interpolation of halo orbit coordinates.

Dark matter particles on quasi-periodic orbits inside a smooth halo
potential trace out trajectories r(t), phi(t) that are naturally
approximated by trigonometric polynomials.  For non-uniformly sampled
data (e.g. outputs from an adaptive N-body integrator) we use the
trigonometric Lagrange basis of Austin & Trefethen (2017):

    L_j(x) = prod_{k != j} sin((x - x_k)/2) / sin((x_j - x_k)/2).

The resulting interpolant is periodic and spectrally accurate for
smooth orbits.  Velocities are obtained by analytic differentiation
of the basis.
"""

from __future__ import annotations
import math
from typing import Callable, Tuple

import numpy as np


def trig_lagrange_basis(x: np.ndarray, xd: np.ndarray, j: int) -> np.ndarray:
    """Evaluate the j-th trigonometric Lagrange basis function at x."""
    nd = xd.size
    num = np.ones_like(x, dtype=float)
    for k in range(nd):
        if k == j:
            continue
        num = num * np.sin(0.5 * (x - xd[k]))
    den = 1.0
    for k in range(nd):
        if k == j:
            continue
        den = den * float(np.sin(0.5 * (xd[j] - xd[k])))
    if abs(den) < 1e-300:
        return np.zeros_like(x)
    return num / den


def trig_interp_lagrange(xd: np.ndarray, yd: np.ndarray,
                         xi: np.ndarray) -> np.ndarray:
    """Trigonometric Lagrange interpolation of data (xd, yd) at xi."""
    nd = xd.size
    yi = np.zeros_like(xi, dtype=float)
    for j in range(nd):
        yi = yi + yd[j] * trig_lagrange_basis(xi, xd, j)
    return yi


def trig_interp_derivative(xd: np.ndarray, yd: np.ndarray,
                           xi: np.ndarray) -> np.ndarray:
    """Analytic derivative of the trigonometric Lagrange interpolant.

    Uses the logarithmic-derivative form

        d/dx sin((x - a)/2) = (1/2) cos((x - a)/2)
    """
    nd = xd.size
    dyi = np.zeros_like(xi, dtype=float)
    for j in range(nd):
        # Precompute denominator for basis j
        den = 1.0
        for k in range(nd):
            if k == j:
                continue
            den = den * float(np.sin(0.5 * (xd[j] - xd[k])))
        if abs(den) < 1e-300:
            continue
        # Derivative = sum_{m != j} (1/2) cot((x - x_m)/2) * basis_j(x)
        #            - sum_{m != j} (1/2) cot((x_j - x_m)/2) * basis_j(x)
        # Simpler: d/dx [ prod_{k!=j} sin((x-x_k)/2) ] / den
        # = basis_j(x) * sum_{m!=j} (1/2) cot((x - x_m)/2)
        basis = np.ones_like(xi)
        for k in range(nd):
            if k == j:
                continue
            basis = basis * np.sin(0.5 * (xi - xd[k]))
        basis = basis / den
        cot_sum = np.zeros_like(xi)
        for m in range(nd):
            if m == j:
                continue
            s = np.sin(0.5 * (xi - xd[m]))
            c = np.cos(0.5 * (xi - xd[m]))
            safe = np.where(np.abs(s) < 1e-12, 1e-12, s)
            cot_sum = cot_sum + 0.5 * c / safe
        dyi = dyi + yd[j] * basis * cot_sum
    return dyi


# ---------- Orbit reconstruction ---------------------------------------------

def make_keplerian_orbit(a: float, ecc: float, period: float,
                         n_samples: int = 17
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (t, r, phi) samples of a quasi-Keplerian orbit with
    small radial epicyclic modulation mimicking a perturbed halo orbit.

    Radial frequency Omega_r and azimuthal frequency Omega_phi differ
    slightly, producing rosette-like precession.
    """
    Omega_r   = 2.0 * math.pi / period
    Omega_phi = Omega_r * (1.0 - 0.03 * ecc)    # mild apsidal precession
    t = np.linspace(0.0, period, n_samples, endpoint=False)
    r = a * (1.0 - ecc * np.cos(Omega_r * t))
    phi = Omega_phi * t + 0.1 * ecc * np.sin(Omega_r * t)
    return t, r, phi


def interpolate_orbit(t_data: np.ndarray,
                      r_data: np.ndarray,
                      phi_data: np.ndarray,
                      t_fine: np.ndarray) -> dict:
    """Return interpolated r(t), phi(t) and their time derivatives."""
    r_fine   = trig_interp_lagrange(t_data, r_data, t_fine)
    phi_fine = trig_interp_lagrange(t_data, phi_data, t_fine)
    dr_dt    = trig_interp_derivative(t_data, r_data, t_fine)
    dphi_dt  = trig_interp_derivative(t_data, phi_data, t_fine)
    return dict(r=r_fine, phi=phi_fine, dr_dt=dr_dt, dphi_dt=dphi_dt)


def self_check() -> dict:
    """Sanity check: interpolate a known orbit and report max error."""
    t, r, phi = make_keplerian_orbit(a=1.0, ecc=0.1, period=2.0 * math.pi,
                                     n_samples=17)
    t_fine = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
    out = interpolate_orbit(t, r, phi, t_fine)
    a, ecc, period = 1.0, 0.1, 2.0 * math.pi
    Omega_r = 2.0 * math.pi / period
    r_exact = a * (1.0 - ecc * np.cos(Omega_r * t_fine))
    err = float(np.max(np.abs(out["r"] - r_exact)))
    return dict(err_r=err, n_data=t.size, n_fine=t_fine.size)
