"""
spacetime_geometry.py — Schwarzschild background and Regge-Wheeler potential.

Core mathematical objects for perturbations of a non-rotating black hole:

  1. Tortoise coordinate
         r_* = r + 2M ln(r/(2M) - 1)
     which maps r in (2M, +inf) to r_* in (-inf, +inf) and absorbs
     the coordinate singularity at the horizon.

  2. Regge-Wheeler potential for axial (odd-parity) perturbations:
         V_l(r) = (1 - 2M/r) [ l(l+1)/r^2 + (1-s^2) 2M / r^3 ]
     with spin-weight s = 2 for gravitational perturbations.

  3. Zerilli potential for polar (even-parity) perturbations is also
     provided for completeness.

  4. The effective potential for radial null geodesics and the
     photon-sphere radius r_ph = 3M.

Every function is vectorised with NumPy and enforces domain guards so that
calls at or inside the horizon produce well-defined limiting values rather
than NaN/Inf.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, Union

from physics_constants import PI, BinaryParameters


# ---------------------------------------------------------------------------
#  Tortoise coordinate
# ---------------------------------------------------------------------------
def tortoise_coordinate(r: Union[float, np.ndarray],
                        M: float,
                        eps: float = 1.0e-12) -> Union[float, np.ndarray]:
    """Compute r_* = r + 2M ln(r/(2M) - 1).

    Parameters
    ----------
    r   : Schwarzschild radial coordinate (same units as M).
    M   : Black-hole mass in geometric units.
    eps : safety margin above the horizon; r = 2M + eps gives a finite r_*.
    """
    r_safe = np.maximum(np.asarray(r, dtype=float), 2.0 * M + eps)
    return r_safe + 2.0 * M * np.log(r_safe / (2.0 * M) - 1.0)


def inverse_tortoise(rstar: Union[float, np.ndarray],
                     M: float,
                     n_iter: int = 30,
                     tol: float = 1.0e-14) -> Union[float, np.ndarray]:
    """Newton inversion of r_*(r) to recover r(r_*).

    Initial guess: r = r_* for r_* >> 2M,  r = 2M + exp((r_*-2M)/(2M)) near horizon.
    Newton step:  r_{n+1} = r_n - (r_*(r_n) - r_*target) / (1 - 2M/r_n)^{-1}
    since dr_*/dr = 1 / (1 - 2M/r).
    """
    rs = np.asarray(rstar, dtype=float)
    # hybrid initial guess
    r = np.where(rs > 4.0 * M,
                 rs,
                 2.0 * M + np.exp((rs - 4.0 * M) / (2.0 * M)) * 2.0 * M)
    r = np.maximum(r, 2.0 * M + 1.0e-14)
    for _ in range(n_iter):
        f = r + 2.0 * M * np.log(r / (2.0 * M) - 1.0) - rs
        df = 1.0 / (1.0 - 2.0 * M / r)
        dr = -f * (1.0 - 2.0 * M / r)
        r = r + dr
        r = np.maximum(r, 2.0 * M + 1.0e-14)
        if np.max(np.abs(dr)) < tol:
            break
    return r


# ---------------------------------------------------------------------------
#  Regge-Wheeler (axial) potential
# ---------------------------------------------------------------------------
def regge_wheeler_potential(r: Union[float, np.ndarray],
                            M: float,
                            ell: int = 2,
                            s: int = 2) -> Union[float, np.ndarray]:
    """Evaluate the spin-weighted Regge-Wheeler potential.

        V_l(r) = (1 - 2M/r) [ l(l+1)/r^2 + (1 - s^2) 2M / r^3 ]

    For gravitational perturbations s = 2 and the second term simplifies.
    Inside the horizon the potential is clipped to zero to keep the PDE
    well-posed under the standard outgoing-wave boundary conditions.
    """
    r_safe = np.maximum(np.asarray(r, dtype=float), 2.0 * M + 1.0e-12)
    lap = float(ell) * float(ell + 1)
    f = 1.0 - 2.0 * M / r_safe
    V = f * (lap / r_safe ** 2 + (1.0 - s * s) * 2.0 * M / r_safe ** 3)
    # enforce causal cutoff: V(r<2M) = 0
    V = np.where(r_safe > 2.0 * M + 1.0e-10, V, 0.0)
    return V


def regge_wheeler_potential_rstar(rstar: Union[float, np.ndarray],
                                  M: float,
                                  ell: int = 2) -> Union[float, np.ndarray]:
    """Regge-Wheeler potential expressed as a function of the tortoise coordinate."""
    r = inverse_tortoise(rstar, M)
    return regge_wheeler_potential(r, M, ell)


# ---------------------------------------------------------------------------
#  Zerilli (polar) potential — retained for binary-parity studies
# ---------------------------------------------------------------------------
def zerilli_potential(r: Union[float, np.ndarray],
                      M: float,
                      ell: int = 2) -> Union[float, np.ndarray]:
    """Zerilli potential for even-parity perturbations.

        lambda = (ell - 1)(ell + 2) / 2
        V_Z(r) = 2 f(r) [ lambda^2 (lambda+1) r^3 + 3 lambda M r^2
                          + 9 M^2 r + 9 M^3 ] / r^3 [ lambda r + 3 M ]^2
    """
    r_safe = np.maximum(np.asarray(r, dtype=float), 2.0 * M + 1.0e-12)
    f = 1.0 - 2.0 * M / r_safe
    lam = 0.5 * (ell - 1) * (ell + 2)
    num = (lam ** 2 * (lam + 1.0) * r_safe ** 3
           + 3.0 * lam * M * r_safe ** 2
           + 9.0 * M ** 2 * r_safe
           + 9.0 * M ** 3)
    den = r_safe ** 3 * (lam * r_safe + 3.0 * M) ** 2
    V = 2.0 * f * num / den
    V = np.where(r_safe > 2.0 * M + 1.0e-10, V, 0.0)
    return V


# ---------------------------------------------------------------------------
#  Photon sphere and null geodesics
# ---------------------------------------------------------------------------
def photon_sphere_radius(M: float) -> float:
    """Radius of the unstable circular null orbit  r_ph = 3M."""
    return 3.0 * M


def light_ring_frequency(M: float) -> float:
    """Quasi-normal-mode real-part proxy: omega_QNM ~ l / (3 sqrt(3) M)."""
    return float(ell := 2) / (3.0 * math.sqrt(3.0) * M)


# ---------------------------------------------------------------------------
#  Grid construction in r_*
# ---------------------------------------------------------------------------
def build_tortoise_grid(M: float,
                        N: int,
                        r_min_factor: float = 1.0 + 1.0e-6,
                        r_max_factor: float = 500.0) -> Tuple[np.ndarray, np.ndarray, float]:
    """Build a uniform grid in r_* that covers r in (2M r_min_factor, 2M r_max_factor).

    Returns
    -------
    rstar : (N,) tortoise coordinates
    r     : (N,) Schwarzschild radii
    drstar: grid spacing in r_*
    """
    r_min = 2.0 * M * r_min_factor
    r_max = 2.0 * M * r_max_factor
    rstar_min = float(tortoise_coordinate(r_min, M))
    rstar_max = float(tortoise_coordinate(r_max, M))
    rstar, drstar = np.linspace(rstar_min, rstar_max, N, retstep=True)
    r = inverse_tortoise(rstar, M)
    return rstar, r, float(drstar)


# ---------------------------------------------------------------------------
#  Potential sampling on a pre-built grid
# ---------------------------------------------------------------------------
def sample_potential_on_grid(r: np.ndarray, M: float, ell: int = 2,
                             potential_kind: str = "regge_wheeler") -> np.ndarray:
    """Evaluate V(r) on a pre-built radial grid."""
    if potential_kind == "regge_wheeler":
        return regge_wheeler_potential(r, M, ell, s=2)
    elif potential_kind == "zerilli":
        return zerilli_potential(r, M, ell)
    else:
        raise ValueError(f"Unknown potential kind: {potential_kind}")
