"""
distance_position.py - Distances in nuclide space and ejecta trajectory.

Adapted from 306_distance_to_position.  Computes distances between
nuclides (A, Z) in the nuclear chart, and converts distances along the
ejecta trajectory to physical positions / times.
"""
from __future__ import annotations
import math
from typing import List, Tuple

def nuclide_distance(A1: int, Z1: int, A2: int, Z2: int) -> float:
    """
    Euclidean distance between two nuclides in (A, Z) space.
    A-scale stretched by factor 0.1 so Z differences matter.
    """
    dA = (A1 - A2) * 0.1
    dZ = (Z1 - Z2)
    return math.sqrt(dA * dA + dZ * dZ)


def mass_weighted_distance(A1: int, Z1: int, A2: int, Z2: int,
                           Y1: float = 1.0, Y2: float = 1.0) -> float:
    """
    Abundance-weighted distance:
      d = sqrt(Y1 Y2) * euclidean(A1, Z1, A2, Z2)
    """
    return math.sqrt(max(0.0, Y1 * Y2)) * nuclide_distance(A1, Z1, A2, Z2)


def distance_to_position(dist_Mpc: float, H0: float = 67.4
                         ) -> Tuple[float, float]:
    """
    Convert comoving distance (Mpc) to lookback time and redshift.
    Approximate:  d = c z / H0  for z << 1.
    Returns (t_lookback_s, z).
    """
    from physical_constants import C_LIGHT
    # H0 in s^-1:  H0 [km/s/Mpc] * (1000 m/km) / (3.086e22 m/Mpc)
    H0_s = H0 * 1.0e3 / 3.085677581491367e22
    z = dist_Mpc * 3.085677581491367e24 * H0_s / C_LIGHT
    t_H = 1.0 / H0_s
    t_lookback = t_H * z / (1.0 + z) if z > 0 else 0.0
    return t_lookback, z


def ejecta_trajectory(t: float, v_ej: float = 0.1, M_ej: float = 0.05,
                      rho0: float = 1.0e10, t0: float = 1.0e-3
                      ) -> Tuple[float, float, float]:
    """
    Homologously expanding ejecta trajectory:
      r(t) = v_ej c t
      rho(t) = M_ej M_sun / (4/3 pi r^3)
      T(t) = T0 (t0/t)  (adiabatic, gamma = 4/3)
    Returns (r_cm, rho_g_ccm, T9).
    """
    from physical_constants import C_LIGHT, M_SUN, M_U
    r = v_ej * C_LIGHT * max(t, 1e-30)
    vol = (4.0 / 3.0) * math.pi * r ** 3 if r > 0 else 1.0
    rho = M_ej * M_SUN / vol
    # Temperature:  T0 ~ 1 MeV at t0 ~ 1 ms  -> T ~ 11.6 GK
    T0_K = 1.16e10
    T_K = T0_K * (t0 / max(t, 1e-30))
    T9 = T_K / 1.0e9
    return r, rho, T9


def path_length(points: List[Tuple[float, float]]) -> float:
    """Total path length along a sequence of (A, Z) points."""
    s = 0.0
    for i in range(len(points) - 1):
        s += nuclide_distance(int(points[i][0]), int(points[i][1]),
                              int(points[i+1][0]), int(points[i+1][1]))
    return s


def nearest_nuclide(A: int, Z: int,
                    candidates: List[Tuple[int, int]]) -> Tuple[int, int]:
    """Return the nuclide in candidates nearest to (A, Z)."""
    best = None
    best_d = 1e300
    for (Ac, Zc) in candidates:
        d = nuclide_distance(A, Z, Ac, Zc)
        if d < best_d:
            best_d = d
            best = (Ac, Zc)
    return best


__all__ = [
    "nuclide_distance", "mass_weighted_distance",
    "distance_to_position", "ejecta_trajectory",
    "path_length", "nearest_nuclide",
]
