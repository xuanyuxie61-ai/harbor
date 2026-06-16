"""
calving_physics.py - Fission fragment recycling and mass-shedding dynamics.

Adapted from 1044_nicsar2_FootlooseCalvingMechanism (elastic beam / ODE
shooting for ice shelf calving).  Re-purposed for fission fragment
distribution and mass-shedding from super-heavy nuclei undergoing
asymmetric fission in the r-process.

Elastic beam analogy:  the fissioning nucleus deforms along an axis,
described by a width function  w(x)  satisfying the beam equation
  B w''''(x) + k w(x) = 0
with boundary conditions at contact point.  The "calving" (fragment
separation) occurs at the first node of w(x).

This yields an asymmetric mass split  (A_L, A_H)  that feeds back into
the r-process network via fission cycling.
"""
from __future__ import annotations
import math
import numpy as np
from typing import Tuple, List
from physical_constants import SYMMETRY_COEFF, SURFACE_ENERGY_COEFF, TINY


def bending_stiffness(A: int, E_nuclear: float = 200.0, nu_nuclear: float = 0.35
                      ) -> float:
    """
    Effective bending stiffness of a deforming nucleus.
    B = E h^3 / (12 (1 - nu^2))
    h ~ R_0 A^{1/3}  with R_0 = 1.25 fm.
    """
    R_fm = 1.25 * max(A, 1) ** (1.0 / 3.0)
    h = R_fm * 1e-13                    # cm
    return E_nuclear * 1.0e33 * h ** 3 / (12.0 * (1.0 - nu_nuclear ** 2))


def buoyancy_wavelength(B: float, k_subgrade: float = 1.0e28) -> float:
    """Theoretical buoyancy wavelength  lw = (B / k)^{1/4}."""
    if k_subgrade <= 0.0:
        return 0.0
    return (B / k_subgrade) ** 0.25


def fission_fragment_masses(A_parent: int, Z_parent: int,
                            B: float = None) -> Tuple[int, int]:
    """
    Predict asymmetric fission fragment masses by finding the first node
    of the elastic beam solution.
      w(x) = exp(-x/lw) cos(x/lw)     (on semi-infinite domain)
    First node at  x_1 = (pi/2) lw.
    Light fragment mass fraction ~ x_1 / (2 R_parent).
    """
    if B is None:
        B = bending_stiffness(A_parent)
    lw = buoyancy_wavelength(B, k_subgrade=1.0e28)
    R_fm = 1.25 * max(A_parent, 1) ** (1.0 / 3.0)
    R_cm = R_fm * 1e-13
    if R_cm <= 0.0:
        return A_parent // 2, A_parent - A_parent // 2
    x1 = (math.pi / 2.0) * max(lw, 1e-30)
    frac = min(0.5, x1 / (2.0 * R_cm + TINY))
    frac = max(0.15, min(frac, 0.45))
    A_L = max(1, int(round(frac * A_parent)))
    A_H = A_parent - A_L
    return A_L, A_H


def fission_yield_distribution(A_parent: int, Z_parent: int,
                               n_points: int = 21) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the fission yield distribution  Y(A)  over a range of fragment
    masses, using a double-Gaussian (asymmetric) + single-Gaussian
    (symmetric) decomposition.
    """
    A_L_peak, A_H_peak = fission_fragment_masses(A_parent, Z_parent)
    sigma_asym = max(3.0, 0.05 * A_parent)
    sigma_sym = max(4.0, 0.07 * A_parent)
    A_center = A_parent // 2
    A_min = max(1, A_center - A_parent // 3)
    A_max = min(A_parent - 1, A_center + A_parent // 3)
    A_grid = np.linspace(A_min, A_max, n_points)
    Y = np.zeros(n_points, dtype=np.float64)
    for i, A in enumerate(A_grid):
        # Asymmetric peaks
        y_L = math.exp(-0.5 * ((A - A_L_peak) / sigma_asym) ** 2)
        y_H = math.exp(-0.5 * ((A - A_H_peak) / sigma_asym) ** 2)
        # Symmetric peak (suppressed for actinides)
        y_S = 0.3 * math.exp(-0.5 * ((A - A_center) / sigma_sym) ** 2)
        Y[i] = y_L + y_H + y_S
    s = np.sum(Y)
    if s > 0.0:
        Y = Y / s
    return A_grid, Y


def shooting_method(target_A_L: int, A_parent: int, Z_parent: int,
                    tol: float = 2.0, max_iter: int = 20) -> int:
    """
    Use a shooting method (from the ice-shelf ODE solver pattern) to
    adjust the stiffness until the predicted light fragment mass matches
    target_A_L.  Returns the adjusted A_parent that gives the target.
    """
    A_try = A_parent
    for _ in range(max_iter):
        A_L, _ = fission_fragment_masses(A_try, Z_parent)
        if abs(A_L - target_A_L) <= tol:
            return A_try
        if A_L < target_A_L:
            A_try = min(A_try + 5, 300)
        else:
            A_try = max(A_try - 5, 50)
    return A_try


def mass_shedding_rate(A: int, Z: int, excitation_MeV: float = 5.0) -> float:
    """
    Rate of neutron emission (mass shedding) from a highly excited nucleus.
    Weisskopf evaporation:  lambda ~ (Gamma_n / hbar) * rho(E - B_n).
    """
    from nuclear_physics import separation_energy, binding_energy
    Sn, _ = separation_energy(A, Z)
    if Sn >= excitation_MeV or Sn <= 0.0:
        return 0.0
    a_level = 0.093 * (A - 1)
    E_star = max(excitation_MeV - Sn, 0.01)
    rho = math.exp(2.0 * math.sqrt(a_level * E_star)) / (12.0 * E_star ** 1.25)
    Gamma_n = 1.0  # MeV typical
    return Gamma_n * rho * 1.0e20


__all__ = [
    "bending_stiffness", "buoyancy_wavelength",
    "fission_fragment_masses", "fission_yield_distribution",
    "shooting_method", "mass_shedding_rate",
]
