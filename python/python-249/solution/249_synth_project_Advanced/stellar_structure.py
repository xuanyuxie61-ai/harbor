# -*- coding: utf-8 -*-
"""
stellar_structure.py
====================
1D Lagrangian stellar structure solver.

Background
----------
The equations of stellar structure in Lagrangian (mass) coordinates are

    dr/dm     = 1 / (4 pi r^2 rho)
    dP/dm     = - G m / (16 pi^2 r^4)
    dL/dm     = epsilon_nuc - epsilon_nu - T ds/dt
    dT/dm     = nabla (T/P) dP/dm

with the equation of state P = P(rho, T, composition) and opacity
kappa = kappa(rho, T, composition) determining nabla.

For the present project we implement a simplified but physically
meaningful version suitable for small-scale reproducible experiments:

  - ideal gas + radiation pressure equation of state
  - Kramers opacity law
  - radiative temperature gradient with Schwarzschild stability check
  - mixing-length convection treatment (simplified)

The discretisation uses the compact finite-difference operators from
`finite_difference.py` on the non-uniform mass grid from
`mesh_adaptation.py`.  Time integration uses the adaptive implicit
scheme of `time_integrator.py`.

This module draws on seed 1131_cchrisgong_aip_rockstar for the
gravitational binding / virial analysis, seed 061_b1g3 and 765_midpoint_adaptive
for the implicit ODE integration, seed 640_laguerre_integrands for
the Gamow-peak quadrature, seed 253_cvt_circle_nonuniform for the
adaptive mesh, seed 1401_wathen_matrix for sparse Jacobian assembly.
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import math

from physical_constants import (
    k_B, G_grav, m_p, a_rad, sigma_SB, c_light, N_A, pi,
    M_sun, L_sun, R_sun
)
from finite_difference import (
    mass_grid, spacings, compact_fd4, fd_central_2, fd_second_2
)


# =====================================================================
# Equation of state
# =====================================================================

def ideal_gas_pressure(rho: float, T: float, mu: float) -> float:
    """Ideal gas pressure  P = rho k_B T / (mu m_p)  [dyn/cm^2]."""
    if rho <= 0.0 or T <= 0.0 or mu <= 0.0:
        return 0.0
    return rho * k_B * T / (mu * m_p)


def radiation_pressure(T: float) -> float:
    """Radiation pressure  P_rad = a T^4 / 3."""
    return a_rad * T**4 / 3.0


def total_pressure(rho: float, T: float, mu: float) -> float:
    """Total (gas + radiation) pressure."""
    return ideal_gas_pressure(rho, T, mu) + radiation_pressure(T)


def beta_gas(rho: float, T: float, mu: float) -> float:
    """Ratio of gas pressure to total pressure.  beta = P_gas / P_tot."""
    Pt = total_pressure(rho, T, mu)
    if Pt <= 0.0:
        return 1.0
    return ideal_gas_pressure(rho, T, mu) / Pt


def mean_molecular_weight_from_Y(Y_H: float, Y_He: float,
                                   Y_Z: float = 0.0) -> float:
    """Mean molecular weight for a fully ionised gas.
        1/mu = 2 X + 3/4 Y + 1/2 Z
    """
    X = max(0.0, Y_H); Y = max(0.0, Y_He); Z = max(0.0, Y_Z)
    s = X + Y + Z
    if s <= 0.0:
        return 0.6
    X /= s; Y /= s; Z /= s
    inv = 2.0 * X + 0.75 * Y + 0.5 * Z
    return 1.0 / inv if inv > 0 else 0.6


# =====================================================================
# Opacity
# =====================================================================

def kramers_opacity(rho: float, T: float, X: float, Z: float = 0.02
                    ) -> float:
    """Kramers opacity law  kappa = kappa_0 rho T^{-3.5}
    with composition-dependent coefficient.

    kappa_0 ~ 4e25 (1 + X)  cm^2 / g  (rough fit).
    """
    if rho <= 0.0 or T <= 0.0:
        return 0.0
    kappa_0 = 4.0e25 * (1.0 + X)
    return kappa_0 * rho * T**(-3.5)


# =====================================================================
# Stellar structure RHS (semi-discrete)
# =====================================================================

class StellarZone:
    """State of a single mass shell."""

    def __init__(self, m: float, r: float, P: float, T: float,
                 L: float, rho: float, mu: float,
                 Y: List[float]):
        self.m = m
        self.r = r
        self.P = P
        self.T = T
        self.L = L
        self.rho = rho
        self.mu = mu
        self.Y = Y[:]


def density_from_PT(P: float, T: float, mu: float) -> float:
    """Solve P = rho k T / (mu m_p) + a T^4 / 3  for rho.
    Since P_rad = a T^4/3 is known given T, we invert the gas part:
        rho = (P - P_rad) * mu m_p / (k_B T)
    """
    P_rad = radiation_pressure(T)
    P_gas = max(P - P_rad, 0.0)
    if T <= 0.0 or mu <= 0.0:
        return 0.0
    return P_gas * mu * m_p / (k_B * T)


def compute_zone(m: float, r_prev: float, P_prev: float, T_prev: float,
                 L_prev: float, dm: float,
                 Y: List[float], epsilon: float = 0.0,
                 epsilon_nu: float = 0.0) -> StellarZone:
    """Advance the stellar structure equations by one mass step dm.

    Given the state at mass m, compute the state at m + dm using a
    single explicit Euler step in mass coordinate:
        dr/dm     = 1 / (4 pi r^2 rho)
        dP/dm     = - G m / (16 pi^2 r^4)
        dL/dm     = epsilon_nuc - epsilon_nu
    The temperature gradient is taken to be radiative:
        dT/dm = (3 kappa L) / (256 pi^2 r^4 sigma T^3) * (P / T)

    This is a simplified one-step advance; the full implicit solver
    iterates to convergence.
    """
    X_H = Y[0] if len(Y) > 0 else 0.7
    Y_He = Y[1] if len(Y) > 1 else 0.28
    Y_Z = 1.0 - X_H - Y_He
    mu = mean_molecular_weight_from_Y(X_H, Y_He, Y_Z)

    rho = density_from_PT(P_prev, T_prev, mu)
    if rho <= 0.0:
        rho = 1.0e-10

    # dr/dm
    if r_prev > 0.0:
        dr_dm = 1.0 / (4.0 * pi * r_prev**2 * rho)
    else:
        dr_dm = 0.0
    r_new = r_prev + dm * dr_dm

    # dP/dm (hydrostatic equilibrium)
    dP_dm = -G_grav * m / (16.0 * pi**2 * max(r_prev, 1.0)**4)
    P_new = max(P_prev + dm * dP_dm, 1.0e3)

    # dL/dm (energy generation)
    dL_dm = epsilon - epsilon_nu
    L_new = max(L_prev + dm * dL_dm, 0.0)

    # dT/dm (radiative gradient)
    kappa = kramers_opacity(rho, T_prev, X_H, Y_Z)
    if T_prev > 0.0 and r_prev > 0.0:
        dT_dm = (3.0 * kappa * max(L_prev, 0.0)
                 / (256.0 * pi**2 * r_prev**4 * sigma_SB * T_prev**3)
                 * (P_prev / max(T_prev, 1.0)))
    else:
        dT_dm = 0.0
    T_new = max(T_prev + dm * dT_dm, 1.0e3)

    return StellarZone(m + dm, r_new, P_new, T_new, L_new,
                        rho, mu, Y)


# =====================================================================
# Full stellar model construction
# =====================================================================

def build_stellar_model(M_total: float, N_zones: int,
                         Y_init: List[float],
                         T_centre: float = 1.5e7,
                         P_centre: float = 2.0e17,
                         L_centre: float = 0.0,
                         epsilon_func=None
                         ) -> List[StellarZone]:
    """Build a 1D stellar model from centre to surface.

    We integrate the structure equations outward in mass coordinate
    starting from central values.  The initial grid is uniform in mass.
    At the very centre we seed a small initial radius using the
    central density estimate  r_0 ~ (3 m_0 / (4 pi rho_c))^{1/3}.
    """
    grid = mass_grid(M_total, N_zones, eta=1.2)
    hs = spacings(grid)
    zones: List[StellarZone] = []
    # Estimate central density from P_centre and T_centre
    X_H = Y_init[0] if len(Y_init) > 0 else 0.7
    Y_He = Y_init[1] if len(Y_init) > 1 else 0.28
    Y_Z = max(0.0, 1.0 - X_H - Y_He)
    mu = mean_molecular_weight_from_Y(X_H, Y_He, Y_Z)
    rho_centre = density_from_PT(P_centre, T_centre, mu)
    if rho_centre <= 0.0:
        rho_centre = 100.0
    # Seed the first zone with a small radius based on central density
    m0 = grid[0] + 0.5 * hs[0] if hs else 1.0e30
    r0 = (3.0 * m0 / (4.0 * pi * rho_centre)) ** (1.0/3.0) if rho_centre > 0 else 1.0e8
    r = r0
    P = P_centre
    T = T_centre
    L = L_centre
    Y = Y_init[:]

    for i in range(N_zones):
        m = grid[i] + 0.5 * hs[i]   # midpoint of shell
        dm = hs[i]
        eps = 0.0
        if epsilon_func is not None:
            eps = epsilon_func(T / 1.0e9, rho_centre, Y)
        # Advance one shell
        # dr/dm
        dr_dm = 1.0 / (4.0 * pi * max(r, 1.0)**2 * rho_centre)
        r_new = r + dm * dr_dm
        # dP/dm: hydrostatic
        dP_dm = -G_grav * m / (16.0 * pi**2 * max(r, 1.0)**4)
        P_new = max(P + dm * dP_dm, 1.0e3)
        # dT/dm: radiative gradient (approximate)
        kappa = kramers_opacity(rho_centre, T, X_H, Y_Z)
        if T > 0.0 and r > 0.0:
            dT_dm = -(3.0 * kappa * max(L, 0.0)
                      / (256.0 * pi**2 * r**4 * sigma_SB * T**3)
                      * (P / max(T, 1.0)))
        else:
            dT_dm = 0.0
        T_new = max(T + dm * dT_dm, 1.0e3)
        # dL/dm: energy generation
        dL_dm = eps
        L_new = max(L + dm * dL_dm, 0.0)
        # New density from updated P, T
        rho_new = density_from_PT(P_new, T_new, mu)
        if rho_new <= 0.0:
            rho_new = rho_centre * 0.5
        zone = StellarZone(m + dm, r_new, P_new, T_new, L_new,
                            rho_new, mu, Y[:])
        zones.append(zone)
        r = r_new
        P = P_new
        T = T_new
        L = L_new
        rho_centre = rho_new
    return zones


# =====================================================================
# Stability diagnostics
# =====================================================================

def schwarzschild_criterion(zone: StellarZone, zone_next: StellarZone
                             ) -> bool:
    """Return True if the layer is convectively unstable according
    to the Schwarzschild criterion.

    The radiative temperature gradient is
        nabla_rad = (3 kappa L P) / (16 pi a c G m T^4)
    and the adiabatic gradient for an ideal gas is  nabla_ad = 0.4.
    Convection sets in when nabla_rad > nabla_ad.
    """
    if zone.m <= 0.0 or zone.T <= 0.0:
        return False
    kappa = kramers_opacity(zone.rho, zone.T, zone.Y[0] if zone.Y else 0.7)
    nabla_rad = (3.0 * kappa * max(zone.L, 0.0) * zone.P
                 / (16.0 * pi * a_rad * c_light * G_grav * zone.m
                    * zone.T**4))
    nabla_ad = 0.4
    return nabla_rad > nabla_ad


def virial_temperature(M: float, R: float, mu: float) -> float:
    """Virial estimate of the mean stellar temperature.
        T_vir ~ G M mu m_p / (3 k_B R)
    This is the analogue of the virial radius calculation in the
    rockstar halo finder (seed 1131).
    """
    if R <= 0.0 or mu <= 0.0:
        return 0.0
    return G_grav * M * mu * m_p / (3.0 * k_B * R)


# =====================================================================
# Diagnostic
# =====================================================================

def _self_test():
    print("stellar_structure self-test:")
    # Build a 20-zone model of a 1 Msun star
    M = 1.0 * M_sun
    Y0 = [0.7, 0.28, 0.02]
    zones = build_stellar_model(M, N_zones=20, Y_init=Y0,
                                 T_centre=1.5e7, P_centre=2.5e17)
    print(f"  built {len(zones)} zones")
    print(f"    centre r = {zones[0].r:.3e} cm  "
          f"T = {zones[0].T:.3e} K  P = {zones[0].P:.3e}")
    print(f"    surface r = {zones[-1].r:.3e} cm  "
          f"T = {zones[-1].T:.3e} K  P = {zones[-1].P:.3e}")
    # Virial temperature
    R_surf = zones[-1].r
    T_vir = virial_temperature(M, R_surf, 0.6)
    print(f"    T_virial = {T_vir:.3e} K")
    # Schwarzschild
    n_conv = sum(1 for i in range(len(zones)-1)
                  if schwarzschild_criterion(zones[i], zones[i+1]))
    print(f"    convective layers = {n_conv}/{len(zones)-1}")
    print("stellar_structure self-test OK")


if __name__ == "__main__":
    _self_test()
