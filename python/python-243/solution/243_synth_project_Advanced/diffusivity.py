"""
diffusivity.py - Neutron diffusion coefficient in r-process ejecta.

Adapted from 1158_shoh5301_Quick-MSD-Diffusivity-Calculator.
Computes the neutron diffusion coefficient in the expanding ejecta:

  D = (1/3) lambda_n v_n
  lambda_n = 1 / (n_target sigma_{n,scatter})

With time-dependent density, the diffusion timescale:
  t_diff = R^2 / (6 D)
"""
from __future__ import annotations
import math
from physical_constants import (
    K_BOLTZMANN, M_NEUTRON, MEV_ERG, M_U, C_LIGHT, TINY,
)


def neutron_density(rho: float, Ye: float, X_n: float = 0.9) -> float:
    """Neutron number density  n_n = X_n rho Ye / m_u   [cm^-3]"""
    if rho <= 0.0:
        return 0.0
    return X_n * rho * Ye / M_U


def scatter_cross_section(E_n_MeV: float = 1.0) -> float:
    """
    Neutron-nucleus scattering cross section  [cm^2].
    Geometric: sigma ~ pi R^2 with R = 1.25 A^{1/3} fm.
    Energy-dependent correction for low-energy resonances.
    """
    R_fm = 1.25 * 150 ** (1.0 / 3.0)      # A ~ 150 typical
    R_cm = R_fm * 1e-13
    sigma_geo = math.pi * R_cm * R_cm
    # Low-energy enhancement: sigma ~ sigma_geo * (1 + v_th / v)
    if E_n_MeV <= 0.0:
        return sigma_geo * 10.0
    v = math.sqrt(2.0 * E_n_MeV * MEV_ERG / M_NEUTRON)
    v_th = 2.2e8                       # cm/s, thermal
    return sigma_geo * (1.0 + v_th / max(v, TINY))


def diffusion_coefficient(rho: float, Ye: float, T9: float,
                          A_typical: int = 130) -> float:
    """
    Neutron diffusion coefficient  D  [cm^2/s].
    D = (1/3) lambda_mfp v_th
    """
    n_n = neutron_density(rho, Ye)
    if n_n < TINY:
        return 1.0e30
    sigma = scatter_cross_section(E_n_MeV=K_BOLTZMANN * T9 * 1e9 / MEV_ERG)
    lambda_mfp = 1.0 / (n_n * sigma + TINY)
    v_th = math.sqrt(8.0 * K_BOLTZMANN * T9 * 1e9 / (math.pi * M_NEUTRON))
    D = (1.0 / 3.0) * lambda_mfp * v_th
    return max(0.0, D)


def diffusion_timescale(R_cm: float, D: float) -> float:
    """Diffusion timescale  t_diff = R^2 / (6 D)  [s]."""
    if D <= 0.0:
        return 1.0e30
    return R_cm * R_cm / (6.0 * D)


def knudsen_number(lambda_mfp: float, L_cm: float) -> float:
    """Kn = lambda / L.  Kn >> 1: free streaming;  Kn << 1: diffusive."""
    if L_cm <= 0.0:
        return 1.0e30
    return lambda_mfp / L_cm


def MSD_from_trajectory(times: list, msd_values: list
                        ) -> float:
    """
    Extract effective diffusivity from mean-square-displacement data.
      MSD(t) = 6 D t  =>  D = slope / 6.
    Adapted from Quick-MSD-Diffusivity-Calculator.
    """
    if len(times) < 2:
        return 0.0
    # Least-squares fit:  MSD = a t
    n = len(times)
    sum_t = sum(times)
    sum_t2 = sum(t * t for t in times)
    sum_msd = sum(msd_values)
    sum_t_msd = sum(t * m for t, m in zip(times, msd_values))
    det = n * sum_t2 - sum_t * sum_t
    if abs(det) < 1e-300:
        return 0.0
    a = (n * sum_t_msd - sum_t * sum_msd) / det
    return max(0.0, a / 6.0)


def opacity_from_diffusion(D: float, rho: float, c: float = None) -> float:
    """
    Rosseland-mean opacity estimate from diffusion:
      kappa ~ c / (3 D rho)
    """
    if c is None:
        c = C_LIGHT
    if D <= 0.0 or rho <= 0.0:
        return 1.0e30
    return c / (3.0 * D * rho)


__all__ = [
    "neutron_density", "scatter_cross_section",
    "diffusion_coefficient", "diffusion_timescale",
    "knudsen_number", "MSD_from_trajectory",
    "opacity_from_diffusion",
]
