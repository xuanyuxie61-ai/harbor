"""
reaction_rates.py - Nuclear reaction rates for r-process network.

Implements:
  - Radiative neutron capture  (n,gamma):  Hauser-Feshbach + direct capture
  - Photodisintegration (gamma,n): detailed balance
  - Beta decay:  empirical systematics
  - Fission rates:  spontaneous + neutron-induced

Based on patterns from 1044_FootlooseCalvingMechanism/physics.py (rate
equations, stiff ODEs) and 1158_Quick-MSD-Diffusivity-Calculator/finetune.py
(feature engineering of rate tables).
"""
from __future__ import annotations
import math
from typing import Dict, Tuple
from physical_constants import (
    K_BOLTZMANN, MEV_ERG, M_NEUTRON, C_LIGHT, H_BAR, FM_CM, TINY,
)
from nuclear_physics import (
    binding_energy, separation_energy, partition_function_simple,
    sample_rate_uncertainty,
)

# ----------------------------------------------------------------------
# Neutron capture  (n, gamma)  --  compound nucleus formation
#   sigma ~ pi * lambda_bar^2 * Sum_J (2J+1)/((2J_n+1)(2J_t+1)) * T_n T_gamma / T_tot
# Simplified:  Weisskopf-Ewing with constant transmission.
# ----------------------------------------------------------------------

def neutron_thermal_velocity(T: float) -> float:
    """Thermal velocity v_th = sqrt(2 k T / m_n)   [cm/s]"""
    if T <= 0.0:
        return 0.0
    return math.sqrt(2.0 * K_BOLTZMANN * T / M_NEUTRON)


def weisskopf_capture_rate(A: int, Z: int, T: float,
                           S_n_MeV: float = None) -> float:
    """
    Approximate radiative neutron capture rate  <sigma v>  [cm^3/s]
    using Weisskopf-Ewing with gamma-width systematics.

      <sigma v> ~ (hbar c / kT)^2 * v_th * (Gamma_gamma / hbar) * rho_comp

    where rho_comp is the level density at the neutron separation energy.
    """
    if T <= 0.0 or A <= 0:
        return 0.0
    if S_n_MeV is None:
        Sn, _ = separation_energy(A + 1, Z)
        S_n_MeV = max(Sn, 0.1)

    v_th = neutron_thermal_velocity(T)
    kT_MeV = K_BOLTZMANN * T / MEV_ERG
    # lambda_bar^2 = (hbar c / E)^2   with E ~ kT
    lambda_bar2 = (H_BAR * C_LIGHT / (kT_MeV * MEV_ERG + TINY))**2  # cm^2

    # Level density  rho  ~ exp(2 sqrt(a E)) / (12 E^{5/4})  (Back-shifted Fermi gas)
    a_level = 0.093 * (A + 1)          # MeV^-1
    E_star = S_n_MeV                   # excitation at Sn
    if E_star <= 0.01:
        E_star = 0.01
    rho = math.exp(2.0 * math.sqrt(a_level * E_star)) / (12.0 * E_star**1.25)

    # Gamma-ray width systematics:  Gamma_gamma ~ 1e-2 * A^{-2/3}  MeV
    Gamma_gamma = 1.0e-2 * (A + 1) ** (-2.0 / 3.0)

    rate = lambda_bar2 * v_th * (Gamma_gamma * MEV_ERG / H_BAR) * rho
    # Clamp to physically sensible range
    rate = max(0.0, min(rate, 1.0e-10))
    return rate


# ----------------------------------------------------------------------
# Photodisintegration  (gamma, n)  via detailed balance (Cameron 1957)
#   lambda_{gamma,n} = <sigma v>_{n,gamma} * (G_t G_n / G_comp)
#                      * (2 pi m_n k T / h^2)^{3/2} * (A/(A+1))^{3/2}
#                      * exp(-S_n / kT)
# ----------------------------------------------------------------------

def photodisintegration_rate(A: int, Z: int, T: float) -> float:
    """Return lambda_{gamma,n}  [1/s]  for (A,Z) -> (A-1,Z) + n."""
    if T <= 0.0 or A <= 1:
        return 0.0
    # Forward capture on (A-1, Z)
    Sn, _ = separation_energy(A, Z)
    if Sn <= 0.0:
        return 0.0
    cap = weisskopf_capture_rate(A - 1, Z, T, S_n_MeV=Sn)

    # Detailed balance factor
    pref = (2.0 * math.pi * M_NEUTRON * K_BOLTZMANN * T / H_BAR**2) ** 1.5
    A_factor = (float(A) / (A + 1.0)) ** 1.5
    exp_arg = -Sn * MEV_ERG / (K_BOLTZMANN * T)
    if exp_arg < -500.0:
        return 0.0
    try:
        exp_val = math.exp(exp_arg)
    except OverflowError:
        return 0.0
    return cap * pref * A_factor * exp_val


# ----------------------------------------------------------------------
# Beta decay  --  empirical systematics  (Moller & Nix)
#   log10 t_{1/2} [s] = a + b log10 Q_beta + c / Q_beta^2
#   lambda = ln(2) / t_{1/2}
# ----------------------------------------------------------------------

def beta_decay_rate(A: int, Z: int) -> float:
    """Beta-decay rate lambda_beta  [1/s]  for (A,Z)."""
    if A <= 0 or Z <= 0 or Z >= A:
        return 0.0
    # Q_beta ~ B(A, Z+1) - B(A, Z)  (rough)
    B0 = binding_energy(A, Z)
    B1 = binding_energy(A, Z + 1)
    Q_beta = max(B1 - B0, 0.01)         # MeV, at least 10 keV
    a, b, c = 1.5, -1.0, 0.5
    try:
        log10_t12 = a + b * math.log10(Q_beta) + c / (Q_beta * Q_beta)
    except ValueError:
        log10_t12 = 10.0
    log10_t12 = max(-3.0, min(log10_t12, 30.0))
    t12 = 10.0 ** log10_t12
    return math.log(2.0) / t12


# ----------------------------------------------------------------------
# Fission  --  spontaneous + neutron-induced
#   lambda_sf = ln(2) / t_{1/2}^{sf}
#   t_{1/2}^{sf} ~ exp( c_f * (B_f / sqrt(A)) )   (Hagemann et al.)
# ----------------------------------------------------------------------

def fission_barrier(A: int, Z: int) -> float:
    """Simple liquid-drop fission barrier B_f  [MeV]."""
    x = (Z * Z / A) / 47.0            # fissility parameter
    if x >= 1.0:
        return 0.0
    return 0.83 * (1.0 - x) * A ** (2.0 / 3.0)


def spontaneous_fission_rate(A: int, Z: int) -> float:
    """Spontaneous fission rate [1/s]."""
    Bf = fission_barrier(A, Z)
    if Bf <= 0.0:
        return 1.0e4                  # rapid but finite
    # log10 t_{1/2} ~ c * Bf / sqrt(A)
    log10_t12 = 20.0 * Bf / math.sqrt(A)
    log10_t12 = max(0.0, min(log10_t12, 100.0))
    t12 = 10.0 ** log10_t12
    return math.log(2.0) / t12


def neutron_induced_fission_rate(A: int, Z: int, T: float) -> float:
    """Neutron-induced fission  <sigma_f v>  [cm^3/s]."""
    Bf = fission_barrier(A + 1, Z)
    if Bf <= 0.0:
        return 1.0e-22
    v_th = neutron_thermal_velocity(T)
    sigma = 1.0e-24 * math.exp(-Bf / (K_BOLTZMANN * T / MEV_ERG + TINY))
    return sigma * v_th


# ----------------------------------------------------------------------
# Rate table builder  --  for all (A, Z) in a network
# ----------------------------------------------------------------------

def build_rate_table(A_min: int, A_max: int, Z_min: int, Z_max: int,
                     T: float, rho: float, Ye: float,
                     apply_uncertainty: bool = False,
                     seed_base: int = 1) -> Dict[str, Dict[Tuple[int,int], float]]:
    """
    Build a dictionary of rates for the network:
      {"n_cap": {(A,Z): rate}, "gamma_n": {...}, "beta": {...}, "fission": {...}}
    Rates can be perturbed by truncated-normal uncertainties.
    """
    table: Dict[str, Dict[Tuple[int,int], float]] = {
        "n_cap": {},
        "gamma_n": {},
        "beta": {},
        "fission": {},
    }
    for Z in range(Z_min, Z_max + 1):
        for A in range(max(A_min, Z), A_max + 1):
            seed = seed_base + A * 100 + Z
            # Neutron capture
            r_ncap = weisskopf_capture_rate(A, Z, T)
            if apply_uncertainty:
                r_ncap = sample_rate_uncertainty(r_ncap, 0.2, seed)
            table["n_cap"][(A, Z)] = r_ncap

            # Photodisintegration
            table["gamma_n"][(A, Z)] = photodisintegration_rate(A, Z, T)

            # Beta decay
            r_beta = beta_decay_rate(A, Z)
            if apply_uncertainty:
                r_beta = sample_rate_uncertainty(r_beta, 0.1, seed + 10000)
            table["beta"][(A, Z)] = r_beta

            # Fission
            r_sf = spontaneous_fission_rate(A, Z)
            r_nf = neutron_induced_fission_rate(A, Z, T)
            # r_nf has units cm^3/s; multiply by neutron density ~ rho * Ye / m_u
            n_n_local = rho * max(Ye, 0.0) / 1.6605e-24
            table["fission"][(A, Z)] = r_sf + r_nf * n_n_local

    return table


__all__ = [
    "neutron_thermal_velocity",
    "weisskopf_capture_rate",
    "photodisintegration_rate",
    "beta_decay_rate",
    "fission_barrier",
    "spontaneous_fission_rate",
    "neutron_induced_fission_rate",
    "build_rate_table",
]
