"""
physical_system.py
==================

High-fidelity physics model of a protoplanetary disk with embedded
pebble-accreting protoplanet.  This module encapsulates the "ground truth"
forward operator that all lower-fidelity surrogates approximate.

Physical setting (following Bitsch+ 2015; Schoonenberg+Ormel 2017)
------------------------------------------------------------------
A viscously accreting alpha-disk around a solar-mass star, threaded by
dust grains and pebbles characterized by a Stokes number St. A protoplanet
of mass M_p orbits at semi-major axis a_p, undergoing Type-I migration
and accreting pebbles in the 2D Hill regime.

Key governing equations (all in code docstrings and inline):

  Kepler frequency:        Omega_K(r) = sqrt(G M_* / r^3)
  Gas surface density:     Sigma_g(r,t) (viscous similarity solution)
  Sound speed:             c_s(r) = c_{s,1} (r / 1 AU)^{-zeta/2}
  Gas scale height:        H(r) = c_s / Omega_K
  Viscosity:               nu(r) = alpha c_s H
  Stokes number:           St = t_stop Omega_K  (assumed constant)
  Pebble surface density:  Sigma_p(r,t) = Z(r,t) Sigma_g(r,t)
  Dust-to-gas ratio Z:     Z(r,t) = Z_0 (r/R_1)^{p_Z} (1 + t/t_visc)^{-q_Z}
  Hill radius:             R_H = a_p (M_p / (3 M_*))^{1/3}
  Pebble accretion rate:   M_dot_peb = 2 R_H Sigma_p Delta_v  (2D Hill)
  Migration torque:        Gamma = -k_mig (q_p h^{-2}) (Sigma_g a_p^4 Omega_K^2)
  Radial drift velocity:   v_r,peb = -2 St eta v_K  (for St << 1)
  Pressure support:        eta = -0.5 (H/r)^2 d ln P / d ln r
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

# Astronomical / physical constants in cgs.
G_CGS = 6.67430e-8            # cm^3 g^-1 s^-2
MSUN_CGS = 1.98892e33         # g
AU_CGS = 1.495978707e13       # cm
YR_CGS = 3.15576e7            # s
MYR_CGS = YR_CGS * 1.0e6
KB_CGS = 1.380649e-16         # erg K^-1
MH_CGS = 1.6735575e-24        # g


# ----------------------------------------------------------------------
# Data classes for disk and protoplanet parameters.
# ----------------------------------------------------------------------
@dataclass
class DiskParams:
    """Physical parameters of a viscous protoplanetary accretion disk."""
    M_star: float = MSUN_CGS              # stellar mass [g]
    M_dot_0: float = 1.0e-7 * MSUN_CGS / YR_CGS  # initial gas accretion rate [g/s]
    alpha_visc: float = 1.0e-2            # Shakura-Sunyaev alpha
    St: float = 0.03                      # characteristic Stokes number
    delta_turb: float = 1.0e-4            # turbulent diffusion parameter
    Z_0: float = 0.01                     # initial dust-to-gas ratio
    R_1: float = 100.0 * AU_CGS           # characteristic disk radius [cm]
    cs_1: float = 6.50e4                  # sound speed at 1 AU [cm/s]
    zeta: float = 0.6                     # temperature power-law index
    t_0: float = 0.2 * MYR_CGS            # disk birth time [s]
    p_Z: float = -0.5                     # radial dust-to-gas gradient
    q_Z: float = 0.5                      # temporal dust depletion index
    mu_mw: float = 2.34                   # mean molecular weight (molecular gas)


@dataclass
class PlanetParams:
    """Parameters of an embedded protoplanet undergoing migration/accretion."""
    M_p: float = 10.0 * 1.0e28            # planet mass [g]  (~few M_Earth)
    a_p0: float = 30.0 * AU_CGS           # initial semi-major axis [cm]
    e_p: float = 0.0                      # eccentricity (assume circular)
    k_mig: float = 2.5                    # migration torque prefactor


@dataclass
class ChemistryParams:
    """Parameters for the reduced CO/H2O/H2 gas-phase chemistry network."""
    n_species: int = 6                    # {H2, CO, H2O, CO2, CH4, OH}
    T_act_CO: float = 20.0                # CO freeze-out temperature [K]
    T_act_H2O: float = 150.0              # H2O snowline temperature [K]
    k_CO_gas: float = 1.0e-10             # CO gas-phase rate coefficient [cm^3/s]
    k_H2O_form: float = 3.0e-12           # water formation rate
    x_cr: float = 1.0e-17                 # cosmic ray ionization rate [s^-1]
    dust_radius: float = 1.0e-2           # characteristic grain radius [cm]
    rho_grain: float = 2.0                # grain material density [g/cm^3]


@dataclass
class DiskState:
    """Complete state vector of the disk at a given epoch."""
    r_grid_cm: 'list[float]' = field(default_factory=list)
    t_s: float = 0.0
    Sigma_g: 'list[float]' = field(default_factory=list)
    Sigma_p: 'list[float]' = field(default_factory=list)
    T_mid: 'list[float]' = field(default_factory=list)
    H_gas: 'list[float]' = field(default_factory=list)
    v_r_peb: 'list[float]' = field(default_factory=list)
    abundances: 'list[list[float]]' = field(default_factory=list)
    a_p: float = 0.0
    M_p: float = 0.0


# ----------------------------------------------------------------------
# Core physical functions.
# ----------------------------------------------------------------------
def kepler_angular(r_cm: float, M_star: float) -> float:
    """Omega_K = sqrt(G M_* / r^3)."""
    if r_cm <= 0.0:
        raise ValueError("kepler_angular: r must be positive.")
    return math.sqrt(G_CGS * M_star / (r_cm ** 3))


def sound_speed(r_cm: float, cs_1: float, R_ref: float, zeta: float) -> float:
    """c_s(r) = c_{s,1} * (r / R_ref)^{-zeta/2}."""
    return cs_1 * math.pow(r_cm / R_ref, -0.5 * zeta)


def midplane_temperature(r_cm: float, cs: float, mu: float) -> float:
    """T_mid = mu m_H c_s^2 / k_B."""
    return mu * MH_CGS * cs * cs / KB_CGS


def gas_scale_height(r_cm: float, cs: float, M_star: float) -> float:
    """H = c_s / Omega_K."""
    return cs / kepler_angular(r_cm, M_star)


def viscosity(r_cm: float, cs: float, H: float, alpha: float) -> float:
    """nu = alpha c_s H (Shakura-Sunyaev prescription)."""
    return alpha * cs * H


def pressure_log_slope(zeta: float, gamma_sg: float) -> float:
    """d ln P / d ln r = -(gamma_sg + zeta/2 + 3/2) for power-law disk."""
    return -(gamma_sg + 0.5 * zeta + 1.5)


def pressure_support(H: float, r_cm: float, dlnP_dlnr: float) -> float:
    """eta = -0.5 (H/r)^2 d ln P / d ln r  (dimensionless pressure gradient)."""
    return -0.5 * (H / r_cm) ** 2 * dlnP_dlnr


def pebble_radial_drift(St: float, eta: float, r_cm: float,
                        M_star: float) -> float:
    """v_r,peb = -2 St eta v_K / (1 + St^2), with v_K = r Omega_K."""
    v_K = r_cm * kepler_angular(r_cm, M_star)
    return -2.0 * St * eta * v_K / (1.0 + St * St)


def hill_radius(a_p: float, M_p: float, M_star: float) -> float:
    """R_H = a_p * (M_p / (3 M_*))^{1/3}."""
    return a_p * math.pow(M_p / (3.0 * M_star), 1.0 / 3.0)


def pebble_accretion_rate_2d(
    R_H: float, Sigma_p_local: float, Delta_v: float, St: float
) -> float:
    """2D Hill-regime pebble accretion rate (Youdin & Goodman 2005 approx).

      M_dot_peb ~ 2 R_H Sigma_p Delta_v * (St / 0.1)^{2/3}
    """
    if R_H <= 0.0 or Sigma_p_local < 0.0:
        return 0.0
    return 2.0 * R_H * Sigma_p_local * max(Delta_v, 1.0e-30) \
        * math.pow(max(St, 1.0e-30) / 0.1, 2.0 / 3.0)


def type1_migration_torque(
    q_p: float, h: float, Sigma_g: float, a_p: float,
    Omega_K: float, k_mig: float
) -> float:
    """Gamma = -k_mig q_p^2 h^{-2} Sigma_g a_p^4 Omega_K^2.

    Note: classical Type-I torque ~ M_p^2 (not q_p^2); we keep q_p^2 here to
    follow the common normalization Gamma_0 = (q_p/h)^2 Sigma_g a^4 Omega_K^2.
    """
    if h <= 0.0:
        raise ValueError("type1_migration_torque: aspect ratio h must be > 0.")
    return -k_mig * (q_p / h) ** 2 * Sigma_g * (a_p ** 4) * (Omega_K ** 2)


def dust_number_density(Sigma_p: float, H_p: float,
                        rho_grain: float, a_grain: float) -> float:
    """n_d ~ Sigma_p / (sqrt(2pi) H_p * (4/3) pi a^3 rho_grain)."""
    if H_p <= 0.0 or a_grain <= 0.0 or rho_grain <= 0.0:
        return 0.0
    vol = (4.0 / 3.0) * math.pi * (a_grain ** 3) * rho_grain
    return Sigma_p / (math.sqrt(2.0 * math.pi) * H_p * vol)


# ----------------------------------------------------------------------
# Gas surface density: viscous similarity solution (Lynden-Bell & Pringle 1974)
#
#   Sigma_g(r,t) = (M_dot_0 / (3 pi nu)) * T^{-(5/2 - gamma) / (2 - gamma)}
#                  * exp( -(r/R_1)^{2-gamma} / T )
#   with T = 1 + (t - t_0) / t_visc,
#        gamma = 3/2 - zeta (for T_mid ~ r^{-zeta/2}),
#        t_visc = R_1^2 / (nu(R_1) * 3 * (2-gamma)^2).
# ----------------------------------------------------------------------
def viscous_similarity_solution(
    r_cm: float, t_s: float, disk: DiskParams
) -> float:
    gamma = 1.5 - disk.zeta
    cs = sound_speed(r_cm, disk.cs_1, AU_CGS, disk.zeta)
    H = gas_scale_height(r_cm, cs, disk.M_star)
    nu = viscosity(r_cm, cs, H, disk.alpha_visc)
    nu_1 = viscosity(AU_CGS, disk.cs_1,
                     gas_scale_height(AU_CGS, disk.cs_1, disk.M_star),
                     disk.alpha_visc)
    t_visc = (AU_CGS * 100.0) ** 2 / (nu_1 * 3.0 * (2.0 - gamma) ** 2)
    T_factor = 1.0 + max(t_s - disk.t_0, 0.0) / t_visc
    power = -(2.5 - gamma) / (2.0 - gamma)
    base = disk.M_dot_0 / (3.0 * math.pi * max(nu, 1.0e-60))
    radial = math.pow(r_cm / disk.R_1, -(2.0 - gamma))  # note r^{-gamma} encoded via base
    # Full expression: base * T^power * exp(-(r/R_1)^{2-gamma}/T)
    exponent = -math.pow(r_cm / disk.R_1, 2.0 - gamma) / max(T_factor, 1.0e-30)
    return base * math.pow(max(T_factor, 1.0e-30), power) * math.exp(exponent)


# ----------------------------------------------------------------------
# Main driver: assemble the high-fidelity disk state at time t.
# ----------------------------------------------------------------------
def build_disk_state(
    disk: DiskParams,
    planet: PlanetParams,
    chem: ChemistryParams,
    t_s: float,
    n_r: int = 64,
    r_min_AU: float = 1.0,
    r_max_AU: float = 200.0,
) -> DiskState:
    """Assemble the high-fidelity protoplanetary-disk state at epoch t_s."""
    if n_r < 2:
        raise ValueError("build_disk_state: n_r must be >= 2.")
    gamma = 1.5 - disk.zeta
    dlnP_dlnr = pressure_log_slope(disk.zeta, gamma)

    # Logarithmic radial grid.
    r_min = r_min_AU * AU_CGS
    r_max = r_max_AU * AU_CGS
    if r_min >= r_max:
        raise ValueError("build_disk_state: r_min must be < r_max.")
    log_r = [
        math.exp(math.log(r_min) + i * (math.log(r_max) - math.log(r_min)) / (n_r - 1))
        for i in range(n_r)
    ]

    Sigma_g: list[float] = []
    Sigma_p: list[float] = []
    T_mid: list[float] = []
    H_gas: list[float] = []
    v_r_peb: list[float] = []
    for r in log_r:
        cs = sound_speed(r, disk.cs_1, AU_CGS, disk.zeta)
        H = gas_scale_height(r, cs, disk.M_star)
        Sg = viscous_similarity_solution(r, t_s, disk)
        # Dust-to-gas ratio evolves as Z(r,t) = Z_0 (r/R_1)^{p_Z} T^{-q_Z}.
        T_factor = 1.0 + max(t_s - disk.t_0, 0.0) / (
            (disk.R_1 ** 2)
            / (viscosity(AU_CGS, disk.cs_1,
                        gas_scale_height(AU_CGS, disk.cs_1, disk.M_star),
                        disk.alpha_visc)
               * 3.0 * (2.0 - gamma) ** 2)
        )
        Z = disk.Z_0 * math.pow(r / disk.R_1, disk.p_Z) \
            * math.pow(max(T_factor, 1.0e-30), -disk.q_Z)
        Sp = max(Sg * Z, 0.0)
        Tm = midplane_temperature(r, cs, disk.mu_mw)
        eta = pressure_support(H, r, dlnP_dlnr)
        vr = pebble_radial_drift(disk.St, eta, r, disk.M_star)
        Sigma_g.append(Sg); Sigma_p.append(Sp)
        T_mid.append(Tm); H_gas.append(H); v_r_peb.append(vr)

    # Reduced chemistry: initialize abundances from simple snowline model.
    abundances: list[list[float]] = []
    for Tm in T_mid:
        # Fractional abundances relative to H_2 (x_i = n_i / n_H2).
        x_CO = 1.0e-4 if Tm > chem.T_act_CO else 1.0e-8
        x_H2O = 1.0e-4 * math.exp(-(chem.T_act_H2O / max(Tm, 1.0)))
        x_CO2 = 0.2 * x_CO
        x_CH4 = 0.1 * x_CO
        x_OH = math.sqrt(chem.x_cr / max(1.0e-10, 1.0e-9 * (Tm / 100.0)))
        abundances.append([1.0, x_CO, x_H2O, x_CO2, x_CH4, x_OH])

    R_H = hill_radius(planet.a_p0, planet.M_p, disk.M_star)
    return DiskState(
        r_grid_cm=log_r, t_s=t_s,
        Sigma_g=Sigma_g, Sigma_p=Sigma_p, T_mid=T_mid,
        H_gas=H_gas, v_r_peb=v_r_peb, abundances=abundances,
        a_p=planet.a_p0, M_p=planet.M_p,
    )


# ----------------------------------------------------------------------
# Scalar QoI: column-integrated CO abundance inside the planet's orbit.
# ----------------------------------------------------------------------
def column_co_abundance(state: DiskState, AU: float = AU_CGS) -> float:
    """QoI: int_0^{a_p} x_CO(r) Sigma_g(r) 2 pi r dr (trapezoidal rule)."""
    r = state.r_grid_cm
    if len(r) < 2:
        return 0.0
    integrand: list[float] = []
    for i, ri in enumerate(r):
        if ri > state.a_p:
            integrand.append(0.0)
            continue
        x_CO = state.abundances[i][1] if state.abundances else 0.0
        integrand.append(x_CO * state.Sigma_g[i] * 2.0 * math.pi * ri)
    total = 0.0
    for i in range(len(r) - 1):
        total += 0.5 * (integrand[i] + integrand[i + 1]) * (r[i + 1] - r[i])
    return total


# ----------------------------------------------------------------------
# Perturbed state: evaluate QoI at a perturbed parameter vector xi.
# xi = [log10(alpha), log10(St), log10(Z_0), zeta].
# ----------------------------------------------------------------------
def evaluate_high_fidelity(
    xi: 'list[float]',
    planet: PlanetParams,
    chem: ChemistryParams,
    t_s: float,
    n_r: int = 64,
) -> float:
    """High-fidelity forward map F_H(xi) -> QoI (CO column)."""
    if len(xi) != 4:
        raise ValueError("evaluate_high_fidelity: xi must have length 4.")
    alpha = math.pow(10.0, xi[0])
    St = math.pow(10.0, xi[1])
    Z0 = math.pow(10.0, xi[2])
    zeta = xi[3]
    # Enforce physically sensible ranges.
    alpha = min(max(alpha, 1.0e-5), 1.0)
    St = min(max(St, 1.0e-4), 1.0)
    Z0 = min(max(Z0, 1.0e-4), 0.1)
    zeta = min(max(zeta, 0.1), 1.5)

    disk = DiskParams(alpha_visc=alpha, St=St, Z_0=Z0, zeta=zeta)
    state = build_disk_state(disk, planet, chem, t_s, n_r=n_r)
    q = column_co_abundance(state)
    # Add a mild regularizer: log-scale output stabilizes GP fitting.
    return math.log10(max(q, 1.0e-60))
