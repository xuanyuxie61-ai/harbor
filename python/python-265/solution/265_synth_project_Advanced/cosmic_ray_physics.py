# -*- coding: utf-8 -*-
"""
cosmic_ray_physics.py
---------------------
Fundamental physical constants, heliospheric background parameters, and
Parker interplanetary magnetic field (IMF) model.

The interplanetary magnetic field (IMF) is taken to be the Archimedean
Parker spiral (Parker 1958, ApJ 128, 664):

    B_r   =  B_0 (r_0 / r)^2
    B_T   = -B_0 (r_0^2 Omega_sun / V_sw) sin(theta) / r
    |B|   =  sqrt(B_r^2 + B_T^2)

The magnetic field unit vector is  b_hat = B / |B|; the field-line
inclination angle psi satisfies  tan(psi) = |B_T| / |B_r|.

All SI units are used internally, except rigidity (GV) and number
densities (1 / (m^3 GV)) for the distribution function f.
"""
from __future__ import annotations
import math
from typing import Tuple

# =====================================================================
# Fundamental constants (CODATA 2018)
# =====================================================================
c_light      = 2.99792458e8          # speed of light [m / s]
q_e          = 1.602176634e-19       # elementary charge [C]
m_p          = 1.67262192369e-27     # proton mass [kg]
m_e          = 9.1093837015e-31      # electron mass [kg]
k_B          = 1.380649e-23          # Boltzmann constant [J / K]
mu_0         = 4.0e-7 * math.pi      # vacuum permeability [N / A^2]
AU           = 1.495978707e11        # astronomical unit [m]
pc           = 3.085677581e16        # parsec [m]
year_seconds = 3.15576e7             # Julian year [s]


# =====================================================================
# Heliospheric background parameters (nominal quiet-Sun)
# =====================================================================
SOLAR_WIND_SPEED_V0 = 4.0e5          # V_sw at r0 = 1 AU [m / s]
B_FIELD_1AU         = 5.0e-9         # radial B-field at 1 AU [T]
OMEGA_SUN           = 2.7e-6         # solar angular velocity [rad / s]
THETA_HELIO         = math.pi / 2.0  # heliographic colatitude (equator)
R_SOLAR             = 6.96e8         # solar radius [m]
R_HELIOPAUSE        = 120.0 * AU     # heliopause distance [m]
R_SOURCE            = 1.0e3 * pc     # GCR source distance [m]


# =====================================================================
# Parker IMF
# =====================================================================
def parker_imf(r: float, theta: float = THETA_HELIO,
               V_sw: float = SOLAR_WIND_SPEED_V0,
               B0: float = B_FIELD_1AU,
               r0: float = AU) -> Tuple[float, float, float, float]:
    """Return (B_r, B_T, |B|, psi) at heliocentric distance r.

    psi is the IMF spiral (garden-hose) angle measured from the radial.
    """
    if r <= 0.0:
        raise ValueError("parker_imf: r must be positive.")
    Br = B0 * (r0 / r) ** 2
    BT = -B0 * (r0 ** 2 * OMEGA_SUN / V_sw) * math.sin(theta) / r
    Bmag = math.sqrt(Br * Br + BT * BT)
    psi = math.atan2(abs(BT), abs(Br))
    return Br, BT, Bmag, psi


def field_line_pitch(r: float, theta: float = THETA_HELIO) -> float:
    """Field-line pitch angle tan(psi) = r * Omega * sin(theta) / V_sw."""
    return math.atan2(r * OMEGA_SUN * math.sin(theta), SOLAR_WIND_SPEED_V0)


# =====================================================================
# Relativistic kinematics (proton)
# =====================================================================
def kinetic_to_rigidity(Ek: float) -> float:
    """Kinetic energy per nucleon Ek [J] -> rigidity R [V].

    R = p c / (Z e);  p = sqrt((Ek + m_p c^2)^2 - (m_p c^2)^2) / c.
    """
    rest = m_p * c_light ** 2
    p = math.sqrt(max((Ek + rest) ** 2 - rest * rest, 0.0)) / c_light
    return p * c_light / q_e


def rigidity_to_kinetic(R: float) -> float:
    """Rigidity [V] -> kinetic energy per nucleon [J]."""
    pc = q_e * R                         # [J] (since Z = 1 for proton)
    rest = m_p * c_light ** 2
    return math.sqrt(pc * pc + rest * rest) - rest


def lorentz_factor(Ek: float) -> float:
    """Lorentz factor gamma = 1 + Ek / (m_p c^2)."""
    return 1.0 + Ek / (m_p * c_light ** 2)


def velocity_from_Ek(Ek: float) -> float:
    """Particle speed v = c * sqrt(1 - 1/gamma^2)."""
    g = lorentz_factor(Ek)
    return c_light * math.sqrt(max(1.0 - 1.0 / (g * g), 0.0))


def rigidity_larmor(R: float, B: float) -> float:
    """Larmor radius r_L = R / (c B). R in V, B in T."""
    if B <= 0.0:
        return math.inf
    return R / (c_light * B)


# =====================================================================
# Force-field modulation (Gleeson & Axford 1968)
# =====================================================================
def force_field_modulation(Ek: float, phi: float, A: int, Z: int) -> float:
    """Return LIS -> modulated ratio  J_mod(Ek) / J_LIS(Ek + phi|Z|/A).

    phi is the modulation potential [V]; A and Z are nucleon and charge
    numbers.  The classic Gleeson-Axford result is

        J_mod(E) = (E^2 + 2 E m c^2) / ((E + phi|Z|/A)^2
                  + 2 (E + phi|Z|/A) m c^2) * J_LIS(E + phi|Z|/A).
    """
    rest = m_p * c_light ** 2
    EL = Ek + phi * abs(Z) / max(A, 1)
    num = Ek * Ek + 2.0 * Ek * rest
    den = EL * EL + 2.0 * EL * rest
    if den <= 0.0:
        return 0.0
    return num / den


# =====================================================================
# Adiabatic cooling and transport coefficients
# =====================================================================
def adiabatic_cooling_rate(dV_dr: float, Ek: float) -> float:
    """Adiabatic energy-loss rate  dE/dt|_ad = - (1/3) (nabla . V) p v / E.

    For a spherically symmetric radial flow  nabla . V_sw = (2/r) V_sw
    + dV_sw/dr.  Here we accept the divergence directly.
    """
    v = velocity_from_Ek(Ek)
    p = m_p * lorentz_factor(Ek) * v
    return -(1.0 / 3.0) * dV_dr * p * v


def diffusion_coefficiency_bohm(R: float, B: float, xi: float = 1.0 / 3.0) -> float:
    """Bohm diffusion  kappa_B = xi * r_L * c / 3  [m^2/s].

    xi is the scattering mean-free-path factor (xi < 1/3 implies strong
    scattering; xi ~ 1/3 approaches Bohm limit).
    """
    rL = rigidity_larmor(R, B)
    return xi * rL * c_light / 3.0


def quasilinear_kappa_parallel(R: float, B: float, delta_B_B: float,
                               l_c: float, v: float) -> float:
    """Quasi-linear parallel diffusion coefficient (Jokipii 1966).

    kappa_|| = (v * r_L / 3) * (B / delta B)^2 * (1 + (l_c / r_L)^(2/3))

    delta_B_B is the turbulence level  delta B / B;  l_c is the
    correlation length of the turbulence [m].
    """
    if delta_B_B <= 0.0:
        return math.inf
    rL = rigidity_larmor(R, B)
    term = 1.0 + (l_c / max(rL, 1.0)) ** (2.0 / 3.0)
    return (v * rL / 3.0) * (1.0 / (delta_B_B * delta_B_B)) * term


def cross_field_kappa_perp(kappa_par: float, B: float,
                           delta_B_B: float) -> float:
    """Perpendicular diffusion from nonlinear guiding centre theory.

    kappa_perp = alpha * kappa_par * (delta B / B)^2 / (1 + (kappa_par /
    (r_L c))^2).  alpha ~ 0.02 for slab/2D composite turbulence.
    """
    alpha = 0.02
    num = alpha * kappa_par * delta_B_B * delta_B_B
    den = 1.0 + (kappa_par / max(rigidity_larmor(1.0, B) * c_light, 1.0)) ** 2
    return num / max(den, 1.0e-300)


# =====================================================================
# Local interstellar spectrum (LIS) parametrisations
# =====================================================================
def lis_proton_vladimir2015(Ek_GeV: float) -> float:
    """Vladimir et al. (2015) parametrisation of the proton LIS.

    Returns the differential intensity J(Ek) in units of
    [1 / (m^2 s sr GV)]  -- here we use kinetic energy [GeV / nucleon].

    J = N * (Ek + Ek0)^(-gamma1) * (1 + (Ek / Ek_b)^alpha)^beta
    """
    N = 4.7e6
    Ek0 = 0.4625
    gamma1 = 2.44
    Ek_b = 2.85
    alpha = 2.26
    beta = -0.21
    if Ek_GeV <= 0.0:
        return 0.0
    return N * (Ek_GeV + Ek0) ** (-gamma1) * (
        1.0 + (Ek_GeV / Ek_b) ** alpha) ** beta


def lis_electron_philip2019(Ek_GeV: float) -> float:
    """Philip et al. (2019) parametrisation of the electron LIS.

    J = N * Ek^(-gamma) * exp(-Ek / Ek_cut)  for Ek > 0.
    """
    N = 1.4e6
    gamma = 3.10
    Ek_cut = 31.5
    if Ek_GeV <= 0.0:
        return 0.0
    return N * Ek_GeV ** (-gamma) * math.exp(-Ek_GeV / Ek_cut)


# =====================================================================
# Self-contained demo (no visualisation)
# =====================================================================
def _demo() -> None:
    """Print a small diagnostic table; no plots."""
    print("[cosmic_ray_physics] Parker IMF diagnostics at 1, 10, 50 AU:")
    for rau in (1.0, 10.0, 50.0):
        Br, BT, Bmag, psi = parker_imf(rau * AU)
        print(f"  r = {rau:4.1f} AU: |B| = {Bmag * 1e9:7.3f} nT, "
              f"psi = {math.degrees(psi):6.2f} deg")
    print("[cosmic_ray_physics] Kinematics check:")
    for Ek_GeV in (0.1, 1.0, 10.0, 100.0):
        Ek = Ek_GeV * 1.0e9 * q_e
        R = kinetic_to_rigidity(Ek) / 1.0e9
        v = velocity_from_Ek(Ek)
        print(f"  Ek = {Ek_GeV:6.2f} GV rigidity R = {R:7.3f} GV, "
              f"v/c = {v / c_light:.4f}")


if __name__ == "__main__":
    _demo()
