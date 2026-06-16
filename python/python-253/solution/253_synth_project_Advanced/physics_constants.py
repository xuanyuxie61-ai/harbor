"""
physics_constants.py — Fundamental physical constants and binary black-hole parameters.

This module collects:
  (a) CODATA 2018 SI constants relevant to gravitational-wave astronomy,
  (b) geometric-unit conversion factors (G = c = 1),
  (c) parameter containers for quasi-circular compact binaries.

Every downstream module imports from here so that the chirp mass M_c,
symmetric mass ratio nu, orbital frequency f_orb, and dimensionless spins
chi_i are defined consistently.

Key formulae encoded:
  - Chirp mass:         M_c = (m1 m2)^{3/5} / (m1 + m2)^{1/5}
  - Total mass:         M   = m1 + m2
  - Symmetric ratio:    nu  = m1 m2 / M^2            in (0, 1/4]
  - Schwarzschild rad:  r_s = 2 G M / c^2
  - ISCO frequency:     f_ISCO = c^3 / (6^{3/2} pi G M)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Tuple


# ---------------------------------------------------------------------------
#  CODATA 2018 constants (SI)
# ---------------------------------------------------------------------------
C_SI: float = 2.99792458e8              # speed of light              [m/s]
G_SI: float = 6.67430e-11               # Newton constant             [m^3 kg^-1 s^-2]
MSUN_SI: float = 1.98892e30             # solar mass                  [kg]
PC_SI: float = 3.0856775814913673e16    # parsec                      [m]
MPC_SI: float = PC_SI * 1.0e6           # megaparsec                  [m]
HBAR_SI: float = 1.054571817e-34        # reduced Planck              [J s]
KB_SI: float = 1.380649e-23             # Boltzmann                   [J/K]
PI: float = math.pi
TWOPI: float = 2.0 * PI


# ---------------------------------------------------------------------------
#  Geometric-unit conversions  (G = c = 1)
# ---------------------------------------------------------------------------
def mass_to_seconds(M_kg: float) -> float:
    """Convert a mass M [kg] to geometric seconds  t_g = G M / c^3."""
    return G_SI * M_kg / C_SI ** 3


def mass_to_metres(M_kg: float) -> float:
    """Convert a mass M [kg] to geometric metres  r_g = G M / c^2."""
    return G_SI * M_kg / C_SI ** 2


def seconds_to_metres(t_s: float) -> float:
    return C_SI * t_s


def frequency_hz_to_geom(f_hz: float, M_kg: float) -> float:
    """Dimensionless frequency  M f = (G M / c^3) f."""
    return mass_to_seconds(M_kg) * f_hz


def strain_at_distance(h_geom: float, D_mpc: float) -> float:
    """Convert geometric strain to dimensionless strain at luminosity distance D [Mpc]."""
    return h_geom / (D_mpc * MPC_SI)


# ---------------------------------------------------------------------------
#  Binary-black-hole parameter container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BinaryParameters:
    """Quasi-circular compact-binary parameters used throughout the pipeline.

    Attributes
    ----------
    m1_si, m2_si : component masses in solar masses
    chi1, chi2   : dimensionless aligned spins   in [-1, +1]
    f_low_hz     : starting GW frequency         [Hz]
    distance_mpc : luminosity distance           [Mpc]
    ell          : spherical-harmonic index of the dominant (2,2) mode
    """
    m1_si: float = 30.0
    m2_si: float = 30.0
    chi1: float = 0.0
    chi2: float = 0.0
    f_low_hz: float = 20.0
    distance_mpc: float = 100.0
    ell: int = 2

    # ----- derived quantities -------------------------------------------
    @property
    def m1_kg(self) -> float:
        return self.m1_si * MSUN_SI

    @property
    def m2_kg(self) -> float:
        return self.m2_si * MSUN_SI

    @property
    def M_kg(self) -> float:
        return self.m1_kg + self.m2_kg

    @property
    def M_tot(self) -> float:
        """Total mass in solar masses."""
        return self.m1_si + self.m2_si

    @property
    def chirp_mass(self) -> float:
        """Chirp mass  M_c = (m1 m2)^{3/5} / (m1+m2)^{1/5}  [Msun]."""
        return (self.m1_si * self.m2_si) ** 0.6 / self.M_tot ** 0.2

    @property
    def symmetric_ratio(self) -> float:
        """Symmetric mass ratio  nu = m1 m2 / M^2  in (0, 1/4]."""
        return (self.m1_si * self.m2_si) / (self.M_tot ** 2)

    @property
    def mass_ratio(self) -> float:
        """Mass ratio  q = m2 / m1 <= 1."""
        return min(self.m1_si, self.m2_si) / max(self.m1_si, self.m2_si)

    @property
    def rs_metres(self) -> float:
        """Schwarzschild radius of total mass  r_s = 2 G M / c^2  [m]."""
        return 2.0 * G_SI * self.M_kg / C_SI ** 2

    @property
    def rs_geom(self) -> float:
        """Schwarzschild radius in geometric units (= 2 M)."""
        return 2.0 * self.M_tot

    @property
    def f_isco_hz(self) -> float:
        """Schwarzschild ISCO GW frequency   f_ISCO = c^3 / (6^{3/2} pi G M)."""
        return C_SI ** 3 / (6.0 ** 1.5 * PI * G_SI * self.M_kg)

    @property
    def chi_eff(self) -> float:
        """Effective aligned spin  chi_eff = (m1 chi1 + m2 chi2) / M."""
        return (self.m1_si * self.chi1 + self.m2_si * self.chi2) / self.M_tot

    def validate(self) -> bool:
        """Check physical admissibility of the parameter set."""
        ok = True
        ok &= self.m1_si > 0.0
        ok &= self.m2_si > 0.0
        ok &= -1.0 <= self.chi1 <= 1.0
        ok &= -1.0 <= self.chi2 <= 1.0
        ok &= 0.0 < self.f_low_hz < self.f_isco_hz
        ok &= self.distance_mpc > 0.0
        ok &= self.ell >= 2
        nu = self.symmetric_ratio
        ok &= 0.0 < nu <= 0.25 + 1.0e-14
        return ok


# ---------------------------------------------------------------------------
#  Convenience constructor for the "standard candle" used in the notebook
# ---------------------------------------------------------------------------
def standard_candle_bbh() -> BinaryParameters:
    """GW150914-like binary: 36 + 29 Msun, non-spinning, 440 Mpc."""
    return BinaryParameters(m1_si=36.0, m2_si=29.0, chi1=0.0, chi2=0.0,
                            f_low_hz=20.0, distance_mpc=440.0, ell=2)


def equal_mass_light_bbh() -> BinaryParameters:
    """Small equal-mass binary used for the reproducible small-scale experiment."""
    return BinaryParameters(m1_si=10.0, m2_si=10.0, chi1=0.0, chi2=0.0,
                            f_low_hz=30.0, distance_mpc=100.0, ell=2)
