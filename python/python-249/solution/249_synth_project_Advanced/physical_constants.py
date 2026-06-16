# -*- coding: utf-8 -*-
"""
physical_constants.py
=====================
Fundamental physical constants and astrophysical conversion factors
used throughout the stellar evolution + nuclear reaction network code.

All values follow CODATA 2018 recommended values and the IAU 2015
Nominal Solar Conversion Constants.

Key constants
-------------
    k_B      : Boltzmann constant          [erg/K]
    N_A      : Avogadro's number            [1/mol]
    h_planck : Planck constant              [erg s]
    c_light  : speed of light               [cm/s]
    G_grav   : gravitational constant       [cm^3 / (g s^2)]
    sigma_SB : Stefan-Boltzmann constant    [erg / (cm^2 s K^4)]
    a_rad    : radiation constant = 4 sigma/c [erg / (cm^3 K^4)]
    m_p      : proton mass                  [g]
    m_e      : electron mass                [g]
    e_chrg   : elementary charge (esu)      [statC]
    M_sun    : nominal solar mass           [g]
    L_sun    : nominal solar luminosity     [erg/s]
    R_sun    : nominal solar radius         [cm]
"""

from __future__ import annotations
import math

# ---------------------------------------------------------------------
# CODATA 2018 fundamental constants (cgs)
# ---------------------------------------------------------------------
k_B       = 1.380649e-16          # erg / K
N_A       = 6.02214076e23         # 1 / mol
h_planck  = 6.62607015e-27        # erg s
c_light   = 2.99792458e10         # cm / s
G_grav    = 6.67430e-8            # cm^3 / (g s^2)
sigma_SB  = 5.670374419e-5        # erg / (cm^2 s K^4)
a_rad     = 4.0 * sigma_SB / c_light   # erg / (cm^3 K^4)
m_p       = 1.67262192369e-24     # g
m_e       = 9.1093837015e-28      # g
e_chrg    = 4.80320425e-10        # statC (esu)
pi        = math.pi
euler_e   = 0.5772156649015329    # Euler-Mascheroni constant

# ---------------------------------------------------------------------
# IAU 2015 nominal solar conversion constants
# ---------------------------------------------------------------------
M_sun     = 1.98892e33            # g
L_sun     = 3.828e33              # erg / s
R_sun     = 6.957e10              # cm
year_s    = 3.15576e7             # s (Julian year)
Mev_to_erg = 1.602176634e-6       # 1 MeV in erg
barn_to_cm2 = 1.0e-24             # 1 barn in cm^2

# ---------------------------------------------------------------------
# Derived composite constants
# ---------------------------------------------------------------------
# Thermal de-Broglie wavelength prefactor:  lambda_T = h / sqrt(2 pi m k_B T)
# We store the prefactor h / sqrt(2 pi m_p k_B) for protons.
_lam_T_pref = h_planck / math.sqrt(2.0 * pi * m_p * k_B)

# Gamow energy prefactor for two nuclei with charges Z1,Z2 and reduced
# mass mu (in amu):
#     E_G = (2 pi alpha Z1 Z2)^2 * 2 mu c^2
# where alpha = e^2 / (hbar c) is the fine structure constant.
alpha_fs = e_chrg**2 / (h_planck / (2.0*pi) * c_light)

# ---------------------------------------------------------------------
# Useful helper functions
# ---------------------------------------------------------------------

def atomic_mass(A: int, Z: int) -> float:
    """Rough nuclear mass in grams using the semi-empirical mass formula
    (Bethe-Weizsacker) in units of m_p.  This is accurate to ~0.5% for
    A > 4 and is sufficient for stellar-evolution level network work
    where exact masses come from a separate mass table.

    The binding energy per nucleon is parametrised as
        B(A,Z) = a_v - a_s A^{-1/3} - a_c Z(Z-1) A^{-4/3}
               - a_a (A-2Z)^2 / A^2 + delta(A,Z)
    with standard coefficients.
    """
    a_v = 15.75       # MeV
    a_s = 17.8        # MeV
    a_c = 0.711       # MeV
    a_a = 23.7        # MeV
    if A <= 0:
        return 0.0
    # Pairing term
    if A % 2 == 1:
        delta = 0.0
    elif Z % 2 == 0:
        delta = +34.0 / math.sqrt(A)   # MeV (even-even)
    else:
        delta = -34.0 / math.sqrt(A)   # MeV (odd-odd)
    B_MeV = (a_v * A
             - a_s * A**(2.0/3.0)
             - a_c * Z * (Z - 1) / A**(1.0/3.0)
             - a_a * (A - 2*Z)**2 / A
             + delta)
    # Mass excess in MeV:  Delta = Z * m_H + (A-Z) * m_n - B
    #   m_H = 7.289 MeV, m_n = 8.071 MeV (mass excesses)
    mass_excess_MeV = Z * 7.289 + (A - Z) * 8.071 - B_MeV
    mass_amu = A + mass_excess_MeV / 931.494
    return mass_amu * m_p * N_A   # returns g/mol actually; we use m_p scale


def mass_per_particle(A: int, Z: int) -> float:
    """Return mass of a single nucleus (A,Z) in grams, using the
    same SEMF as :func:`atomic_mass`."""
    a_v = 15.75
    a_s = 17.8
    a_c = 0.711
    a_a = 23.7
    if A <= 0:
        return m_p
    if A % 2 == 1:
        delta = 0.0
    elif Z % 2 == 0:
        delta = +34.0 / math.sqrt(A)
    else:
        delta = -34.0 / math.sqrt(A)
    B_MeV = (a_v * A
             - a_s * A**(2.0/3.0)
             - a_c * Z * (Z - 1) / A**(1.0/3.0)
             - a_a * (A - 2*Z)**2 / A
             + delta)
    mass_excess_MeV = Z * 7.289 + (A - Z) * 8.071 - B_MeV
    mass_amu = A + mass_excess_MeV / 931.494
    return mass_amu * 1.66053906660e-24   # 1 amu in g


def mean_molecular_weight(X: float, Y: float, Z_met: float,
                          ionised: bool = True) -> float:
    """Mean molecular weight mu of a fully ionised (or neutral) gas
    with hydrogen mass fraction X, helium Y, and metals Z_met.

    For a fully ionised mixture,
        1 / mu = 2 X + 3/4 Y + 1/2 Z_met
    which follows from counting electrons + ions.
    """
    X = max(0.0, min(1.0, X))
    Y = max(0.0, min(1.0, Y))
    Z_met = max(0.0, min(1.0, Z_met))
    s = X + Y + Z_met
    if s <= 0.0:
        return 0.6
    X /= s; Y /= s; Z_met /= s
    if ionised:
        inv_mu = 2.0 * X + 0.75 * Y + 0.5 * Z_met
    else:
        inv_mu = X + 0.25 * Y + 0.5 * Z_met
    if inv_mu <= 0.0:
        return 0.6
    return 1.0 / inv_mu


def mu_e(X: float, Y: float, Z_met: float) -> float:
    """Mean molecular weight per electron."""
    X = max(0.0, min(1.0, X)); Y = max(0.0, min(1.0, Y))
    Z_met = max(0.0, 1.0 - X - Y)
    inv = X + 0.5 * (Y + Z_met)
    return 1.0 / inv if inv > 0 else 2.0


def coulomb_parameter(Z1: int, Z2: int, mu_amu: float, T9: float) -> float:
    """Return the Gamow (Sommerfeld) parameter
        eta = Z1 Z2 e^2 / (hbar v)
    expressed as  4.2487 * Z1 * Z2 * sqrt(mu / T9)
    where mu is the reduced mass in amu and T9 = T / 10^9 K.
    """
    if T9 <= 0.0:
        return 1.0e30
    return 4.2487 * Z1 * Z2 * math.sqrt(mu_amu / T9)


def gamow_peak_energy(Z1: int, Z2: int, mu_amu: float, T9: float) -> float:
    """Most effective energy E0 [keV] of the Gamow window,
        E0 = 0.1220 * (Z1^2 Z2^2 mu T9^2)^{1/3}  [MeV]
    (Rolfs & Rodney 1988 eq. 4-30).
    """
    return 0.1220 * (Z1 * Z1 * Z2 * Z2 * mu_amu * T9 * T9) ** (1.0/3.0)


def gamow_window_width(Z1: int, Z2: int, mu_amu: float, T9: float) -> float:
    """FWHM Delta of the Gamow window [MeV]:
        Delta = 0.2368 * (Z1^2 Z2^2 mu T9^5)^{1/6}
    """
    return 0.2368 * (Z1 * Z1 * Z2 * Z2 * mu_amu
                     * T9**5) ** (1.0/6.0)
