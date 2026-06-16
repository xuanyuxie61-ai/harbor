"""
physical_constants.py
=====================
Central repository of physical constants, central-object parameters and
Shakura-Sunyaev accretion-disk reference scales.

Every numerical value used elsewhere in the project is sourced from this
module so that dimensional consistency can be audited from a single place.
The parameter layout follows the persistent-default pattern of the
dosage_parameters module (312_dosage_ode) but is extended to carry the
full set of MHD-normalisation scales needed for the shearing-box model.

Non-dimensionalisation
----------------------
We adopt the standard shearing-box normalisation

    length   ->  H   (disk scale height at the fiducial radius R0)
    time     ->  Omega0^{-1}  (inverse Keplerian angular velocity at R0)
    density  ->  rho0  (mid-plane volume density at R0)
    velocity ->  c_s0  (isothermal sound speed at R0)
    magnetic ->  sqrt(mu0 * rho0) * c_s0

so that the fiducial Mach number of the Keplerian flow is unity.  The
plasma beta is then beta = 2 c_s0^2 / v_A^2 and the initial toroidal
field is set by beta_0.

References
----------
- Shakura, N. I. & Sunyaev, R. A. 1973, A&A, 24, 337
- Balbus, S. A. & Hawley, J. F. 1991, ApJ, 376, 214  (MRI)
- Stone, J. M. et al. 1996, ApJ, 462, 823            (shearing box)
"""

from __future__ import annotations
import math
from typing import Any, Dict, Optional
import numpy as np


# ---------------------------------------------------------------------------
#                            CODATA 2018 constants
# ---------------------------------------------------------------------------
CGS_G      = 6.67430e-8        # cm^3 g^-1 s^-2  gravitational constant
CGS_C      = 2.99792458e10     # cm s^-1         speed of light
CGS_KB     = 1.380649e-16      # erg K^-1        Boltzmann constant
CGS_MP     = 1.67262192e-24    # g               proton mass
CGS_SIGMA  = 5.670374419e-5    # erg cm^-2 s^-1 K^-4   Stefan-Boltzmann
CGS_MU0_SI = 4.0e-7 * math.pi  # H m^-1 (SI)     vacuum permeability


# ---------------------------------------------------------------------------
#                       Astrophysical reference units
# ---------------------------------------------------------------------------
M_SUN   = 1.98892e33           # g
R_SUN   = 6.957e10             # cm
L_SUN   = 3.828e33             # erg s^-1
AU      = 1.495978707e13       # cm
PC      = 3.085677581e18       # cm
YR      = 3.15576e7            # s
DAY     = 8.64e4               # s


# ---------------------------------------------------------------------------
#                         Default configuration record
# ---------------------------------------------------------------------------
_DEFAULTS: Dict[str, Any] = {
    # Central compact object (stellar-mass black hole)
    "M_bh":        10.0,        # M_sun
    # Fiducial disk radius
    "R0":          1.0e11,      # cm  (roughly 0.7 R_sun; ~100 R_g)
    # Mid-plane conditions at R0
    "rho0":        1.0e-7,      # g cm^-3
    "T0":          1.0e6,       # K
    "mu_mw":       0.62,        # mean molecular weight (ionised solar)
    "gamma_eos":   5.0 / 3.0,   # adiabatic index
    # Shakura-Sunyaev alpha (used only for initial profile scaling)
    "alpha_SS":    0.1,
    # Initial plasma beta (sets the seed toroidal field amplitude)
    "beta0":       1.0e2,
    # Perturbation levels
    "amp_pressure": 1.0e-4,    # relative amplitude of p' seed
    "amp_velocity": 1.0e-4,    # relative amplitude of v' seed
    "amp_magnetic": 1.0e-4,    # relative amplitude of B' seed
    # Random seed for reproducible stochastic initial conditions
    "seed":        20260607,
    # Shearing-box domain size in units of H
    "Lx_over_H":   0.5,        # radial (sub-sonic shear)
    "Ly_over_H":   0.5,        # azimuthal
    "Lz_over_H":   0.25,       # vertical (stratification kept mild)
    # Base grid resolution per dimension (cells)
    "Nx": 32,
    "Ny": 32,
    "Nz": 8,
    # Time integration
    "t_end":       40.0,        # in Omega0^{-1}
    "cfl":         0.4,
    "rk_stage":    3,           # SSP-RK order
    # Diagnostics
    "dt_snapshot":  2.0,        # in Omega0^{-1}
    "dt_history":   0.1,        # in Omega0^{-1}
}


def reset_defaults(**overrides: Any) -> Dict[str, Any]:
    """Overwrite default configuration entries and return the full dict.

    This mirrors the persistent-default behaviour of the dosage_parameters
    module (312_dosage_ode): subsequent calls to ``get()`` will reflect
    the overridden values.
    """
    for key, val in overrides.items():
        if key not in _DEFAULTS:
            raise KeyError(f"physical_constants: unknown parameter '{key}'")
        _DEFAULTS[key] = val
    return dict(_DEFAULTS)


def get(key: Optional[str] = None) -> Any:
    """Return either a single parameter or the entire configuration."""
    if key is None:
        return dict(_DEFAULTS)
    if key not in _DEFAULTS:
        raise KeyError(f"physical_constants: unknown parameter '{key}'")
    return _DEFAULTS[key]


# ---------------------------------------------------------------------------
#        Derived shearing-box normalisation scales (CGS unless stated)
# ---------------------------------------------------------------------------
def derived_scales() -> Dict[str, float]:
    """Return the dimensional scales that follow from the primitive inputs.

    Returns a dictionary with at least the following entries:

        Omega0     : Keplerian angular velocity at R0  [s^-1]
        cs0        : isothermal sound speed            [cm s^-1]
        H          : pressure scale height             [cm]
        vA0        : Alfven speed for B set by beta0   [cm s^-1]
        B0         : initial toroidal field strength   [G]
        rho_g0     : vertical gravity gradient rho Omega^2 [cgs]
        q shear    : logarithmic shear dq = -d ln Omega / d ln R = 3/2
    """
    p = _DEFAULTS
    M    = p["M_bh"] * M_SUN
    R0   = p["R0"]
    rho0 = p["rho0"]
    T0   = p["T0"]
    mu   = p["mu_mw"]
    gam  = p["gamma_eos"]
    beta0 = p["beta0"]

    Omega0 = math.sqrt(CGS_G * M / R0**3)          # s^-1
    cs0    = math.sqrt(CGS_KB * T0 / (mu * CGS_MP)) # cm s^-1
    H      = cs0 / Omega0                           # cm
    # Plasma-beta definition: beta = 2 c_s^2 / v_A^2 with v_A = B / sqrt(4 pi rho)
    # => B0 = c_s0 * sqrt(8 pi rho0 / beta0)   (CGS, Gaussian units)
    B0 = cs0 * math.sqrt(8.0 * math.pi * rho0 / beta0)
    vA0 = B0 / math.sqrt(4.0 * math.pi * rho0)
    rho_g0 = rho0 * Omega0**2                       # vertical gradient amplitude

    return {
        "Omega0": Omega0,
        "cs0":    cs0,
        "H":      H,
        "vA0":    vA0,
        "B0":     B0,
        "rho_g0": rho_g0,
        "q_shear": 1.5,
        "M_bh_g": M,
        "R0_cm":  R0,
        "rho0":   rho0,
    }


# ---------------------------------------------------------------------------
#        MRI most-unstable wavelength (Balbus & Hawley 1991)
# ---------------------------------------------------------------------------
def mri_most_unstable_wavelength() -> float:
    """Return the wavelength of the fastest-growing MRI mode in cm.

    For a purely toroidal field in the incompressible limit the maximum
    growth rate is (3/4) Omega and occurs at

        lambda_max = 2 pi v_A / sqrt( (16/15) Omega )

    which reduces to a factor ~ 4.82 * v_A / Omega.
    """
    s = derived_scales()
    return 2.0 * math.pi * s["vA0"] / math.sqrt((16.0 / 15.0) * s["Omega0"])


# ---------------------------------------------------------------------------
#        Thermal and radiative scales
# ---------------------------------------------------------------------------
def radiation_pressure(T: float) -> float:
    """Radiation pressure p_rad = a T^4 / 3  in cgs (erg cm^-3)."""
    a = 4.0 * CGS_SIGMA / CGS_C
    return a * T**4 / 3.0


def gas_pressure(rho: float, T: float, mu: float = 0.62) -> float:
    """Ideal-gas pressure p = rho k_B T / (mu m_p)  in cgs."""
    return rho * CGS_KB * T / (mu * CGS_MP)


def sound_speed(rho: float, p: float, gamma: float = 5.0 / 3.0) -> float:
    """Adiabatic sound speed c_s = sqrt(gamma p / rho).

    A small floor is added for numerical safety.
    """
    floor = 1.0e-30
    return math.sqrt(gamma * max(p, floor) / max(rho, 1.0e-30))


def alfven_speed(B2: float, rho: float) -> float:
    """Alfven speed v_A = sqrt(B^2 / (4 pi rho))  (cgs, Gaussian)."""
    return math.sqrt(max(B2, 0.0) / (4.0 * math.pi * max(rho, 1.0e-30)))


def fast_magnetosonic(B2: float, rho: float, p: float,
                      gamma: float = 5.0 / 3.0) -> float:
    """Fast magnetosonic speed c_f = sqrt( (c_s^2 + v_A^2) ) for B perp k."""
    cs2 = gamma * max(p, 1.0e-30) / max(rho, 1.0e-30)
    va2 = max(B2, 0.0) / (4.0 * math.pi * max(rho, 1.0e-30))
    return math.sqrt(cs2 + va2)


def plasma_beta(p, B2):
    """Thermal plasma beta = 2 p / (B^2 / 8 pi).

    Works for scalar or array inputs (uses np.maximum).
    """
    p = np.asarray(p)
    B2 = np.asarray(B2)
    return 2.0 * np.maximum(p, 1.0e-30) / np.maximum(B2 / (8.0 * math.pi), 1.0e-30)
