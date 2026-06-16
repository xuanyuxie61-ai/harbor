"""
physical_constants.py - CODATA 2018 & nuclear astrophysics constants
All CGS unless noted. Used throughout the r-process network solver.
"""
import math

# ---------------- CODATA 2018 ----------------
C_LIGHT      = 2.99792458e10          # cm / s
K_BOLTZMANN  = 1.380649e-16           # erg / K
H_PLANCK     = 6.62607015e-27         # erg s
H_BAR        = H_PLANCK / (2.0 * math.pi)
N_AVOGADRO   = 6.02214076e23          # 1 / mol
G_GRAV       = 6.67430e-8             # cm^3 g^-1 s^-2
M_ELECTRON   = 9.1093837016e-28       # g
M_PROTON     = 1.67262192369e-24      # g
M_NEUTRON    = 1.67492749804e-24      # g
M_U          = 1.66053906660e-24      # atomic mass unit, g
E_CHARGE     = 4.80320425e-10         # statC
EV_ERG       = 1.602176634e-12        # erg / eV
MEV_ERG      = EV_ERG * 1.0e6
FM_CM        = 1.0e-13                # fm -> cm

# ---------------- Solar / astrophysical ----------------
M_SUN        = 1.98892e33             # g
Y_SOLAR_R    = 0.014                  # solar r-process mass fraction (approx)

# ---------------- Conversion helpers ----------------
def T9_to_T(T9: float) -> float:
    """Convert temperature in 10^9 K to Kelvin."""
    return T9 * 1.0e9

def T_to_T9(T: float) -> float:
    return T / 1.0e9

def MeV_to_erg(E_MeV: float) -> float:
    return E_MeV * MEV_ERG

def erg_to_MeV(E_erg: float) -> float:
    return E_erg / MEV_ERG

def amu_to_g(A: float) -> float:
    return A * M_U

# ---------------- Thermodynamic ----------------
def radiation_constant() -> float:
    """a = 8 pi^5 k_B^4 / (15 h^3 c^3)  [erg cm^-3 K^-4]"""
    num = 8.0 * math.pi**5 * K_BOLTZMANN**4
    den = 15.0 * H_PLANCK**3 * C_LIGHT**3
    return num / den

A_RAD = radiation_constant()

def fermi_coupling_constant() -> float:
    """G_F / (hbar c)^3 in MeV^-2  (standard value)."""
    return 1.1663787e-11

G_F = fermi_coupling_constant()

# ---------------- Weak-interaction scale ----------------
WEAK_MAGNETIC_CORRECTION = 1.0 + 0.032   # leading radiative correction

# ---------------- Nuclear scale ----------------
R0_FM = 1.25                              # fm, nuclear radius parameter
SURFACE_ENERGY_COEFF = 17.0               # MeV, liquid-drop surface term
COULOMB_COEFF = 0.71                      # MeV
SYMMETRY_COEFF = 23.0                     # MeV
PAIRING_COEFF = 12.0                      # MeV

# ---------------- Safety ----------------
TINY = 1.0e-300
HUGE = 1.0e300

__all__ = [
    "C_LIGHT", "K_BOLTZMANN", "H_PLANCK", "H_BAR", "N_AVOGADRO",
    "G_GRAV", "M_ELECTRON", "M_PROTON", "M_NEUTRON", "M_U",
    "E_CHARGE", "EV_ERG", "MEV_ERG", "FM_CM",
    "M_SUN", "Y_SOLAR_R",
    "T9_to_T", "T_to_T9", "MeV_to_erg", "erg_to_MeV", "amu_to_g",
    "radiation_constant", "A_RAD", "fermi_coupling_constant", "G_F",
    "WEAK_MAGNETIC_CORRECTION",
    "R0_FM", "SURFACE_ENERGY_COEFF", "COULOMB_COEFF",
    "SYMMETRY_COEFF", "PAIRING_COEFF",
    "TINY", "HUGE",
]
