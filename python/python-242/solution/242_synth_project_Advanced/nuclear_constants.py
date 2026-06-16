"""
nuclear_constants.py  --  Fundamental physical constants and conversion factors
============================================================================
Used throughout the shell-model calculation. All quantities in MeV / fm
units (natural nuclear-physics conventions) unless otherwise noted.

Key constants (PDG 2024):
    hbar * c       = 197.3269804 MeV fm
    m_p            = 938.27208816  MeV/c^2
    m_n            = 939.56542052  MeV/c^2
    e^2            = 1.43997643    MeV fm   (Coulomb constant in nuclear units)
    r_0            = 1.25          fm       (Woods-Saxon radius parameter)
    a_surf         = 0.55          fm       (Woods-Saxon diffuseness)
    V_0            = 50.0          MeV      (central WS depth, typical)
    V_so           = 5.0           MeV      (spin-orbit WS depth)
    m_reduced      ~ A_eff * m_N            (reduced nucleon mass in finite well)

Conversion factors (Bohr magneton for nucleons):
    mu_N = e hbar / (2 m_p c) = 0.105155 nuclear magnetons
    quadrupole operator scale: e * r^2 with e_eff(proton)=1.5, e_eff(neutron)=0.5

Reference:
    Bohr & Mottelson, Nuclear Structure Vol. I & II (1998)
    Heyde, Basic Ideas and Concepts in Nuclear Physics (2004)
"""

from __future__ import annotations
import math

# ----------------------------------------------------------------------
#  Fundamental physical constants (MeV / fm / s convention)
# ----------------------------------------------------------------------
HBAR_C: float = 197.3269804          # hbar * c  [MeV fm]
HBAR: float = 6.582119569e-22        # hbar      [MeV s]
C_LIGHT: float = 2.99792458e23       # c         [fm / s]
M_PROTON: float = 938.27208816       # proton mass [MeV/c^2]
M_NEUTRON: float = 939.56542052      # neutron mass [MeV/c^2]
M_ELECTRON: float = 0.51099895       # electron mass [MeV/c^2]
E_CHARGED_SQ: float = 1.43997643     # e^2 in MeV fm (Coulomb kernel)
PI: float = math.pi
TWO_PI: float = 2.0 * PI
SQRT_PI: float = math.sqrt(PI)
FOUR_PI: float = 4.0 * PI

# ----------------------------------------------------------------------
#  Woods-Saxon parameterisation (global fit, Satchel 1954)
# ----------------------------------------------------------------------
R0_FM: float = 1.25                  # fm  (r = R0 * A**(1/3))
DIFFUSIVENESS_A: float = 0.55        # fm  (surface diffuseness)
V0_CENTRAL: float = 50.0             # MeV (central depth)
V_SO: float = 5.0                    # MeV (spin-orbit strength)
V_COULOMB_R0: float = 1.25           # fm  (Coulomb radius parameter, protons)

# ----------------------------------------------------------------------
#  Effective charges and g-factors for E2 / M1 transitions
# ----------------------------------------------------------------------
E_EFF_PROTON: float = 1.5            # e (free proton effective charge)
E_EFF_NEUTRON: float = 0.5           # e (free neutron effective charge)
G_L_PROTON: float = 1.0              # orbital g-factor
G_L_NEUTRON: float = 0.0             # orbital g-factor
G_S_PROTON_FREE: float = 5.585694    # spin g-factor, free
G_S_NEUTRON_FREE: float = -3.826083  # spin g-factor, free
G_S_PROTON_QUENCH: float = 0.7 * G_S_PROTON_FREE   # quenched in medium
G_S_NEUTRON_QUENCH: float = 0.7 * G_S_NEUTRON_FREE

# ----------------------------------------------------------------------
#  Nuclear magneton and Bohr-magneton-like constants
# ----------------------------------------------------------------------
MU_NUCLEAR: float = E_CHARGED_SQ * HBAR / (2.0 * M_PROTON * C_LIGHT)

# ----------------------------------------------------------------------
#  Model-space parameters for 1d0f shell (16O -> 40Ca region)
# ----------------------------------------------------------------------
MODEL_SPACE_NAME: str = "sd-shell"
A_CORE: int = 16                     # inert core mass number
Z_CORE: int = 8                      # core proton number
N_CORE: int = 8                      # core neutron number

# ----------------------------------------------------------------------
#  Numerical tolerances
# ----------------------------------------------------------------------
ENERGY_TOL_MEV: float = 1.0e-8       # MeV convergence for eigenvalues
WAVE_TOL: float = 1.0e-10            # L2 norm tolerance for wavefunctions
MAX_RADIAL_GRID: float = 25.0        # fm, outer boundary of radial mesh
NR_RADIAL: int = 401                 # number of radial points
R_EPSILON: float = 1.0e-12           # regularisation for r=0
STABILITY_CFL: float = 0.5           # Courant number for time propagation

# ----------------------------------------------------------------------
#  Derived quantities
# ----------------------------------------------------------------------
def radius_A(A: int) -> float:
    """Nuclear radius R = R0 * A^(1/3) in fm."""
    return R0_FM * (A ** (1.0 / 3.0))


def coulomb_barrier(Z: int, A: int) -> float:
    """Coulomb barrier height for a proton on a (Z-1, A-1) core."""
    R = V_COULOMB_R0 * ((A - 1) ** (1.0 / 3.0))
    return E_CHARGED_SQ * (Z - 1) / R


def reduced_mass(A_core: int, nucleon_kind: str = "neutron") -> float:
    """Reduced mass in MeV/c^2 of a single nucleon coupled to a core."""
    m_core = A_core * 0.5 * (M_PROTON + M_NEUTRON)
    m_nuc = M_NEUTRON if nucleon_kind == "neutron" else M_PROTON
    return (m_core * m_nuc) / (m_core + m_nuc)


def fermi_energy(A: int) -> float:
    """Fermi energy E_F = hbar^2 k_F^2 / (2m) with k_F ~ 1.36 fm^-1."""
    k_F = 1.36
    m_eff = 0.5 * (M_PROTON + M_NEUTRON)
    return (HBAR_C ** 2) * (k_F ** 2) / (2.0 * m_eff)
