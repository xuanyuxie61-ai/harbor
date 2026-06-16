"""
perovskite_constants.py
=======================
Physical, chemical, and numerical constants for perovskite solar cell
defect-state calculations. All values are in SI unless otherwise noted.

Key physical model parameters
-----------------------------
- MAPbI3 (methylammonium lead iodide) bandgap          : E_g = 1.55 eV
- Relative permittivity (static)                       : eps_r ~ 25 (high-k)
- Effective masses m_e*, m_h*                          : 0.12-0.15 m_0
- Thermal velocity v_th = sqrt(3 kT / m*)              : ~1e7 cm/s at 300 K
- Defect capture cross section sigma                   : 1e-15 cm^2 typical
- Shockley-Read-Hall (SRH) lifetime                    : tau_SRH = 1/(v_th * sigma * N_t)

The high-k nature of MAPbI3 (eps_r ~ 25) strongly screens charged defects;
combined with soft lattice phonons, this produces shallow defect states whose
transition levels can be computed via Poisson + drift-diffusion solvers
discretized with high-order finite differences.

References
----------
[1] Yin, W.-J., Shi, T., Yan, Y. "Unusual defect physics in CH3NH3PbI3."
    Appl. Phys. Lett. 104, 063903 (2014).
[2] Meggiolaro, D. et al. "Iodine chemistry determines the defect tolerance
    of lead-halide perovskites." Energy Environ. Sci. 11, 706 (2018).
"""

from __future__ import annotations
import math

# ---------------------------------------------------------------------------
# Fundamental constants (CODATA 2018)
# ---------------------------------------------------------------------------
E_CHARGE: float = 1.602176634e-19           # elementary charge [C]
HBAR: float = 1.054571817e-34               # reduced Planck [J.s]
K_B: float = 1.380649e-23                   # Boltzmann [J/K]
M_0: float = 9.1093837015e-31               # electron rest mass [kg]
EPS_0: float = 8.8541878128e-12             # vacuum permittivity [F/m]
C_LIGHT: float = 2.99792458e8               # speed of light [m/s]
N_A: float = 6.02214076e23                  # Avogadro constant [1/mol]

# ---------------------------------------------------------------------------
# Temperature
# ---------------------------------------------------------------------------
T_K: float = 300.0                          # room temperature [K]
THERMAL_VOLTAGE: float = K_B * T_K / E_CHARGE  # V_t = kT/q ~ 25.85 mV
BETA_T: float = 1.0 / (K_B * T_K)          # inverse thermal energy [1/J]

# ---------------------------------------------------------------------------
# MAPbI3 material parameters (room temperature)
# ---------------------------------------------------------------------------
BAND_GAP_EV: float = 1.55                   # bandgap [eV]
BAND_GAP_J: float = BAND_GAP_EV * E_CHARGE  # bandgap [J]
EPS_R_STATIC: float = 25.7                  # static dielectric constant
EPS_R_HIGH_FREQ: float = 5.1                # high-frequency dielectric
EPS_PERP: float = EPS_R_STATIC * EPS_0      # absolute permittivity [F/m]

EFFECTIVE_MASS_E: float = 0.12 * M_0        # electron eff. mass [kg]
EFFECTIVE_MASS_H: float = 0.15 * M_0        # hole eff. mass [kg]

# Thermal velocities v_th = sqrt(3 kT / m*)
V_TH_ELECTRON: float = math.sqrt(3.0 * K_B * T_K / EFFECTIVE_MASS_E)  # [m/s]
V_TH_HOLE: float = math.sqrt(3.0 * K_B * T_K / EFFECTIVE_MASS_H)      # [m/s]

# Density of states (3D parabolic): N_c,v = 2 (m* kT / 2 pi hbar^2)^(3/2)
_N_PREFAC_E = 2.0 * (EFFECTIVE_MASS_E * K_B * T_K
                     / (2.0 * math.pi * HBAR**2))**1.5
_N_PREFAC_H = 2.0 * (EFFECTIVE_MASS_H * K_B * T_K
                     / (2.0 * math.pi * HBAR**2))**1.5
DOS_CONDUCTION: float = _N_PREFAC_E          # N_c [1/m^3]
DOS_VALENCE: float = _N_PREFAC_H             # N_v [1/m^3]

# ---------------------------------------------------------------------------
# Defect parameters (typical iodine vacancy V_I in MAPbI3)
# ---------------------------------------------------------------------------
DEFECT_DEPTH_EV: float = 0.12               # shallow trap depth [eV]
DEFECT_CAPTURE_E: float = 1.0e-15 * 1.0e-4  # cm^2 -> m^2 capture cross-section
DEFECT_CAPTURE_H: float = 1.0e-15 * 1.0e-4
DEFECT_DENSITY_DEFAULT: float = 1.0e15 * 1.0e6  # 1e15 cm^-3 -> m^-3

# SRH lifetime: tau = 1 / (v_th * sigma * N_t)
SRH_LIFETIME_E: float = 1.0 / (
    V_TH_ELECTRON * DEFECT_CAPTURE_E * DEFECT_DENSITY_DEFAULT)
SRH_LIFETIME_H: float = 1.0 / (
    V_TH_HOLE * DEFECT_CAPTURE_H * DEFECT_DENSITY_DEFAULT)

# ---------------------------------------------------------------------------
# Device geometry (small-scale reproducible experiment, 1D slab)
# ---------------------------------------------------------------------------
DEVICE_LENGTH_NM: float = 500.0             # active layer thickness [nm]
DEVICE_LENGTH_M: float = DEVICE_LENGTH_NM * 1.0e-9
V_BI: float = 1.0                           # built-in voltage [V]
V_APPLIED: float = 0.9                      # operating voltage [V]
V_NET: float = V_BI - V_APPLIED             # net voltage across slab [V]

# ---------------------------------------------------------------------------
# Numerical parameters for the high-order finite-difference scheme
# ---------------------------------------------------------------------------
FD_ORDER: int = 6                           # stencil half-width p (O(h^{2p}))
FD_HALO: int = FD_ORDER                     # ghost cells on each side
DEFAULT_NX: int = 128                       # default number of grid points
DEFAULT_NY: int = 32                        # for 2D quadrature patches

# von Neumann stability limit for explicit diffusion:
#   dt <= dx^2 / (2 D)    (CFL for FTCS)
# High-order centered schemes tighten the limit by a factor of
#   S_p = sum_{k=1}^{p} [2 (p!)^2 / ((p+k)!(p-k)! k^2)]^-1
def fd_stability_factor(p: int) -> float:
    """Return the von Neumann stability factor S_p for the 2p-th order
    centered finite-difference Laplacian. For p = 1 (standard 3-point),
    S_1 = 0.5, recovering the classical FTCS bound."""
    s_inv = 0.0
    fact_p = math.factorial(p)
    for k in range(1, p + 1):
        num = 2.0 * fact_p * fact_p
        den = math.factorial(p + k) * math.factorial(p - k) * k * k
        s_inv += num / den
    return 1.0 / s_inv if s_inv > 0.0 else 0.5


STABILITY_FACTOR: float = fd_stability_factor(FD_ORDER)

# ---------------------------------------------------------------------------
# Langevin / Brownian sampler constants for defect configuration sampling
# ---------------------------------------------------------------------------
LANGEVIN_GAMMA: float = 1.0e12              # friction [1/s] (phonon bath)
LANGEVIN_H: float = 1.0e-15                 # time step [s]

# ---------------------------------------------------------------------------
# ML defect-predictor hyperparameters (RandomForest defaults)
# ---------------------------------------------------------------------------
ML_N_ESTIMATORS: int = 120
ML_MAX_DEPTH: int = 14
ML_RANDOM_STATE: int = 283

# ---------------------------------------------------------------------------
# Helper: thermal de Broglie wavelength (useful for DOS sanity check)
# ---------------------------------------------------------------------------
def thermal_debroglie(m_star: float) -> float:
    """Lambda_th = h / sqrt(2 pi m* kT)."""
    return HBAR * 2.0 * math.pi / math.sqrt(2.0 * math.pi * m_star * K_B * T_K)


LAMBDA_TH_E: float = thermal_debroglie(EFFECTIVE_MASS_E)
LAMBDA_TH_H: float = thermal_debroglie(EFFECTIVE_MASS_H)


def sanity_check() -> dict:
    """Return a dict of sanity values for regression testing."""
    return {
        "V_t [mV]": round(THERMAL_VOLTAGE * 1e3, 3),
        "N_c [1/m^3]": f"{DOS_CONDUCTION:.3e}",
        "N_v [1/m^3]": f"{DOS_VALENCE:.3e}",
        "tau_SRH_e [ns]": round(SRH_LIFETIME_E * 1e9, 3),
        "tau_SRH_h [ns]": round(SRH_LIFETIME_H * 1e9, 3),
        "S_p (p=6)": round(STABILITY_FACTOR, 6),
        "Lambda_th_e [nm]": round(LAMBDA_TH_E * 1e9, 4),
        "Lambda_th_h [nm]": round(LAMBDA_TH_H * 1e9, 4),
    }
