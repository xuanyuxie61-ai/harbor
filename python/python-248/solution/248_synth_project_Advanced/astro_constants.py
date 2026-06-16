"""
astro_constants.py
==================

CGS physical constants, cosmological parameters, and default simulation
hyper-parameters for the galaxy-formation hydrodynamics test problem.

All quantities are in CGS unless otherwise noted.  Default simulation
choices (grid resolution, ADIABATIC_INDEX, etc.) correspond to a small,
reproducible 3-D test volume described in README_博士级合成说明.md.

Reference
---------
- Draine, B. T. 2011, Physics of the Interstellar and Intergalactic Medium
- Springel, V. 2010, ARA&A 48, 325 (cosmological SPH review)
- Sharma, P., & Nath, B. B. 2007, MNRAS 380, 1459 (cooling + feedback)
"""

from __future__ import annotations
import math


# =====================================================================
#                           CGS PHYSICAL CONSTANTS
# =====================================================================

BOLTZMANN_CGS = 1.380649e-16            # erg K^{-1}
PROTON_MASS_CGS = 1.6726219e-24         # g
ELECTRON_MASS_CGS = 9.109383e-28        # g
GRAVITATIONAL_CGS = 6.67430e-8          # cm^3 g^{-1} s^{-2}
SPEED_OF_LIGHT_CGS = 2.99792458e10      # cm s^{-1}
SOLAR_MASS_CGS = 1.98892e33             # g
PARSEC_CGS = 3.0856776e18               # cm
KILOPARSEC_CGS = 3.0856776e21           # cm
MEGAPARSEC_CGS = 3.0856776e24           # cm
YEAR_CGS = 3.15576e7                    # s
MEGAYEAR_CGS = 3.15576e13               # s
# back-compat alias
MEWAYEAR_CGS = MEGAYEAR_CGS
RYDBERG_CGS = 2.179872e-11              # erg
THOMSON_CGS = 6.652459e-25              # cm^2
STEFAN_BOLTZMANN_CGS = 5.670374e-5      # erg cm^{-2} s^{-1} K^{-4}

# =====================================================================
#                    COSMOLOGICAL PARAMETERS (Planck 2018)
# =====================================================================

HUBBLE_100 = 1.0e5 * 100.0 / MEGAPARSEC_CGS    # 100 km/s/Mpc in s^{-1}
HUBBLE_LITTLE_H = 0.6774
OMEGA_MATTER = 0.3089
OMEGA_LAMBDA = 0.6911
OMEGA_BARYON = 0.0490
CMB_TEMPERATURE_K = 2.7255
SIGMA_8 = 0.8159
SPECTRAL_INDEX_NS = 0.9665

# =====================================================================
#            GALAXY-FORMATION ISM / IGM DEFAULT PARAMETERS
# =====================================================================

# Primordial + metal-line gas composition
MEAN_MOLECULAR_WEIGHT_NEUTRAL = 1.22          # mu for neutral primordial gas
MEAN_MOLECULAR_WEIGHT_IONIZED = 0.59          # mu for fully ionized gas
METAL_MASS_FRACTION_SOLAR = 0.0142            # Z_sun (Asplund et al. 2009)
HELIUM_MASS_FRACTION = 0.248                  # Y_p (BBN)
HYDROGEN_MASS_FRACTION = 1.0 - HELIUM_MASS_FRACTION - METAL_MASS_FRACTION_SOLAR

# Reference ISM scales
TYPICAL_ISM_NUMBER_DENSITY_CM3 = 1.0          # cm^{-3}
TYPICAL_ISM_TEMPERATURE_K = 8000.0            # K (warm neutral medium)
TYPICAL_TURBULENT_VELOCITY_KMS = 10.0         # km/s (McKee & Ostriker 1977)

# =====================================================================
#                      SIMULATION HYPER-PARAMETERS
# =====================================================================

# High-order finite-difference stencil choices
FD_ORDER_DEFAULT = 4              # 4th-order centered differences
FD_ORDER_MAX = 6                  # 6th-order (for smooth-region benchmarks)
WENO_ORDER = 5                    # WENO5-J / WENO5-Z reconstruction

# SSP-RK3 time integration
RK_ORDER = 3
RK_CFL_NUMBER = 0.4               # safety factor below CFL hard limit
RK_CFL_MAX = 0.9                  # absolute CFL ceiling

# WENO non-linear weights
WENO_EPS_DEFAULT = 1.0e-36        # classical WENO-J epsilon
WENO_EPS_ZETA = 1.0e-99           # WENO-Z extra power epsilon (Borges 2008)
WENO_Z_POWER = 2                  # WENO-Z power p (p=2 gives WENO-Z)

# Cooling function tabulation
COOLING_LOG_TEMP_MIN = 4.0        # log10(T/K)
COOLING_LOG_TEMP_MAX = 9.0
COOLING_TABLE_SIZE = 256

# Voronoi / Knapsack refinement
MAX_AMR_LEVELS = 4
KNAPSACK_WEIGHT_BUDGET_FRAC = 0.10   # 10% memory budget per refinement pass
CAUSAL_BOOTSTRAP_SAMPLES = 64     # bootstrap resamples for causal analysis

# Default 3-D test problem (small reproducible volume)
DOMAIN_SIZE_KPC = 10.0            # box side in proper kpc
BASELINE_GRID_CELLS_PER_AXIS = 16 # 16^3 base grid -> 4096 cells
OUTPUT_INTERVAL_MYR = 5.0         # myr between state snapshots
TOTAL_SIMULATION_TIME_MYR = 50.0  # myr of evolved time

# =====================================================================
#                         DERIVED CONSTANTS
# =====================================================================

def sound_speed_cgs(temperature_k: float, mu: float = MEAN_MOLECULAR_WEIGHT_NEUTRAL, gamma: float = 5.0 / 3.0) -> float:
    """Adiabatic sound speed c_s = sqrt(gamma * k_B * T / (mu * m_p))."""
    return math.sqrt(gamma * BOLTZMANN_CGS * temperature_k / (mu * PROTON_MASS_CGS))


def jeans_length_cgs(density_cgs: float, temperature_k: float, mu: float = MEAN_MOLECULAR_WEIGHT_NEUTRAL, gamma: float = 5.0 / 3.0) -> float:
    """
    Jeans length  L_J = c_s * sqrt(pi / (G * rho)).
    Gas parcels larger than L_J collapse gravitationally.
    """
    cs = sound_speed_cgs(temperature_k, mu, gamma)
    return cs * math.sqrt(math.pi / (GRAVITATIONAL_CGS * max(density_cgs, 1.0e-60)))


def free_fall_time_cgs(density_cgs: float) -> float:
    """Free-fall time  t_ff = sqrt(3 pi / (32 G rho))."""
    return math.sqrt(3.0 * math.pi / (32.0 * GRAVITATIONAL_CGS * max(density_cgs, 1.0e-60)))


def thermal_energy_per_mass_cgs(temperature_k: float, mu: float = MEAN_MOLECULAR_WEIGHT_NEUTRAL, gamma: float = 5.0 / 3.0) -> float:
    """
    Specific thermal energy  e_th = k_B T / ((gamma - 1) mu m_p).
    """
    return BOLTZMANN_CGS * temperature_k / ((gamma - 1.0) * mu * PROTON_MASS_CGS)
