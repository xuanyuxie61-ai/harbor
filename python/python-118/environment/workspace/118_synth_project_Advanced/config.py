"""
config.py

Simulation configuration and physical constants for molecular dynamics simulation
of alloy solid-liquid interface dynamics.

Physical Model:
    Binary alloy A-B (e.g., Cu-Ni) with FCC crystal structure.
    The system consists of a solid slab in contact with a liquid melt.
    
Key Parameters:
    - Lattice constant: a0 = 3.615 Angstrom (Cu)
    - Masses: m_A = 63.546 amu, m_B = 58.693 amu
    - Temperature: T = 1500 K (near melting)
    - Time step: dt = 1.0 fs
    
Potential Energy Model (EAM-like):
    The total energy is given by:
    
    E_total = sum_i [ sum_{j>i} phi_{alpha beta}(r_{ij}) 
                       + F_alpha( rho_i^bar ) ]
    
    where the embedding density is:
    
    rho_i^bar = sum_{j != i} f_beta( r_{ij} )
    
    phi_{alpha beta}(r) = 4 * epsilon_{alpha beta} * [ (sigma/r)^12 - (sigma/r)^6 ]
    
    The cross-interaction follows the Lorentz-Berthelot mixing rules:
    sigma_{AB} = (sigma_AA + sigma_BB) / 2
    epsilon_{AB} = sqrt( epsilon_AA * epsilon_BB )
"""

import numpy as np

# =============================================================================
# Physical Constants (SI units where applicable)
# =============================================================================
BOLTZMANN_KB = 1.380649e-23          # J/K
AVOGADRO_NA = 6.02214076e23          # mol^-1
ELEMENTARY_CHARGE = 1.602176634e-19  # C
ANGSTROM = 1.0e-10                   # m
FEMTOSECOND = 1.0e-15                # s
AMU = 1.66053906660e-27              # kg
EV_TO_J = 1.602176634e-19            # J/eV

# =============================================================================
# Simulation Box Parameters
# =============================================================================
LATTICE_CONSTANT = 3.615             # Angstrom (Cu FCC)
BOX_X = 5 * LATTICE_CONSTANT         # Angstrom
BOX_Y = 5 * LATTICE_CONSTANT
BOX_Z = 10 * LATTICE_CONSTANT

# =============================================================================
# Atomic Parameters for Binary Alloy A-B (Cu-Ni like)
# =============================================================================
MASS_A = 63.546 * AMU / AMU          # atomic mass units
MASS_B = 58.693 * AMU / AMU

# Lennard-Jones parameters
# In reduced units: epsilon in eV, sigma in Angstrom
EPSILON_AA = 0.415                   # eV
EPSILON_BB = 0.420                   # eV
SIGMA_AA = 2.315                     # Angstrom
SIGMA_BB = 2.227                     # Angstrom

# Mixing rules (Lorentz-Berthelot)
SIGMA_AB = (SIGMA_AA + SIGMA_BB) / 2.0
EPSILON_AB = np.sqrt(EPSILON_AA * EPSILON_BB)

# EAM embedding function parameters
# F(rho) = -A * sqrt(rho) + B * rho^2
# where rho is the local electron density
EAM_A = 1.5                          # eV * Angstrom^3
EAM_B = 0.05                         # eV * Angstrom^6
EAM_RHO0 = 0.1                       # reference density

# Pair potential cutoff
R_CUTOFF = 2.5 * max(SIGMA_AA, SIGMA_BB)  # Angstrom
R_CUTOFF_SQ = R_CUTOFF ** 2

# =============================================================================
# Thermodynamic State
# =============================================================================
TARGET_TEMPERATURE = 1500.0          # K
MELTING_TEMPERATURE = 1357.0         # K (Cu)
SUPERCOOLING_DELTA_T = 50.0          # K

# =============================================================================
# MD Integration Parameters
# =============================================================================
TIME_STEP = 1.0                      # femtoseconds
N_STEPS_EQUILIBRATION = 100
N_STEPS_PRODUCTION = 100
N_STEPS_THERMOSTAT = 10              # apply thermostat every N steps

# Nose-Hoover thermostat parameters
NOSE_HOOVER_Q = 10.0                 # fictitious mass parameter
NOSE_HOOVER_XI = 0.0                 # initial thermostat variable

# =============================================================================
# Monte Carlo Parameters (from Ising model concepts)
# =============================================================================
MC_SWAP_PROBABILITY = 0.1            # probability of attempting a swap
MC_N_SWAPS_PER_CYCLE = 5

# =============================================================================
# Order Parameter Parameters
# =============================================================================
# Steinhardt order parameters for solid/liquid identification
# Q_l = sqrt( 4*pi/(2*l+1) * sum_{m=-l}^l |Q_lm|^2 )
# Q_lm = (1/N_b) * sum_{j=1}^{N_b} Y_lm( theta_j, phi_j )
# where Y_lm are spherical harmonics
STEINHARDT_L = 6
Q6_THRESHOLD_SOLID = 0.45            # Q6 > threshold => solid
Q6_THRESHOLD_LIQUID = 0.25           # Q6 < threshold => liquid
INTERFACE_WIDTH_CUTOFF = 5.0         # Angstrom

# =============================================================================
# Sparse Grid Quadrature Parameters
# =============================================================================
SPARSE_GRID_DIM = 3
SPARSE_GRID_LEVEL_MAX = 3

# =============================================================================
# Transport / Diffusion Parameters
# =============================================================================
# Fick's law: J = -D * grad(c)
# Species conservation: dc/dt = D * nabla^2(c)
DIFFUSION_COEFFICIENT_SOLID = 1.0e-12   # m^2/s (approximate)
DIFFUSION_COEFFICIENT_LIQUID = 1.0e-9   # m^2/s
INTERFACE_MOBILITY = 0.1                # m/(J*s)

# =============================================================================
# Predator-Prey like species dynamics parameters
# At the interface, dissolution (A->liquid) and precipitation (B->solid)
# can be modeled as coupled reactions:
# dC_solid/dt = -k_dissolve * C_solid + k_precip * C_liquid
# dC_liquid/dt = +k_dissolve * C_solid - k_precip * C_liquid
DISSOLUTION_RATE = 0.01
PRECIPITATION_RATE = 0.005

# =============================================================================
# Numerical Tolerances and Bounds
# =============================================================================
EPSILON_MACHINE = np.finfo(float).eps
MIN_DISTANCE = 0.5                   # Angstrom, minimum allowed interatomic distance
MAX_FORCE = 100.0                    # eV/Angstrom, force capping

# =============================================================================
# Random Number Generation
# =============================================================================
RANDOM_SEED = 42


def get_reduced_units():
    """
    Convert to reduced (Lennard-Jones) units for numerical stability.
    
    In reduced units:
        length* = length / sigma
        energy* = energy / epsilon
        time* = time / sqrt(m * sigma^2 / epsilon)
        temperature* = k_B * T / epsilon
        force* = force * sigma / epsilon
    
    Returns:
        dict with conversion factors.
    """
    sigma_ref = SIGMA_AA
    epsilon_ref = EPSILON_AA
    mass_ref = MASS_A
    
    time_ref = np.sqrt(mass_ref * sigma_ref ** 2 / epsilon_ref)  # in fs units if mass in amu
    # Actually time_ref in internal units: sqrt(AMU * Angstrom^2 / eV)
    # = sqrt(1.66e-27 * 1e-20 / 1.6e-19) seconds = sqrt(1.0375e-28) = 1.018e-14 s = 10.18 fs
    time_ref = np.sqrt(AMU * ANGSTROM ** 2 / EV_TO_J) / FEMTOSECOND
    
    return {
        'sigma': sigma_ref,
        'epsilon': epsilon_ref,
        'mass': mass_ref,
        'time': time_ref,
        'temperature': epsilon_ref / (BOLTZMANN_KB / EV_TO_J),
    }
