"""
physics_constants.py
====================
Fundamental nuclear, plasma and mathematical constants used throughout the
tritium-breeding-blanket neutron transport calculation.

All units follow the standard fusion-neutronics convention:
  - energies in MeV,
  - cross sections in barn (1 barn = 1e-24 cm^2),
  - number densities in atoms/barn-cm,
  - lengths in cm,
  - angles in radians.

The module is intentionally dependency-free so it can be imported by every
other submodule without circular-import risk.
"""

from __future__ import annotations
import math
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# Universal constants
# ---------------------------------------------------------------------------
PI: float = math.pi
E: float = math.e
SQRT2: float = math.sqrt(2.0)
LN2: float = math.log(2.0)
AVOGADRO: float = 6.02214076e23          # mol^{-1}
BARN_TO_CM2: float = 1.0e-24             # cm^2 per barn
MEV_TO_ERG: float = 1.602176634e-6       # erg per MeV
MEV_TO_JOULE: float = 1.602176634e-13    # J per MeV
AMU_TO_G: float = 1.66053906660e-24      # g per amu
C_LIGHT: float = 2.99792458e10           # cm/s
K_BOLTZMANN_MEV: float = 8.617333262e-11 # MeV/K


# ---------------------------------------------------------------------------
# Nuclear masses and Q-values for the TBR reactions (ENDF/B-VIII.0)
# ---------------------------------------------------------------------------
MASS_NEUTRON: float = 1.00866491588      # amu
MASS_PROTON: float = 1.007276466812      # amu
MASS_DEUTERON: float = 2.01355321274     # amu
MASS_TRITIUM: float = 3.0155007134       # amu
MASS_HE4: float = 4.00260325413          # amu
MASS_LI6: float = 6.015122795            # amu
MASS_LI7: float = 7.0160034366           # amu

# D-T fusion kinematics -----------------------------------------------
# D + T -> n(14.06 MeV) + alpha(3.52 MeV), Q = 17.59 MeV
Q_DT: float = 17.5897                    # MeV
E_NEUTRON_DT: float = 14.06              # MeV (mono-energetic source)
E_ALPHA_DT: float = 3.52                 # MeV

# TBR reaction Q-values
Q_LI6_NT: float = 4.783                  # MeV  :  6Li(n,t)4He
Q_LI7_NNT: float = -2.468                # MeV  :  7Li(n,n't)4He (threshold)


# ---------------------------------------------------------------------------
# Multigroup energy structure (Vitamin-J 175-group, truncated here to the
# 14 fast-neutron groups that dominate transport in a breeding blanket).
# Upper and lower bounds in MeV.
# ---------------------------------------------------------------------------
GROUP_BOUNDS_MEV: Tuple[float, ...] = (
    14.918, 13.738, 12.214, 10.667, 9.138, 7.408, 6.065,
    4.724, 3.400, 2.466, 1.738, 1.227, 0.871, 0.554, 0.0
)
N_GROUPS: int = len(GROUP_BOUNDS_MEV) - 1     # 14 fast groups
DELTAS_MEV: Tuple[float, ...] = tuple(
    GROUP_BOUNDS_MEV[g] - GROUP_BOUNDS_MEV[g + 1]
    for g in range(N_GROUPS)
)

# Representative lethargy widths:  u = ln(E_{g-1}/E_g)
LETHARGY_WIDTHS: Tuple[float, ...] = tuple(
    math.log(GROUP_BOUNDS_MEV[g] / max(GROUP_BOUNDS_MEV[g + 1], 1.0e-10))
    for g in range(N_GROUPS)
)


# ---------------------------------------------------------------------------
# Material specifications (demo HCPB / HCLL hybrid blanket)
# ---------------------------------------------------------------------------
# Atomic densities (atoms / barn-cm).  Li enrichment: 60% Li-6.
DENSITY_LI2O_GCM3: float = 2.013
MOLAR_MASS_LI2O: float = 29.88
ATOM_DENSITY_LI2O: float = (
    AVOGADRO * DENSITY_LI2O_GCM3 / MOLAR_MASS_LI2O * BARN_TO_CM2
)
# Stoichiometry: Li2O -> 2 Li, 1 O per molecule
ATOM_DENSITY_LI: float = 2.0 * ATOM_DENSITY_LI2O
ATOM_DENSITY_O: float = 1.0 * ATOM_DENSITY_LI2O
FRAC_LI6: float = 0.60
FRAC_LI7: float = 0.40

# LiPb eutectic (Pb-17Li)
DENSITY_LIPB_GCM3: float = 9.5
MOLAR_MASS_LIPB: float = 110.9
ATOM_DENSITY_LIPB: float = (
    AVOGADRO * DENSITY_LIPB_GCM3 / MOLAR_MASS_LIPB * BARN_TO_CM2
)
FRAC_PB: float = 0.83
FRAC_LI_IN_LIPB: float = 0.17

# Eurofer-97 structural steel (simplified)
DENSITY_EUROFER: float = 7.87
MOLAR_MASS_EUROFER: float = 55.0
ATOM_DENSITY_EUROFER: float = (
    AVOGADRO * DENSITY_EUROFER / MOLAR_MASS_EUROFER * BARN_TO_CM2
)


# ---------------------------------------------------------------------------
# Representative one-group macroscopic cross sections (barn-cm = cm^{-1})
# These are effective collapsed values; the full energy dependence is built
# by cross_sections.py using the Dirichlet-multigroup model.
# ---------------------------------------------------------------------------
# Microscopic (barn): 1 barn * atom_density -> macroscopic (cm^{-1})
def microscopic_to_macroscopic(sigma_barn: float,
                               atom_density: float) -> float:
    """Return Sigma (cm^{-1}) given sigma (barn) and N (atoms/barn-cm).

    The relation is  Sigma = N * sigma  with sigma in barn and N chosen so
    that the barn unit cancels.  Because N here is in atoms/barn-cm the
    product has units cm^{-1} directly.
    """
    return sigma_barn * atom_density


# Representative infinite-dilution capture cross sections (barn)
SIGMA_T_LI6: Dict[int, float] = {
    # group index -> total cross section (barn)
    0: 2.40, 1: 2.55, 2: 2.72, 3: 2.91, 4: 3.12,
    5: 3.34, 6: 3.58, 7: 3.84, 8: 4.12, 9: 4.44,
    10: 4.80, 11: 5.21, 12: 5.68, 13: 6.22,
}
SIGMA_T_LI7: Dict[int, float] = {
    0: 2.10, 1: 2.22, 2: 2.36, 3: 2.52, 4: 2.70,
    5: 2.90, 6: 3.12, 7: 3.37, 8: 3.65, 9: 3.97,
    10: 4.33, 11: 4.74, 12: 5.21, 13: 5.76,
}

# (n,t) reaction on Li-6 (1/v dominated, thermal ~940 barn, fast ~mb)
SIGMA_LI6_NT: Dict[int, float] = {
    0: 0.012, 1: 0.014, 2: 0.016, 3: 0.019, 4: 0.022,
    5: 0.027, 6: 0.033, 7: 0.041, 8: 0.052, 9: 0.068,
    10: 0.093, 11: 0.135, 12: 0.210, 13: 0.380,
}


# ---------------------------------------------------------------------------
# Blanket geometry parameters (small-scale demonstrator)
# ---------------------------------------------------------------------------
BLANKET_THICKNESS_CM: float = 80.0        # radial thickness of the TBM
FIRST_WALL_THICKNESS_CM: float = 2.0      # Eurofer first wall
BACK_WALL_THICKNESS_CM: float = 1.5
PLASMA_FACE_X_CM: float = 0.0             # x = 0: plasma-facing surface


# ---------------------------------------------------------------------------
# Numerical default parameters
# ---------------------------------------------------------------------------
DEFAULT_NX: int = 128                     # spatial cells
DEFAULT_SN_ORDER: int = 8                 # S_N quadrature order (must be even)
DEFAULT_MAX_ITER: int = 2000
DEFAULT_TOLERANCE: float = 1.0e-9
EPS_NUMERICAL: float = 1.0e-300           # underflow guard
EPS_REL: float = 1.0e-12


# ---------------------------------------------------------------------------
# Derived helpers
# ---------------------------------------------------------------------------
def group_midpoint_energy(group_index: int) -> float:
    """Return the arithmetic midpoint energy of group *group_index* (MeV)."""
    if not 0 <= group_index < N_GROUPS:
        raise ValueError("group_index out of range")
    return 0.5 * (GROUP_BOUNDS_MEV[group_index] +
                  GROUP_BOUNDS_MEV[group_index + 1])


def group_lethargy_midpoint(group_index: int) -> float:
    """Return ln(E_{g-1}/E_g)/2 + ln(E_g): i.e. midpoint in lethargy."""
    E_hi = GROUP_BOUNDS_MEV[group_index]
    E_lo = GROUP_BOUNDS_MEV[group_index + 1]
    if E_lo <= 0.0:
        E_lo = EPS_NUMERICAL
    return 0.5 * math.log(E_hi / E_lo) + math.log(E_lo)


def thermal_velocity_cm_s(energy_mev: float, temperature_K: float) -> float:
    """Maxwellian-averaged velocity for neutrons of given energy at T.

    v = sqrt(2 E / m_n) * (1 + 3 kT / (4 E))^{1/2}    (low-order correction).
    """
    if energy_mev <= 0.0:
        return 0.0
    mass_n_mev = MASS_NEUTRON * 931.494                     # amu -> MeV
    v0 = C_LIGHT * math.sqrt(2.0 * energy_mev / mass_n_mev)
    correction = math.sqrt(1.0 + 3.0 * K_BOLTZMANN_MEV * temperature_K
                            / (4.0 * energy_mev))
    return v0 * correction
