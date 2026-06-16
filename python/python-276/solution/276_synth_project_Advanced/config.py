"""
config.py — Physical constants, numerical parameters and model configuration
============================================================================

This module centralizes every physical constant and numerical parameter that
appears in the crystal-defect formation-energy pipeline. The code is kept
explicit (rather than hidden behind a settings file) so that the relationship
between the formulae in the README and the implementation is one-to-one.

All quantities are in Hartree atomic units unless explicitly noted:
    length : Bohr   (a0 = 0.52917721067 Ang)
    energy : Hartree (Eh = 27.21138602 eV)
    mass   : electron mass me
    charge : elementary charge e

The default physical system is a graphene-like 2D hexagonal monolayer
(spatial dimension d = 2) hosting a single vacancy or self-interstitial.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field


# -------------------------------------------------------------------------
# Fundamental constants (CODATA 2018, atomic units ⇒ many are unity)
# -------------------------------------------------------------------------
BOHR_TO_ANG = 0.52917721067         # 1 Bohr in Angstrom
HARTREE_TO_EV = 27.21138602         # 1 Hartree in eV
K_B_EV = 8.617333262145e-5          # Boltzmann constant in eV/K
PI = math.pi
TWO_PI = 2.0 * PI
SQRT_PI = math.sqrt(PI)
EULER_GAMMA = 0.5772156649015328606 # Euler–Mascheroni γ


# -------------------------------------------------------------------------
# Crystal model parameters
# -------------------------------------------------------------------------
@dataclass(frozen=True)
class CrystalParams:
    """Geometric and physical parameters of the host crystal."""
    # Lattice
    lattice_constant: float = 4.65          # a, Bohr (graphene ~ 4.65 a0)
    n_cells: int = 8                        # N×N primitive cells in supercell
    basis_type: str = "hexagonal"           # "hexagonal" or "square"

    # Pseudopotential (model: Gaussian-core ion)
    #   V_ion(r) = -Z_eff * erf(r / r_core) / r
    z_eff: float = 4.0                      # effective valence charge
    r_core: float = 0.90                    # core radius, Bohr

    # Exchange–correlation (2D Kohn–Sham LDA, spin-unpolarised)
    #   ε_xc(n) = -0.3867 - 0.1554 * ln(n)   (Attaccalite et al. 2011, 2D)
    xc_a: float = -0.3867
    xc_b: float = -0.1554

    # Electronic temperature for Fermi smearing
    kT_ev: float = 0.025                    # ~ room temperature
    n_electrons_per_cell: int = 4           # for graphene π-band

    @property
    def supercell_length(self) -> float:
        """Linear size L = N * a of the supercell (Bohr)."""
        return self.n_cells * self.lattice_constant

    @property
    def kt_hartree(self) -> float:
        return self.kT_ev / HARTREE_TO_EV


# -------------------------------------------------------------------------
# Numerical-grid parameters
# -------------------------------------------------------------------------
@dataclass(frozen=True)
class GridParams:
    """Real-space grid parameters."""
    n_grid: int = 64                # grid points per dimension (must be even)
    fd_order: int = 6               # finite-difference order: 2, 4, 6, 8
    boundary: str = "periodic"      # "periodic" or "dirichlet"

    @property
    def h(self) -> float:
        """Grid spacing (computed externally; placeholder here)."""
        return 0.0  # overridden at runtime

    def spacing(self, L: float) -> float:
        return L / self.n_grid


# -------------------------------------------------------------------------
# Defect parameters
# -------------------------------------------------------------------------
@dataclass(frozen=True)
class DefectParams:
    """Single-defect parameters."""
    kind: str = "vacancy"           # "vacancy" or "interstitial"
    charge_state: int = 0           # q ∈ {-2, -1, 0, +1, +2}
    site_index: tuple = (0, 0)      # lattice-site index of the defect
    relaxation_steps: int = 2000    # imaginary-time relaxation steps
    relaxation_dt: float = 0.005    # Δτ in Hartree^-1


# -------------------------------------------------------------------------
# Eshelby / elastic correction parameters
# -------------------------------------------------------------------------
@dataclass(frozen=True)
class EshelbyParams:
    """Eshelby inclusion parameters for elastic far-field correction."""
    young_modulus_ev: float = 340.0     # E, eV/Å² (graphene in-plane)
    poisson_ratio: float = 0.165        # ν (graphene)
    defect_volume_ang3: float = 1.5     # Ω_def, Å³ (vacancy relaxation volume)
    inclusion_shape: str = "circular"   # "circular" or "elliptical"
    aspect_ratio: float = 1.0           # b / a for elliptical inclusion

    @property
    def young_hartree(self) -> float:
        return self.young_modulus_ev / HARTREE_TO_EV

    @property
    def shear_modulus(self) -> float:
        """G = E / (2(1 + ν))"""
        return self.young_modulus_ev / (2.0 * (1.0 + self.poisson_ratio))


# -------------------------------------------------------------------------
# Statistical / QC parameters
# -------------------------------------------------------------------------
@dataclass(frozen=True)
class StatisticalParams:
    """Parameters for ensemble averaging and uncertainty quantification."""
    n_samples: int = 32                 # number of defect configurations
    confidence_level: float = 0.95      # 95% CI
    seed: int = 276                     # reproducible RNG seed
    temperature_sweep: tuple = (100.0, 300.0, 600.0, 900.0)  # K


# -------------------------------------------------------------------------
# Aggregate configuration
# -------------------------------------------------------------------------
@dataclass
class ProjectConfig:
    crystal: CrystalParams = field(default_factory=CrystalParams)
    grid: GridParams = field(default_factory=GridParams)
    defect: DefectParams = field(default_factory=DefectParams)
    eshelby: EshelbyParams = field(default_factory=EshelbyParams)
    stats: StatisticalParams = field(default_factory=StatisticalParams)


def default_config() -> ProjectConfig:
    """Return the default ProjectConfig used by main.py."""
    return ProjectConfig()
