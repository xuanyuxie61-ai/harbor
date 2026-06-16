"""
Project 280 — Configuration module for multi-scale damage evolution simulation.

Stores all physical constants, numerical parameters, material properties,
and simulation settings for the high-order finite-difference damage model.
"""

import math
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional


# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------
PLANCK_REDUCED: float = 1.054571817e-34   # J·s
BOLTZMANN: float = 1.380649e-23           # J/K
AVOGADRO: float = 6.02214076e23           # 1/mol
ELEMENTARY_CHARGE: float = 1.602176634e-19  # C


# ---------------------------------------------------------------------------
# Material parameters — representative quasi-brittle (concrete / rock / ceramic)
# ---------------------------------------------------------------------------
@dataclass
class MaterialParams:
    """Constitutive parameters for the graded quasi-brittle material."""
    young_modulus: float = 30.0e9          # E  [Pa]
    poisson_ratio: float = 0.20            # ν
    fracture_energy: float = 100.0         # G_f [J/m²]
    tensile_strength: float = 3.0e6        # f_t [Pa]
    compressive_strength: float = 30.0e6   # f_c [Pa]
    characteristic_length: float = 0.05    # ℓ_c [m]  — nonlocal interaction radius
    damage_threshold_strain: float = 1.0e-4  # κ_0  — initial damage threshold
    softening_exponent: float = 2.0        # n_s  — exponent in damage law
    mass_density: float = 2400.0           # ρ [kg/m³]
    thermal_expansion: float = 1.0e-5      # α_T [1/K]
    reference_temperature: float = 293.15  # T_ref [K]
    weibull_modulus: float = 8.0           # m_W — Weibull shape parameter
    weibull_scale_strain: float = 1.5e-4   # ε_0W — Weibull scale

    @property
    def lame_mu(self) -> float:
        """First Lamé parameter μ (shear modulus)."""
        E, nu = self.young_modulus, self.poisson_ratio
        return E / (2.0 * (1.0 + nu))

    @property
    def lame_lambda(self) -> float:
        """Second Lamé parameter λ."""
        E, nu = self.young_modulus, self.poisson_ratio
        return E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    @property
    def bulk_modulus(self) -> float:
        """Bulk modulus K."""
        E, nu = self.young_modulus, self.poisson_ratio
        return E / (3.0 * (1.0 - 2.0 * nu))

    @property
    def p_wave_speed(self) -> float:
        """P-wave speed  c_p = sqrt((λ + 2μ) / ρ)."""
        return math.sqrt((self.lame_lambda + 2.0 * self.lame_mu) / self.mass_density)

    @property
    def s_wave_speed(self) -> float:
        """S-wave speed  c_s = sqrt(μ / ρ)."""
        return math.sqrt(self.lame_mu / self.mass_density)

    @property
    def rayleigh_wave_speed_approx(self) -> float:
        """Approximate Rayleigh wave speed (Victorov formula)."""
        cs = self.s_wave_speed
        kappa = (0.862 + 1.14 * self.poisson_ratio) / (1.0 + self.poisson_ratio)
        return kappa * cs


# ---------------------------------------------------------------------------
# Numerical parameters
# ---------------------------------------------------------------------------
@dataclass
class NumericalParams:
    """Discretization and solver parameters."""
    # Grid
    nx: int = 60
    ny: int = 60
    domain_x: Tuple[float, float] = (0.0, 0.3)   # [m]
    domain_y: Tuple[float, float] = (0.0, 0.3)
    grid_refinement_levels: int = 2
    annulus_inner_radius: float = 0.01    # [m]
    annulus_outer_radius: float = 0.10    # [m]
    annulus_points_radial: int = 12
    annulus_points_angular: int = 24

    # Finite difference
    fd_order: int = 4           # interior scheme order
    fd_boundary_order: int = 2  # boundary scheme order
    fd_stretch_factor: float = 1.0  # grid stretching (1 = uniform)

    # Time integration
    total_time: float = 1.0e-3          # [s]
    initial_dt: float = 1.0e-7          # [s]
    cfl_number: float = 0.4
    max_dt_growth_factor: float = 1.2
    min_dt: float = 1.0e-12
    time_scheme: str = "imex_rk2"       # euler, rk4, imex_rk2, newmark
    newmark_beta: float = 0.25
    newmark_gamma: float = 0.50

    # Nonlocal damage
    nonlocal_weight_type: str = "gaussian"  # gaussian, hat, bell
    nonlocal_integration_tol: float = 1.0e-8
    nonlocal_max_iterations: int = 200

    # Percolation
    percolation_damage_threshold: float = 0.60
    connectivity_mode: str = "von_neumann"  # von_neumann, moore

    # Stability
    stability_check_interval: int = 10
    eigenvalue_check_enabled: bool = True
    energy_dissipation_check: bool = True

    # MCMC calibration
    mcmc_chain_length: int = 500
    mcmc_burnin: int = 100
    mcmc_step_size: float = 0.05
    mcmc_target_acceptance: float = 0.35

    # Benchmark
    benchmark_num_cases: int = 5
    benchmark_mesh_sizes: List[int] = field(default_factory=lambda: [20, 30, 40, 50, 60])

    # Monte Carlo defects
    defect_seed: int = 42
    defect_intensity: float = 200.0       # defects per m²
    defect_max_radius: float = 5.0e-4     # [m]

    # Stopping criteria
    max_global_damage: float = 0.95
    force_displacement_record_interval: int = 5


# ---------------------------------------------------------------------------
# Crack configuration
# ---------------------------------------------------------------------------
@dataclass
class CrackConfig:
    """Initial crack geometry (mode-I edge crack)."""
    tip_x: float = 0.10       # [m]
    tip_y: float = 0.15       # [m]
    length: float = 0.05      # [m]
    angle: float = 0.0        # [rad] — 0 = horizontal
    tip_radius: float = 1.0e-4  # [m] — regularised tip radius

    @property
    def tail_x(self) -> float:
        return self.tip_x - self.length * math.cos(self.angle)

    @property
    def tail_y(self) -> float:
        return self.tip_y - self.length * math.sin(self.angle)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@dataclass
class LoadingConfig:
    """Remote loading applied to the specimen."""
    mode: str = "displacement_control"   # displacement_control, force_control
    applied_strain_rate: float = 0.1     # [1/s]
    max_applied_strain: float = 3.0e-3
    lateral_constraint: str = "plane_strain"  # plane_strain, plane_stress
    temperature_change: float = 0.0      # ΔT [K]


# ---------------------------------------------------------------------------
# Aggregate simulation config
# ---------------------------------------------------------------------------
@dataclass
class SimulationConfig:
    """Top-level configuration container."""
    material: MaterialParams = field(default_factory=MaterialParams)
    numerical: NumericalParams = field(default_factory=NumericalParams)
    crack: CrackConfig = field(default_factory=CrackConfig)
    loading: LoadingConfig = field(default_factory=LoadingConfig)
    output_dir: str = "results_280"
    verbose: bool = True

    def dx(self) -> float:
        """Characteristic grid spacing."""
        Lx = self.numerical.domain_x[1] - self.numerical.domain_x[0]
        return Lx / max(self.numerical.nx - 1, 1)

    def dy(self) -> float:
        Ly = self.numerical.domain_y[1] - self.numerical.domain_y[0]
        return Ly / max(self.numerical.ny - 1, 1)


# Global singleton for convenience
DEFAULT_CONFIG = SimulationConfig()
