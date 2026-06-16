"""
PROJECT_290 - 计算等离子体：阿尔芬波与高能粒子相互作用
高阶有限差分与稳定性分析（小规模可复现实验）

本包实现了阿尔芬波-高能粒子耦合系统的数值模拟和稳定性分析。
"""

from .plasma_config import PlasmaParameters
from .magnetic_geometry import MagneticGeometry, CVTAdaptiveGrid
from .high_order_operators import (
    BandMatrixSPD,
    apply_first_derivative,
    apply_second_derivative,
    apply_radial_laplacian_cylindrical,
)
from .boundary_handler import cyclic_wrap, periodic_index, helical_boundary_map
from .alfven_wave_solver import AlfvenWaveState, AlfvenWaveSolver
from .energetic_particle_kinetics import EnergeticParticleDistribution, MultiSpeciesEPDriver
from .dispersion_analysis import (
    plasma_dispersion_function,
    alfven_continuum_gap_structure,
    tae_frequency_estimate,
    monte_carlo_orbit_statistics,
)
from .stability_eigenvalue import StabilityAnalyzer
from .conservation_monitor import ConservationMonitor, ConstrainedDistributionReconstructor
from .statistical_diagnostics import (
    estimate_growth_rate,
    spectral_analysis,
    fisher_exact_test_2x2,
    difference_in_differences_analysis,
    convergence_order_analysis,
)
from .solution_io import SimulationResultWriter, SolutionFileReader
