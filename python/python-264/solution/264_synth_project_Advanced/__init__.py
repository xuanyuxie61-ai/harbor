# -*- coding: utf-8 -*-
"""
264_synth_project_Advanced
===========================

磁层粒子输运模拟: 高阶有限差分与稳定性分析

计算空间物理博士级科研代码项目.
"""

from . import physical_constants
from .magnetosphere_grid import MagnetosphereGrid
from .high_order_fd import (
    central_diff_2nd, central_diff_4th, central_diff_6th,
    central_diff2_2nd, central_diff2_4th,
    compact_first_derivative, compact_second_derivative,
    weno5_reconstruct, weno5_flux_divergence,
)
from .boundary_conditions import BoundaryConditions
from .stochastic_sampler import (
    ZigguratSampler, DiscreteCDFSampler,
    PolynomialChaosExpansion, halton_sequence,
)
from .stability_analysis import (
    von_neumann_analysis_diffusion, von_neumann_analysis_advection,
    compute_cfl_timestep, build_diffusion_matrix,
    matrix_stability_analysis, modified_gram_schmidt,
)
from .vlasov_solver import VlasovSolver
from .field_solver import DipoleField, ChirikovMap, trace_field_line
from .phase_space_diagnostics import (
    cross_correlation, phase_space_count,
    phase_space_moments, lyness_triangle_quadrature,
)
from .combinatorial_modes import (
    cyclotron_resonance_condition, enumerate_resonance_modes,
    LShellCouplingNetwork,
)
from .prime_grid import sieve_of_eratosthenes, PrimeResonanceGrid
from .partial_digest_resonance import WaveModeIdentifier
from .data_io import (
    compute_checksum, write_xyz, read_xyz,
    save_grid_data, load_grid_data,
)
from .test_particle_orbit import TestParticle, GuidingCenterIntegrator

__version__ = "1.0.0"
__all__ = [
    'physical_constants',
    'MagnetosphereGrid',
    'BoundaryConditions',
    'VlasovSolver',
    'DipoleField',
    'ChirikovMap',
    'ZigguratSampler',
    'PolynomialChaosExpansion',
]
