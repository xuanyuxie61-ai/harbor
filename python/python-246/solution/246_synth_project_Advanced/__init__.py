"""
cosmo_nbody  —  宇宙大尺度结构 N 体模拟
======================================

高阶有限差分 + 稳定性分析 + 可复现实验
"""
from .cosmo_config import (
    CosmoParams, load_cosmo_params, save_params, get_paths,
    z_to_a, hubble_factor, RHO_CRIT_H2, H100_SI,
)
from .legendre_kernel import (
    legendre_shifted_values, gauss_legendre_nodes,
    gauss_legendre_quadrature, legendre_cic_kernel,
)
from .finite_difference import (
    fd_coefficients_2nd_derivative, fd_stencil_2nd,
    compact_fd_coefficients, thomas_solve,
    compact_second_derivative_1d, laplacian_3d_periodic,
    gradient_3d_periodic, fd_coefficients_1st_derivative,
)
from .poisson_solver import (
    poisson_fft_3d, poisson_1d_banded, condition_number_estimate_1d,
)
from .initial_conditions import (
    generate_density_field, generate_delta_k,
    zeldovich_displacement, sample_particle_positions,
    eisenstein_hu_transfer, primordial_power_spectrum,
)
from .mesh_topology import (
    MeshTopology, polygon_triangulate, triangle_area,
    cube_surface_distance_stats, particle_pair_distance_stats,
)
from .time_integrator import (
    kick_drift_kick_step, forest_ruth_step, compute_timestep,
    kdv_exact_sech, kdv_exact_rational, kdv_residual, kdv_parameters,
)
from .stability_analysis import (
    zero_itp, power_method, power_method2,
    amplification_matrix_leapfrog, spectral_radius_leapfrog,
    critical_cfl_number, von_neumann_stability_scan,
)
from .parameter_calibration import (
    calibrate_parameters, lbfgs_refine, sheth_tormen_mass_function,
    calibration_cost_function,
)
from .nas_search import (
    PeriodicHashGrid, cic_deposit, tsc_deposit,
    neighbor_count_statistics,
)
from .diagnostics import (
    power_spectrum, two_point_correlation, density_pdf,
    kinetic_energy, potential_energy, full_diagnostics,
)

__version__ = "1.0.0"
