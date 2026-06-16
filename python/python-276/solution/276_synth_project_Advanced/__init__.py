"""
crystal_defect_formation_energy — PhD-level synthesis project
=============================================================

Computes the formation energy of point defects (vacancies, interstitials)
in a 2-D hexagonal crystal using high-order finite differences, sparse
Green's functions, FFT-based Poisson solvers, Eshelby elastic corrections
and statistical uncertainty quantification.
"""

__version__ = "1.0.0"
__all__ = [
    "config",
    "crystal_lattice",
    "high_order_fd",
    "stability_analysis",
    "sparse_green_defect",
    "fft_poisson",
    "defect_formation_energy",
    "eshelby_strain",
    "statistical_qc",
    "mesh_defect",
    "denoise_filter",
    "analytical_benchmarks",
    "linear_solver",
]
