"""
Tritium Breeding Blanket Neutron Transport Solver
=================================================

A high-order compact finite-difference discrete-ordinates (S_N) solver
for the multigroup neutron transport equation in a fusion breeding
blanket, with von Neumann / spectral stability analysis and Feynman-Kac
stochastic verification.
"""

from . import (
    physics_constants,
    cross_sections,
    energy_spectra,
    blanket_geometry,
    angular_quadrature,
    fd_transport,
    stability_analysis,
    monte_carlo_fk,
    tbr_calculator,
    latent_decomposer,
)

__all__ = [
    'physics_constants',
    'cross_sections',
    'energy_spectra',
    'blanket_geometry',
    'angular_quadrature',
    'fd_transport',
    'stability_analysis',
    'monte_carlo_fk',
    'tbr_calculator',
    'latent_decomposer',
]
