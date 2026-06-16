"""
level_density.py  --  Nuclear level density from the computed spectrum
=====================================================================
Implements the constant-temperature and back-shifted Fermi-gas models for
the level density rho(E), using the single-particle spectrum computed by
radial_fd_solver.py.

    rho(E) = (1 / sqrt(48 E)) * exp(2 sqrt(a E))  (Fermi gas)
    with level-density parameter a ~ A / 8 MeV^{-1}.
"""
import math
import numpy as np
from nuclear_constants import R_EPSILON


def fermi_gas_level_density(E: float, A: int, E_shift: float = 1.0) -> float:
    """Back-shifted Fermi-gas level density rho(E) in MeV^{-1}."""
    a = A / 8.0  # MeV^{-1}
    U = max(E - E_shift, 0.0)
    if U <= 0.0:
        return 0.0
    return math.exp(2.0 * math.sqrt(a * U)) / math.sqrt(max(48.0 * U, R_EPSILON))


def constant_temperature_rho(E: float, T: float, E0: float) -> float:
    """Constant-temperature level density: rho(E) = (1/T) exp((E - E0)/T)."""
    if T <= 0.0:
        return 0.0
    return math.exp((E - E0) / T) / T


def empirical_level_density_from_spectrum(
    energies: np.ndarray, E_bins: np.ndarray,
) -> np.ndarray:
    """Bin the computed spectrum into a histogram approximation of rho(E)."""
    counts, _ = np.histogram(energies, bins=E_bins)
    widths = np.diff(E_bins)
    widths = np.where(widths < R_EPSILON, R_EPSILON, widths)
    return counts.astype(np.float64) / widths


def bethe_formula(a: float, E: float) -> float:
    """Bethe formula rho ~ exp(2 sqrt(a E)) / (12 sqrt(a) E^{5/4})."""
    if E <= 0.0 or a <= 0.0:
        return 0.0
    return math.exp(2.0 * math.sqrt(a * E)) / (12.0 * math.sqrt(a) * (E ** 1.25))
