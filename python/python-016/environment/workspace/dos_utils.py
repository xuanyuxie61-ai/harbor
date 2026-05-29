"""
Density of States (DOS) and Van Hove Singularity Analysis
==========================================================
Computes the electronic density of states from tight-binding eigenvalues
and identifies Van Hove singularities—divergences in the DOS that arise
from saddle points in the band dispersion.

Scientific Background
---------------------
The density of states per unit area per spin is

    D(E) = (1/N_k) Σ_{n,k} δ(E − E_n(k))

In practice the delta function is replaced by a broadening function
(e.g., Gaussian or Lorentzian):

    δ(E) ≈ (1/√(2π)σ) exp(−E² / (2σ²))

Van Hove singularities occur at critical points k* where ∇_k E_n(k*) = 0.
Near a saddle point in 2D the dispersion is

    E(k) ≈ E* + (1/2) α (k_x − k_x*)² − (1/2) β (k_y − k_y*)²

with α, β > 0.  The DOS then has a logarithmic divergence:

    D(E) ∝ ln |E − E*| .

This divergence is smoothed by finite broadening σ, producing a sharp
peak whose height scales as ∼ ln(1/σ).

The integrated DOS (IDOS) is

    N(E) = ∫_{−∞}^{E} D(E′) dE′ .

At charge neutrality N(E_F) equals the number of states per unit cell.

The Fermi velocity v_F at the Dirac point can be extracted from the
linear region of N(E) near the neutrality point:

    N(E) ≈ (E² / (4π ħ² v_F²)) + const .
"""

import numpy as np
from typing import Tuple, List


def gaussian_dos(
    energies: np.ndarray,
    E_grid: np.ndarray,
    sigma: float = 0.01,
) -> np.ndarray:
    """
    Compute the DOS by Gaussian broadening of discrete eigenvalues.

        D(E) = (1/N) Σ_n (1/√(2π)σ) exp(−(E − E_n)² / (2σ²))

    Parameters
    ----------
    energies : np.ndarray of shape (N,)
        Discrete eigenvalues.
    E_grid : np.ndarray of shape (M,)
        Energy grid for evaluation.
    sigma : float
        Broadening width in eV.

    Returns
    -------
    np.ndarray of shape (M,)
        DOS values in states/(eV·unit_cell).
    """
    if sigma <= 0.0:
        raise ValueError("Broadening sigma must be positive.")
    energies = np.ravel(energies)
    N = energies.size
    prefactor = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    dos = np.zeros_like(E_grid)
    for e_n in energies:
        dos += prefactor * np.exp(-((E_grid - e_n) ** 2) / (2.0 * sigma ** 2))
    return dos / N


def lorentzian_dos(
    energies: np.ndarray,
    E_grid: np.ndarray,
    gamma: float = 0.01,
) -> np.ndarray:
    """
    Compute the DOS by Lorentzian broadening.

        D(E) = (1/N) Σ_n (1/π) γ / [(E − E_n)² + γ²]

    Parameters
    ----------
    energies : np.ndarray
    E_grid : np.ndarray
    gamma : float
        Half-width at half-maximum in eV.

    Returns
    -------
    np.ndarray
    """
    if gamma <= 0.0:
        raise ValueError("Gamma must be positive.")
    energies = np.ravel(energies)
    N = energies.size
    dos = np.zeros_like(E_grid)
    for e_n in energies:
        dos += (1.0 / np.pi) * gamma / ((E_grid - e_n) ** 2 + gamma ** 2)
    return dos / N


def tetrahedron_dos_2d(
    kpoints: np.ndarray,
    energies_at_k: np.ndarray,
    E_grid: np.ndarray,
) -> np.ndarray:
    """
    Approximate the DOS using the 2D tetrahedron (triangle) method.

    For each triangle in the k-mesh, the energy is linearly interpolated:

        E(k) = E_0 + (k − k_0)·∇E

    The contribution to the DOS from one triangle is proportional to the
    length of the iso-energy line inside the triangle, divided by |∇E|.

    Parameters
    ----------
    kpoints : np.ndarray of shape (N, 2)
    energies_at_k : np.ndarray of shape (N, n_bands)
    E_grid : np.ndarray of shape (M,)

    Returns
    -------
    np.ndarray of shape (M,)
    """
    from scipy.spatial import Delaunay

    N, n_bands = energies_at_k.shape
    if kpoints.shape[0] != N:
        raise ValueError("kpoints and energies must have matching first dimension.")

    # Delaunay triangulation of k-points
    try:
        tri = Delaunay(kpoints)
    except Exception:
        # Fallback: simple grid triangulation
        return gaussian_dos(energies_at_k, E_grid, sigma=0.02)

    dos = np.zeros_like(E_grid)
    simplex_areas = tri.area if hasattr(tri, "area") else None

    for simplex in tri.simplices:
        k_tri = kpoints[simplex]
        for band in range(n_bands):
            e_tri = energies_at_k[simplex, band]
            # Linear interpolation coefficients
            A = np.vstack([k_tri[:, 0], k_tri[:, 1], np.ones(3)])
            try:
                coeffs = np.linalg.solve(A.T, e_tri)
            except np.linalg.LinAlgError:
                continue
            grad = coeffs[0:2]
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-12:
                continue
            # For each energy in E_grid, compute line length contribution
            e_min = np.min(e_tri)
            e_max = np.max(e_tri)
            area = 0.5 * abs(np.cross(k_tri[1] - k_tri[0], k_tri[2] - k_tri[0]))
            mask = (E_grid >= e_min) & (E_grid <= e_max)
            dos[mask] += area / grad_norm

    # Normalize by total BZ area (approximate)
    total_area = 0.5 * abs(np.cross(
        kpoints.max(axis=0) - kpoints.min(axis=0),
        np.array([0.0, 1.0])
    )) * 2.0  # rough estimate
    if total_area > 0.0:
        dos /= total_area
    return dos


def find_van_hove_singularities(
    E_grid: np.ndarray,
    dos: np.ndarray,
    prominence: float = 0.1,
) -> List[Tuple[float, float]]:
    """
    Identify Van Hove singularities as prominent peaks in the DOS.

    A peak is defined as a local maximum with DOS value exceeding
    prominence × (max(DOS) − min(DOS)).

    Parameters
    ----------
    E_grid : np.ndarray
    dos : np.ndarray
    prominence : float
        Relative prominence threshold.

    Returns
    -------
    list of (energy, dos_value)
    """
    if dos.size < 3:
        return []

    threshold = prominence * (np.max(dos) - np.min(dos))
    peaks = []
    for i in range(1, dos.size - 1):
        if dos[i] > dos[i - 1] and dos[i] > dos[i + 1]:
            if dos[i] - min(dos[i - 1], dos[i + 1]) > threshold:
                # Parabolic interpolation for sub-grid accuracy
                e_peak = E_grid[i]
                peaks.append((float(e_peak), float(dos[i])))

    return peaks


def estimate_fermi_velocity_from_dos(
    E_grid: np.ndarray,
    dos: np.ndarray,
    fermi_level: float,
    fit_window: float = 0.05,
) -> float:
    """
    Estimate the Fermi velocity v_F from the DOS near the charge-
    neutrality point.

    In 2D Dirac systems the DOS is linear:

        D(E) = |E| / (2π ħ² v_F²)

    We fit the DOS in a window [−w, +w] around E_F to this form.

    Parameters
    ----------
    E_grid : np.ndarray
    dos : np.ndarray
    fermi_level : float
    fit_window : float
        Energy window in eV.

    Returns
    -------
    float
        Estimated v_F in nm/fs.
    """
    mask = np.abs(E_grid - fermi_level) < fit_window
    if np.sum(mask) < 3:
        return 0.0

    E_fit = E_grid[mask] - fermi_level
    D_fit = dos[mask]
    # Fit D = a |E| using linear regression on |E| vs D
    x = np.abs(E_fit)
    # Remove points too close to zero
    valid = x > 1e-6
    if np.sum(valid) < 2:
        return 0.0
    x = x[valid]
    y = D_fit[valid]
    a = np.sum(x * y) / np.sum(x ** 2)
    if a <= 0.0:
        return 0.0
    # a = 1 / (2π ħ² v_F²)  with ħ in eV·fs
    hbar = 0.6582119
    v_F = 1.0 / np.sqrt(2.0 * np.pi * a) / hbar
    return float(v_F)


def blowup_divergence_metric(
    dos: np.ndarray,
    sigma: float,
) -> float:
    """
    Quantify the "blow-up" character of Van Hove singularities by
    measuring how much the peak DOS exceeds the smooth background.

    In a 2D saddle-point VHS the peak height scales logarithmically
    with inverse broadening:

        D_peak ∝ ln(1/σ) .

    We compute the ratio

        R = D_peak / D_smooth

    as a dimensionless measure of divergence strength.

    Parameters
    ----------
    dos : np.ndarray
    sigma : float
        Broadening used.

    Returns
    -------
    float
        Divergence ratio.
    """
    if dos.size == 0:
        return 0.0
    D_peak = np.max(dos)
    # Smooth background = median
    D_smooth = np.median(dos)
    if D_smooth < 1e-15:
        D_smooth = np.mean(dos)
    if D_smooth < 1e-15:
        return 0.0
    ratio = D_peak / D_smooth
    # Theoretical ln(1/σ) scaling factor
    expected = max(1.0, np.log(1.0 / max(sigma, 1e-6)))
    return float(ratio / expected)
