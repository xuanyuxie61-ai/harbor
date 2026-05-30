
import numpy as np
from typing import Tuple, List


def gaussian_dos(
    energies: np.ndarray,
    E_grid: np.ndarray,
    sigma: float = 0.01,
) -> np.ndarray:
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
    from scipy.spatial import Delaunay

    N, n_bands = energies_at_k.shape
    if kpoints.shape[0] != N:
        raise ValueError("kpoints and energies must have matching first dimension.")


    try:
        tri = Delaunay(kpoints)
    except Exception:

        return gaussian_dos(energies_at_k, E_grid, sigma=0.02)

    dos = np.zeros_like(E_grid)
    simplex_areas = tri.area if hasattr(tri, "area") else None

    for simplex in tri.simplices:
        k_tri = kpoints[simplex]
        for band in range(n_bands):
            e_tri = energies_at_k[simplex, band]

            A = np.vstack([k_tri[:, 0], k_tri[:, 1], np.ones(3)])
            try:
                coeffs = np.linalg.solve(A.T, e_tri)
            except np.linalg.LinAlgError:
                continue
            grad = coeffs[0:2]
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-12:
                continue

            e_min = np.min(e_tri)
            e_max = np.max(e_tri)
            area = 0.5 * abs(np.cross(k_tri[1] - k_tri[0], k_tri[2] - k_tri[0]))
            mask = (E_grid >= e_min) & (E_grid <= e_max)
            dos[mask] += area / grad_norm


    total_area = 0.5 * abs(np.cross(
        kpoints.max(axis=0) - kpoints.min(axis=0),
        np.array([0.0, 1.0])
    )) * 2.0
    if total_area > 0.0:
        dos /= total_area
    return dos


def find_van_hove_singularities(
    E_grid: np.ndarray,
    dos: np.ndarray,
    prominence: float = 0.1,
) -> List[Tuple[float, float]]:
    if dos.size < 3:
        return []

    threshold = prominence * (np.max(dos) - np.min(dos))
    peaks = []
    for i in range(1, dos.size - 1):
        if dos[i] > dos[i - 1] and dos[i] > dos[i + 1]:
            if dos[i] - min(dos[i - 1], dos[i + 1]) > threshold:

                e_peak = E_grid[i]
                peaks.append((float(e_peak), float(dos[i])))

    return peaks


def estimate_fermi_velocity_from_dos(
    E_grid: np.ndarray,
    dos: np.ndarray,
    fermi_level: float,
    fit_window: float = 0.05,
) -> float:
    mask = np.abs(E_grid - fermi_level) < fit_window
    if np.sum(mask) < 3:
        return 0.0

    E_fit = E_grid[mask] - fermi_level
    D_fit = dos[mask]

    x = np.abs(E_fit)

    valid = x > 1e-6
    if np.sum(valid) < 2:
        return 0.0
    x = x[valid]
    y = D_fit[valid]
    a = np.sum(x * y) / np.sum(x ** 2)
    if a <= 0.0:
        return 0.0

    hbar = 0.6582119
    v_F = 1.0 / np.sqrt(2.0 * np.pi * a) / hbar
    return float(v_F)


def blowup_divergence_metric(
    dos: np.ndarray,
    sigma: float,
) -> float:
    if dos.size == 0:
        return 0.0
    D_peak = np.max(dos)

    D_smooth = np.median(dos)
    if D_smooth < 1e-15:
        D_smooth = np.mean(dos)
    if D_smooth < 1e-15:
        return 0.0
    ratio = D_peak / D_smooth

    expected = max(1.0, np.log(1.0 / max(sigma, 1e-6)))
    return float(ratio / expected)
