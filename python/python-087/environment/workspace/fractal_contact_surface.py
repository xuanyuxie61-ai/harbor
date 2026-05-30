
import numpy as np
from scipy.optimize import brent
from typing import Tuple, Optional, Callable


def ifs_fractal_surface_1d(length: float = 1.0, n_points: int = 1024,
                           D: float = 1.6, gamma: float = 1.5,
                           n_terms: int = 10, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    if not (1.0 < D < 2.0):
        raise ValueError("分形维数 D 必须在 (1, 2) 区间内")
    if gamma <= 1.0:
        raise ValueError("gamma 必须 > 1")
    if seed is not None:
        np.random.seed(seed)
    x = np.linspace(0, length, n_points)
    z = np.zeros_like(x)

    G = length * 1e-4
    phases = np.random.uniform(0, 2 * np.pi, n_terms)
    for n in range(n_terms):
        amplitude = (G ** (D - 1)) * (gamma ** ((D - 2) * n))
        frequency = 2 * np.pi * (gamma ** n) / length
        z += amplitude * np.cos(frequency * x + phases[n])

    z -= z.mean()
    return x, z


def ifs_fractal_surface_2d(size: float = 1.0, n_grid: int = 256,
                           D: float = 1.6, gamma: float = 1.5,
                           n_terms: int = 8, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (1.0 < D < 2.0):
        raise ValueError("D 必须在 (1, 2)")
    if seed is not None:
        np.random.seed(seed)
    x = np.linspace(0, size, n_grid)
    y = np.linspace(0, size, n_grid)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    G = size * 1e-4
    for n in range(n_terms):
        amp = (G ** (D - 1)) * (gamma ** ((D - 2) * n))
        freq = 2 * np.pi * (gamma ** n) / size
        phi1 = np.random.uniform(0, 2 * np.pi)
        phi2 = np.random.uniform(0, 2 * np.pi)
        Z += amp * (np.cos(freq * (X + Y) + phi1) + np.cos(freq * (X - Y) + phi2))
    Z -= Z.mean()
    return X, Y, Z


def surface_statistics(z: np.ndarray) -> dict:
    if z.size == 0:
        raise ValueError("空数组")
    rms = np.sqrt(np.mean(z ** 2))
    Ra = np.mean(np.abs(z - np.mean(z)))
    sigma = rms if rms > 1e-18 else 1.0
    skewness = np.mean((z - np.mean(z)) ** 3) / (sigma ** 3)
    kurtosis = np.mean((z - np.mean(z)) ** 4) / (sigma ** 4)

    peaks = 0
    for i in range(1, len(z) - 1):
        if z[i] > z[i - 1] and z[i] > z[i + 1]:
            peaks += 1
    peak_density = peaks / len(z)
    return {
        "rms": float(rms),
        "Ra": float(Ra),
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
        "peak_density": float(peak_density)
    }


def contact_gap_function(surface1_z: Callable[[np.ndarray], np.ndarray],
                         surface2_z: Callable[[np.ndarray], np.ndarray],
                         x: np.ndarray) -> np.ndarray:
    z1 = surface1_z(x)
    z2 = surface2_z(x)
    return z2 - z1


def minimize_contact_gap(surface1_z: Callable[[np.ndarray], np.ndarray],
                         surface2_z: Callable[[np.ndarray], np.ndarray],
                         x_min: float, x_max: float,
                         tol: float = 1e-9, maxiter: int = 100) -> Tuple[float, float]:
    def gap_scalar(x):

        if np.isscalar(x):
            return float(surface2_z(np.array([x]))[0] - surface1_z(np.array([x]))[0])
        return surface2_z(x) - surface1_z(x)

    x_star = brent(gap_scalar, brack=(x_min, x_max), tol=tol, maxiter=maxiter)
    g_min = gap_scalar(x_star)
    return float(x_star), float(g_min)


def hertz_mindlin_normal_force(delta: float, E_star: float, R_eq: float) -> float:
    if delta < 0:
        return 0.0
    if delta < 1e-18:
        return 0.0
    return (4.0 / 3.0) * E_star * np.sqrt(R_eq) * (delta ** 1.5)


def equivalent_modulus_radius(E1: float, nu1: float, E2: float, nu2: float,
                               R1: float, R2: float) -> Tuple[float, float]:
    E_star = 1.0 / ((1.0 - nu1 ** 2) / E1 + (1.0 - nu2 ** 2) / E2)
    R_eq = 1.0 / (1.0 / R1 + 1.0 / R2)
    return float(E_star), float(R_eq)


def gw_contact_force(stats: dict, separation: float, E_star: float,
                     R_eq: float, eta: float, A_n: float) -> float:
    sigma = stats["rms"]
    if sigma <= 1e-18:
        return 0.0
    d_norm = separation / sigma

    s = np.linspace(max(d_norm, -5.0), 5.0, 1001)
    if len(s) < 2:
        return 0.0
    ds = s[1] - s[0]
    phi = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * s ** 2)
    integrand = np.maximum(s - d_norm, 0.0) ** 1.5 * phi
    integral = np.trapezoid(integrand, s)
    prefactor = eta * A_n * (4.0 / 3.0) * E_star * np.sqrt(R_eq) * (sigma ** 1.5)
    return float(prefactor * integral)
