
import numpy as np
from typing import Tuple, List


R0_FACTOR = 1.2
C3_ASYMMETRY = np.sqrt(7.0 / (4.0 * np.pi))


def nuclear_radius(mass_number: int) -> float:
    if mass_number <= 0:
        raise ValueError("mass_number must be positive")
    return R0_FACTOR * float(mass_number) ** (1.0 / 3.0)


def spherical_harmonic_y20(theta: np.ndarray) -> np.ndarray:
    return np.sqrt(5.0 / (16.0 * np.pi)) * (3.0 * np.cos(theta) ** 2 - 1.0)


def spherical_harmonic_y30(theta: np.ndarray) -> np.ndarray:
    c = np.cos(theta)
    return np.sqrt(7.0 / (16.0 * np.pi)) * (5.0 * c ** 3 - 3.0 * c)


def spherical_harmonic_y40(theta: np.ndarray) -> np.ndarray:
    c = np.cos(theta)
    return np.sqrt(9.0 / (256.0 * np.pi)) * (35.0 * c ** 4 - 30.0 * c ** 2 + 3.0)


def spherical_harmonic_y50(theta: np.ndarray) -> np.ndarray:
    c = np.cos(theta)
    return np.sqrt(11.0 / (256.0 * np.pi)) * (63.0 * c ** 5 - 70.0 * c ** 3 + 15.0 * c)


def nuclear_surface_profile(theta: np.ndarray, beta: np.ndarray, mass_number: int) -> np.ndarray:
    if len(beta) < 4:
        raise ValueError("beta must have at least 4 elements")
    R0 = nuclear_radius(mass_number)

    y2 = spherical_harmonic_y20(theta)
    y3 = spherical_harmonic_y30(theta)
    y4 = spherical_harmonic_y40(theta)
    y5 = spherical_harmonic_y50(theta)
    shape_factor = 1.0 + beta[0] * y2 + beta[1] * y3 + beta[2] * y4 + beta[3] * y5

    shape_factor = np.maximum(shape_factor, 0.1)
    return R0 * shape_factor


def mass_asymmetry_to_fragment_mass(beta3: float, mass_number: int) -> Tuple[float, float]:
    if mass_number <= 0:
        raise ValueError("mass_number must be positive")






    raise NotImplementedError("Hole_1: mass_asymmetry_to_fragment_mass 待修复")
    return 0.0, 0.0


def fragment_mass_to_asymmetry(A_light: float, mass_number: int) -> float:
    if A_light <= 0 or A_light >= mass_number:
        raise ValueError("A_light out of valid range")
    numerator = mass_number - 2.0 * A_light
    denominator = 2.0 * np.sqrt(A_light * (mass_number - A_light))
    if abs(denominator) < 1e-14:
        return 0.0
    x = numerator / denominator
    return x / C3_ASYMMETRY


def closest_point_brute(points: np.ndarray, target: np.ndarray) -> Tuple[int, float]:
    if points.ndim != 2 or target.ndim != 1:
        raise ValueError("points must be 2D array and target must be 1D array")
    if points.shape[1] != target.shape[0]:
        raise ValueError("dimension mismatch between points and target")
    if len(points) == 0:
        raise ValueError("points array is empty")

    min_dist_sq = np.inf
    nearest_idx = -1
    for i in range(len(points)):
        dist_sq = np.sum((points[i] - target) ** 2)
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            nearest_idx = i
    return nearest_idx, np.sqrt(min_dist_sq)


def collective_coordinate_bounds(mass_number: int) -> dict:

    scale = min(1.0, mass_number / 300.0)
    bounds = {
        'beta2': (-0.5 * scale, 2.5 * scale),
        'beta3': (-1.2 * scale, 1.2 * scale),
        'beta4': (-0.8 * scale, 0.8 * scale),
        'beta5': (-0.4 * scale, 0.4 * scale),
        'delta': (0.0, 3.0),
    }
    return bounds


def clip_to_physical_domain(q: np.ndarray, bounds: dict) -> np.ndarray:
    keys = ['beta2', 'beta3', 'beta4', 'beta5', 'delta']
    q_clipped = q.copy()
    for i, key in enumerate(keys):
        if i < len(q):
            lo, hi = bounds[key]
            q_clipped[i] = np.clip(q_clipped[i], lo, hi)
    return q_clipped


def triangulate_configuration_space_1d(n_nodes: int, q_min: float, q_max: float) -> np.ndarray:
    if n_nodes < 2:
        raise ValueError("n_nodes must be at least 2")
    return np.linspace(q_min, q_max, n_nodes + 1)
