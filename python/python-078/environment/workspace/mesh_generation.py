
import numpy as np
from typing import Tuple






def find_closest(generators: np.ndarray, samples: np.ndarray) -> np.ndarray:
    n_gen = generators.shape[0]
    n_samp = samples.shape[0]
    nearest = np.zeros(n_samp, dtype=int)

    for i in range(n_samp):
        dists = np.sum((generators - samples[i]) ** 2, axis=1)
        nearest[i] = int(np.argmin(dists))
    return nearest


def cvt_iterate(generators: np.ndarray, n_samples: int,
                density_func=None, seed: int = None) -> Tuple[np.ndarray, float, float]:
    if seed is not None:
        np.random.seed(seed)

    n_gen = generators.shape[0]
    if n_gen < 1:
        return generators.copy(), 0.0, 0.0


    if density_func is None:
        samples = np.random.rand(n_samples, 2)
    else:


        candidates = np.random.rand(n_samples * 5, 2)
        weights = density_func(candidates[:, 0], candidates[:, 1])
        weights = np.maximum(weights, 1e-15)
        weights /= weights.sum()
        idx = np.random.choice(len(candidates), size=n_samples, p=weights, replace=True)
        samples = candidates[idx]


    samples = np.clip(samples, 0.0, 1.0)

    nearest = find_closest(generators, samples)
    new_generators = np.zeros_like(generators)
    counts = np.zeros(n_gen)

    for j in range(n_gen):
        mask = (nearest == j)
        if np.any(mask):
            new_generators[j] = np.mean(samples[mask], axis=0)
            counts[j] = np.sum(mask)
        else:

            new_generators[j] = np.random.rand(2)
            counts[j] = 1


    diff = float(np.sum(np.linalg.norm(new_generators - generators, axis=1)))


    energy = 0.0
    for i in range(n_samples):
        j = nearest[i]
        d2 = np.sum((samples[i] - generators[j]) ** 2)
        if density_func is not None:
            rho = density_func(np.array([samples[i, 0]]), np.array([samples[i, 1]]))[0]
            energy += rho * d2
        else:
            energy += d2
    energy /= n_samples

    return new_generators, diff, energy


def generate_cvt_mesh(n_generators: int, n_samples_per_gen: int = 100,
                      max_iter: int = 100, tol: float = 1e-5,
                      density_func=None, seed: int = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)

    generators = np.random.rand(n_generators, 2)
    n_samples = n_generators * n_samples_per_gen

    for it in range(max_iter):
        generators, diff, energy = cvt_iterate(
            generators, n_samples, density_func=density_func, seed=None
        )
        if diff < tol:
            break

    return generators






def vessel_wall_density(x: np.ndarray, y: np.ndarray,
                        thickness_center: float = 0.5,
                        thickness_amplitude: float = 0.3,
                        n_modes: int = 3) -> np.ndarray:
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    thickness = thickness_center * np.ones_like(x)

    for k in range(1, n_modes + 1):
        thickness += thickness_amplitude * (
            np.sin(2.0 * np.pi * k * x) * np.cos(2.0 * np.pi * k * y)
        ) / k

    thickness = np.clip(thickness, 0.1 * thickness_center, 3.0 * thickness_center)
    density = 1.0 / thickness
    return density


def map_cvt_to_annulus(generators: np.ndarray,
                       inner_radius: float, outer_radius: float) -> np.ndarray:
    x = generators[:, 0]
    y = generators[:, 1]

    theta = 2.0 * np.pi * x
    r = inner_radius + y * (outer_radius - inner_radius)

    x_cart = r * np.cos(theta)
    y_cart = r * np.sin(theta)
    return np.column_stack([x_cart, y_cart])






class VascularCVTMesh:
    def __init__(self, generators: np.ndarray, inner_r: float, outer_r: float):
        self.generators = generators.copy()
        self.inner_radius = inner_r
        self.outer_radius = outer_r
        self.cartesian = map_cvt_to_annulus(generators, inner_r, outer_r)

    def radial_coordinates(self) -> np.ndarray:
        pts = self.cartesian
        return np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)

    def angular_coordinates(self) -> np.ndarray:
        pts = self.cartesian
        return np.arctan2(pts[:, 1], pts[:, 0])

    def wall_thickness_distribution(self,
                                     thickness_center: float = 1.0e-3,
                                     amplitude: float = 0.3e-3) -> np.ndarray:
        theta = self.angular_coordinates()

        thickness = thickness_center + amplitude * np.cos(theta)
        return np.clip(thickness, 0.2e-3, 2.0e-3)
