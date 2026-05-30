
import numpy as np
from typing import Tuple, List


def hexagon_domain_sample(n_samples: int, radius: float = 1.0) -> np.ndarray:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if radius <= 0.0:
        raise ValueError("radius must be positive.")

    points = []


    apothem = radius * np.cos(np.pi / 6.0)
    while len(points) < n_samples:

        batch = np.random.uniform(-radius, radius, size=(n_samples * 4, 2))
        for p in batch:
            x, y = p
            if abs(x) <= apothem:
                slope = np.tan(np.pi / 6.0)
                y_limit = slope * (apothem - abs(x))
                if abs(y) <= y_limit:
                    points.append(p)
            if len(points) >= n_samples:
                break
    return np.array(points[:n_samples])


def voronoi_cell_centroid_2d_hexagonal(
    generators: np.ndarray,
    density_func,
    n_mc: int = 5000,
    radius: float = 1.0,
) -> np.ndarray:
    n = generators.shape[0]
    if n == 0:
        return np.zeros((0, 2))

    centroids = np.zeros((n, 2))
    counts = np.zeros(n)


    samples = hexagon_domain_sample(n_mc, radius)
    for s in samples:

        dists = np.sum((generators - s) ** 2, axis=1)
        j = int(np.argmin(dists))
        weight = max(density_func(s), 0.0)
        centroids[j] += weight * s
        counts[j] += weight


    for j in range(n):
        if counts[j] > 0.0:
            centroids[j] /= counts[j]
        else:
            centroids[j] = generators[j]

    return centroids


def lloyd_cvt_hexagonal(
    n_generators: int,
    density_func,
    radius: float = 1.0,
    n_iterations: int = 30,
    init_mode: str = "random",
) -> Tuple[np.ndarray, np.ndarray]:
    if n_generators < 1:
        raise ValueError("n_generators must be positive.")
    if n_iterations < 1:
        raise ValueError("n_iterations must be positive.")


    if init_mode == "random":
        generators = hexagon_domain_sample(n_generators, radius)
    elif init_mode == "compressed":

        r_small = radius * 0.3
        generators = []
        while len(generators) < n_generators:
            pts = np.random.uniform(-r_small, r_small, size=(n_generators * 2, 2))
            for p in pts:
                if np.linalg.norm(p) <= r_small:
                    generators.append(p)
                if len(generators) >= n_generators:
                    break
        generators = np.array(generators[:n_generators])
    else:
        raise ValueError("init_mode must be 'random' or 'compressed'.")

    energy_history = np.zeros(n_iterations)

    for it in range(n_iterations):
        centroids = voronoi_cell_centroid_2d_hexagonal(
            generators, density_func, n_mc=8000, radius=radius
        )


        samples = hexagon_domain_sample(10000, radius)
        total_energy = 0.0
        total_weight = 0.0
        for s in samples:
            dists = np.sum((generators - s) ** 2, axis=1)
            j = int(np.argmin(dists))
            w = max(density_func(s), 0.0)
            total_energy += w * dists[j]
            total_weight += w
        if total_weight > 0.0:
            energy_history[it] = total_energy / total_weight
        else:
            energy_history[it] = total_energy

        generators = centroids

    return generators, energy_history


def mbz_cvt_kpoints(
    theta_deg: float,
    n_k: int = 64,
    n_iterations: int = 20,
    dos_weight_func=None,
) -> np.ndarray:
    from tight_binding import moire_lattice_constant

    L_m = moire_lattice_constant(theta_deg)
    b_mag = 4.0 * np.pi / (np.sqrt(3.0) * L_m)
    radius = b_mag

    if dos_weight_func is None:
        def uniform_weight(k):
            return 1.0
        dos_weight_func = uniform_weight

    generators, _ = lloyd_cvt_hexagonal(
        n_k, dos_weight_func, radius=radius, n_iterations=n_iterations
    )
    return generators


def irreducible_wedge_kpoints(
    kpoints: np.ndarray,
    tolerance: float = 1e-8,
) -> np.ndarray:
    wedge_points = []
    for k in kpoints:
        x, y = k


        for _ in range(6):

            angle = np.pi / 3.0
            xr = x * np.cos(angle) + y * np.sin(angle)
            yr = -x * np.sin(angle) + y * np.cos(angle)
            x, y = xr, yr
            if x >= -tolerance and y >= -tolerance:
                tan_pi6 = np.tan(np.pi / 6.0)
                if y <= x * tan_pi6 + tolerance:
                    wedge_points.append([x, y])
                    break
        else:

            wedge_points.append([k[0], k[1]])


    if len(wedge_points) == 0:
        return np.zeros((0, 2))
    arr = np.array(wedge_points)
    unique = []
    for p in arr:
        if not unique or min(np.linalg.norm(np.array(unique) - p, axis=1)) > tolerance:
            unique.append(p)
    return np.array(unique)
