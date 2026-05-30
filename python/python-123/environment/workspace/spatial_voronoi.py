
import numpy as np
from typing import Tuple


def disk_sample_uniform(num_samples: int, radius: float = 1.0,
                        seed: int = 42) -> np.ndarray:
    if num_samples < 0:
        raise ValueError("disk_sample_uniform: num_samples >= 0")
    if radius <= 0:
        raise ValueError("disk_sample_uniform: radius > 0")

    rng = np.random.default_rng(seed=seed)
    u = rng.random(num_samples)
    v = rng.random(num_samples)
    r = radius * np.sqrt(u)
    theta = 2.0 * np.pi * v
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])


def find_closest(sample_points: np.ndarray, generators: np.ndarray) -> np.ndarray:
    S = sample_points.shape[0]
    G = generators.shape[0]
    nearest = np.zeros(S, dtype=int)

    for s in range(S):
        min_dist = np.inf
        min_idx = 0
        for g in range(G):
            dx = sample_points[s, 0] - generators[g, 0]
            dy = sample_points[s, 1] - generators[g, 1]
            d2 = dx * dx + dy * dy
            if d2 < min_dist:
                min_dist = d2
                min_idx = g
        nearest[s] = min_idx

    return nearest


def cvt_disk_iterate(
    radius: float, num_samples: int, generators: np.ndarray,
    p_type: np.ndarray, num_iterations: int = 30
) -> np.ndarray:
    N = generators.shape[0]
    if p_type.shape[0] != N:
        raise ValueError("cvt_disk_iterate: p_type 长度与生成子数不匹配")
    if num_samples < N:
        raise ValueError("cvt_disk_iterate: num_samples >= 生成子数")

    for _ in range(num_iterations):
        sample_points = disk_sample_uniform(num_samples, radius)
        nearest = find_closest(sample_points, generators)

        v_xy = generators.copy()
        counts = np.ones(N)

        for s in range(num_samples):
            g = nearest[s]
            v_xy[g, 0] += sample_points[s, 0]
            v_xy[g, 1] += sample_points[s, 1]
            counts[g] += 1.0


        for g in range(N):
            if counts[g] > 1e-12:
                v_xy[g, 0] /= counts[g]
                v_xy[g, 1] /= counts[g]


        for g in range(N):
            if p_type[g] == 1:
                r2 = v_xy[g, 0] ** 2 + v_xy[g, 1] ** 2
                r = np.sqrt(max(r2, 1e-15))
                v_xy[g, 0] = radius * v_xy[g, 0] / r
                v_xy[g, 1] = radius * v_xy[g, 1] / r
            else:

                r2 = v_xy[g, 0] ** 2 + v_xy[g, 1] ** 2
                if r2 > radius ** 2:
                    r = np.sqrt(r2)
                    v_xy[g, 0] = radius * v_xy[g, 0] / r * 0.999
                    v_xy[g, 1] = radius * v_xy[g, 1] / r * 0.999

        generators = v_xy

    return generators


def initialize_tumor_generators(
    n_boundary: int, n_interior: int, radius: float, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    if n_boundary < 3:
        raise ValueError("initialize_tumor_generators: n_boundary >= 3")
    if n_interior < 0:
        raise ValueError("initialize_tumor_generators: n_interior >= 0")
    if radius <= 0:
        raise ValueError("initialize_tumor_generators: radius > 0")

    np_total = n_boundary + n_interior
    rng = np.random.default_rng(seed=seed)


    interior = disk_sample_uniform(n_interior, radius, seed=seed)


    theta = np.linspace(0.0, 2.0 * np.pi, n_boundary, endpoint=False)
    boundary = np.column_stack([
        radius * np.cos(theta),
        radius * np.sin(theta)
    ])

    generators = np.vstack([boundary, interior])
    p_type = np.zeros(np_total, dtype=int)
    p_type[:n_boundary] = 1
    p_type[n_boundary:] = 2

    return generators, p_type


def radial_growth_expand(
    generators: np.ndarray, p_type: np.ndarray,
    radius: float, new_boundary_count: int
) -> Tuple[float, np.ndarray, np.ndarray]:
    n_old_boundary = int(np.sum(p_type == 1))
    if n_old_boundary <= 0:
        n_old_boundary = 1

    n_new_boundary = n_old_boundary + new_boundary_count
    factor = n_new_boundary / n_old_boundary

    new_radius = radius * factor
    new_generators = generators.copy() * factor


    new_p_type = np.abs(p_type)

    return new_radius, new_generators, new_p_type


def add_boundary_generators(
    generators: np.ndarray, p_type: np.ndarray,
    radius: float, n_add: int, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed=seed)
    theta_new = rng.uniform(0.0, 2.0 * np.pi, n_add)
    new_points = np.column_stack([
        radius * np.cos(theta_new),
        radius * np.sin(theta_new)
    ])

    new_generators = np.vstack([generators, new_points])
    new_p_type = np.concatenate([
        p_type,
        np.ones(n_add, dtype=int)
    ])

    return new_generators, new_p_type


def partition_metabolic_activity(
    weights: np.ndarray
) -> Tuple[np.ndarray, float]:
    weights = np.asarray(weights, dtype=float)
    if np.any(weights < 0):
        raise ValueError("partition_metabolic_activity: 权重必须非负")

    n = weights.shape[0]
    if n == 0:
        return np.array([], dtype=int), 0.0


    idx_desc = np.argsort(-weights)
    labels = np.zeros(n, dtype=int)

    s0_sum = 0.0
    s1_sum = 0.0
    for i in range(n):
        j = idx_desc[i]
        if s0_sum < s1_sum:
            labels[j] = 0
            s0_sum += weights[j]
        else:
            labels[j] = 1
            s1_sum += weights[j]

    discrepancy = abs(s0_sum - s1_sum)
    return labels, float(discrepancy)


def compute_voronoi_energy(generators: np.ndarray,
                           sample_points: np.ndarray) -> float:
    nearest = find_closest(sample_points, generators)
    energy = 0.0
    for s in range(sample_points.shape[0]):
        g = nearest[s]
        dx = sample_points[s, 0] - generators[g, 0]
        dy = sample_points[s, 1] - generators[g, 1]
        energy += dx * dx + dy * dy
    return float(energy)
