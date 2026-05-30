
import numpy as np
from typing import List, Tuple


def uniform_on_sphere01(n_points: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xyz = rng.standard_normal((n_points, 3), dtype=np.float64)
    norms = np.linalg.norm(xyz, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    points = xyz / norms
    return points


def spherical_triangle_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
                             radius: float = 1.0) -> float:
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)


    p1 = p1 / np.linalg.norm(p1)
    p2 = p2 / np.linalg.norm(p2)
    p3 = p3 / np.linalg.norm(p3)


    n12 = np.cross(p1, p2)
    n23 = np.cross(p2, p3)
    n31 = np.cross(p3, p1)

    n12_norm = np.linalg.norm(n12)
    n23_norm = np.linalg.norm(n23)
    n31_norm = np.linalg.norm(n31)

    if n12_norm < 1e-14 or n23_norm < 1e-14 or n31_norm < 1e-14:
        return 0.0

    n12 = n12 / n12_norm
    n23 = n23 / n23_norm
    n31 = n31 / n31_norm


    cos_alpha = np.clip(np.dot(-n12, n31), -1.0, 1.0)
    cos_beta = np.clip(np.dot(-n23, n12), -1.0, 1.0)
    cos_gamma = np.clip(np.dot(-n31, n23), -1.0, 1.0)

    alpha = np.arccos(cos_alpha)
    beta = np.arccos(cos_beta)
    gamma = np.arccos(cos_gamma)

    area = (alpha + beta + gamma - np.pi) * (radius ** 2)
    return float(np.maximum(area, 0.0))


def spherical_polygon_centroid(points: np.ndarray,
                                radius: float = 1.0) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    m = len(points)
    if m < 3:
        raise ValueError("Polygon must have at least 3 vertices.")


    center = np.mean(points, axis=0)
    center = center / np.linalg.norm(center)

    areas = []
    weighted_sum = np.zeros(3, dtype=np.float64)

    for k in range(m):
        p1 = points[k]
        p2 = points[(k + 1) % m]
        a = spherical_triangle_area(p1, p2, center, radius)
        areas.append(a)

        tri_centroid = (p1 + p2 + center) / 3.0
        weighted_sum += a * tri_centroid

    total_area = sum(areas)
    if total_area < 1e-15:
        return center

    centroid = weighted_sum / total_area
    centroid = centroid / np.linalg.norm(centroid)
    return centroid


def sphere_cvt_step(generators: np.ndarray,
                     radius: float = 1.0) -> np.ndarray:
    generators = np.asarray(generators, dtype=np.float64)
    n = len(generators)


    norms = np.linalg.norm(generators, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    generators = generators / norms

    new_generators = np.zeros_like(generators)

    for i in range(n):
        gi = generators[i]

        dists = np.linalg.norm(generators - gi, axis=1)


        k = min(12, n - 1)
        neighbor_idx = np.argpartition(dists, k)[:k + 1]
        neighbors = generators[neighbor_idx]


        local_centroid = np.mean(neighbors, axis=0)
        local_centroid = local_centroid / np.linalg.norm(local_centroid)
        new_generators[i] = local_centroid

    return new_generators * radius


def sphere_cvt_iterate(n_generators: int,
                        n_iterations: int = 100,
                        radius: float = 1.0,
                        seed: int = 42) -> np.ndarray:
    generators = uniform_on_sphere01(n_generators, seed)
    generators = generators * radius

    for it in range(n_iterations):
        generators = sphere_cvt_step(generators, radius)

    return generators


def cvt_energy(generators: np.ndarray) -> float:
    generators = np.asarray(generators, dtype=np.float64)
    n = len(generators)
    if n < 2:
        return 0.0

    energy = 0.0
    for i in range(n):
        dists = np.linalg.norm(generators - generators[i], axis=1)
        dists[i] = np.inf
        energy += np.min(dists) ** 2

    return float(energy / n)


def project_to_ice_dome_region(points: np.ndarray,
                                latitude_range: Tuple[float, float] = (-90.0, -60.0),
                                longitude_range: Tuple[float, float] = (-180.0, 180.0),
                                earth_radius: float = 6371e3) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    lat_min, lat_max = latitude_range
    lon_min, lon_max = longitude_range

    filtered = []
    for p in points:
        x, y, z = p
        lat = np.degrees(np.arcsin(np.clip(z / earth_radius, -1.0, 1.0)))
        lon = np.degrees(np.arctan2(y, x))

        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            filtered.append(p)

    if not filtered:

        return points[:max(1, len(points) // 4)]

    return np.array(filtered, dtype=np.float64)
