
import numpy as np
from typing import List, Tuple, Optional


def hypersphere_positive_sample(m: int) -> np.ndarray:
    if m <= 0:
        raise ValueError("Dimension m must be positive.")
    z = np.abs(np.random.randn(m))
    norm = np.linalg.norm(z)
    if norm < 1e-15:
        norm = 1e-15
    return z / norm


def hypersphere_positive_distance_stats(m: int, n_samples: int = 5000) -> Tuple[float, float]:
    if n_samples < 2:
        n_samples = 2
    distances = []
    for _ in range(n_samples):
        p1 = hypersphere_positive_sample(m)
        p2 = hypersphere_positive_sample(m)
        d = np.linalg.norm(p1 - p2)
        distances.append(d)
    distances = np.array(distances, dtype=float)
    return float(np.mean(distances)), float(np.var(distances, ddof=1))


def theoretical_mean_distance(m: int) -> float:
    from math import gamma, sqrt, pi
    if m <= 1:
        return 2.0

    E_full = (2.0 ** (m - 1)) * (gamma(m / 2.0) ** 2) / (sqrt(pi) * gamma(m - 0.5))

    return E_full / np.sqrt(2.0)


def compute_partition_quality(volumes: np.ndarray) -> float:
    mean_vol = np.mean(volumes)
    if mean_vol < 1e-15:
        return 0.0
    std_vol = np.std(volumes, ddof=1)
    Q = max(0.0, 1.0 - std_vol / mean_vol)
    return float(Q)


def plane_tetrahedron_intersect(
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    tetra_vertices: np.ndarray
) -> Optional[np.ndarray]:
    n = plane_normal / (np.linalg.norm(plane_normal) + 1e-15)
    d = np.dot(tetra_vertices - plane_point, n)


    on_plane = []
    for k in range(4):
        if abs(d[k]) < 1e-10:
            on_plane.append(tetra_vertices[k].copy())


    edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    for a, b in edges:
        if d[a] * d[b] < -1e-15:

            t = d[a] / (d[a] - d[b])
            pt = tetra_vertices[a] + t * (tetra_vertices[b] - tetra_vertices[a])
            on_plane.append(pt)

    if len(on_plane) == 0:
        return None
    if len(on_plane) == 1:
        return np.array([on_plane[0]])



    if abs(n[2]) < 0.9:
        e1 = np.cross(n, np.array([0.0, 0.0, 1.0]))
    else:
        e1 = np.cross(n, np.array([0.0, 1.0, 0.0]))
    e1 = e1 / (np.linalg.norm(e1) + 1e-15)
    e2 = np.cross(n, e1)

    pts_2d = []
    for pt in on_plane:
        v = pt - plane_point
        pts_2d.append([np.dot(v, e1), np.dot(v, e2)])
    pts_2d = np.array(pts_2d, dtype=float)


    centroid = np.mean(pts_2d, axis=0)
    angles = np.arctan2(pts_2d[:, 1] - centroid[1], pts_2d[:, 0] - centroid[0])
    order = np.argsort(angles)


    unique = [order[0]]
    for idx in order[1:]:
        dist = np.linalg.norm(pts_2d[idx] - pts_2d[unique[-1]])
        if dist > 1e-10:
            unique.append(idx)

    polygon = np.array([on_plane[i] for i in unique], dtype=float)
    return polygon


def polygon_area_3d(polygon: np.ndarray) -> float:
    if polygon.shape[0] < 3:
        return 0.0
    n = polygon.shape[0]
    centroid = np.mean(polygon, axis=0)
    area_vec = np.zeros(3, dtype=float)
    for i in range(n):
        v1 = polygon[i] - centroid
        v2 = polygon[(i + 1) % n] - centroid
        area_vec += np.cross(v1, v2)
    return 0.5 * np.linalg.norm(area_vec)


def partial_digest_reconstruct(
    distances: List[float],
    max_coord: float = 1.0,
    tol: float = 1e-9
) -> Optional[np.ndarray]:
    if len(distances) < 1:
        return np.array([0.0, max_coord])
    d_sorted = sorted(distances, reverse=True)
    n_est = int((1 + np.sqrt(1 + 8 * len(distances))) / 2 + 0.5)

    def remove_distances(remaining: List[float], dists_to_remove: List[float]) -> Optional[List[float]]:
        rem = remaining.copy()
        for d in dists_to_remove:
            found = False
            for i, r in enumerate(rem):
                if abs(r - d) <= tol:
                    rem.pop(i)
                    found = True
                    break
            if not found:
                return None
        return rem

    def backtrack(remaining: List[float], points: List[float]) -> Optional[List[float]]:
        if len(remaining) == 0:
            return points
        max_d = remaining[0]

        candidates = set()
        for p in points:
            cand = p + max_d
            if cand <= max_coord + tol:
                candidates.add(round(cand / tol) * tol)
            cand2 = p - max_d
            if cand2 >= -tol:
                candidates.add(round(max(cand2, 0.0) / tol) * tol)

        for cand in sorted(candidates, reverse=True):
            new_dists = [abs(cand - q) for q in points]
            rem_copy = remove_distances(remaining, new_dists)
            if rem_copy is None:
                continue
            res = backtrack(rem_copy, points + [cand])
            if res is not None:
                return res
        return None

    result = backtrack(d_sorted, [0.0])
    if result is None:
        return None
    return np.array(sorted(result), dtype=float)


def interface_matching_score(
    nodes_i: np.ndarray,
    nodes_j: np.ndarray,
    tol: float = 1e-6
) -> float:
    if nodes_i.shape[0] == 0 or nodes_j.shape[0] == 0:
        return 0.0
    matched_i = 0
    for p in nodes_i:
        dists = np.linalg.norm(nodes_j - p, axis=1)
        if np.min(dists) <= tol:
            matched_i += 1
    matched_j = 0
    for p in nodes_j:
        dists = np.linalg.norm(nodes_i - p, axis=1)
        if np.min(dists) <= tol:
            matched_j += 1
    score = 0.5 * (matched_i / nodes_i.shape[0] + matched_j / nodes_j.shape[0])
    return float(score)
