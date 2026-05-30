
import numpy as np
from typing import Tuple, Optional


def ellipsoid_grid_count(n: int, r: np.ndarray) -> int:
    D = len(r)
    r_min = np.min(r)
    if r_min < 1e-15:
        raise ValueError("半轴长度必须为正")
    h = 2.0 * r_min / (2.0 * n + 1.0)
    counts = np.ceil(r / r_min * n).astype(int)

    volume_ratio = np.prod(r) / (r_min ** D)
    est_points = int(volume_ratio * (2 * n + 1) ** D)
    return est_points


def ellipsoid_grid(n: int, r: np.ndarray, c: np.ndarray) -> np.ndarray:
    D = len(r)
    r_min = np.min(r)
    if r_min < 1e-15:
        raise ValueError("半轴长度必须为正")
    h = 2.0 * r_min / (2.0 * n + 1.0)
    ni = np.ceil(r / r_min * n).astype(int)
    points = []

    ranges = [range(ni[d] + 1) for d in range(D)]

    def gen_points(dim: int, current: list):
        if dim == D:
            x = np.array(current, dtype=np.float64)
            x_scaled = (x - c) / r
            if np.sum(x_scaled ** 2) <= 1.0 + 1e-12:

                p = c + x * h

                for mask in range(1 << D):
                    q = p.copy()
                    for d in range(D):
                        if (mask >> d) & 1:
                            q[d] = 2.0 * c[d] - q[d]

                    valid = True
                    for d in range(D):
                        if current[d] == 0 and ((mask >> d) & 1):
                            valid = False
                            break
                    if valid:
                        points.append(q)
            return
        for i in range(ni[dim] + 1):
            current.append(i)
            gen_points(dim + 1, current)
            current.pop()
    gen_points(0, [])
    if len(points) == 0:
        return np.array([c])
    return np.array(points)


def anisotropic_metric_tensor(data: np.ndarray, center: np.ndarray,
                               bandwidth: float = 1.0) -> np.ndarray:
    D = data.shape[1]
    diff = data - center
    dist_sq = np.sum(diff ** 2, axis=1)

    weights = np.exp(-dist_sq / (2.0 * bandwidth ** 2))
    weights = weights / (np.sum(weights) + 1e-15)

    g = np.zeros((D, D), dtype=np.float64)
    for k in range(len(data)):
        g += weights[k] * np.outer(diff[k], diff[k])

    g += 1e-6 * np.eye(D)
    return g


def board_grid_discretize(bounds: np.ndarray, n_cells: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    D = len(n_cells)
    edges = []
    for d in range(D):
        e = np.linspace(bounds[d, 0], bounds[d, 1], n_cells[d] + 1)
        edges.append(e)

    centers_list = []
    def gen_centers(dim: int, current: list):
        if dim == D:
            centers_list.append(np.array(current, dtype=np.float64))
            return
        for i in range(n_cells[dim]):
            center = (edges[dim][i] + edges[dim][i + 1]) / 2.0
            current.append(center)
            gen_centers(dim + 1, current)
            current.pop()
    gen_centers(0, [])
    return np.array(centers_list), edges


def adaptive_ellipsoid_sample(data: np.ndarray, target_n_points: int = 500,
                               n_levels: int = 3) -> np.ndarray:
    mean = np.mean(data, axis=0)
    cov = np.cov(data.T)

    eigvals, eigvecs = np.linalg.eigh(cov)

    r = np.sqrt(np.maximum(eigvals, 1e-10))

    r = r / np.max(r)
    samples = []
    for level in range(n_levels):
        n = 2 + level * 2

        scale = 1.0 / (level + 1)
        pts = ellipsoid_grid(n, r * scale, mean)

        pts = (eigvecs @ pts.T).T + mean
        samples.append(pts)
    all_samples = np.vstack(samples)

    if len(all_samples) > target_n_points:
        idx = np.random.choice(len(all_samples), target_n_points, replace=False)
        all_samples = all_samples[idx]
    return all_samples


def local_tangent_space(data: np.ndarray, center: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    diff = data - center
    dist_sq = np.sum(diff ** 2, axis=1)
    idx = np.argsort(dist_sq)[:k]
    local_data = diff[idx]
    cov = local_data.T @ local_data / k
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    return eigvecs[:, idx], eigvals[idx]
