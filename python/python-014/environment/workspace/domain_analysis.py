
import numpy as np
from typing import Tuple, Dict, List
from utils import histogram_stats_1d, triangle_area_histogram_2d, EPS_MACHINE
from spin_lattice import connected_components_2d_spin_map


def extract_domain_statistics(
    spin_map: np.ndarray, threshold: float = 0.5
) -> Dict:
    labels = connected_components_2d_spin_map(spin_map, threshold)
    max_label = int(labels.max())
    if max_label == 0:
        return {
            "n_domains": 0,
            "domain_sizes": np.array([]),
            "max_domain_size": 0,
            "mean_domain_size": 0.0,
            "domain_size_entropy": 0.0,
            "histogram": {},
        }

    sizes = np.array([np.sum(labels == k) for k in range(1, max_label + 1)], dtype=int)
    total = np.sum(sizes)
    probs = sizes / total
    entropy = -np.sum(probs * np.log(probs + EPS_MACHINE))

    counts, edges, stats = histogram_stats_1d(sizes.astype(float), bins=min(20, max(5, max_label)))

    return {
        "n_domains": int(max_label),
        "domain_sizes": sizes,
        "max_domain_size": int(np.max(sizes)),
        "mean_domain_size": float(np.mean(sizes)),
        "domain_size_entropy": float(entropy),
        "histogram": {
            "counts": counts,
            "edges": edges,
            "stats": stats,
        },
    }


def spin_orientation_histogram(
    spins: np.ndarray, n_bins: int = 18
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    N = spins.shape[0]
    sz = np.clip(spins[:, 2], -1.0, 1.0)
    theta = np.arccos(sz)
    counts, edges = np.histogram(theta, bins=n_bins, range=(0.0, np.pi))
    theta_centers = 0.5 * (edges[:-1] + edges[1:])
    counts_f = counts.astype(float)
    _, _, stats = histogram_stats_1d(theta, bins=n_bins)
    return counts, theta_centers, stats


def radial_distribution_function_2d(
    positions: np.ndarray, spins: np.ndarray, max_r: float = 0.5, n_bins: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must be (N, 2)")
    N = positions.shape[0]
    dr = max_r / n_bins
    g = np.zeros(n_bins, dtype=float)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(N):
        dx = positions[:, 0] - positions[i, 0]
        dy = positions[:, 1] - positions[i, 1]

        dx = np.minimum(np.abs(dx), 1.0 - np.abs(dx))
        dy = np.minimum(np.abs(dy), 1.0 - np.abs(dy))
        r_vals = np.sqrt(dx * dx + dy * dy)
        for j in range(i + 1, N):
            r = r_vals[j]
            if r >= max_r or r < EPS_MACHINE:
                continue
            idx = int(r / dr)
            if idx >= n_bins:
                continue
            corr = np.dot(spins[i], spins[j])
            g[idx] += corr
            counts[idx] += 1

    mask = counts > 0
    g[mask] /= counts[mask]
    r_centers = np.linspace(0.5 * dr, max_r - 0.5 * dr, n_bins)
    return r_centers, g


def entropy_rate_from_trajectory(mz_trajectory: np.ndarray, delay: int = 1) -> float:
    delta = np.diff(mz_trajectory)
    if delta.size == 0:
        return 0.0

    dmin, dmax = delta.min(), delta.max()
    if abs(dmax - dmin) < EPS_MACHINE:
        return 0.0
    bins = 20
    counts, _ = np.histogram(delta, bins=bins)
    probs = counts / np.sum(counts)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def analyze_triangle_spin_distribution(
    spins_xy: np.ndarray, n_sub: int = 5
) -> Tuple[np.ndarray, Dict]:

    pts = spins_xy.copy()

    pts = np.abs(pts)
    s = np.sum(pts, axis=1, keepdims=True) + EPS_MACHINE
    pts = pts / s
    histo, info = triangle_area_histogram_2d(pts, n_sub)
    return histo, info
