
import numpy as np
from typing import Tuple


def piecewise_constant_1d(x_breaks: np.ndarray, y_values: np.ndarray,
                          x_query: np.ndarray) -> np.ndarray:
    n = len(y_values)
    result = np.zeros(len(x_query))
    for i, xq in enumerate(x_query):

        idx = np.searchsorted(x_breaks, xq, side='right') - 1
        idx = np.clip(idx, 0, n - 1)
        result[i] = y_values[idx]
    return result


def piecewise_constant_nd(data: np.ndarray, n_bins: int = 10) -> Tuple[np.ndarray, list, list]:
    D = data.shape[1]
    N = len(data)

    effective_bins = n_bins
    if D > 6:
        cov = np.cov(data.T)
        eigvals, eigvecs = np.linalg.eigh(cov)

        data = data @ eigvecs[:, :6]
        D = 6
    bin_edges = []
    bin_centers = []
    for d in range(D):
        xmin, xmax = np.min(data[:, d]), np.max(data[:, d])
        edges = np.linspace(xmin, xmax, effective_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        bin_edges.append(edges)
        bin_centers.append(centers)

    counts = np.zeros([effective_bins] * D, dtype=int)
    for pt in data:
        idx = []
        for d in range(D):
            ix = int((pt[d] - bin_edges[d][0]) / (bin_edges[d][-1] - bin_edges[d][0] + 1e-10) * effective_bins)
            ix = min(ix, effective_bins - 1)
            idx.append(ix)
        counts[tuple(idx)] += 1

    vol = 1.0
    for d in range(D):
        vol *= (bin_edges[d][1] - bin_edges[d][0])
    density = counts / (N * vol + 1e-15)
    return density, bin_edges, bin_centers


def adaptive_piecewise_density(data: np.ndarray, min_bins: int = 4,
                                max_bins: int = 32) -> Tuple[np.ndarray, list]:
    N, D = data.shape

    n_bins = min(max_bins, max(min_bins, int(N ** (1.0 / D) / 2)))
    density, edges, _ = piecewise_constant_nd(data, n_bins)

    for _ in range(2):

        threshold = np.mean(density[density > 0])
        high_density_mask = density > threshold

        n_bins = min(max_bins, n_bins + 2)
        density, edges, _ = piecewise_constant_nd(data, n_bins)
    return density, edges


def pwc_histogram_entropy(data: np.ndarray, n_bins: int = 20) -> float:
    density, edges, _ = piecewise_constant_nd(data, n_bins)

    cell_vol = 1.0
    for d in range(len(edges)):
        cell_vol *= (edges[d][1] - edges[d][0])
    entropy = 0.0
    for idx in np.ndindex(*density.shape):
        p = density[idx] * cell_vol
        if p > 1e-15:
            entropy -= p * np.log(density[idx] + 1e-15)
    return float(entropy)


def pwc_mutual_information(data_x: np.ndarray, data_y: np.ndarray,
                            n_bins: int = 10) -> float:
    data_joint = np.hstack([data_x, data_y])

    safe_bins = min(n_bins, 6)
    H_x = pwc_histogram_entropy(data_x, safe_bins)
    H_y = pwc_histogram_entropy(data_y, safe_bins)
    H_joint = pwc_histogram_entropy(data_joint, safe_bins)
    return max(0.0, H_x + H_y - H_joint)
