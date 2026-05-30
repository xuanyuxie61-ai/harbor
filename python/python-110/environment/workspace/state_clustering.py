
import numpy as np
from typing import Tuple, List, Dict
from utils import validate_array_1d, validate_array_2d


def criterion_variance(
    data: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
) -> float:
    data = validate_array_2d(data, "data")
    labels = validate_array_1d(labels, "labels")
    n_samples = data.shape[1]
    if labels.size != n_samples:
        raise ValueError("labels size must match number of samples")
    total = 0.0
    for k in range(n_clusters):
        mask = labels == k
        if not np.any(mask):
            continue
        cluster_data = data[:, mask]
        mu_k = np.mean(cluster_data, axis=1, keepdims=True)
        diff = cluster_data - mu_k
        total += float(np.sum(diff ** 2))
    return total


def _compute_cluster_means(data: np.ndarray, labels: np.ndarray, n_clusters: int):
    dim, n = data.shape
    means = np.zeros((dim, n_clusters), dtype=float)
    for k in range(n_clusters):
        mask = labels == k
        if np.any(mask):
            means[:, k] = np.mean(data[:, mask], axis=1)
    return means


def transfer_step(
    data: np.ndarray,
    labels: np.ndarray,
    cluster_sizes: np.ndarray,
    n_clusters: int,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    data = validate_array_2d(data, "data")
    labels = np.array(labels, dtype=int).copy()
    cluster_sizes = np.array(cluster_sizes, dtype=int).copy()
    n_samples = data.shape[1]
    if labels.size != n_samples or cluster_sizes.size != n_clusters:
        raise ValueError("Dimension mismatch")

    eps = 1e-12
    ntrans = 0
    current_crit = criterion_variance(data, labels, n_clusters)

    for i in range(n_samples):
        m = labels[i]
        if cluster_sizes[m] <= 1:
            continue
        best_l = m
        best_crit = current_crit
        for l in range(n_clusters):
            if l == m:
                continue

            labels[i] = l
            new_crit = criterion_variance(data, labels, n_clusters)
            if new_crit < best_crit - eps:
                best_crit = new_crit
                best_l = l
            labels[i] = m
        if best_l != m:
            labels[i] = best_l
            cluster_sizes[m] -= 1
            cluster_sizes[best_l] += 1
            current_crit = best_crit
            ntrans += 1

    return labels, cluster_sizes, current_crit, ntrans


def swap_step(
    data: np.ndarray,
    labels: np.ndarray,
    cluster_sizes: np.ndarray,
    n_clusters: int,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    data = validate_array_2d(data, "data")
    labels = np.array(labels, dtype=int).copy()
    cluster_sizes = np.array(cluster_sizes, dtype=int).copy()
    n_samples = data.shape[1]
    eps = 1e-12
    ntrans = 0
    current_crit = criterion_variance(data, labels, n_clusters)

    for i in range(n_samples):
        l = labels[i]
        for j in range(i):
            m = labels[j]
            if l == m:
                continue
            if cluster_sizes[l] <= 1 or cluster_sizes[m] <= 1:
                continue

            labels[i] = m
            labels[j] = l
            new_crit = criterion_variance(data, labels, n_clusters)
            if new_crit < current_crit - eps:
                current_crit = new_crit
                ntrans += 1
            else:

                labels[i] = l
                labels[j] = m

    return labels, cluster_sizes, current_crit, ntrans


def optimize_clustering(
    data: np.ndarray,
    n_clusters: int,
    max_iter: int = 50,
) -> Dict[str, np.ndarray]:
    data = validate_array_2d(data, "data")
    n_samples = data.shape[1]
    if n_clusters < 2 or n_clusters > n_samples:
        raise ValueError("Invalid number of clusters")


    labels = np.zeros(n_samples, dtype=int)
    centers_init = np.zeros((data.shape[0], n_clusters), dtype=float)

    first_idx = np.random.randint(0, n_samples)
    centers_init[:, 0] = data[:, first_idx]
    for k in range(1, n_clusters):
        dists = np.full(n_samples, float('inf'))
        for i in range(n_samples):
            d_min = float('inf')
            for j in range(k):
                d = np.sum((data[:, i] - centers_init[:, j]) ** 2)
                if d < d_min:
                    d_min = d
            dists[i] = d_min
        probs = dists / (np.sum(dists) + 1e-15)
        next_idx = np.random.choice(n_samples, p=probs)
        centers_init[:, k] = data[:, next_idx]

    for i in range(n_samples):
        best_k = 0
        best_d = float('inf')
        for k in range(n_clusters):
            d = np.sum((data[:, i] - centers_init[:, k]) ** 2)
            if d < best_d:
                best_d = d
                best_k = k
        labels[i] = best_k
    cluster_sizes = np.array([np.sum(labels == k) for k in range(n_clusters)], dtype=int)

    crit_history = []
    for it in range(max_iter):
        labels, cluster_sizes, crit, nt = transfer_step(data, labels, cluster_sizes, n_clusters)
        crit_history.append(crit)
        if nt == 0:
            labels, cluster_sizes, crit, ns = swap_step(data, labels, cluster_sizes, n_clusters)
            crit_history.append(crit)
            if ns == 0:
                break


    centers = np.zeros((data.shape[0], n_clusters), dtype=float)
    for k in range(n_clusters):
        mask = labels == k
        if np.any(mask):
            centers[:, k] = np.mean(data[:, mask], axis=1)

    return {
        "labels": labels,
        "cluster_centers": centers,
        "cluster_sizes": cluster_sizes,
        "criterion_history": np.array(crit_history, dtype=float),
    }


def classify_quantum_dot_ensemble(
    energies_eV: np.ndarray,
    linewidths_meV: np.ndarray,
    n_clusters: int = 3,
) -> Dict[str, np.ndarray]:
    energies_eV = validate_array_1d(energies_eV, "energies_eV")
    linewidths_meV = validate_array_1d(linewidths_meV, "linewidths_meV")
    if energies_eV.size != linewidths_meV.size:
        raise ValueError("energies and linewidths must have same size")
    n = energies_eV.size
    if n < n_clusters:
        raise ValueError("Not enough samples for clustering")

    data = np.vstack([energies_eV, linewidths_meV])
    result = optimize_clustering(data, n_clusters)
    return result


def spectral_purity_index(
    cluster_center_energy: float,
    cluster_mean_linewidth: float,
    temperature_K: float = 4.0,
) -> float:
    kB = 1.380649e-23
    if cluster_mean_linewidth <= 0 or temperature_K <= 0:
        raise ValueError("Linewidth and temperature must be positive")

    Delta_omega = cluster_mean_linewidth * 1e-3 * 1.602176634e-19 / 1.054571817e-34
    PI = (kB * temperature_K) / (1.054571817e-34 * Delta_omega)
    return float(PI)
