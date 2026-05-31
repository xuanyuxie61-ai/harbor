"""
Atomic-Stacking Classification and K-Means Clustering
======================================================
Classifies atoms in the moiré supercell into distinct stacking regions
(AA, AB, BA) using K-Means and H-Means clustering on structural
features, and computes cluster energetics.

Scientific Background
---------------------
In twisted bilayer graphene the local interlayer stacking varies
periodically across the moiré unit cell.  Three high-symmetry stackings
are dominant:

    AA : A sublattice of top layer sits directly above A of bottom layer
    AB : A of top sits above B of bottom (Bernal stacking)
    BA : B of top sits above A of bottom

The local stacking determines the interlayer hopping amplitude and thus
the local electronic density.  By clustering atoms according to their
in-plane displacement relative to the opposing layer, we can map the
moiré pattern into distinct domains.

For N atoms with feature vectors x_i ∈ ℝ^d, K-Means minimizes

    J = Σ_{j=1}^{K} Σ_{i∈C_j} ||x_i − μ_j||²

where C_j is the j-th cluster and μ_j its centroid.  The algorithm
alternates between assignment and update steps (Lloyd iteration):

    Assignment:  C_j = { i : ||x_i − μ_j|| ≤ ||x_i − μ_l|| ∀ l }
    Update:      μ_j = (1/|C_j|) Σ_{i∈C_j} x_i

H-Means (Hartigan-Wong) adds an optimal-transfer phase that moves a
point to another cluster if the decrease in J exceeds a threshold.
"""

import numpy as np
from typing import Tuple, List


def extract_stacking_features(
    positions: np.ndarray,
    layer_index: np.ndarray,
) -> np.ndarray:
    """
    Extract feature vectors for each atom based on its local stacking
    environment.

    Feature vector for atom i:
        f_i = [x_i, y_i, Δz_i, local_density_i]

    where local_density_i is the number of opposing-layer atoms within
    a cutoff radius.

    Parameters
    ----------
    positions : np.ndarray of shape (N, 3)
    layer_index : np.ndarray of shape (N,)

    Returns
    -------
    np.ndarray of shape (N, 4)
        Feature vectors.
    """
    N = positions.shape[0]
    features = np.zeros((N, 4))
    features[:, 0:3] = positions

    cutoff = 0.2  # nm
    for i in range(N):
        layer_i = layer_index[i]
        opposite_layer = 1 - layer_i
        mask = layer_index == opposite_layer
        count = 0
        for j in np.where(mask)[0]:
            dr = positions[i, :2] - positions[j, :2]
            dist = np.linalg.norm(dr)
            if dist < cutoff:
                count += 1
        features[i, 3] = float(count)

    # Normalize features
    for d in range(4):
        std = np.std(features[:, d])
        if std > 1e-10:
            features[:, d] = (features[:, d] - np.mean(features[:, d])) / std

    return features


def kmeans_lloyd(
    data: np.ndarray,
    n_clusters: int,
    max_iter: int = 100,
    tol: float = 1e-6,
    init: str = "k-means++",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    K-Means clustering using Lloyd's algorithm.

    Parameters
    ----------
    data : np.ndarray of shape (N, d)
    n_clusters : int
    max_iter : int
    tol : float
        Convergence tolerance on centroid displacement.
    init : str
        "k-means++" or "random".

    Returns
    -------
    labels : np.ndarray of shape (N,)
    centroids : np.ndarray of shape (n_clusters, d)
    inertia : float
        Final sum of squared distances.
    """
    N, d = data.shape
    if n_clusters < 1 or n_clusters > N:
        raise ValueError("Invalid number of clusters.")

    # Initialize centroids
    if init == "k-means++":
        centroids = np.zeros((n_clusters, d))
        centroids[0] = data[np.random.randint(N)]
        for k in range(1, n_clusters):
            dists = np.min(
                np.sum((data[:, None, :] - centroids[None, :k, :]) ** 2, axis=2),
                axis=1,
            )
            probs = dists / np.sum(dists)
            idx = np.random.choice(N, p=probs)
            centroids[k] = data[idx]
    else:
        indices = np.random.choice(N, size=n_clusters, replace=False)
        centroids = data[indices].copy()

    labels = np.zeros(N, dtype=int)

    for _ in range(max_iter):
        # Assignment step
        distances = np.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)

        # Update step
        new_centroids = np.zeros_like(centroids)
        for k in range(n_clusters):
            mask = labels == k
            if np.sum(mask) > 0:
                new_centroids[k] = np.mean(data[mask], axis=0)
            else:
                # Empty cluster: reinitialize to farthest point
                dists = np.sum((data - centroids[k]) ** 2, axis=1)
                new_centroids[k] = data[np.argmax(dists)]

        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift < tol:
            break

    inertia = np.sum(
        np.min(np.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=2), axis=1)
    )
    return labels, centroids, inertia


def hmeans_hartigan(
    data: np.ndarray,
    n_clusters: int,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    H-Means (Hartigan-Wong) clustering with optimal transfer.

    After each Lloyd update, scan all points and move any point i from
    cluster j to cluster l if

        |C_j|/(|C_j|−1) ||x_i − μ_j||²  >  |C_l|/(|C_l|+1) ||x_i − μ_l||² .

    Parameters
    ----------
    data : np.ndarray of shape (N, d)
    n_clusters : int
    max_iter : int
    tol : float

    Returns
    -------
    labels, centroids, inertia
    """
    labels, centroids, _ = kmeans_lloyd(data, n_clusters, max_iter=max_iter // 2, tol=tol)
    N, d = data.shape

    for _ in range(max_iter // 2):
        changed = False
        counts = np.array([np.sum(labels == k) for k in range(n_clusters)])
        for i in range(N):
            j = labels[i]
            if counts[j] <= 1:
                continue
            x = data[i]
            d_j = np.sum((x - centroids[j]) ** 2)
            best_gain = 0.0
            best_l = -1
            for l in range(n_clusters):
                if l == j:
                    continue
                d_l = np.sum((x - centroids[l]) ** 2)
                gain = (counts[j] / (counts[j] - 1.0)) * d_j - \
                       (counts[l] / (counts[l] + 1.0)) * d_l
                if gain > best_gain:
                    best_gain = gain
                    best_l = l
            if best_l >= 0 and best_gain > tol:
                labels[i] = best_l
                changed = True
                # Update centroids incrementally
                centroids[j] = (centroids[j] * counts[j] - x) / (counts[j] - 1.0)
                centroids[best_l] = (centroids[best_l] * counts[best_l] + x) / (counts[best_l] + 1.0)
                counts[j] -= 1
                counts[best_l] += 1
        if not changed:
            break

    inertia = np.sum(
        np.min(np.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=2), axis=1)
    )
    return labels, centroids, inertia


def classify_stacking_regions(
    positions: np.ndarray,
    layer_index: np.ndarray,
    n_clusters: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Classify atoms into moiré stacking regions using K-Means clustering
    on structural features.

    Parameters
    ----------
    positions : np.ndarray
    layer_index : np.ndarray
    n_clusters : int
        Typically 3 (AA, AB, BA).

    Returns
    -------
    labels : np.ndarray of shape (N,)
    centroids : np.ndarray of shape (n_clusters, 4)
    features : np.ndarray of shape (N, 4)
    """
    features = extract_stacking_features(positions, layer_index)
    labels, centroids, _ = kmeans_lloyd(features, n_clusters, init="k-means++")
    return labels, centroids, features


def cluster_energy_analysis(
    labels: np.ndarray,
    energies: np.ndarray,
    positions: np.ndarray,
) -> dict:
    """
    Compute energy statistics per cluster.

    Parameters
    ----------
    labels : np.ndarray of shape (N,)
    energies : np.ndarray of shape (N,)
        Local energy contribution per atom.
    positions : np.ndarray of shape (N, 3)

    Returns
    -------
    dict
        Dictionary with cluster-indexed statistics.
    """
    n_clusters = int(np.max(labels)) + 1
    stats = {}
    for k in range(n_clusters):
        mask = labels == k
        stats[k] = {
            "count": int(np.sum(mask)),
            "mean_energy": float(np.mean(energies[mask])) if np.sum(mask) > 0 else 0.0,
            "std_energy": float(np.std(energies[mask])) if np.sum(mask) > 0 else 0.0,
            "mean_x": float(np.mean(positions[mask, 0])) if np.sum(mask) > 0 else 0.0,
            "mean_y": float(np.mean(positions[mask, 1])) if np.sum(mask) > 0 else 0.0,
        }
    return stats
