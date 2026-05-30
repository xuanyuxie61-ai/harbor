
import numpy as np
from typing import Tuple, List


def extract_stacking_features(
    positions: np.ndarray,
    layer_index: np.ndarray,
) -> np.ndarray:
    N = positions.shape[0]
    features = np.zeros((N, 4))
    features[:, 0:3] = positions

    cutoff = 0.2
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
    N, d = data.shape
    if n_clusters < 1 or n_clusters > N:
        raise ValueError("Invalid number of clusters.")


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

        distances = np.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)


        new_centroids = np.zeros_like(centroids)
        for k in range(n_clusters):
            mask = labels == k
            if np.sum(mask) > 0:
                new_centroids[k] = np.mean(data[mask], axis=0)
            else:

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
    features = extract_stacking_features(positions, layer_index)
    labels, centroids, _ = kmeans_lloyd(features, n_clusters, init="k-means++")
    return labels, centroids, features


def cluster_energy_analysis(
    labels: np.ndarray,
    energies: np.ndarray,
    positions: np.ndarray,
) -> dict:
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
