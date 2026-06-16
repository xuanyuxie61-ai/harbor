"""
mode_classifier.py
==================
Classification of CMB spherical-harmonic modes into:
  - E-modes  (gradient-like, parity-even)
  - B-modes  (curl-like, parity-odd)
  - Systematics  (noise / foreground residuals)

The classification uses agglomerative hierarchical clustering
(from 1026_JLefortBesnard_MCI_cluster_prediction clustering.py).

Features extracted per (l, m) mode:
  - |a_{lm}|^2  (power)
  - Re(a_{lm}) / |a_{lm}|  (phase)
  - Parity sign:  a_{l,-m} vs (-1)^l a_{lm}^*
  - Expected ratio to C_l theory

Agglomerative clustering uses Ward's linkage on the standardized
feature matrix.  The number of clusters K is chosen by the
gap statistic (or set to K=3 by default for {E, B, syst}).

Reference:
  Lefort-Besnard et al. (2023) clustering.py - MCI subgroup analysis
"""

from __future__ import annotations
import math
from typing import List, Tuple, Dict


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_mode_features(alm: Dict[Tuple[int, int], complex],
                            cl_theory: List[float]) -> List[List[float]]:
    """
    For each (l, m) mode, compute a 4D feature vector:
      f0 = |a_{lm}|^2
      f1 = Re(a_{lm}) / |a_{lm}|
      f2 = parity asymmetry  = |a_{l,-m} - (-1)^l a_{lm}^*| / |a_{lm}|
      f3 = power ratio       = |a_{lm}|^2 / C_l^theory
    """
    features = []
    for (l, m), a in alm.items():
        if l < 2:
            continue
        abs_a = abs(a)
        if abs_a < 1e-30:
            continue
        f0 = abs_a * abs_a
        f1 = a.real / abs_a
        # parity: for E-modes, a_{l,-m} = (-1)^m a_{lm}  (real)
        a_lm_conj = a.conjugate()
        sign = (-1) ** l
        a_parity = alm.get((l, -m), 0.0)
        f2 = abs(a_parity - sign * a_lm_conj) / abs_a
        # power ratio
        cl_th = cl_theory[l] if l < len(cl_theory) and cl_theory[l] > 1e-30 else 1e-30
        f3 = f0 / cl_th
        features.append([f0, f1, f2, f3])
    return features


# ---------------------------------------------------------------------------
# Standardisation (zero mean, unit variance)
# ---------------------------------------------------------------------------
def standardise(X: List[List[float]]) -> Tuple[List[List[float]], List[float], List[float]]:
    n = len(X)
    if n == 0:
        return X, [], []
    d = len(X[0])
    mean = [0.0] * d
    for x in X:
        for j in range(d):
            mean[j] += x[j]
    for j in range(d):
        mean[j] /= n
    var = [0.0] * d
    for x in X:
        for j in range(d):
            var[j] += (x[j] - mean[j]) ** 2
    for j in range(d):
        var[j] = math.sqrt(var[j] / max(1, n - 1))
        if var[j] < 1e-30:
            var[j] = 1.0
    X_std = [[(x[j] - mean[j]) / var[j] for j in range(d)] for x in X]
    return X_std, mean, var


# ---------------------------------------------------------------------------
# Euclidean distance
# ---------------------------------------------------------------------------
def euclidean(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


# ---------------------------------------------------------------------------
# Agglomerative clustering (Ward's linkage)
# ---------------------------------------------------------------------------
def agglomerative_ward(X: List[List[float]], n_clusters: int = 3) -> List[int]:
    """
    Agglomerative hierarchical clustering with Ward's variance criterion.
    Returns cluster labels 0..K-1 for each sample.
    """
    n = len(X)
    if n <= n_clusters:
        return list(range(n))
    d = len(X[0])

    # Initialise each point as its own cluster
    clusters = [[i] for i in range(n)]
    # Precompute centroids and sizes
    centroids = [X[i][:] for i in range(n)]
    sizes = [1] * n
    active = list(range(n))

    while len(active) > n_clusters:
        # Find pair (i, j) with minimum Ward increase
        best_d = float("inf")
        best_pair = (active[0], active[1])
        for ii in range(len(active)):
            for jj in range(ii + 1, len(active)):
                i = active[ii]
                j = active[jj]
                ni, nj = sizes[i], sizes[j]
                dd = euclidean(centroids[i], centroids[j])
                # Ward distance
                ward_d = (ni * nj) / (ni + nj) * dd * dd
                if ward_d < best_d:
                    best_d = ward_d
                    best_pair = (i, j)
        # Merge
        i, j = best_pair
        new_cluster = clusters[i] + clusters[j]
        ni, nj = sizes[i], sizes[j]
        new_centroid = [(centroids[i][k] * ni + centroids[j][k] * nj) / (ni + nj) for k in range(d)]
        # Update: replace i with merged, remove j
        clusters[i] = new_cluster
        centroids[i] = new_centroid
        sizes[i] = ni + nj
        clusters[j] = []
        sizes[j] = 0
        active.remove(j)

    # Build labels
    labels = [0] * n
    for idx, cl_id in enumerate(active):
        for sample_idx in clusters[cl_id]:
            labels[sample_idx] = active.index(cl_id)
    return labels


# ---------------------------------------------------------------------------
# Gap statistic for choosing K
# ---------------------------------------------------------------------------
def within_cluster_dispersion(X: List[List[float]], labels: List[int]) -> float:
    """Sum of pairwise squared distances within each cluster, divided by 2n_k."""
    K = max(labels) + 1 if labels else 0
    n = len(X)
    d = len(X[0]) if n > 0 else 0
    total = 0.0
    for k in range(K):
        members = [i for i in range(n) if labels[i] == k]
        nk = len(members)
        if nk <= 1:
            continue
        s = 0.0
        for i in members:
            for j in members:
                s += euclidean(X[i], X[j]) ** 2
        total += s / (2.0 * nk)
    return total


def gap_statistic(X: List[List[float]], k_max: int = 6,
                    n_references: int = 10, seed: int = 42) -> int:
    """
    Choose K by the gap statistic (Tibshirani et al. 2001):
      Gap(K) = E[log W_k^ref] - log W_k
    Choose K = smallest K such that Gap(K) >= Gap(K+1) - s_{K+1}.
    """
    import random
    rng = random.Random(seed)
    n = len(X)
    d = len(X[0]) if n > 0 else 0
    log_wk = []
    for K in range(1, k_max + 1):
        labels = agglomerative_ward(X, n_clusters=K)
        wk = within_cluster_dispersion(X, labels)
        log_wk.append(math.log(max(1e-30, wk)))
    # Reference datasets (uniform in bounding box)
    mins = [min(X[i][j] for i in range(n)) for j in range(d)] if n > 0 else []
    maxs = [max(X[i][j] for i in range(n)) for j in range(d)] if n > 0 else []
    ref_log_wk = [[0.0] * k_max for _ in range(n_references)]
    for r in range(n_references):
        X_ref = [[mins[j] + (maxs[j] - mins[j]) * rng.random() for j in range(d)] for _ in range(n)]
        for K in range(1, k_max + 1):
            labels = agglomerative_ward(X_ref, n_clusters=K)
            wk = within_cluster_dispersion(X_ref, labels)
            ref_log_wk[r][K - 1] = math.log(max(1e-30, wk))
    gap = [0.0] * k_max
    for K in range(k_max):
        mean_ref = sum(ref_log_wk[r][K] for r in range(n_references)) / n_references
        gap[K] = mean_ref - log_wk[K]
    # Choose first K with gap[K] >= gap[K+1] - s_{K+1}
    best_k = 1
    for K in range(k_max - 1):
        s_k1 = math.sqrt(
            sum((ref_log_wk[r][K + 1] - sum(ref_log_wk[r2][K + 1] for r2 in range(n_references)) / n_references) ** 2
                for r in range(n_references)) / max(1, n_references - 1)
        ) * math.sqrt(1.0 + 1.0 / n_references)
        if gap[K] >= gap[K + 1] - s_k1:
            best_k = K + 1
            break
    return best_k


# ---------------------------------------------------------------------------
# Classify E/B/systematics by cluster properties
# ---------------------------------------------------------------------------
def classify_clusters(X: List[List[float]], labels: List[int],
                        alm_keys: List[Tuple[int, int]]) -> Dict[str, List[Tuple[int, int]]]:
    """
    Assign each cluster one of {E-mode, B-mode, systematic} based on
    mean parity asymmetry (feature 2):
      low parity   -> E-mode
      high parity  -> B-mode
      extreme power ratio -> systematic
    """
    K = max(labels) + 1 if labels else 0
    # Compute mean features per cluster
    cluster_means = []
    for k in range(K):
        members = [i for i in range(len(labels)) if labels[i] == k]
        if not members:
            cluster_means.append([0.0] * len(X[0]))
            continue
        mean = [0.0] * len(X[0])
        for i in members:
            for j in range(len(X[0])):
                mean[j] += X[i][j]
        for j in range(len(X[0])):
            mean[j] /= len(members)
        cluster_means.append(mean)
    # Sort by parity (feature 2)
    order = sorted(range(K), key=lambda k: cluster_means[k][2])
    # Assign: lowest parity -> E, middle -> B, highest -> systematic
    e_modes = []
    b_modes = []
    syst = []
    for k in range(K):
        members = [alm_keys[i] for i in range(len(labels)) if labels[i] == k]
        if k == order[0]:
            e_modes.extend(members)
        elif k == order[-1] and K >= 3:
            syst.extend(members)
        else:
            b_modes.extend(members)
    return {"E": e_modes, "B": b_modes, "systematic": syst}


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random
    rng = random.Random(42)
    # Synthetic 3-cluster data
    X = []
    for _ in range(30):
        X.append([rng.gauss(1, 0.2), 0.1, 0.2, 1.0])   # E-like
    for _ in range(30):
        X.append([rng.gauss(1, 0.2), 0.5, 0.8, 1.0])   # B-like
    for _ in range(30):
        X.append([rng.gauss(5, 0.5), 0.0, 3.0, 10.0])  # systematic
    X_std, _, _ = standardise(X)
    K = gap_statistic(X_std, k_max=5)
    print(f"Optimal K = {K}")
    labels = agglomerative_ward(X_std, n_clusters=3)
    print(f"Cluster counts: { [labels.count(k) for k in range(3)] }")
