# -*- coding: utf-8 -*-
"""
burning_classifier.py
=====================
Classification of stellar burning regimes by clustering the local
thermodynamic state (T, rho, composition) using K-means.

Background
----------
A stellar model has many mass zones, each with its own (T, rho, Y_i)
state.  Different zones undergo different burning regimes:

  - quiescent (no burning, T < 5e6 K)
  - hydrogen burning (pp-chain or CNO)
  - helium burning (triple-alpha)
  - carbon / neon / oxygen burning
  - silicon burning (nuclear statistical equilibrium)

Rather than hard-threshold the classification, we use *K-means
clustering* in the 4D feature space

    x = (log10 T, log10 rho, Y_H, Y_He)

with K = 5 classes.  This mirrors the image quantization via k-means
in seed 583_image_quantization, but applied to stellar state space
instead of greyscale pixels.

The K-means algorithm follows Lloyd's iteration:
  1. Initialise K centres by random selection from the data.
  2. Assign each point to its nearest centre (Euclidean distance).
  3. Recompute centres as the mean of their assigned points.
  4. Repeat until convergence or max_iter.

We also provide a *pipeline structure* inspired by seed 1055 (DeepHP):
prepare_data -> cluster -> evaluate -> report, following the same
modular organisation.
"""

from __future__ import annotations
from typing import List, Tuple, Dict
import math
import random


# =====================================================================
# Feature extraction
# =====================================================================

def extract_features(T: List[float], rho: List[float],
                      Y_H: List[float], Y_He: List[float]
                      ) -> List[List[float]]:
    """Build the feature matrix for K-means.

    Feature vector for each mass shell:
        x = (log10 T, log10 rho, Y_H, Y_He)
    with floor values to avoid log(0).
    """
    N = len(T)
    X = []
    for i in range(N):
        lT  = math.log10(max(T[i],  1.0))
        lrh = math.log10(max(rho[i], 1.0e-30))
        x = [lT, lrh, max(0.0, Y_H[i]), max(0.0, Y_He[i])]
        X.append(x)
    return X


# =====================================================================
# K-means clustering (from seed 583)
# =====================================================================

def kmeans(X: List[List[float]], K: int, max_iter: int = 100,
           seed: int = 42) -> Tuple[List[int], List[List[float]], int]:
    """Run K-means clustering on the feature matrix X.

    Parameters
    ----------
    X : N x D feature matrix.
    K : number of clusters.
    max_iter : maximum iterations.
    seed : random seed.

    Returns
    -------
    labels : list of N ints in [0, K-1].
    centres : list of K centres (each a D-vector).
    n_iter : number of iterations run.
    """
    if not X:
        return [], [], 0
    N = len(X)
    D = len(X[0])
    rng = random.Random(seed)

    # Initialise centres by random selection
    idx = rng.sample(range(N), min(K, N))
    centres = [X[i][:] for i in idx]
    while len(centres) < K:
        centres.append(centres[-1][:])
    labels = [0] * N

    for it in range(max_iter):
        # Assignment step: nearest centre (Euclidean)
        new_labels = [0] * N
        for i in range(N):
            best_k = 0
            best_d = float("inf")
            for k in range(K):
                d = sum((X[i][d] - centres[k][d])**2 for d in range(D))
                if d < best_d:
                    best_d = d
                    best_k = k
            new_labels[i] = best_k
        # Check convergence
        if new_labels == labels and it > 0:
            labels = new_labels
            break
        labels = new_labels
        # Update step: recompute centres
        sums = [[0.0]*D for _ in range(K)]
        counts = [0] * K
        for i in range(N):
            k = labels[i]
            counts[k] += 1
            for d in range(D):
                sums[k][d] += X[i][d]
        for k in range(K):
            if counts[k] > 0:
                centres[k] = [sums[k][d] / counts[k] for d in range(D)]
    return labels, centres, it + 1


# =====================================================================
# Burning regime labelling
# =================================================================-----

REGIME_NAMES = [
    "quiescent",        # T < 5e6 K
    "H_burning",        # H-rich, T ~ 1-3e7 K
    "He_burning",       # He-rich, T ~ 1-3e8 K
    "advanced_burning", # T > 5e8 K
    "NSE",              # T > 3e9 K, nuclear statistical equilibrium
]


def label_regime(centre: List[float]) -> str:
    """Map a cluster centre to a physical burning regime label."""
    lT, lrh, Y_H, Y_He = centre
    T = 10.0 ** lT
    if T < 5.0e6:
        return "quiescent"
    if T >= 3.0e9:
        return "NSE"
    if Y_H > 0.3 and T >= 5.0e6:
        return "H_burning"
    if Y_He > 0.3 and T >= 1.0e8:
        return "He_burning"
    return "advanced_burning"


# =====================================================================
# Pipeline (following seed 1055 DeepHP structure)
# =====================================================================

def classify_burning_regimes(T: List[float], rho: List[float],
                              Y_H: List[float], Y_He: List[float],
                              K: int = 5, seed: int = 42
                              ) -> Dict[str, object]:
    """Full classification pipeline.

    Returns dict with
      labels : per-shell regime index
      centres : K cluster centres in feature space
      names : regime name per cluster
      counts : number of shells per class
    """
    X = extract_features(T, rho, Y_H, Y_He)
    labels, centres, n_iter = kmeans(X, K=K, seed=seed)
    names = [label_regime(c) for c in centres]
    counts = [0] * K
    for lab in labels:
        if 0 <= lab < K:
            counts[lab] += 1
    return {
        "labels": labels,
        "centres": centres,
        "names": names,
        "counts": counts,
        "n_iter": n_iter,
    }


# =====================================================================
# Diagnostic
# ======================================================================

def _self_test():
    print("burning_classifier self-test:")
    # Synthetic data: 30 shells spanning quiescent to H burning
    T   = [1.0e6] * 10 + [1.5e7] * 10 + [2.0e8] * 10
    rho = [1.0]   * 10 + [150.0]* 10 + [1.0e4]* 10
    Y_H = [0.7]   * 10 + [0.35] * 10 + [0.0]  * 10
    Y_He= [0.28]  * 10 + [0.63] * 10 + [0.98] * 10
    res = classify_burning_regimes(T, rho, Y_H, Y_He, K=3, seed=1)
    print(f"  K-means converged in {res['n_iter']} iterations")
    for k, (nm, cnt) in enumerate(zip(res["names"], res["counts"])):
        print(f"    class {k}: {nm}  ({cnt} shells)")
    print("burning_classifier self-test OK")


if __name__ == "__main__":
    _self_test()
