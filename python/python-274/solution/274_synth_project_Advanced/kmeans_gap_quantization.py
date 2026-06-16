# -*- coding: utf-8 -*-
"""
kmeans_gap_quantization.py
--------------------------
Vector-quantisation (k-means clustering) of the Eliashberg gap function
Delta(omega_n) on the Matsubara axis.  The continuous gap profile is
compressed into K representative "gap levels" which are then used to
construct a reduced-rank approximation of the Eliashberg kernel.

Scientific origin of the fused algorithms
-----------------------------------------
* Grayscale image quantisation   (seed project 583_image_quantization)
    -> k-means clustering of pixel intensities into K shades.
    -> adapted: cluster *gap values* on the Matsubara grid into K levels,
       effectively discretising the gap function.

Core physics / mathematics
--------------------------
* The isotropic Eliashberg equations are a nonlinear integral equation
      Delta_n = F[Delta](omega_n).
  After k-means quantisation with K centres c_1, ..., c_K each Matsubara
  frequency n is assigned to the closest centre, yielding a *block structure*
  in the kernel:
      Delta_n approx c_{q(n)}   where  q(n) = argmin_j |Delta_n - c_j|.
* The quantisation error
      E_Q = sum_n  |Delta_n - c_{q(n)}|^2
  bounds the residual of the reduced-rank Eliashberg equation.
* Physically this corresponds to grouping Matsubara frequencies into
  "energy shells" within which the gap is approximately constant, a common
  trick in multi-gap superconductors (e.g. MgB_2).

Stability / boundary notes
--------------------------
* K-means is initialised with k-means++ to avoid degenerate centres.
* Empty clusters are re-seeded from the point farthest from any centre.
* The algorithm stops when centre displacement < tol or max_iter is reached.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1.  k-means++ initialisation
# ---------------------------------------------------------------------------
def _kmeans_pp_init(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ initialisation of k centres from the data x (N, d)."""
    n = x.shape[0]
    centres = np.empty((k, x.shape[1]), dtype=x.dtype)
    idx = rng.integers(0, n)
    centres[0] = x[idx]
    for j in range(1, k):
        # Squared distance of each point to its closest centre so far
        d2 = np.array([
            min(((x[i] - centres[m]) ** 2).sum() for m in range(j))
            for i in range(n)
        ])
        d2_sum = d2.sum()
        if d2_sum <= 0.0:
            # All remaining points coincide with existing centres
            centres[j] = x[rng.integers(0, n)]
            continue
        probs = d2 / d2_sum
        idx = rng.choice(n, p=probs)
        centres[j] = x[idx]
    return centres


# ---------------------------------------------------------------------------
# 2.  k-means clustering (Lloyd's algorithm)
# ---------------------------------------------------------------------------
@dataclass
class KMeansResult:
    centres: np.ndarray          # (K, d)
    labels: np.ndarray           # (N,)  cluster index for each point
    inertia: float               # sum of squared distances to nearest centre
    n_iter: int


def kmeans(
    x: np.ndarray,
    k: int,
    max_iter: int = 200,
    tol: float = 1e-10,
    seed: int = 274,
) -> KMeansResult:
    """Cluster the (N, d) data x into k clusters using Lloyd's algorithm."""
    if k < 1:
        raise ValueError("kmeans: k must be >= 1")
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n, d = x.shape
    rng = np.random.default_rng(seed)

    centres = _kmeans_pp_init(x, k, rng)
    labels = np.zeros(n, dtype=int)

    for it in range(max_iter):
        # Assign each point to its nearest centre
        dists = np.array([((x - centres[j]) ** 2).sum(axis=1) for j in range(k)]).T
        labels = dists.argmin(axis=1)

        # Update centres
        new_centres = np.zeros_like(centres)
        for j in range(k):
            mask = labels == j
            if mask.sum() == 0:
                # Empty cluster: re-seed from the farthest point
                min_dists = dists.min(axis=1)
                farthest = int(min_dists.argmax())
                new_centres[j] = x[farthest]
            else:
                new_centres[j] = x[mask].mean(axis=0)

        shift = float(((new_centres - centres) ** 2).sum())
        centres = new_centres
        if shift < tol:
            break

    # Final inertia
    inertia = float(((x - centres[labels]) ** 2).sum())
    return KMeansResult(
        centres=centres, labels=labels, inertia=inertia, n_iter=it + 1
    )


# ---------------------------------------------------------------------------
# 3.  Gap quantisation driver
# ---------------------------------------------------------------------------
@dataclass
class GapQuantization:
    """Result of quantising a gap function Delta(omega_n) into K levels."""
    omega_n: np.ndarray       # (M,) Matsubara frequencies
    Delta_original: np.ndarray    # (M,) original gap
    centres: np.ndarray       # (K,) gap levels
    labels: np.ndarray        # (M,) cluster assignments
    Delta_quantized: np.ndarray   # (M,) gap after quantisation
    quantisation_error: float
    compression_ratio: float  # M / K


def quantise_gap(
    omega_n: np.ndarray,
    Delta: np.ndarray,
    n_levels: int = 8,
    seed: int = 274,
) -> GapQuantization:
    """Quantise the gap function Delta on the Matsubara grid into n_levels
    discrete gap values using k-means clustering.
    """
    if n_levels < 1:
        raise ValueError("quantise_gap: n_levels must be >= 1")
    M = omega_n.size
    if Delta.size != M:
        raise ValueError("quantise_gap: omega_n and Delta must have the same length")

    # Normalise gap to [-1, 1] for stable clustering
    dmax = max(abs(Delta.max()), abs(Delta.min()), 1e-300)
    Delta_norm = Delta / dmax

    res = kmeans(Delta_norm.reshape(-1, 1), n_levels, seed=seed)
    centres_norm = res.centres[:, 0]
    Delta_q_norm = centres_norm[res.labels]

    # Denormalise
    centres = centres_norm * dmax
    Delta_q = Delta_q_norm * dmax
    qerr = float(((Delta - Delta_q) ** 2).sum())
    return GapQuantization(
        omega_n=omega_n,
        Delta_original=Delta,
        centres=centres,
        labels=res.labels,
        Delta_quantized=Delta_q,
        quantisation_error=qerr,
        compression_ratio=float(M) / n_levels,
    )


# ---------------------------------------------------------------------------
# 4.  Reduced-rank Eliashberg kernel from quantised gap
# ---------------------------------------------------------------------------
def reduced_kernel_from_quantization(
    K_full: np.ndarray,
    gq: GapQuantization,
) -> np.ndarray:
    """Return the block-averaged kernel K_red of shape (K_levels, K_levels)
    obtained by averaging K_full over the Matsubara blocks defined by the
    gap quantisation.
    """
    K = int(gq.centres.size)
    K_red = np.zeros((K, K))
    for a in range(K):
        mask_a = gq.labels == a
        for b in range(K):
            mask_b = gq.labels == b
            block = K_full[np.ix_(mask_a, mask_b)]
            if block.size > 0:
                K_red[a, b] = block.mean()
    return K_red
