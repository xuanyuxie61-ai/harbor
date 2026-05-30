#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple, List


def svd_low_rank_approximation(
    data_matrix: np.ndarray,
    rank: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    m, n = data_matrix.shape
    rank = max(1, min(rank, min(m, n)))

    u, s, vh = np.linalg.svd(data_matrix, full_matrices=False)

    u_r = u[:, :rank]
    s_r = s[:rank]
    vh_r = vh[:rank, :]


    approx = u_r @ np.diag(s_r) @ vh_r

    return u_r, s_r, vh_r, approx


def singular_value_entropy(singular_values: np.ndarray) -> float:
    s = np.asarray(singular_values)
    s = s[s > 1e-15]
    if s.size == 0:
        return 0.0

    p = s ** 2
    p_sum = np.sum(p)
    if p_sum < 1e-15:
        return 0.0

    p = p / p_sum

    p = p[p > 1e-15]

    entropy = -np.sum(p * np.log(p))
    max_entropy = np.log(s.size)

    if max_entropy < 1e-15:
        return 0.0

    return float(entropy / max_entropy)


def signal_background_discriminator(
    hit_maps: List[np.ndarray],
    labels: np.ndarray,
    n_components: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    n_events = len(hit_maps)
    if n_events == 0:
        raise ValueError("没有输入事例")


    pixel_size = hit_maps[0].size
    X = np.zeros((n_events, pixel_size))
    for i, hm in enumerate(hit_maps):
        X[i, :] = hm.ravel()


    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean


    _, _, vh = np.linalg.svd(X_centered, full_matrices=False)
    basis = vh[:n_components, :]


    projections = X_centered @ basis.T
    scores = np.linalg.norm(projections, axis=1)


    mean_sig = np.mean(scores[labels == 1]) if np.any(labels == 1) else 0.0
    mean_bkg = np.mean(scores[labels == 0]) if np.any(labels == 0) else 1.0
    if mean_sig < mean_bkg:
        scores = -scores

    return basis, scores


def pca_denoise(
    data_matrix: np.ndarray,
    variance_threshold: float = 0.95
) -> np.ndarray:
    u, s, vh = np.linalg.svd(data_matrix, full_matrices=False)

    total_var = np.sum(s ** 2)
    if total_var < 1e-15:
        return data_matrix

    cumvar = np.cumsum(s ** 2) / total_var
    k = np.searchsorted(cumvar, variance_threshold) + 1
    k = min(k, s.size)

    u_k = u[:, :k]
    s_k = s[:k]
    vh_k = vh[:k, :]

    return u_k @ np.diag(s_k) @ vh_k


def resonance_peak_finder(
    invariant_mass: np.ndarray,
    counts: np.ndarray,
    window_width: float = 10.0
) -> Tuple[float, float, float]:
    n = invariant_mass.size
    if n < 3:
        return 0.0, 0.0, 0.0

    best_sig = 0.0
    best_mass = invariant_mass[n // 2]
    best_height = 0.0

    for i in range(n):
        m_central = invariant_mass[i]

        in_window = np.abs(invariant_mass - m_central) <= window_width

        in_sideband = (np.abs(invariant_mass - m_central) > window_width) & \
                      (np.abs(invariant_mass - m_central) <= 2.0 * window_width)

        n_window = np.sum(counts[in_window])
        n_side = np.sum(counts[in_sideband])
        n_side_bins = np.sum(in_sideband)

        if n_side_bins > 0 and n_side > 0:

            bkg_est = n_side * np.sum(in_window) / n_side_bins
            excess = max(n_window - bkg_est, 0.0)
            sig = excess / np.sqrt(max(bkg_est, 1.0))

            if sig > best_sig:
                best_sig = sig
                best_mass = m_central
                best_height = excess

    return best_mass, best_height, best_sig
