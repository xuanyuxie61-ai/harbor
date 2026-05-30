# -*- coding: utf-8 -*-

import numpy as np
from special_functions import fraunhofer_diffraction_particle_size


def kmeans_1d(data, k, max_iter=100, tol=1e-6, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    data = np.asarray(data, dtype=float)
    n = data.size
    if n == 0 or k <= 0:
        return np.array([]), np.array([], dtype=int), 0.0
    k = min(k, n)


    indices = rng.choice(n, size=k, replace=False)
    centers = data[indices].copy()

    for _ in range(max_iter):

        distances = np.abs(data[:, np.newaxis] - centers)
        labels = np.argmin(distances, axis=1)


        new_centers = np.zeros_like(centers)
        for i in range(k):
            mask = labels == i
            if np.any(mask):
                new_centers[i] = np.mean(data[mask])
            else:

                new_centers[i] = data[rng.integers(n)]


        shift = np.max(np.abs(new_centers - centers))
        centers = new_centers
        if shift < tol:
            break


    inertia = 0.0
    for i in range(k):
        mask = labels == i
        if np.any(mask):
            inertia += np.sum((data[mask] - centers[i]) ** 2)

    return centers, labels, inertia


def discretize_csd_kmeans(L_grid, f_values, k_classes):
    L_grid = np.asarray(L_grid, dtype=float)
    f_values = np.asarray(f_values, dtype=float)


    weights = np.maximum(f_values, 0.0)
    total_weight = np.trapezoid(weights, L_grid)
    if total_weight <= 0:
        return np.zeros(k_classes), np.zeros(k_classes), np.zeros(k_classes + 1)


    rng = np.random.default_rng(42)
    n_samples = 10000

    cum_weights = np.cumsum(weights)
    cum_weights /= cum_weights[-1]
    u = rng.random(n_samples)
    sampled_indices = np.searchsorted(cum_weights, u)
    sampled_L = L_grid[sampled_indices]


    centers, labels, _ = kmeans_1d(sampled_L, k_classes, rng=rng)
    centers = np.sort(centers)


    class_counts = np.zeros(k_classes)
    boundaries = np.zeros(k_classes + 1)
    boundaries[0] = L_grid[0]
    boundaries[-1] = L_grid[-1]


    for i in range(1, k_classes):
        boundaries[i] = 0.5 * (centers[i - 1] + centers[i])

    for i in range(k_classes):
        mask = (L_grid >= boundaries[i]) & (L_grid < boundaries[i + 1])
        if i == k_classes - 1:
            mask = (L_grid >= boundaries[i]) & (L_grid <= boundaries[i + 1])
        if np.any(mask):
            class_counts[i] = np.trapezoid(f_values[mask], L_grid[mask])

    return centers, class_counts, boundaries


def diffraction_inversion_feret(theta, intensity, wavelength,
                                 L_min=1e-6, L_max=1000e-6, n_bins=100):
    from scipy.optimize import nnls

    theta = np.asarray(theta, dtype=float)
    intensity = np.asarray(intensity, dtype=float)

    L_bins = np.linspace(L_min, L_max, n_bins)
    dL = L_bins[1] - L_bins[0]


    A = np.zeros((len(theta), n_bins), dtype=float)
    for j, L in enumerate(L_bins):
        A[:, j] = fraunhofer_diffraction_particle_size(L / 2.0, wavelength, theta)


    n_L, residual = nnls(A, intensity)
    n_L = n_L / dL

    return L_bins, n_L


def csd_statistical_moments(L_grid, f_values):
    L_grid = np.asarray(L_grid, dtype=float)
    f_values = np.asarray(f_values, dtype=float)


    total = np.trapezoid(f_values, L_grid)
    if total <= 0:
        return {'mean': 0.0, 'std': 0.0, 'skewness': 0.0, 'kurtosis': 0.0}

    pdf = f_values / total

    mean_L = np.trapezoid(L_grid * pdf, L_grid)
    var_L = np.trapezoid((L_grid - mean_L) ** 2 * pdf, L_grid)
    std_L = np.sqrt(max(var_L, 0.0))

    if std_L < 1e-30:
        return {'mean': mean_L, 'std': 0.0, 'skewness': 0.0, 'kurtosis': 0.0}

    skewness = np.trapezoid(((L_grid - mean_L) / std_L) ** 3 * pdf, L_grid)
    kurtosis = np.trapezoid(((L_grid - mean_L) / std_L) ** 4 * pdf, L_grid)

    return {
        'mean': mean_L,
        'std': std_L,
        'skewness': skewness,
        'kurtosis': kurtosis
    }
