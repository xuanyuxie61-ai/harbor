# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import norm


def _sample_density_square(density_func, n_samples, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0)):
    samples = []
    max_attempts = n_samples * 20
    attempts = 0

    xs = np.linspace(xlim[0], xlim[1], 50)
    ys = np.linspace(ylim[0], ylim[1], 50)
    max_rho = 0.0
    for xx in xs:
        for yy in ys:
            val = density_func(xx, yy)
            if val > max_rho:
                max_rho = val
    if max_rho <= 0.0:
        max_rho = 1.0
    envelope = max_rho * 1.5

    while len(samples) < n_samples and attempts < max_attempts:
        x = np.random.uniform(xlim[0], xlim[1])
        y = np.random.uniform(ylim[0], ylim[1])
        u = np.random.uniform(0.0, envelope)
        if u <= density_func(x, y):
            samples.append([x, y])
        attempts += 1

    if len(samples) < n_samples:

        n_extra = n_samples - len(samples)
        x_extra = np.random.uniform(xlim[0], xlim[1], n_extra)
        y_extra = np.random.uniform(ylim[0], ylim[1], n_extra)
        for i in range(n_extra):
            samples.append([x_extra[i], y_extra[i]])

    return np.array(samples, dtype=float)


def _voronoi_centroids_square(generators, samples):
    n = generators.shape[0]
    centroids = np.zeros_like(generators)
    counts = np.zeros(n)
    for s in samples:
        dists = np.sum((generators - s) ** 2, axis=1)
        idx = int(np.argmin(dists))
        centroids[idx] += s
        counts[idx] += 1.0

    for i in range(n):
        if counts[i] > 0:
            centroids[i] /= counts[i]
        else:

            centroids[i] = generators[i].copy()
    return centroids


def _voronoi_centroids_weighted(generators, samples, weights):
    n = generators.shape[0]
    centroids = np.zeros_like(generators)
    weight_sums = np.zeros(n)
    for s, w in zip(samples, weights):
        dists = np.sum((generators - s) ** 2, axis=1)
        idx = int(np.argmin(dists))
        centroids[idx] += w * s
        weight_sums[idx] += w

    for i in range(n):
        if weight_sums[i] > 1e-12:
            centroids[i] /= weight_sums[i]
        else:
            centroids[i] = generators[i].copy()
    return centroids


def cvt_step_square(generators, density_func=None, n_samples=5000,
                    xlim=(-1.0, 1.0), ylim=(-1.0, 1.0)):
    generators = np.asarray(generators, dtype=float)
    if density_func is None:
        samples = np.random.uniform(
            [xlim[0], ylim[0]], [xlim[1], ylim[1]], size=(n_samples, 2)
        )
        centroids = _voronoi_centroids_square(generators, samples)
    else:
        samples = _sample_density_square(density_func, n_samples, xlim, ylim)
        weights = np.array([density_func(s[0], s[1]) for s in samples])

        wmax = np.max(weights)
        if wmax > 0:
            weights = weights / wmax
        else:
            weights = np.ones_like(weights)
        centroids = _voronoi_centroids_weighted(generators, samples, weights)
    return centroids


def lloyd_relaxation_square(n_generators, n_steps=15, density_func=None,
                            n_samples=8000, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0),
                            seed=42):
    np.random.seed(seed)

    generators = np.random.uniform(
        [xlim[0], ylim[0]], [xlim[1], ylim[1]], size=(n_generators, 2)
    )
    for step in range(n_steps):
        generators = cvt_step_square(generators, density_func, n_samples, xlim, ylim)
    return generators


def _sample_uniform_sphere(n):
    points = np.random.normal(size=(n, 3))
    norms = norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return points / norms


def _voronoi_centroids_sphere(generators, samples):
    n = generators.shape[0]
    centroids = np.zeros_like(generators)
    counts = np.zeros(n)
    for s in samples:

        dists = np.sum((generators - s) ** 2, axis=1)
        idx = int(np.argmin(dists))
        centroids[idx] += s
        counts[idx] += 1.0

    for i in range(n):
        if counts[i] > 0:
            centroids[i] /= counts[i]

            r = norm(centroids[i])
            if r > 1e-12:
                centroids[i] /= r
            else:
                centroids[i] = generators[i].copy()
        else:
            centroids[i] = generators[i].copy()
    return centroids


def cvt_step_sphere(generators, n_samples=5000):
    generators = np.asarray(generators, dtype=float)
    samples = _sample_uniform_sphere(n_samples)
    centroids = _voronoi_centroids_sphere(generators, samples)
    return centroids


def lloyd_relaxation_sphere(n_generators, n_steps=15, n_samples=8000, seed=42):
    np.random.seed(seed)
    generators = _sample_uniform_sphere(n_generators)
    for step in range(n_steps):
        generators = cvt_step_sphere(generators, n_samples)
    return generators


def compute_voronoi_areas_square(generators, n_samples=20000,
                                 xlim=(-1.0, 1.0), ylim=(-1.0, 1.0)):
    n = generators.shape[0]
    areas = np.zeros(n)
    total_area = (xlim[1] - xlim[0]) * (ylim[1] - ylim[0])
    samples = np.random.uniform(
        [xlim[0], ylim[0]], [xlim[1], ylim[1]], size=(n_samples, 2)
    )
    for s in samples:
        dists = np.sum((generators - s) ** 2, axis=1)
        idx = int(np.argmin(dists))
        areas[idx] += 1.0
    areas = areas / n_samples * total_area
    return areas


def compute_voronoi_areas_sphere(generators, n_samples=20000):
    n = generators.shape[0]
    counts = np.zeros(n)
    samples = _sample_uniform_sphere(n_samples)
    for s in samples:
        dists = np.sum((generators - s) ** 2, axis=1)
        idx = int(np.argmin(dists))
        counts[idx] += 1.0
    areas = counts / n_samples * 4.0 * np.pi
    return areas
