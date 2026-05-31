"""
coverage_optimization.py
========================
Centroidal Voronoi Tessellation (CVT) for optimal swarm coverage.

Incorporates:
  - cvt_circle_nonuniform (from 253_cvt_circle_nonuniform)

Scientific role:
  CVT provides the energy-minimizing configuration of generators (robots)
  with respect to a density function rho(x). The CVT energy is
      E = sum_i integral_{V_i} rho(x) ||x - p_i||^2 dx
  where V_i is the Voronoi cell of generator p_i. Lloyd's algorithm
  iteratively moves each generator to the mass centroid of its cell,
  converging to a locally optimal coverage configuration.

  In swarm robotics, CVT is the canonical model for area coverage,
  surveillance, and environmental monitoring. Nonuniform density allows
  robots to concentrate in high-priority regions.
"""

import numpy as np


def cvt_lloyd_2d(generators: np.ndarray, density_func, bounds: tuple,
                 n_samples: int = 5000, n_iterations: int = 30):
    """
    Lloyd's algorithm for CVT in 2D with arbitrary density.

    Parameters
    ----------
    generators : ndarray, shape (N, 2)
        Initial generator positions.
    density_func : callable
        rho(x, y) -> float, must be non-negative.
    bounds : tuple
        (xmin, xmax, ymin, ymax).
    n_samples : int
        Number of Monte-Carlo samples per iteration.
    n_iterations : int
        Number of Lloyd iterations.

    Returns
    -------
    generators : ndarray, shape (N, 2)
        Optimized generator positions.
    energy_history : list
        CVT energy at each iteration.
    """
    generators = np.asarray(generators, dtype=float)
    N = generators.shape[0]
    xmin, xmax, ymin, ymax = bounds

    energy_history = []

    for it in range(n_iterations):
        # Monte-Carlo sample points in domain
        xs = np.random.uniform(xmin, xmax, size=n_samples)
        ys = np.random.uniform(ymin, ymax, size=n_samples)
        samples = np.column_stack((xs, ys))

        # evaluate density
        rhos = np.array([density_func(s[0], s[1]) for s in samples], dtype=float)
        rhos = np.clip(rhos, 0.0, None)

        # nearest generator for each sample
        diffs = samples[:, np.newaxis, :] - generators[np.newaxis, :, :]
        dists = np.sum(diffs ** 2, axis=2)
        nearest = np.argmin(dists, axis=1)

        # compute weighted centroids
        new_generators = np.zeros_like(generators)
        weights = np.zeros(N, dtype=float)
        for i in range(N):
            mask = nearest == i
            if np.any(mask):
                new_generators[i] = np.sum(samples[mask] * rhos[mask][:, np.newaxis], axis=0) / np.sum(rhos[mask])
                weights[i] = np.sum(rhos[mask])
            else:
                # empty cell: random reinitialization within bounds
                new_generators[i] = [np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax)]

        # compute energy
        min_dists = np.min(dists, axis=1)
        energy = np.sum(rhos * min_dists)
        energy_history.append(float(energy))

        generators = new_generators.copy()

    return generators, energy_history


def cvt_circle_nonuniform_density(n: int, radius: float = 1.0,
                                  n_iterations: int = 20, n_samples: int = None):
    """
    Compute CVT generators inside a circle with nonuniform density
    peaked at the center.

    Density: rho(r) = 1 + 5*exp(-5*r^2/R^2)

    Parameters
    ----------
    n : int
        Number of generators.
    radius : float
        Circle radius.
    n_iterations : int
        Lloyd iterations.
    n_samples : int or None
        Monte-Carlo samples.

    Returns
    -------
    generators : ndarray, shape (n, 2)
    energy_history : list
    """
    if n_samples is None:
        n_samples = 2000 * n

    # initial generators: rejection sampling in circle
    gens = []
    while len(gens) < n:
        candidates = np.random.uniform(-radius, radius, size=(2 * n, 2))
        mask = np.sum(candidates ** 2, axis=1) <= radius ** 2
        gens.extend(candidates[mask].tolist())
    generators = np.array(gens[:n], dtype=float)

    def density_func(x, y):
        r2 = (x ** 2 + y ** 2) / (radius ** 2)
        return 1.0 + 5.0 * np.exp(-5.0 * r2)

    bounds = (-radius, radius, -radius, radius)
    generators, energy_history = cvt_lloyd_2d(generators, density_func, bounds,
                                             n_samples=n_samples, n_iterations=n_iterations)
    # project back to circle
    dists = np.linalg.norm(generators, axis=1)
    mask = dists > radius
    generators[mask] *= (radius / dists[mask])[:, np.newaxis]
    return generators, energy_history


def coverage_metric(positions: np.ndarray, density_func, bounds: tuple, n_samples: int = 10000):
    """
    Compute the coverage cost (CVT energy) for a given set of positions.

    Parameters
    ----------
    positions : ndarray, shape (N, 2)
    density_func : callable
    bounds : tuple
        (xmin, xmax, ymin, ymax).
    n_samples : int

    Returns
    -------
    energy : float
    """
    xmin, xmax, ymin, ymax = bounds
    xs = np.random.uniform(xmin, xmax, size=n_samples)
    ys = np.random.uniform(ymin, ymax, size=n_samples)
    samples = np.column_stack((xs, ys))
    rhos = np.array([density_func(s[0], s[1]) for s in samples], dtype=float)
    rhos = np.clip(rhos, 0.0, None)
    diffs = samples[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sum(diffs ** 2, axis=2)
    min_dists = np.min(dists, axis=1)
    energy = float(np.sum(rhos * min_dists) / np.sum(rhos))
    return energy
