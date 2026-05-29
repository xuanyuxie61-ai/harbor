"""
Centroidal Voronoi Tessellation (CVT) Sampling Module
======================================================
Based on project 238_cvt.

Implements CVT-based optimal sampling in stochastic parameter space
for variance reduction in Monte Carlo integration of structural
response statistics.

Key concepts:
- CVT energy: E = sum_i integral_{V_i} ||xi - z_i||^2 rho(xi) dxi
- Lloyd iteration: z_i^{new} = centroid(V_i)
- Optimal sampling: CVT generators provide low-discrepancy, space-filling
  designs with density adaptation.
"""

import numpy as np


def cvt_energy(generators, samples):
    """
    Compute discrete CVT energy for given generators and samples.
    
    E = (1/M) sum_{j=1}^M min_i ||s_j - g_i||^2
    
    Parameters
    ----------
    generators : ndarray, shape (n_gen, dim)
        CVT generator points.
    samples : ndarray, shape (n_samples, dim)
        Sample points.
    
    Returns
    -------
    energy : float
        CVT energy.
    assignments : ndarray
        Index of closest generator for each sample.
    """
    # Compute pairwise distances efficiently
    # dists[i,j] = ||samples[i] - generators[j]||^2
    diff = samples[:, np.newaxis, :] - generators[np.newaxis, :, :]
    dists = np.sum(diff ** 2, axis=2)
    assignments = np.argmin(dists, axis=1)
    min_dists = dists[np.arange(len(samples)), assignments]
    energy = np.mean(min_dists)
    return energy, assignments


def cvt_iterate(generators, samples, density=None):
    """
    Perform one Lloyd iteration of CVT.
    
    New generators are centroids of their Voronoi regions:
    g_i^{new} = (sum_{s in V_i} w(s) * s) / (sum_{s in V_i} w(s))
    
    Parameters
    ----------
    generators : ndarray, shape (n_gen, dim)
    samples : ndarray, shape (n_samples, dim)
    density : ndarray, optional
        Sample weights for non-uniform density.
    
    Returns
    -------
    new_generators : ndarray
    diff : float
        L2 norm of generator displacement.
    """
    n_gen = generators.shape[0]
    energy, assignments = cvt_energy(generators, samples)
    
    new_generators = np.zeros_like(generators)
    counts = np.zeros(n_gen)
    
    if density is None:
        density = np.ones(len(samples))
    
    for i in range(n_gen):
        mask = (assignments == i)
        if np.any(mask):
            weights = density[mask]
            new_generators[i] = np.sum(weights[:, np.newaxis] * samples[mask], axis=0) / np.sum(weights)
            counts[i] = np.sum(mask)
        else:
            # Empty region: keep old generator or reinitialize
            new_generators[i] = generators[i]
    
    diff = np.linalg.norm(new_generators - generators)
    return new_generators, diff, energy


def generate_cvt_samples(dim, n_gen, n_samples=10000, it_max=50, 
                         bounds=None, seed=None):
    """
    Generate CVT generator points in a hypercube.
    
    Parameters
    ----------
    dim : int
        Dimension of sampling space.
    n_gen : int
        Number of generators (sampling points).
    n_samples : int
        Number of Monte Carlo samples per iteration.
    it_max : int
        Maximum Lloyd iterations.
    bounds : list of tuples, optional
        [(low_1, high_1), ..., (low_dim, high_dim)].
    seed : int, optional
    
    Returns
    -------
    generators : ndarray, shape (n_gen, dim)
        Optimal CVT sampling points.
    """
    if seed is not None:
        np.random.seed(seed)
    
    if bounds is None:
        bounds = [(-1.0, 1.0)] * dim
    bounds = np.asarray(bounds)
    
    # Initialize generators uniformly
    generators = np.random.rand(n_gen, dim)
    generators = bounds[:, 0] + generators * (bounds[:, 1] - bounds[:, 0])
    
    for it in range(it_max):
        # Generate samples
        samples = np.random.rand(n_samples, dim)
        samples = bounds[:, 0] + samples * (bounds[:, 1] - bounds[:, 0])
        
        generators, diff, energy = cvt_iterate(generators, samples)
        
        if diff < 1e-6:
            break
    
    return generators


def cvt_quadrature_weights(generators, samples=None, n_samples=50000):
    """
    Compute CVT-based quadrature weights by Monte Carlo Voronoi area estimation.
    
    w_i ≈ (1/M) * |{s_j : s_j in V_i}|
    
    Parameters
    ----------
    generators : ndarray, shape (n_gen, dim)
    samples : ndarray, optional
    n_samples : int
    
    Returns
    -------
    weights : ndarray
        Quadrature weights (sum to 1).
    """
    if samples is None:
        dim = generators.shape[1]
        samples = np.random.rand(n_samples, dim) * 2.0 - 1.0
    
    _, assignments = cvt_energy(generators, samples)
    n_gen = generators.shape[0]
    weights = np.zeros(n_gen)
    for i in range(n_gen):
        weights[i] = np.mean(assignments == i)
    return weights


def adaptive_cvt_for_reliability(dim, n_gen, beta_sphere_radius,
                                  n_samples=20000, it_max=40, seed=None):
    """
    Generate CVT samples concentrated near the reliability index sphere.
    
    In standard normal space, the design point lies on the beta-sphere.
    We use a radially-adapted density to place more samples near |xi| ≈ beta.
    
    Parameters
    ----------
    dim : int
        Stochastic dimension.
    n_gen : int
        Number of generators.
    beta_sphere_radius : float
        Reliability index (distance to design point).
    n_samples : int
    it_max : int
    seed : int, optional
    
    Returns
    -------
    generators : ndarray, shape (n_gen, dim)
    weights : ndarray
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize with normal-distributed samples
    generators = np.random.randn(n_gen, dim)
    norms = np.linalg.norm(generators, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    generators = generators / norms * beta_sphere_radius
    
    for it in range(it_max):
        # Generate samples with radial bias toward beta-sphere
        samples = np.random.randn(n_samples, dim)
        sample_norms = np.linalg.norm(samples, axis=1)
        sample_norms = np.maximum(sample_norms, 1e-10)
        
        # Radial density: higher near beta
        radial_factor = np.exp(-0.5 * (sample_norms - beta_sphere_radius) ** 2)
        
        generators, diff, energy = cvt_iterate(generators, samples, density=radial_factor)
        
        if diff < 1e-5:
            break
    
    weights = cvt_quadrature_weights(generators, n_samples=n_samples)
    return generators, weights
