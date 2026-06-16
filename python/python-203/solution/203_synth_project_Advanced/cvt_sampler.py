"""
cvt_sampler.py -- Centroidal Voronoi Tessellation for Adaptive Stochastic Sampling
===================================================================================
Implements CVT-based adaptive refinement of quasi-Monte Carlo sample points
in the stochastic space. The CVT iteration moves generators to the centroids
of their Voronoi cells, producing an optimally uniform point distribution
that minimizes the quantization error:
    E = sum_i integral_{V_i} rho(x) ||x - g_i||^2 dx

This is used to generate optimal stochastic collocation points for UQ.

Seed references:
  - 264_cvtp: cvtp_iteration, cvtp_find_closest, cvtp_region_sampler
  - 235_cube_monte_carlo: uniform cube sampling
  - 883_polygon_average: polygon averaging iteration (convergence concept)

Scientific context:
  In high-dimensional stochastic spaces, standard QMC sequences may leave
  gaps or create clusters. CVT-based sampling iteratively adjusts the points
  to achieve a centroidal configuration where each point is the center of
  mass of its Voronoi cell, providing optimal L2 quantization.
"""
import numpy as np
from typing import Tuple


def cvt_iteration(generators: np.ndarray, sample_num: int,
                  domain_min: np.ndarray, domain_max: np.ndarray,
                  modular: bool = False) -> Tuple[np.ndarray, float]:
    """
    One Lloyd-type iteration of the Centroidal Voronoi Tessellation algorithm.

    Algorithm:
    1. Generate sample_num random points in the domain
    2. Assign each sample to the nearest generator
    3. Move each generator to the centroid of its assigned samples
    4. Compute the L2 change in generator positions

    The Voronoi cell V_i of generator g_i is:
        V_i = {x : ||x - g_i|| <= ||x - g_j|| for all j != i}

    Parameters
    ----------
    generators : ndarray, shape (n_gen, dim)
        Current generator positions.
    sample_num : int
        Number of Monte Carlo samples per iteration.
    domain_min, domain_max : ndarray, shape (dim,)
        Domain bounds.
    modular : bool
        If True, use periodic (toroidal) distance metric.

    Returns
    -------
    new_generators : ndarray, shape (n_gen, dim)
        Updated generator positions.
    change_l2 : float
        Frobenius norm of the change: ||g_new - g_old||_F.
    """
    n_gen, dim = generators.shape
    width = domain_max - domain_min

    # Generate uniform random samples in the domain
    samples = np.zeros((sample_num, dim))
    for d in range(dim):
        samples[:, d] = domain_min[d] + (domain_max[d] - domain_min[d]) * np.random.rand(sample_num)

    # Assign each sample to the nearest generator
    accum = np.zeros((n_gen, dim))
    count = np.zeros(n_gen, dtype=int)

    for s in range(sample_num):
        best_gen, best_dist = _find_closest(samples[s], generators, width, modular)
        accum[best_gen] += samples[s]
        count[best_gen] += 1

    # Update generators to centroids
    new_generators = generators.copy()
    for g in range(n_gen):
        if count[g] > 0:
            new_generators[g] = accum[g] / count[g]

    # Apply boundary conditions
    if modular:
        # Periodic wrap-around
        for d in range(dim):
            new_generators[:, d] = domain_min[d] + np.mod(
                new_generators[:, d] - domain_min[d], width[d])
    else:
        # Clamp to domain
        for d in range(dim):
            new_generators[:, d] = np.clip(new_generators[:, d],
                                           domain_min[d], domain_max[d])

    change_l2 = float(np.linalg.norm(new_generators - generators, 'fro'))
    return new_generators, change_l2


def _find_closest(point: np.ndarray, generators: np.ndarray,
                  width: np.ndarray, modular: bool) -> Tuple[int, float]:
    """
    Find the nearest generator to a given point.
    If modular=True, consider periodic images in each dimension.

    Parameters
    ----------
    point : ndarray, shape (dim,)
    generators : ndarray, shape (n_gen, dim)
    width : ndarray, shape (dim,)
    modular : bool

    Returns
    -------
    best_gen : int
        Index of closest generator.
    best_dist : float
        Squared distance to closest generator.
    """
    n_gen = generators.shape[0]
    best_gen = 0
    best_dist = np.inf

    for g in range(n_gen):
        if modular:
            # Periodic distance: consider 3 images per dimension
            diff = point - generators[g]
            # Wrap to shortest image
            diff = diff - width * np.round(diff / width)
            dist = np.sum(diff ** 2)
        else:
            dist = np.sum((point - generators[g]) ** 2)

        if dist < best_dist:
            best_dist = dist
            best_gen = g

    return best_gen, best_dist


def cvt_run(n_generators: int, dim: int, max_iter: int = 50,
            tol: float = 1e-8, sample_factor: int = 100,
            modular: bool = False, seed: int = 42) -> dict:
    """
    Run the CVT iteration to convergence.

    Parameters
    ----------
    n_generators : int
        Number of CVT generators (collocation points).
    dim : int
        Dimension of the stochastic space.
    max_iter : int
        Maximum number of Lloyd iterations.
    tol : float
        Convergence tolerance on ||g_new - g_old||_F.
    sample_factor : int
        Number of MC samples = sample_factor * n_generators.
    modular : bool
        Use periodic domain.
    seed : int
        Random seed for initialization.

    Returns
    -------
    result : dict
        'generators': final CVT points, shape (n_gen, dim)
        'history': L2 changes per iteration
        'n_iter': number of iterations taken
        'converged': bool
        'energy': final CVT energy (quantization error estimate)
    """
    np.random.seed(seed)

    domain_min = np.zeros(dim)
    domain_max = np.ones(dim)
    sample_num = sample_factor * n_generators

    # Initialize generators randomly
    generators = np.random.rand(n_generators, dim)

    history = []
    converged = False

    for iteration in range(max_iter):
        generators, change = cvt_iteration(
            generators, sample_num, domain_min, domain_max, modular)
        history.append(change)

        if change < tol:
            converged = True
            break

    # Compute CVT energy: average squared distance from samples to nearest generator
    energy = _compute_cvt_energy(generators, sample_num, domain_min, domain_max, modular)

    return {
        'generators': generators,
        'history': history,
        'n_iter': len(history),
        'converged': converged,
        'energy': energy
    }


def _compute_cvt_energy(generators: np.ndarray, n_samples: int,
                        domain_min: np.ndarray, domain_max: np.ndarray,
                        modular: bool) -> float:
    """
    Estimate the CVT quantization energy:
        E = (1/N) sum_{s=1}^{N} min_i ||x_s - g_i||^2
    """
    dim = generators.shape[1]
    width = domain_max - domain_min
    samples = np.random.rand(n_samples, dim)
    for d in range(dim):
        samples[:, d] = domain_min[d] + (domain_max[d] - domain_min[d]) * samples[:, d]

    total_energy = 0.0
    for s in range(n_samples):
        _, best_dist = _find_closest(samples[s], generators, width, modular)
        total_energy += best_dist

    return total_energy / n_samples


def cvt_discrepancy_study(dim_list: list, n_gen: int = 50) -> dict:
    """
    Compare CVT discrepancy against Halton and random sampling across
    dimensions. Demonstrates that CVT produces more uniform point sets.

    Returns dict with discrepancy values for each method and dimension.
    """
    from quasi_mc import halton, star_discrepancy

    results = {'dimensions': dim_list, 'cvt_disc': [], 'halton_disc': [], 'random_disc': []}

    for dim in dim_list:
        # CVT points
        cvt_result = cvt_run(n_gen, dim, max_iter=30, seed=123)
        cvt_pts = cvt_result['generators']
        results['cvt_disc'].append(star_discrepancy(cvt_pts))

        # Halton points
        h_pts = halton(n_gen, dim)
        results['halton_disc'].append(star_discrepancy(h_pts))

        # Random points (average over 5 trials)
        rng = np.random.RandomState(42)
        disc_sum = 0.0
        for _ in range(5):
            r_pts = rng.rand(n_gen, dim)
            disc_sum += star_discrepancy(r_pts)
        results['random_disc'].append(disc_sum / 5)

    return results
