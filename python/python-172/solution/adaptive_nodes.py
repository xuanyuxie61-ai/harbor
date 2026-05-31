# -*- coding: utf-8 -*-
"""
Adaptive Node Generation and Optimization
==========================================
Centroidal Voronoi Tessellation (CVT) for adaptive spectral node placement
and quadratic interpolation for superconvergent point extraction.

Inspired by:
- cvt_ellipse_uniform: Lloyd's algorithm for CVT
- opt_quadratic: quadratic interpolation for critical point detection

Mathematical formulation:
- Given a density function rho(x) > 0, CVT minimizes:
    F({z_i}) = sum_i integral_{V_i} rho(x) ||x - z_i||^2 dx
  where V_i are Voronoi cells. The optimal generators z_i* are centroids:
    z_i* = integral_{V_i} x rho(x) dx / integral_{V_i} rho(x) dx.
- For spectral methods, adaptive CGL-type nodes cluster where solution
  gradients are large.
- Superconvergent points are extracted by quadratic interpolation through
  three neighboring nodes and finding the critical point.
"""

import numpy as np


def density_function(x, alpha=2.0, center=0.0):
    """
    Adaptive density function: higher density near the center.
    rho(x) = exp(-alpha * (x - center)^2) + epsilon.

    Parameters
    ----------
    x : ndarray
        Spatial coordinates.
    alpha : float
        Concentration parameter.
    center : float
        Center of density peak.

    Returns
    -------
    rho : ndarray
        Density values.
    """
    return np.exp(-alpha * (x - center) ** 2) + 0.01


def cvt_1d_lloyd(n_generators, n_samples, it_num, domain=(-1.0, 1.0),
                 rho_func=None, seed=42):
    """
    Compute a 1D Centroidal Voronoi Tessellation via Lloyd's algorithm.

    Parameters
    ----------
    n_generators : int
        Number of generators.
    n_samples : int
        Number of Monte Carlo sample points per iteration.
    it_num : int
        Number of Lloyd iterations.
    domain : tuple
        (xmin, xmax).
    rho_func : callable, optional
        Density function rho(x). Defaults to uniform.
    seed : int
        Random seed.

    Returns
    -------
    z : ndarray
        Optimized generator locations.
    """
    rng = np.random.default_rng(seed)
    xmin, xmax = domain
    # Initialize generators uniformly
    z = np.linspace(xmin, xmax, n_generators)

    if rho_func is None:
        def rho_func(x):
            return np.ones_like(x)

    for it in range(it_num):
        # Sample points in domain
        samples = rng.uniform(xmin, xmax, n_samples)
        weights = rho_func(samples)

        # Assign each sample to nearest generator
        # For 1D, sort generators and use bins
        z_sorted = np.sort(z)
        boundaries = np.zeros(n_generators + 1)
        boundaries[0] = xmin - 1e-10
        boundaries[-1] = xmax + 1e-10
        for i in range(1, n_generators):
            boundaries[i] = 0.5 * (z_sorted[i - 1] + z_sorted[i])

        # Compute centroids
        z_new = np.zeros(n_generators)
        for i in range(n_generators):
            mask = (samples >= boundaries[i]) & (samples < boundaries[i + 1])
            if np.any(mask):
                z_new[i] = np.average(samples[mask], weights=weights[mask])
            else:
                z_new[i] = z_sorted[i]
        z = z_new

    return np.sort(z)


def map_cvt_to_cgl(cvt_nodes, n_cgl):
    """
    Map CVT adaptive nodes to a standard CGL grid for spectral differentiation
    while preserving clustering. Uses a nonlinear warping.

    Parameters
    ----------
    cvt_nodes : ndarray
        CVT nodes.
    n_cgl : int
        Target number of CGL nodes.

    Returns
    -------
    warped_nodes : ndarray
        Adaptively warped nodes suitable for spectral differentiation.
    mapping_derivative : ndarray
        Derivative of the mapping (needed for chain rule in spectral operators).
    """
    cvt_nodes = np.asarray(cvt_nodes)
    n = len(cvt_nodes)
    # Standard CGL nodes
    j = np.arange(n_cgl)
    cgl = np.cos(np.pi * j / (n_cgl - 1))

    # Warping: map CGL nodes to CVT distribution via empirical CDF
    cvt_sorted = np.sort(cvt_nodes)
    cvt_cdf = np.arange(1, n + 1) / n

    # Interpolate inverse CDF
    warped = np.interp((j + 0.5) / n_cgl, cvt_cdf, cvt_sorted)
    # Ensure boundaries
    warped[0] = cvt_sorted[-1]
    warped[-1] = cvt_sorted[0]

    # Derivative of mapping (approximate)
    mapping_derivative = np.gradient(warped, cgl)
    mapping_derivative = np.clip(np.abs(mapping_derivative), 0.1, 10.0)

    return warped, mapping_derivative


def quadratic_superconvergent_point(x_vals, y_vals):
    """
    Fit a quadratic polynomial through three points (x_i, y_i) and return
    its critical point (vertex). Adapted from opt_quadratic.

    Given parabola y = a x^2 + b x + c through three points, vertex at:
        x* = -b / (2a)
    where [a,b,c]^T = V^{-1} [y1,y2,y3]^T and V is the Vandermonde matrix.

    Parameters
    ----------
    x_vals : ndarray, shape (3,)
    y_vals : ndarray, shape (3,)

    Returns
    -------
    x_star : float
        Critical point location.
    y_star : float
        Estimated function value at critical point.
    """
    x_vals = np.asarray(x_vals, dtype=np.float64)
    y_vals = np.asarray(y_vals, dtype=np.float64)
    if len(x_vals) != 3 or len(y_vals) != 3:
        raise ValueError("Exactly three points required.")
    # Vandermonde matrix for quadratic fit
    V = np.vander(x_vals, 3)
    try:
        coeffs = np.linalg.solve(V, y_vals)
    except np.linalg.LinAlgError:
        # Degenerate case: points are collinear
        return np.mean(x_vals), np.mean(y_vals)
    a, b, c = coeffs[0], coeffs[1], coeffs[2]
    if abs(a) < 1e-12:
        return np.mean(x_vals), np.mean(y_vals)
    x_star = -b / (2.0 * a)
    y_star = a * x_star ** 2 + b * x_star + c
    return x_star, y_star


def extract_superconvergent_points(x, u):
    """
    Scan through neighboring triplets and extract superconvergent points
    where the interpolated quadratic has extrema.

    Parameters
    ----------
    x : ndarray
        Grid points.
    u : ndarray
        Function values.

    Returns
    -------
    sc_x : ndarray
        Superconvergent point locations.
    sc_u : ndarray
        Estimated values at superconvergent points.
    """
    x = np.asarray(x, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    sc_x = []
    sc_u = []
    for i in range(len(x) - 2):
        xs, ys = quadratic_superconvergent_point(x[i:i + 3], u[i:i + 3])
        if x[i + 2] <= xs <= x[i]:
            sc_x.append(xs)
            sc_u.append(ys)
    return np.array(sc_x), np.array(sc_u)
