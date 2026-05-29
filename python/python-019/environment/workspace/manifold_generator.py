"""
manifold_generator.py
---------------------
Generate parameter manifolds (curves and surfaces) for exceptional-point
studies in non-Hermitian physics.

Adapted from seed project 1052_sammon_data (data generation: circle,
helix, linear, nonlinear, simplex).

Scientific Background
=====================
Exceptional points in non-Hermitian Hamiltonians often reside on
parameter manifolds of codimension 2 in the full parameter space.
When we encircle an exceptional point by adiabatically varying
parameters along a closed loop, the eigenstates undergo a cyclic
exchange:

    |ψ_1^R⟩ → |ψ_2^R⟩,   |ψ_2^R⟩ → |ψ_1^R⟩.

This is the hallmark of a topological defect. To study such phenomena,
one needs to generate smooth closed loops (circles), helical paths
(representing adiabatic cycles with a slow drift), and higher-dimensional
parameter simplices for systematic sampling.

For a 2D parameter space (m, γ), an EP loop can be parameterized as

    m(θ) = m_0 + r cos θ,
    γ(θ) = γ_0 + r sin θ,

and a helix in 3D parameter space (m, γ, t) is

    m(θ) = m_0 + r cos θ,
    γ(θ) = γ_0 + r sin θ,
    t(θ) = t_0 + h θ / (2π).
"""

import numpy as np


def circle_loop(center, radius, n_points=100):
    """
    Generate points on a circular loop in 2D parameter space.

    Parameters
    ----------
    center : tuple (m0, gamma0)
    radius : float
    n_points : int

    Returns
    -------
    params : ndarray, shape (n_points, 2)
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    m = center[0] + radius * np.cos(theta)
    gamma = center[1] + radius * np.sin(theta)
    return np.column_stack((m, gamma))


def helix_loop(center, radius, pitch, n_points=200, turns=2):
    """
    Generate points on a helical path in 3D parameter space.

    Parameters
    ----------
    center : tuple (m0, gamma0, t0)
    radius : float
    pitch : float
        Vertical rise per turn.
    n_points : int
    turns : float
        Number of turns.

    Returns
    -------
    params : ndarray, shape (n_points, 3)
    """
    theta = np.linspace(0.0, 2.0 * np.pi * turns, n_points)
    m = center[0] + radius * np.cos(theta)
    gamma = center[1] + radius * np.sin(theta)
    t = center[2] + pitch * theta / (2.0 * np.pi)
    return np.column_stack((m, gamma, t))


def nonlinear_curve(n_points=100, dim=5):
    """
    Generate a nonlinear curve in high-dimensional parameter space,
    analogous to the nonlinear data generator in sammon_data.

    The curve is parameterized by z and embeds into dim dimensions as
    a superposition of sinusoids with increasing frequency.

    Parameters
    ----------
    n_points : int
    dim : int

    Returns
    -------
    params : ndarray, shape (n_points, dim)
    """
    z = np.linspace(0.0, 2.0 * np.pi, n_points)
    params = np.zeros((n_points, dim))
    for d in range(dim):
        freq = d + 1
        params[:, d] = np.cos(freq * z) / np.sqrt(freq)
    return params


def simplex_parameter_space(dim, n_points=75, std=0.2):
    """
    Generate parameter points clustered near the vertices of a regular
    simplex in dim dimensions. This is useful for sampling distinct
    phases of a non-Hermitian phase diagram.

    Parameters
    ----------
    dim : int
        Spatial dimension of parameter space.
    n_points : int
        Total number of points.
    std : float
        Gaussian spread around each vertex.

    Returns
    -------
    params : ndarray, shape (n_points, dim)
    labels : ndarray, shape (n_points,)
        Vertex index for each point.
    """
    vertices = _regular_simplex_vertices(dim)
    params = np.zeros((n_points, dim))
    labels = np.zeros(n_points, dtype=int)
    points_per_vertex = n_points // (dim + 1)
    idx = 0
    for v in range(dim + 1):
        n = points_per_vertex if v < dim else (n_points - idx)
        params[idx:idx + n] = vertices[v] + std * np.random.randn(n, dim)
        labels[idx:idx + n] = v
        idx += n
    return params, labels


def _regular_simplex_vertices(n):
    """
    Compute vertices of a regular simplex centered at the origin in n
    dimensions with unit distance from centroid.
    """
    x = np.zeros((n, n + 1))
    for j in range(n):
        x[j, j] = 1.0
    a = (1.0 - np.sqrt(1.0 + n)) / n
    x[:, n] = a
    c = x.sum(axis=1) / (n + 1)
    x = x - c[:, None]
    s = np.linalg.norm(x[:, 0])
    x = x / s
    return x.T


def adiabatic_cycle_around_ep(ep_center, ep_radius, n_points=200):
    """
    Generate a smooth adiabatic cycle that encircles an exceptional
    point in the (m, γ) plane. The cycle is a circle offset so that
    the EP lies inside.

    Parameters
    ----------
    ep_center : tuple (m_EP, gamma_EP)
    ep_radius : float
        Radius of the loop around the EP.
    n_points : int

    Returns
    -------
    cycle : ndarray, shape (n_points, 2)
    """
    return circle_loop(ep_center, ep_radius, n_points)
