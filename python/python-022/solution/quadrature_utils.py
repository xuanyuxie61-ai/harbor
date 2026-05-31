"""
Gauss-Jacobi Quadrature Utilities for ICF Simulation

Based on: jacobi_rule (Project 608)

Provides high-precision numerical integration using Gauss-Jacobi quadrature,
essential for opacity Rosseland means, equation-of-state integrals,
and laser absorption path integrals in plasma physics.
"""

import numpy as np
from scipy.special import roots_jacobi


def gauss_jacobi_rule(order, alpha, beta, a, b):
    """
    Compute Gauss-Jacobi quadrature rule on [a, b].

    Parameters
    ----------
    order : int
        Number of quadrature points.
    alpha, beta : float
        Exponents for (b-x)^alpha * (x-a)^beta.
    a, b : float
        Interval endpoints.

    Returns
    -------
    x, w : ndarray
        Quadrature nodes and weights.
    """
    if order < 1:
        raise ValueError("gauss_jacobi_rule: order must be >= 1.")
    if alpha <= -1.0 or beta <= -1.0:
        raise ValueError("gauss_jacobi_rule: alpha, beta must be > -1.")
    if abs(b - a) < np.finfo(float).eps:
        raise ValueError("gauss_jacobi_rule: interval [a,b] is degenerate.")

    x, w = roots_jacobi(order, alpha, beta)
    # Scale from [-1, 1] to [a, b]
    slp = (b - a) / 2.0
    shft = (a + b) / 2.0
    x_scaled = shft + slp * x
    w_scaled = slp * w
    return x_scaled, w_scaled


def gauss_legendre_rule(order, a, b):
    """
    Compute Gauss-Legendre quadrature rule on [a, b].
    """
    if order < 1:
        raise ValueError("gauss_legendre_rule: order must be >= 1.")
    if abs(b - a) < np.finfo(float).eps:
        raise ValueError("gauss_legendre_rule: interval [a,b] is degenerate.")

    from numpy.polynomial.legendre import leggauss
    x, w = leggauss(order)
    slp = (b - a) / 2.0
    shft = (a + b) / 2.0
    x_scaled = shft + slp * x
    w_scaled = slp * w
    return x_scaled, w_scaled


def integrate_gauss_jacobi(f, order, alpha, beta, a, b):
    """
    Integrate f(x) * (b-x)^alpha * (x-a)^beta over [a, b].
    """
    x, w = gauss_jacobi_rule(order, alpha, beta, a, b)
    fx = np.array([f(xi) for xi in x])
    return np.sum(w * fx)
