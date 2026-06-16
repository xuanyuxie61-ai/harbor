"""
least_squares_quadrature.py
===========================
Least-squares quadrature weights for halo mass integrals.

Background (Burkardt / Hyman)
-----------------------------
Given function values F_i = f(x_i) at N (possibly noisy) sample
points x_i in [a, b], we wish to estimate

    I(f) = int_a^b f(x) dx

by a quadrature rule  Q(f) = sum_i W_i F_i.  The weights W are
determined by requiring Q to be exact for polynomials of degree D
in the least-squares sense.  When N = D + 1 the procedure reduces
to classical interpolatory quadrature; for N > D + 1 it yields a
robust, noise-tolerant rule suited to stochastic halo sampling.

Formally, the weights satisfy   V' V C = V' F,   where V is the
Vandermonde matrix.  Solving for W gives

    W = Q  (V' V)^{-1}  V',

where Q is the vector of monomial integrals over [a, b].
"""

from __future__ import annotations
import math
from typing import Tuple

import numpy as np


def monomial_integrals(a: float, b: float, degree: int) -> np.ndarray:
    """Return the vector Q with Q_k = int_a^b x^k dx,  k = 0..degree."""
    Q = np.zeros(degree + 1)
    for k in range(degree + 1):
        Q[k] = (b ** (k + 1) - a ** (k + 1)) / (k + 1)
    return Q


def vandermonde(x: np.ndarray, degree: int) -> np.ndarray:
    """Build the Vandermonde matrix V_{i, k} = x_i^k."""
    V = np.zeros((x.size, degree + 1))
    V[:, 0] = 1.0
    for k in range(1, degree + 1):
        V[:, k] = V[:, k - 1] * x
    return V


def least_squares_weights(x: np.ndarray, a: float, b: float,
                          degree: int) -> np.ndarray:
    """Compute least-squares quadrature weights for the nodes x
    on the interval [a, b] with polynomial degree D."""
    V = vandermonde(x, degree)
    Q = monomial_integrals(a, b, degree)
    # Solve V'V R = V' for R, then W = Q @ R
    try:
        # Regularised normal equations for robustness
        lam = 1e-12 * np.eye(V.shape[1])
        G = V.T @ V + lam
        R = np.linalg.solve(G, V.T)
        W = Q @ R
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse
        W = Q @ np.linalg.pinv(V)
    return W


# ---------- Halo mass integral -----------------------------------------------

def halo_mass_integral(r_samples: np.ndarray,
                       rho_samples: np.ndarray,
                       degree: int = 4) -> Tuple[float, np.ndarray]:
    """Estimate the enclosed halo mass  M(< R) = int_0^R 4 pi r^2 rho(r) dr
    from noisy samples using a least-squares quadrature rule.

    Returns (mass_estimate, weights).
    """
    a, b = float(r_samples.min()), float(r_samples.max())
    if b <= a:
        return 0.0, np.zeros_like(r_samples)
    W = least_squares_weights(r_samples, a, b, degree)
    integrand = 4.0 * math.pi * r_samples ** 2 * rho_samples
    # rescale to [a, b]
    mass = float((b - a) * np.sum(W * integrand))
    return mass, W


# ---------- Self-check --------------------------------------------------------

def self_check() -> dict:
    rng = np.random.default_rng(0)
    # Test on a known integral: int_0^1 x^2 dx = 1/3
    x = rng.uniform(0.0, 1.0, size=20)
    f = x ** 2
    W = least_squares_weights(x, 0.0, 1.0, degree=3)
    est = float(np.sum(W * f))
    return dict(integral_estimate=est, exact=1.0 / 3.0,
                abs_error=abs(est - 1.0 / 3.0))
