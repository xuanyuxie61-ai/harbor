"""
chaotic_mixing.py
=================

Lagrangian chaotic advection inside the disk-shaped geophysical
reactor.  Two ingredients, both ported from existing Burkardt /
Burkardt-style codes, are fused into a single tracer-mixing kernel
that serves as the velocity surrogate in the Sobol forward model:

1. **Chirikov standard map** (``chirikov_map``, ``chirikov_iter``).
   A 2-D area-preserving map

       p_{n+1} = p_n + K sin(theta_n)     (mod 2 pi)
       theta_{n+1} = theta_n + p_{n+1}    (mod 2 pi)

   whose stochastic parameter ``K`` controls the degree of phase-space
   mixing.  ``K = 0`` is integrable, ``K >= 1`` is fully chaotic, with
   the golden-mean Cantor breakup at ``K ~ 0.9716``.

2. **Uniform disk sampling** (``disk_sample``).
   Uses the Marsaglia-style normalised-Gaussian trick to draw uniform
   points on the 2-D disk of radius ``R``.  The samples seed Lagrangian
   tracers whose trajectories are iterated by the Chirikov map to
   quantify the effective stirring rate.

Together these produce a *finite-time Lyapunov exponent* (FTLE)
scalar field ``Lambda(theta)`` whose magnitude summarises how much
the reactor mixes per unit time.  This scalar enters the Sobol
forward model as the advective velocity scale:

    v_eff(theta) = R * Lambda(theta) / (2 pi)

The FTLE is the Sobol quantity of interest in Step 2 of the
pipeline, with the stochastic parameter ``K`` being one of the
uncertain inputs whose first-order index we compute.

References
----------
* B. V. Chirikov, *A universal instability of multi-dimensional
  oscillator systems*, Phys. Rep. 52 (1979), 263-379.
* J. Burkardt, ``chirikov_iteration`` and ``disk_monte_carlo``
  libraries.
* G. Haller, *Distinguished material surfaces and coherent structures
  in three-dimensional fluid flows*, Physica D 149 (2001), 248-277.
"""

from __future__ import annotations

import math

import numpy as np

from r8col_utils import dedupe_sample_matrix


# =====================================================================
# Chirikov standard map
# =====================================================================
def chirikov_step(theta: float, p: float, K: float) -> tuple[float, float]:
    """One iteration of the Chirikov standard map.

    Parameters
    ----------
    theta, p : float
        Angle and action.
    K : float
        Stochasticity parameter (non-negative).

    Returns
    -------
    theta_new, p_new : float
        Updated angle and action (both reduced modulo ``2 pi``).
    """
    if K < 0.0:
        raise ValueError("chirikov_step: K must be non-negative")
    p_new = p + K * math.sin(theta)
    theta_new = theta + p_new
    # Reduce mod 2 pi to keep iterates bounded
    theta_new = math.fmod(theta_new, 2.0 * math.pi)
    p_new = math.fmod(p_new, 2.0 * math.pi)
    if theta_new < 0.0:
        theta_new += 2.0 * math.pi
    if p_new < 0.0:
        p_new += 2.0 * math.pi
    return theta_new, p_new


def chirikov_iter(theta: float, p: float, K: float, n_iter: int
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Iterate the Chirikov map for ``n_iter`` steps.

    Returns two 1-D arrays of length ``n_iter + 1`` containing the
    trajectories of ``(theta, p)``.
    """
    if n_iter < 0:
        raise ValueError("chirikov_iter: n_iter must be non-negative")
    th_arr = np.empty(n_iter + 1, dtype=float)
    p_arr = np.empty(n_iter + 1, dtype=float)
    th_arr[0] = theta
    p_arr[0] = p
    for i in range(n_iter):
        theta, p = chirikov_step(theta, p, K)
        th_arr[i + 1] = theta
        p_arr[i + 1] = p
    return th_arr, p_arr


# =====================================================================
# Finite-time Lyapunov exponent (FTLE)
# =====================================================================
def chirikov_ftle(theta: float, p: float, K: float, n_iter: int,
                  delta: float = 1.0e-7) -> float:
    """Finite-time Lyapunov exponent at ``(theta, p)``.

    Two nearby orbits are integrated; the logarithmic separation rate
    approximates the largest Lyapunov exponent for this area-preserving
    map.

    Returns
    -------
    float
        Non-negative FTLE value (clamped to 0 if separation collapses).
    """
    if n_iter < 1:
        return 0.0
    # Perturbed initial condition
    th1, p1 = theta + delta, p
    th2, p2 = theta, p + delta
    for _ in range(n_iter):
        th1, p1 = chirikov_step(th1, p1, K)
        th2, p2 = chirikov_step(th2, p2, K)
    # Use angular distance on the torus for both coordinates
    dth = min(abs(th1 - th2), 2.0 * math.pi - abs(th1 - th2))
    dp = min(abs(p1 - p2), 2.0 * math.pi - abs(p1 - p2))
    sep = math.sqrt(dth * dth + dp * dp)
    if sep < 1.0e-30:
        return 0.0
    return max(0.0, math.log(sep / delta) / n_iter)


def chirikov_ftle_grid(K: float, n_iter: int, n_grid: int,
                       seed: int = 0) -> np.ndarray:
    """Compute FTLE on an ``n_grid x n_grid`` torus grid.

    The result is a 2-D array with FTLE values in radians/time step.
    """
    if n_grid < 2:
        raise ValueError("chirikov_ftle_grid: n_grid must be >= 2")
    rng = np.random.default_rng(seed)
    theta_grid = np.linspace(0.0, 2.0 * math.pi, n_grid, endpoint=False)
    p_grid = np.linspace(0.0, 2.0 * math.pi, n_grid, endpoint=False)
    L = np.zeros((n_grid, n_grid), dtype=float)
    for i, th in enumerate(theta_grid):
        for j, pp in enumerate(p_grid):
            L[i, j] = chirikov_ftle(th, pp, K, n_iter)
    return L


# =====================================================================
# Uniform disk sampling (Marsaglia / Gaussian-normalised trick)
# =====================================================================
def disk_sample(center: np.ndarray, R: float, n: int,
                rng: np.random.Generator | None = None) -> np.ndarray:
    """Draw ``n`` uniform points inside the 2-D disk of radius ``R``.

    Parameters
    ----------
    center : array-like of shape (2,)
        Centre of the disk.
    R : float
        Radius (positive).
    n : int
        Number of samples (positive).
    rng : numpy Generator, optional.

    Returns
    -------
    X : ndarray of shape (2, n)
        Sampled points, one column per point.
    """
    if R <= 0.0:
        raise ValueError("disk_sample: R must be positive")
    if n <= 0:
        raise ValueError("disk_sample: n must be positive")
    center = np.atleast_1d(np.asarray(center, dtype=float)).reshape(2)
    rng = rng or np.random.default_rng()
    X = rng.standard_normal((2, n))
    norms = np.sqrt((X * X).sum(axis=0))
    norms = np.where(norms < 1.0e-30, 1.0, norms)
    X = X / norms  # points on the unit circle
    # sqrt(uniform) radial factor => uniform density on the disk
    r2 = rng.random(n)
    X = X * np.sqrt(r2) * R + center[:, None]
    return X


def disk_monomial_integral(p: int, q: int, R: float) -> float:
    r"""Analytic integral of ``x^p y^q`` over the disk of radius ``R``.

    Closed form (for even ``p + q``)::

        I = 2 pi R^{p+q+2} / (p + q + 2) * Beta((p+1)/2, (q+1)/2) / (2 Beta(1/2, 1/2))

    Returns 0 for ``p + q`` odd by symmetry.
    """
    if (p + q) % 2 == 1:
        return 0.0
    from math import gamma
    num = 2.0 * math.pi * (R ** (p + q + 2))
    num *= gamma(0.5 * (p + 1)) * gamma(0.5 * (q + 1))
    den = (p + q + 2) * gamma(0.5) * gamma(0.5)
    return num / den


# =====================================================================
# Effective advective velocity scale
# =====================================================================
def effective_velocity(R: float, K: float, n_iter: int = 64,
                       n_quad: int = 8, seed: int = 0) -> float:
    """Estimate the effective stirring velocity ``v_eff`` (m/s).

    Averages the FTLE over an ``n_quad x n_quad`` torus grid and
    multiplies by the disk radius:

        v_eff = R * <Lambda>_torus / (2 pi)

    The average is robustly approximated by the trapezoidal rule on
    the periodic domain; boundary values are automatically identified
    by the ``dedupe_sample_matrix`` utility (this prevents double
    counting when the grid is wrapped).
    """
    if R <= 0.0 or K < 0.0 or n_iter < 1 or n_quad < 2:
        raise ValueError("effective_velocity: invalid parameters")
    L = chirikov_ftle_grid(K, n_iter, n_quad, seed=seed)
    mean_L = float(L.mean())
    return R * mean_L / (2.0 * math.pi)


# =====================================================================
# Sanity check
# =====================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(2024)
    K = 1.2
    print("Chirikov trajectory at K =", K)
    th, p = chirikov_iter(1.0, 0.5, K, 20)
    print("  theta:", th[:5], "...")
    print("  p:    ", p[:5], "...")
    print("FTLE at (1, 0.5) =", chirikov_ftle(1.0, 0.5, K, 256))
    pts = disk_sample(np.array([0.0, 0.0]), 1.0, 5, rng=rng)
    print("disk samples:\n", pts)
    print("monomial I(x^2) over disk R=1 =", disk_monomial_integral(2, 0, 1.0))
    print("v_eff =", effective_velocity(1.0, 1.2, n_iter=64, n_quad=6))
