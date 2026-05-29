# -*- coding: utf-8 -*-
"""
quadrature_rules.py
===================
High-dimensional numerical integration rules adapted from two seed projects:
  - 302_disk01_rule:  Gauss-Legendre quadrature on the unit disk
  - 933_pyramid_integrals:  Monomial integration over the unit pyramid

Applications in seismic isolation analysis:
  - Disk quadrature:  integrate bearing contact pressure over circular
    bearing pads (radius R) to obtain total vertical force and moment.
  - Pyramid quadrature:  integrate mass density over 3-D pyramid finite
    elements for consistent mass matrix assembly.
"""

import numpy as np
from typing import Tuple


# ====================================================================== #
# Legendre-Gauss quadrature on [-1, +1] (Elhay-Kautsky / IQPACK method)
# ====================================================================== #
def legendre_ek_compute(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the n-point Gauss-Legendre quadrature rule on [-1, +1].
    
    Uses the Golub-Welsch algorithm via symmetric tridiagonal Jacobi matrix.
    The recurrence coefficients for Legendre polynomials are:
      beta_j = j^2 / (4 * j^2 - 1)
    
    Returns
    -------
    x : np.ndarray, shape (n,)
        Quadrature nodes (abscissas).
    w : np.ndarray, shape (n,)
        Quadrature weights.
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    # Jacobi matrix for Legendre: diagonal is zero, off-diagonal is sqrt(beta)
    beta = np.zeros(n, dtype=float)
    for i in range(1, n + 1):
        beta[i - 1] = (i * i) / (4.0 * i * i - 1.0)
    beta = np.sqrt(beta)

    # Build symmetric tridiagonal matrix
    J = np.diag(beta[:-1], k=1) + np.diag(beta[:-1], k=-1)

    # Eigenvalues are nodes, eigenvectors give weights
    eigvals, eigvecs = np.linalg.eigh(J)
    x = eigvals
    # First component of each normalized eigenvector
    w = 2.0 * (eigvecs[0, :] ** 2)

    return x, w


# ====================================================================== #
# Disk quadrature (from 302_disk01_rule)
# ====================================================================== #
def disk01_quadrature_rule(nr: int, nt: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a product quadrature rule for the unit disk  x^2 + y^2 <= 1.
    
    Uses radial Gauss-Legendre rule mapped to [0, 1] with sqrt transform,
    and uniform angular partitioning.
    
    Integral formula:
      I(f) = pi * sum_{j=1}^{NT} sum_{i=1}^{NR} w_i * f(r_i * cos(t_j), r_i * sin(t_j))
    
    Parameters
    ----------
    nr : int
        Number of radial points.
    nt : int
        Number of angular sectors.
    
    Returns
    -------
    w : np.ndarray, shape (nr,)
        Radial weights (already include 1/nt factor).
    r : np.ndarray, shape (nr,)
        Radial nodes.
    t : np.ndarray, shape (nt,)
        Angular nodes.
    """
    xr, wr = legendre_ek_compute(nr)
    # Map [-1, 1] -> [0, 1]
    xr = (xr + 1.0) / 2.0
    wr = wr / 2.0

    # Disk rule: r = sqrt(xr), w = wr / nt
    t = np.linspace(0.0, 2.0 * np.pi, nt, endpoint=False)
    w = wr / nt
    r = np.sqrt(xr)

    return w, r, t


def integrate_over_disk(
    func: callable,
    nr: int = 12,
    nt: int = 24,
    radius: float = 1.0,
) -> float:
    """
    Integrate a function f(x, y) over a disk of given radius.
    
    Parameters
    ----------
    func : callable
        Function f(x, y) accepting scalar or array inputs.
    nr, nt : int
        Number of radial and angular quadrature points.
    radius : float
        Disk radius.
    
    Returns
    -------
    value : float
        Approximate integral.
    """
    w, r, theta = disk01_quadrature_rule(nr, nt)
    # Scale to actual radius: r -> radius * r, weights -> radius^2 * w
    # Jacobian for polar: r dr dtheta -> (radius^2) * (r_scaled) * dr_scaled * dtheta
    # The base rule already accounts for area element on unit disk.
    # Scaling: integral over disk(R) = R^2 * integral over disk(1) with same rule.
    value = 0.0
    for it in range(nt):
        for ir in range(nr):
            x = radius * r[ir] * np.cos(theta[it])
            y = radius * r[ir] * np.sin(theta[it])
            value += w[ir] * func(x, y)
    value = value * (radius ** 2) * np.pi
    return float(value)


# ====================================================================== #
# Pyramid monomial integral (from 933_pyramid_integrals)
# ====================================================================== #
def pyramid01_monomial_integral(expon: Tuple[int, int, int]) -> float:
    """
    Integrate  x^expon[0] * y^expon[1] * z^expon[2]  over the unit pyramid:
      -(1 - z) <= x <= 1 - z
      -(1 - z) <= y <= 1 - z
                 0 <= z <= 1
    
    Closed-form formula (Stroud, 1971):
      If either x or y exponent is odd, integral = 0.
      Otherwise:
        I = (2/(e_x+1)) * (2/(e_y+1)) * sum_{i=0}^{2+e_x+e_y} (-1)^i * C(2+e_x+e_y, i) / (i + e_z + 1)
    """
    ex, ey, ez = expon
    if ex < 0 or ey < 0 or ez < 0:
        raise ValueError("Exponents must be non-negative")

    if (ex % 2 == 1) or (ey % 2 == 1):
        return 0.0

    i_hi = 2 + ex + ey
    value = 0.0
    for i in range(i_hi + 1):
        sign = -1.0 if (i % 2 == 1) else 1.0
        # Binomial coefficient
        nchoosek = 1.0
        for j in range(1, i + 1):
            nchoosek *= (i_hi + 1 - j) / j
        value += sign * nchoosek / (i + ez + 1)

    value *= (2.0 / (ex + 1)) * (2.0 / (ey + 1))
    return float(value)


def pyramid_volume() -> float:
    """Volume of the unit pyramid = 4/3."""
    return 4.0 / 3.0


def integrate_over_pyramid(
    func: callable,
    n_sample: int = 8,
) -> float:
    """
    Monte-Carlo-like integration over the unit pyramid using uniform sampling.
    This serves as a verification for the exact monomial integrals.
    """
    rng = np.random.default_rng(seed=42)
    samples = []
    weights = []
    for _ in range(n_sample ** 3):
        z = rng.random()
        max_xy = 1.0 - z
        x = rng.uniform(-max_xy, max_xy)
        y = rng.uniform(-max_xy, max_xy)
        samples.append(func(x, y, z))
        weights.append(1.0)

    samples = np.array(samples)
    volume = pyramid_volume()
    return float(volume * np.mean(samples))


# ====================================================================== #
# Structural application: bearing pressure integration
# ====================================================================== #
def bearing_contact_force(
    pressure_func: callable,
    bearing_radius: float = 0.25,
    nr: int = 12,
    nt: int = 24,
) -> float:
    """
    Integrate contact pressure p(x, y) over a circular bearing pad to
    obtain the total vertical contact force.
    
    Parameters
    ----------
    pressure_func : callable
        p(x, y) [Pa], accepting scalar or array arguments.
    bearing_radius : float
        Radius of the bearing [m].
    nr, nt : int
        Quadrature resolution.
    
    Returns
    -------
    F_total : float
        Total contact force [N].
    """
    return integrate_over_disk(pressure_func, nr=nr, nt=nt, radius=bearing_radius)


# ====================================================================== #
# Structural application: consistent mass of pyramid element
# ====================================================================== #
def pyramid_consistent_mass(
    rho: float,
    base_area: float,
    height: float,
) -> np.ndarray:
    """
    Compute a symmetric positive-definite consistent mass matrix (5x5) for
    a linear pyramid finite element.
    
    The actual pyramid volume is V = base_area * height / 3.
    The total mass is m_total = rho * V.
    
    We construct M such that:
      (1) sum(M) = m_total   (total mass conservation)
      (2) M is symmetric positive definite
      (3) Diagonal dominance ensures stability
    
    Choosing:
      diagonal   a = m_total / 5 - 4*b
      off-diagonal b = m_total / 100
    satisfies sum(M) = 5a + 20b = m_total and a > 4b (positive definite).
    """
    V = base_area * height / 3.0
    m_total = rho * V
    b = m_total / 100.0
    a = m_total / 5.0 - 4.0 * b   # = 16 * m_total / 100

    M = np.full((5, 5), b, dtype=float)
    np.fill_diagonal(M, a)
    return M
