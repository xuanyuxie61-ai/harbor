# -*- coding: utf-8 -*-
"""
================================================================================
Synaptic Vesicle Release Probability Module
================================================================================

This module computes neurotransmitter release probabilities by integrating
over the geometry of presynaptic boutons, modeled as spheres and circles.

Biological Context:
-------------------
Presynaptic boutons are roughly spherical structures of radius R ≈ 0.5-1.0 μm.
Vesicle release probability depends on:
1. Ca²⁺ concentration at active zones (localized on the membrane)
2. Distance from Ca²⁺ channels to release sites
3. Surface curvature effects

Mathematical Model:
-------------------
The release probability density on the bouton surface S is:

    p(θ, φ) = p₀ · exp(-d² / 2σ²) · f(κ)

where:
    p₀   = baseline release probability
    d    = geodesic distance to nearest active zone
    σ    = Ca²⁺ spread parameter
    f(κ) = curvature correction factor
    κ    = local Gaussian curvature

For a sphere of radius R:
    κ = 1/R² (constant)

The total release probability is the surface integral:

    P_release = ∫_S p(θ, φ) dA = R² ∫_0^{2π} ∫_0^π p(θ, φ) sinθ dθ dφ

Quadrature Rules:
-----------------
1. Circle Rule: Uniform quadrature on the unit circle
   ∫_0^{2π} f(θ) dθ ≈ (2π/N) · Σ_{i=1}^N f(θ_i)
   where θ_i = 2π(i-1)/N

2. Sphere Monomial Integral: Exact integration of monomials x^a y^b z^c
   on the unit sphere using Gamma functions:

   If any exponent is odd: I = 0
   Else: I = 2 · Γ((a+1)/2) · Γ((b+1)/2) · Γ((c+1)/2) / Γ((a+b+c+3)/2)

3. Gauss-Legendre for elevation, uniform for azimuth.

================================================================================
"""

import numpy as np
from scipy.special import gamma
from typing import Tuple


def circle_rule_quadrature(n_theta: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a uniform quadrature rule for the unit circle.

    The quadrature approximates:

        ∫_0^{2π} f(θ) dθ ≈ 2π · Σ_i w_i · f(θ_i)

    with equal weights w_i = 1/N and nodes θ_i = 2π(i-1)/N.

    Parameters
    ----------
    n_theta : int
        Number of quadrature points. Must be >= 1.

    Returns
    -------
    weights : np.ndarray
        Quadrature weights, shape (n_theta,).
    angles : np.ndarray
        Quadrature angles, shape (n_theta,).
    """
    if n_theta < 1:
        raise ValueError("n_theta must be >= 1.")

    weights = np.ones(n_theta) / n_theta
    angles = 2.0 * np.pi * np.arange(n_theta) / n_theta
    return weights, angles


def sphere_monomial_integral(exponents: Tuple[int, int, int]) -> float:
    """
    Compute the exact integral of x^a y^b z^c over the unit sphere S².

    Formula:
    --------
    For monomial f(x,y,z) = x^a · y^b · z^c on x²+y²+z² = 1:

    If any of a, b, c is negative:
        Error (not integrable)

    If a = b = c = 0:
        I = 2·π^(3/2) / Γ(3/2) = 4π

    If any exponent is odd:
        I = 0   (by symmetry)

    Otherwise (all even):
        I = 2 · Γ((a+1)/2) · Γ((b+1)/2) · Γ((c+1)/2) / Γ((a+b+c+3)/2)

    Parameters
    ----------
    exponents : tuple of int
        (a, b, c) exponents for x, y, z.

    Returns
    -------
    integral : float
        Exact integral value.
    """
    a, b, c = exponents
    if a < 0 or b < 0 or c < 0:
        raise ValueError("All exponents must be non-negative.")

    if a == 0 and b == 0 and c == 0:
        return 2.0 * np.sqrt(np.pi ** 3) / gamma(1.5)

    if (a % 2 == 1) or (b % 2 == 1) or (c % 2 == 1):
        return 0.0

    integral = 2.0
    integral *= gamma(0.5 * (a + 1))
    integral *= gamma(0.5 * (b + 1))
    integral *= gamma(0.5 * (c + 1))
    integral /= gamma(0.5 * (a + b + c + 3))

    return float(integral)


def integrate_release_probability_sphere(
    n_theta: int = 32,
    n_phi: int = 16,
    R: float = 0.5,
    p0: float = 0.3,
    sigma: float = 0.2,
    active_zone_theta: float = 0.0,
    active_zone_phi: float = 0.0,
) -> float:
    """
    Compute the total vesicle release probability by integrating over
    a spherical bouton surface.

    The probability density is modeled as:

        p(θ, φ) = p₀ · exp(-d² / (2σ²))

    where d is the geodesic distance to the active zone.

    Parameters
    ----------
    n_theta : int
        Number of azimuthal points.
    n_phi : int
        Number of polar points.
    R : float
        Bouton radius [μm]. Must be positive.
    p0 : float
        Baseline release probability. Must be in [0,1].
    sigma : float
        Ca²⁺ spread parameter [rad].
    active_zone_theta : float
        Azimuthal angle of active zone [rad].
    active_zone_phi : float
        Polar angle of active zone [rad].

    Returns
    -------
    P_release : float
        Total integrated release probability.
    """
    if n_theta < 1 or n_phi < 1:
        raise ValueError("n_theta and n_phi must be >= 1.")
    if R <= 0.0:
        raise ValueError("Radius R must be positive.")
    if not (0.0 <= p0 <= 1.0):
        raise ValueError("p0 must be in [0,1].")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")

    # Convert active zone to Cartesian
    az = np.array([
        np.sin(active_zone_phi) * np.cos(active_zone_theta),
        np.sin(active_zone_phi) * np.sin(active_zone_theta),
        np.cos(active_zone_phi),
    ])

    # Quadrature points
    theta = 2.0 * np.pi * np.arange(n_theta) / n_theta
    phi = np.pi * np.arange(1, n_phi + 1) / (n_phi + 1)  # exclude poles

    dtheta = 2.0 * np.pi / n_theta
    dphi = np.pi / (n_phi + 1)

    P_release = 0.0
    for t in theta:
        for p in phi:
            # Surface point
            sp = np.array([
                np.sin(p) * np.cos(t),
                np.sin(p) * np.sin(t),
                np.cos(p),
            ])
            # Geodesic distance (angle between vectors)
            cos_d = np.clip(np.dot(az, sp), -1.0, 1.0)
            d = np.arccos(cos_d)

            p_density = p0 * np.exp(-d ** 2 / (2.0 * sigma ** 2))
            # Surface area element: R² sin(φ) dφ dθ
            P_release += p_density * (R ** 2) * np.sin(p) * dphi * dtheta

    return P_release


def integrate_release_probability_circle(
    n_points: int = 64,
    R: float = 0.5,
    p0: float = 0.3,
    sigma: float = 0.3,
    active_zone_angle: float = 0.0,
) -> float:
    """
    Compute release probability by integrating over a circular cross-section
    of the bouton (2D simplification).

    For a circle of radius R:

        P_release = R · ∫_0^{2π} p(θ) dθ

    Parameters
    ----------
    n_points : int
        Number of quadrature points.
    R : float
        Circle radius [μm].
    p0 : float
        Baseline probability.
    sigma : float
        Angular spread [rad].
    active_zone_angle : float
        Angle of active zone [rad].

    Returns
    -------
    P_release : float
        Integrated release probability.
    """
    if n_points < 1:
        raise ValueError("n_points must be >= 1.")
    if R <= 0.0:
        raise ValueError("R must be positive.")
    if not (0.0 <= p0 <= 1.0):
        raise ValueError("p0 must be in [0,1].")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")

    weights, angles = circle_rule_quadrature(n_points)

    # Angular distance (circular)
    d_angles = np.abs(angles - active_zone_angle)
    d_angles = np.minimum(d_angles, 2.0 * np.pi - d_angles)

    p_density = p0 * np.exp(-d_angles ** 2 / (2.0 * sigma ** 2))

    # Line integral on circle: ds = R dθ
    P_release = 2.0 * np.pi * R * np.sum(weights * p_density)

    return P_release


def compute_quantal_content(
    P_release: float,
    n_vesicles: int = 10,
) -> Tuple[float, float]:
    """
    Compute the expected quantal content and variance.

    Under the binomial model of vesicle release:

        E[Q] = N · P_release
        Var[Q] = N · P_release · (1 - P_release)

    where N is the readily releasable pool (RRP) size.

    Parameters
    ----------
    P_release : float
        Release probability per vesicle. Must be in [0,1].
    n_vesicles : int
        Number of vesicles in RRP. Must be >= 0.

    Returns
    -------
    mean_q : float
        Expected quantal content.
    var_q : float
        Variance of quantal content.
    """
    if not (0.0 <= P_release <= 1.0):
        raise ValueError("P_release must be in [0,1].")
    if n_vesicles < 0:
        raise ValueError("n_vesicles must be non-negative.")

    mean_q = n_vesicles * P_release
    var_q = n_vesicles * P_release * (1.0 - P_release)
    return mean_q, var_q


def simulate_vesicle_release_batch(
    n_boutons: int = 20,
    R_mean: float = 0.5,
    R_std: float = 0.1,
    p0_mean: float = 0.25,
    p0_std: float = 0.05,
) -> dict:
    """
    Simulate vesicle release across a population of boutons with
    heterogeneous sizes and baseline probabilities.

    Parameters
    ----------
    n_boutons : int
        Number of boutons.
    R_mean : float
        Mean bouton radius [μm].
    R_std : float
        Std dev of radius.
    p0_mean : float
        Mean baseline probability.
    p0_std : float
        Std dev of baseline probability.

    Returns
    -------
    results : dict
        Dictionary with 'P_sphere', 'P_circle', 'mean_q', 'var_q' arrays.
    """
    if n_boutons < 1:
        raise ValueError("n_boutons must be >= 1.")
    if R_mean <= 0.0 or R_std < 0.0:
        raise ValueError("Invalid radius parameters.")
    if not (0.0 <= p0_mean <= 1.0) or p0_std < 0.0:
        raise ValueError("Invalid probability parameters.")

    rng = np.random.default_rng(seed=42)
    R_vals = np.clip(rng.normal(R_mean, R_std, n_boutons), 0.1, 2.0)
    p0_vals = np.clip(rng.normal(p0_mean, p0_std, n_boutons), 0.01, 0.99)

    P_sphere = np.zeros(n_boutons)
    P_circle = np.zeros(n_boutons)
    mean_q = np.zeros(n_boutons)
    var_q = np.zeros(n_boutons)

    for i in range(n_boutons):
        P_sphere[i] = integrate_release_probability_sphere(
            n_theta=16, n_phi=8, R=R_vals[i], p0=p0_vals[i]
        )
        P_circle[i] = integrate_release_probability_circle(
            n_points=32, R=R_vals[i], p0=p0_vals[i]
        )
        mean_q[i], var_q[i] = compute_quantal_content(P_sphere[i])

    return {
        "P_sphere": P_sphere,
        "P_circle": P_circle,
        "mean_q": mean_q,
        "var_q": var_q,
        "R_vals": R_vals,
        "p0_vals": p0_vals,
    }


if __name__ == "__main__":
    # Test sphere monomial integral
    I000 = sphere_monomial_integral((0, 0, 0))
    print(f"Sphere area (a=b=c=0): {I000:.6f} (exact: 4π={4*np.pi:.6f})")

    I200 = sphere_monomial_integral((2, 0, 0))
    print(f"∫ x² dΩ: {I200:.6f}")

    P = integrate_release_probability_sphere()
    print(f"Spherical bouton release probability: {P:.6f}")

    Pc = integrate_release_probability_circle()
    print(f"Circular cross-section release probability: {Pc:.6f}")
