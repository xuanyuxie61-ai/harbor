# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import gamma
from typing import Tuple


def circle_rule_quadrature(n_theta: int) -> Tuple[np.ndarray, np.ndarray]:
    if n_theta < 1:
        raise ValueError("n_theta must be >= 1.")

    weights = np.ones(n_theta) / n_theta
    angles = 2.0 * np.pi * np.arange(n_theta) / n_theta
    return weights, angles


def sphere_monomial_integral(exponents: Tuple[int, int, int]) -> float:
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
    if n_theta < 1 or n_phi < 1:
        raise ValueError("n_theta and n_phi must be >= 1.")
    if R <= 0.0:
        raise ValueError("Radius R must be positive.")
    if not (0.0 <= p0 <= 1.0):
        raise ValueError("p0 must be in [0,1].")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")


    az = np.array([
        np.sin(active_zone_phi) * np.cos(active_zone_theta),
        np.sin(active_zone_phi) * np.sin(active_zone_theta),
        np.cos(active_zone_phi),
    ])


    theta = 2.0 * np.pi * np.arange(n_theta) / n_theta
    phi = np.pi * np.arange(1, n_phi + 1) / (n_phi + 1)

    dtheta = 2.0 * np.pi / n_theta
    dphi = np.pi / (n_phi + 1)

    P_release = 0.0
    for t in theta:
        for p in phi:

            sp = np.array([
                np.sin(p) * np.cos(t),
                np.sin(p) * np.sin(t),
                np.cos(p),
            ])

            cos_d = np.clip(np.dot(az, sp), -1.0, 1.0)
            d = np.arccos(cos_d)

            p_density = p0 * np.exp(-d ** 2 / (2.0 * sigma ** 2))

            P_release += p_density * (R ** 2) * np.sin(p) * dphi * dtheta

    return P_release


def integrate_release_probability_circle(
    n_points: int = 64,
    R: float = 0.5,
    p0: float = 0.3,
    sigma: float = 0.3,
    active_zone_angle: float = 0.0,
) -> float:
    if n_points < 1:
        raise ValueError("n_points must be >= 1.")
    if R <= 0.0:
        raise ValueError("R must be positive.")
    if not (0.0 <= p0 <= 1.0):
        raise ValueError("p0 must be in [0,1].")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")

    weights, angles = circle_rule_quadrature(n_points)


    d_angles = np.abs(angles - active_zone_angle)
    d_angles = np.minimum(d_angles, 2.0 * np.pi - d_angles)

    p_density = p0 * np.exp(-d_angles ** 2 / (2.0 * sigma ** 2))


    P_release = 2.0 * np.pi * R * np.sum(weights * p_density)

    return P_release


def compute_quantal_content(
    P_release: float,
    n_vesicles: int = 10,
) -> Tuple[float, float]:
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

    I000 = sphere_monomial_integral((0, 0, 0))
    print(f"Sphere area (a=b=c=0): {I000:.6f} (exact: 4π={4*np.pi:.6f})")

    I200 = sphere_monomial_integral((2, 0, 0))
    print(f"∫ x² dΩ: {I200:.6f}")

    P = integrate_release_probability_sphere()
    print(f"Spherical bouton release probability: {P:.6f}")

    Pc = integrate_release_probability_circle()
    print(f"Circular cross-section release probability: {Pc:.6f}")
