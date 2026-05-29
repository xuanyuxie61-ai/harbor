"""
SED Triangle Integration Module
===============================
Based on seed project 1308_triangle_integrands:
- p00_fun.m, p00_monte_carlo.m, triangle_area.m, triangle_sample.m

Physics:
--------
The spectral energy distribution (SED) of GRB afterglow emission
requires integration over the electron Lorentz factor γ and the
pitch angle θ between the electron velocity and the magnetic field.
The domain of integration is a triangle in the (γ, θ) plane:

    T = { (γ, θ) : γ_min ≤ γ ≤ γ_max,
                  θ_min ≤ θ ≤ θ_max,
                  θ ≤ θ_max - (θ_max - θ_min)(γ - γ_min)/(γ_max - γ_min) }

The synchrotron emissivity for a single electron is:

    j_ν(γ,θ) = (√3 e³ B sin θ) / (4π m_e c²)
               · F(ν/ν_c)

where the critical frequency is:

    ν_c(γ) = (3 e B sin θ) / (4π m_e c) · γ²

and F(x) is the synchrotron function:

    F(x) = x · ∫_x^{∞} K_{5/3}(ξ) dξ

The total flux at observed frequency ν is:

    F_ν = ∫_T j_ν(γ,θ) N(γ) dγ dθ

We evaluate this via Monte Carlo integration over the triangle,
sampling uniformly in barycentric coordinates.
"""

import numpy as np


def triangle_area(vertices):
    """
    Computes the area of a triangle with vertices V0, V1, V2.

        A = ½ | (V1 - V0) × (V2 - V0) |

    Parameters
    ----------
    vertices : ndarray, shape (3, 2)
        Triangle vertices.

    Returns
    -------
    area : float
        Triangle area.
    """
    v0 = vertices[0]
    v1 = vertices[1]
    v2 = vertices[2]
    area = 0.5 * abs((v1[0] - v0[0]) * (v2[1] - v0[1])
                     - (v2[0] - v0[0]) * (v1[1] - v0[1]))
    return area


def triangle_sample(vertices, n):
    """
    Generates N uniformly distributed random points inside a triangle
    using barycentric coordinates.

    For random r1, r2 ∈ [0,1]:
        λ1 = 1 - √r1
        λ2 = √r1 (1 - r2)
        λ3 = √r1 · r2
        P = λ1 V0 + λ2 V1 + λ3 V2

    Parameters
    ----------
    vertices : ndarray, shape (3, 2)
        Triangle vertices.
    n : int
        Number of sample points.

    Returns
    -------
    p : ndarray, shape (2, n)
        Sampled points.
    """
    v0 = vertices[0, :].reshape(2, 1)
    v1 = vertices[1, :].reshape(2, 1)
    v2 = vertices[2, :].reshape(2, 1)

    r1 = np.random.rand(n)
    r2 = np.random.rand(n)

    sqrt_r1 = np.sqrt(r1)
    lam1 = 1.0 - sqrt_r1
    lam2 = sqrt_r1 * (1.0 - r2)
    lam3 = sqrt_r1 * r2

    p = lam1 * v0 + lam2 * v1 + lam3 * v2
    return p


def synchrotron_function_approx(x):
    """
    Approximation to the synchrotron function F(x) using the
    Padé-like approximation (Crusius & Schlickeiser 1986):

        F(x) ≈ 1.808 · x^{1/3} · exp(-x)   for x < 1e-3
        F(x) ≈ √(πx/2) · exp(-x)           for x > 10
        F(x) ≈ polynomial fit               intermediate
    """
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)

    # Small-x asymptote
    mask_small = x < 1e-3
    if np.any(mask_small):
        result[mask_small] = 1.808 * x[mask_small] ** (1.0 / 3.0) * np.exp(-x[mask_small])

    # Large-x asymptote
    mask_large = x > 10.0
    if np.any(mask_large):
        result[mask_large] = np.sqrt(np.pi * x[mask_large] / 2.0) * np.exp(-x[mask_large])

    # Intermediate: polynomial fit
    mask_mid = ~(mask_small | mask_large)
    if np.any(mask_mid):
        xi = x[mask_mid]
        # Rational approximation
        result[mask_mid] = (1.808 * xi ** (1.0 / 3.0)
                            * np.exp(-xi)
                            * (1.0 + 0.16 * xi ** (2.0 / 3.0))
                            / (1.0 + 0.53 * xi ** (2.0 / 3.0)))

    return result


def integrand_sed(gamma, theta, nu_obs, B, N_gamma):
    """
    SED integrand for synchrotron emission from a power-law
    electron distribution N(γ) = N₀ γ^{-p}.

    j_ν(γ,θ) ∝ B sin θ · F(ν/ν_c) · γ^{-p}

    Parameters
    ----------
    gamma, theta : ndarray
        Electron Lorentz factor and pitch angle (rad).
    nu_obs : float
        Observed frequency (Hz).
    B : float
        Magnetic field (Gauss).
    N_gamma : float
        Normalization of electron distribution.

    Returns
    -------
    f : ndarray
        Integrand values.
    """
    m_e = 9.10938356e-28
    c = 2.99792458e10
    e = 4.80320427e-10
    p_index = 2.5

    sin_theta = np.sin(theta)
    sin_theta = np.clip(sin_theta, 1e-6, 1.0)

    nu_c = (3.0 * e * B * sin_theta) / (4.0 * np.pi * m_e * c) * gamma ** 2
    nu_c = np.clip(nu_c, 1e-20, None)

    x = nu_obs / nu_c
    F_x = synchrotron_function_approx(x)

    prefactor = (np.sqrt(3.0) * e ** 3 * B * sin_theta) / (4.0 * np.pi * m_e * c ** 2)
    n_e = N_gamma * gamma ** (-p_index)

    f = prefactor * F_x * n_e
    f = np.clip(f, 0.0, 1e200)
    return f


def monte_carlo_sed(vertices, n_samples, nu_obs, B, N_gamma):
    """
    Monte Carlo integration of the SED over a triangular domain.

        F_ν ≈ A · (1/N) Σ_i f(P_i)

    Parameters
    ----------
    vertices : ndarray, shape (3, 2)
        Triangle vertices in (γ, θ) space.
    n_samples : int
        Number of Monte Carlo samples.
    nu_obs : float
        Observed frequency.
    B : float
        Magnetic field.
    N_gamma : float
        Electron distribution normalization.

    Returns
    -------
    flux : float
        Integrated flux.
    std_err : float
        Standard error estimate.
    """
    area = triangle_area(vertices)
    p = triangle_sample(vertices, n_samples)
    gamma = p[0, :]
    theta = p[1, :]

    # Robustness: clip to physical ranges
    gamma = np.clip(gamma, 1.0, 1e12)
    theta = np.clip(theta, 1e-6, np.pi - 1e-6)

    f_vals = integrand_sed(gamma, theta, nu_obs, B, N_gamma)

    mean_f = np.mean(f_vals)
    std_f = np.std(f_vals, ddof=1)

    flux = area * mean_f
    std_err = area * std_f / np.sqrt(n_samples)
    return flux, std_err
