"""
Gaussian Vortex Initialization and Hermite Spectral Filtering
=============================================================
Derived from seed project 454_gaussian (Gaussian evaluation and
Hermite polynomial recursion).

Mesoscale eddies are often initialized as Gaussian vortices with
the streamfunction:

    ψ(r) = A · exp( −r² / (2σ²) )

where r² = (x−x₀)² + (y−y₀)². The corresponding vorticity is:

    ζ(r) = ∇²ψ = A/σ² · (r²/σ² − 2) · exp( −r² / (2σ²) )

and the azimuthal velocity:

    v_θ(r) = −∂ψ/∂r = (A·r/σ²) · exp( −r² / (2σ²) )

Hermite polynomials are used for spectral filtering of high-wavenumber
noise. The n-th derivative of a Gaussian is:

    dⁿ/dxⁿ G(x) = (−1)ⁿ / (σ√2)ⁿ · Heₙ( (x−μ)/(σ√2) ) · G(x)

where Heₙ(x) satisfies the recursion:
    He₀(x) = 1
    He₁(x) = x
    Heₙ(x) = x·Heₙ₋₁(x) − (n−1)·Heₙ₋₂(x)
"""

import numpy as np

def hermite_polynomial_value(n, x):
    """
    Evaluate the probabilists' Hermite polynomial He_n(x) using
    the three-term recurrence relation.

    Parameters
    ----------
    n : int
        Polynomial degree (≥ 0).
    x : ndarray
        Evaluation points.

    Returns
    -------
    He_n(x) : ndarray
    """
    if n < 0:
        raise ValueError("n must be non-negative.")
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()

    H_prev2 = np.ones_like(x)
    H_prev1 = x.copy()
    for k in range(2, n + 1):
        H_curr = x * H_prev1 - (k - 1) * H_prev2
        H_prev2 = H_prev1
        H_prev1 = H_curr
    return H_prev1

def gaussian_value(x, mu=0.0, sigma=1.0):
    """
    Evaluate the normalized Gaussian probability density:
        G(x) = 1/(σ√(2π)) · exp( −(x−μ)²/(2σ²) )
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    return np.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * np.sqrt(2.0 * np.pi))

def gaussian_derivative(x, n, mu=0.0, sigma=1.0):
    """
    Evaluate the n-th derivative of a Gaussian using Hermite polynomials:
        dⁿG/dxⁿ = (−1)ⁿ / (σ√2)ⁿ · Heₙ((x−μ)/(σ√2)) · G(x)
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    z = (x - mu) / (sigma * np.sqrt(2.0))
    G = gaussian_value(x, mu, sigma)
    Hn = hermite_polynomial_value(n, z)
    return ((-1.0)**n / (sigma * np.sqrt(2.0))**n) * Hn * G

def initialize_gaussian_vortex_2d(Nx, Ny, Lx, Ly, x0, y0, A, sigma,
                                  vorticity_sign=1.0):
    """
    Initialize a 2D Gaussian vortex on a periodic domain.

    Parameters
    ----------
    Nx, Ny : int
        Grid resolution.
    Lx, Ly : float
        Domain sizes.
    x0, y0 : float
        Vortex centre.
    A : float
        Streamfunction amplitude [m²/s].
    sigma : float
        Vortex radius [m].
    vorticity_sign : float
        +1 for cyclonic, −1 for anticyclonic.

    Returns
    -------
    psi : ndarray
        Streamfunction.
    zeta : ndarray
        Relative vorticity.
    """
    x = np.linspace(0, Lx, Nx, endpoint=False)
    y = np.linspace(0, Ly, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Periodic distance
    dx = X - x0
    dy = Y - y0
    dx = np.mod(dx + Lx/2, Lx) - Lx/2
    dy = np.mod(dy + Ly/2, Ly) - Ly/2
    r2 = dx**2 + dy**2

    psi = A * np.exp(-r2 / (2.0 * sigma**2))
    zeta = vorticity_sign * (A / sigma**2) * (r2 / sigma**2 - 2.0) * np.exp(-r2 / (2.0 * sigma**2))

    return psi, zeta

def hermite_spectral_filter(spec_field, KX, KY, k_cutoff, order=4):
    """
    Apply a Hermite-polynomial-inspired spectral filter:
        F̂(k) = exp( −α · (|k|/k_c)^{2q} ) · f̂(k)
    where q is the filter order (analogous to Hermite degree).

    This suppresses high-wavenumber noise while preserving smooth
    large-scale eddy structures.
    """
    k = np.sqrt(KX**2 + KY**2)
    alpha = np.log(2.0)
    filter_mask = np.exp(-alpha * (k / k_cutoff)**(2 * order))
    return spec_field * filter_mask
