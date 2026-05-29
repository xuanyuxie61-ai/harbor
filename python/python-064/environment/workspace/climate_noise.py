"""
Climate Noise Generation Module
===============================
Generates stochastic climate forcing using Gaussian Random Fields (GRF)
and other noise models.

Incorporates:
- Gaussian Random Field generation (from 489_grf_display)

Scientific Background:
----------------------
Climate variability contains both deterministic (orbital) and stochastic
(noise) components. Red noise (AR1) is characteristic of climate:
    x_t = phi * x_{t-1} + epsilon_t
where phi ~ 0.7-0.9 and epsilon_t ~ N(0, sigma^2).

Gaussian Random Fields on the sphere model spatially correlated noise:
    Cov(x, y) = sigma^2 * exp(-d(x,y)^2 / (2*L^2))
where d is geodesic distance and L is decorrelation length scale.
"""

import numpy as np


def generate_grf_1d(n_points, length_scale=0.1, sigma=1.0):
    """
    Generate 1D Gaussian Random Field with exponential covariance.

    Covariance: C(x_i, x_j) = sigma^2 * exp(-|x_i - x_j| / L)

    Parameters
    ----------
    n_points : int
        Number of points.
    length_scale : float
        Correlation length.
    sigma : float
        Standard deviation.

    Returns
    -------
    ndarray
        GRF values.
    """
    x = np.linspace(0, 1, n_points)
    # Build covariance matrix
    dx = np.abs(x[:, None] - x[None, :])
    C = sigma ** 2 * np.exp(-dx / length_scale)

    # Add small regularization
    C += 1e-10 * np.eye(n_points)

    # Cholesky decomposition
    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        # Fallback: use eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-10)
        L = eigvecs @ np.diag(np.sqrt(eigvals))

    z = np.random.randn(n_points)
    return L @ z


def generate_grf_spherical(n_lat=36, n_lon=72, length_scale_km=1000.0, sigma=1.0):
    """
    Generate Gaussian Random Field on a sphere.

    Covariance uses chord distance on unit sphere:
    C = sigma^2 * exp(-d^2 / (2*L^2))
    where d = 2*sin(delta/2), delta is angular separation.

    Parameters
    ----------
    n_lat, n_lon : int
        Grid dimensions.
    length_scale_km : float
        Decorrelation length scale in km.
    sigma : float
        Standard deviation.

    Returns
    -------
    ndarray, shape (n_lat, n_lon)
        GRF on spherical grid.
    """
    R = 6371.0  # Earth radius in km
    L_rad = length_scale_km / R

    lats = np.deg2rad(np.linspace(-90, 90, n_lat))
    lons = np.deg2rad(np.linspace(0, 360, n_lon))

    n = n_lat * n_lon
    points = np.zeros((n, 3))
    idx = 0
    for i in range(n_lat):
        for j in range(n_lon):
            points[idx] = [
                np.cos(lats[i]) * np.cos(lons[j]),
                np.cos(lats[i]) * np.sin(lons[j]),
                np.sin(lats[i])
            ]
            idx += 1

    # Compute covariance matrix
    C = np.zeros((n, n))
    for i in range(n):
        # Chord distances to all points
        dot = np.dot(points, points[i])
        dot = np.clip(dot, -1.0, 1.0)
        chord = np.sqrt(2.0 * (1.0 - dot))
        C[i, :] = sigma ** 2 * np.exp(-chord ** 2 / (2.0 * L_rad ** 2))

    C += 1e-10 * np.eye(n)

    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-10)
        L = eigvecs @ np.diag(np.sqrt(eigvals))

    z = np.random.randn(n)
    grf = L @ z
    return grf.reshape((n_lat, n_lon))


def ar1_noise(n, phi=0.85, sigma=1.0, x0=0.0):
    """
    Generate red noise (AR1 process).
    Typical climate red noise has phi ~ 0.7-0.9.

    Formula:
    x_t = phi * x_{t-1} + epsilon_t
    epsilon_t ~ N(0, sigma^2 * (1 - phi^2))

    Parameters
    ----------
    n : int
        Number of time steps.
    phi : float
        Autoregressive coefficient [0, 1].
    sigma : float
        Standard deviation of stationary distribution.
    x0 : float
        Initial value.

    Returns
    -------
    ndarray
        AR1 noise sequence.
    """
    phi = np.clip(phi, -0.999, 0.999)
    sigma_e = sigma * np.sqrt(1.0 - phi ** 2)
    x = np.zeros(n)
    x[0] = x0
    for t in range(1, n):
        x[t] = phi * x[t - 1] + np.random.randn() * sigma_e
    return x


def fbm_noise(n, hurst=0.8, sigma=1.0):
    """
    Generate fractional Brownian motion noise.
    Hurst exponent H > 0.5 indicates long-range persistence.

    Parameters
    ----------
    n : int
        Number of points.
    hurst : float
        Hurst exponent (0 < H < 1).
    sigma : float
        Standard deviation.

    Returns
    -------
    ndarray
        fBm sequence.
    """
    hurst = np.clip(hurst, 0.01, 0.99)
    # Circulant embedding method approximation
    k = np.arange(n)
    r = 0.5 * (np.abs(k + 1) ** (2 * hurst) + np.abs(k - 1) ** (2 * hurst) -
               2 * np.abs(k) ** (2 * hurst))
    r[0] = 1.0

    # FFT-based generation
    n_fft = 2 * n
    r_ext = np.zeros(n_fft)
    r_ext[0:n] = r
    r_ext[n_fft - n + 1:n_fft] = r[1:n][::-1]

    lambda_vals = np.real(np.fft.fft(r_ext))
    lambda_vals = np.maximum(lambda_vals, 0.0)

    z = np.random.randn(n_fft) + 1j * np.random.randn(n_fft)
    y = np.fft.ifft(np.sqrt(lambda_vals) * z)
    fbm = np.real(y[0:n])
    fbm = sigma * (fbm - np.mean(fbm)) / max(np.std(fbm), 1e-15)
    return fbm


def seasonal_noise(n_years, annual_amplitude=5.0, phase=0.0,
                   interannual_variability=1.0):
    """
    Generate realistic seasonal + interannual climate noise.

    Parameters
    ----------
    n_years : int
        Number of years.
    annual_amplitude : float
        Seasonal amplitude in K.
    phase : float
        Seasonal phase shift in radians.
    interannual_variability : float
        Interannual standard deviation in K.

    Returns
    -------
    ndarray
        Noise sequence (monthly resolution).
    """
    n_months = n_years * 12
    t = np.linspace(0, 2.0 * np.pi * n_years, n_months)

    # Seasonal cycle
    seasonal = annual_amplitude * np.sin(t + phase)

    # Interannual variability (AR1 with 1-year decorrelation)
    interannual = ar1_noise(n_months, phi=0.95, sigma=interannual_variability)

    return seasonal + interannual
