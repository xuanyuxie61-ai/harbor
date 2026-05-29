"""
turbulence_statistics.py
========================
Turbulent Scalar Statistics and Probability Density Functions for DNS Data.

Based on seed projects:
  699 (log_normal_truncated_ab) - Truncated log-normal PDF for scalar fluctuations
  178 (circle_distance)           - Monte Carlo sampling on iso-surfaces

Scientific Context:
-------------------
In turbulent combustion, scalar fields (mixture fraction Z, progress variable c)
exhibit strong fluctuations. The probability density function (PDF) P(Z) describes
the probability of finding a fluid parcel with mixture fraction Z.

The mixture fraction Z is typically bounded in [0,1] and its PDF is often
modeled by a beta function or a log-normal distribution truncated to [0,1].

For a log-normal variable X ~ LN(μ, σ²), the PDF is:
  f(x; μ, σ) = 1/(x σ √(2π)) * exp( -(ln x - μ)² / (2σ²) ),  x > 0

The truncated log-normal PDF on [a,b] is:
  f_{trunc}(x; μ, σ, a, b) = f(x; μ, σ) / (Φ((ln b - μ)/σ) - Φ((ln a - μ)/σ))

where Φ is the standard normal CDF.

The mean and variance of the truncated log-normal are:
  E[X] = exp(μ + σ²/2) * [Φ(β - σ) - Φ(α - σ)] / [Φ(β) - Φ(α)]
  Var[X] = E[X²] - E[X]²
  where α = (ln a - μ)/σ, β = (ln b - μ)/σ

Scalar dissipation rate χ = 2D |∇Z|² [1/s] also follows a log-normal distribution.

Circle Sampling for Flame Curvature:
-------------------------------------
For iso-surface analysis, we sample points on circles in the plane perpendicular
to the flame normal to compute local curvature κ:
  κ = ∇ · n̂ = ∇ · (∇c/|∇c|)

The mean curvature from N circle samples:
  <κ> = (1/N) Σ_i κ(θ_i)
"""

import numpy as np
from math import erf, sqrt, exp, log, pi

SQRT2 = sqrt(2.0)
SQRT2PI = sqrt(2.0 * pi)


def normal_cdf(x):
    """Standard normal CDF: Φ(x) = 0.5 * [1 + erf(x/√2)]."""
    return 0.5 * (1.0 + erf(x / SQRT2))


def normal_cdf_inv(p):
    """Inverse standard normal CDF using rational approximation."""
    if p <= 0.0:
        return -1e10
    if p >= 1.0:
        return 1e10
    # Acklam's approximation
    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01
    a6 = 2.506628277459239e+00
    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01
    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e+00
    c4 = -2.549732539343734e+00
    c5 = 4.374664141464968e+00
    c6 = 2.938163982698783e+00
    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e+00
    d4 = 3.754408661907416e+00
    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = sqrt(-2.0 * log(p))
        x = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        x = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    else:
        q = sqrt(-2.0 * log(1.0 - p))
        x = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    return x


def log_normal_pdf(x, mu, sigma):
    """Log-normal PDF: f(x; μ, σ)."""
    if x <= 0.0:
        return 0.0
    return exp(-0.5 * ((log(x) - mu) / sigma) ** 2) / (x * sigma * SQRT2PI)


def log_normal_cdf(x, mu, sigma):
    """Log-normal CDF: F(x; μ, σ) = Φ((ln x - μ)/σ)."""
    if x <= 0.0:
        return 0.0
    return normal_cdf((log(x) - mu) / sigma)


def log_normal_truncated_ab_pdf(x, mu, sigma, a, b):
    """
    Truncated log-normal PDF on [a,b].
    Based on seed 699 (log_normal_truncated_ab_pdf.m).
    """
    if x <= a or x >= b:
        return 0.0
    cdf_a = log_normal_cdf(a, mu, sigma)
    cdf_b = log_normal_cdf(b, mu, sigma)
    denom = cdf_b - cdf_a
    if denom < 1e-30:
        return 0.0
    return log_normal_pdf(x, mu, sigma) / denom


def log_normal_truncated_ab_sample(mu, sigma, a, b):
    """
    Sample from truncated log-normal on [a,b] via inverse transform.
    Based on seed 699 (log_normal_truncated_ab_sample.m).
    """
    cdf_a = log_normal_cdf(a, mu, sigma)
    cdf_b = log_normal_cdf(b, mu, sigma)
    u = np.random.uniform(0.0, 1.0)
    cdf = cdf_a + u * (cdf_b - cdf_a)
    # x = exp(μ + σ * Φ^{-1}(cdf))
    z = normal_cdf_inv(cdf)
    return exp(mu + sigma * z)


def log_normal_truncated_ab_mean(mu, sigma, a, b):
    """Mean of truncated log-normal distribution."""
    alpha = (log(a) - mu) / sigma if a > 0 else -1e10
    beta = (log(b) - mu) / sigma if b > 0 else -1e10
    phi_alpha = normal_cdf(alpha)
    phi_beta = normal_cdf(beta)
    phi_diff = phi_beta - phi_alpha
    if phi_diff < 1e-30:
        return 0.0
    return exp(mu + 0.5 * sigma**2) * (normal_cdf(beta - sigma) - normal_cdf(alpha - sigma)) / phi_diff


def log_normal_truncated_ab_variance(mu, sigma, a, b):
    """Variance of truncated log-normal distribution."""
    alpha = (log(a) - mu) / sigma if a > 0 else -1e10
    beta = (log(b) - mu) / sigma if b > 0 else -1e10
    phi_alpha = normal_cdf(alpha)
    phi_beta = normal_cdf(beta)
    phi_diff = phi_beta - phi_alpha
    if phi_diff < 1e-30:
        return 0.0
    mean = log_normal_truncated_ab_mean(mu, sigma, a, b)
    e2 = exp(2.0 * mu + 2.0 * sigma**2)
    term2 = (normal_cdf(beta - 2.0 * sigma) - normal_cdf(alpha - 2.0 * sigma)) / phi_diff
    return e2 * term2 - mean**2


def fit_truncated_log_normal_to_data(data, a=0.0, b=1.0):
    """
    Fit truncated log-normal parameters (μ, σ) to data using method of moments.
    """
    data = np.asarray(data)
    data = data[(data > a) & (data < b)]
    if len(data) < 3:
        return 0.0, 1.0
    # Use log-transformed data statistics
    log_data = np.log(np.clip(data, 1e-12, 1.0))
    mu_est = np.mean(log_data)
    sigma_est = np.std(log_data)
    sigma_est = max(sigma_est, 0.01)
    return mu_est, sigma_est


def sample_circle_unit(n):
    """
    Sample n points uniformly on the unit circle.
    Based on seed 178 (circle_unit_sample.m).
    """
    theta = 2.0 * np.pi * np.random.rand(n)
    x = np.cos(theta)
    y = np.sin(theta)
    return x, y


def estimate_flame_curvature_from_circle_samples(c_field, dx, dy, n_samples=16):
    """
    Estimate mean flame curvature by sampling on circles around high-gradient points.
    Based on seed 178 (circle_distance_stats.m) — statistics from random samples.

    Curvature κ = ∇ · n̂ where n̂ = ∇c / |∇c|.
    """
    # Compute gradient
    dcdx = np.gradient(c_field, axis=0) / dx
    dcdy = np.gradient(c_field, axis=1) / dy
    grad_mag = np.sqrt(dcdx**2 + dcdy**2)

    # Find flame front (max gradient locations)
    threshold = 0.3 * np.max(grad_mag)
    flame_mask = grad_mag > threshold
    if not np.any(flame_mask):
        return 0.0, 0.0

    indices = np.argwhere(flame_mask)
    if len(indices) > 100:
        indices = indices[np.random.choice(len(indices), 100, replace=False)]

    curvatures = []
    for ix, iy in indices:
        if ix <= 0 or ix >= c_field.shape[0] - 1 or iy <= 0 or iy >= c_field.shape[1] - 1:
            continue
        # Sample circle around this point
        cx, cy = sample_circle_unit(n_samples)
        # Interpolate c at circle points (simple bilinear)
        local_curv = []
        for k in range(n_samples):
            # Position on circle with small radius
            px = ix + 0.5 * cx[k]
            py = iy + 0.5 * cy[k]
            ix0, iy0 = int(px), int(py)
            ix1, iy1 = min(ix0 + 1, c_field.shape[0] - 1), min(iy0 + 1, c_field.shape[1] - 1)
            wx = px - ix0
            wy = py - iy0
            c_val = ((1 - wx) * (1 - wy) * c_field[ix0, iy0]
                     + wx * (1 - wy) * c_field[ix1, iy0]
                     + (1 - wx) * wy * c_field[ix0, iy1]
                     + wx * wy * c_field[ix1, iy1])
            local_curv.append(c_val)
        if len(local_curv) > 0:
            # Curvature ≈ (c_center - mean_circle) / r²
            r = 0.5 * dx
            kappa = (c_field[ix, iy] - np.mean(local_curv)) / (r**2)
            curvatures.append(kappa)

    if len(curvatures) == 0:
        return 0.0, 0.0
    return np.mean(curvatures), np.var(curvatures)


def scalar_dissipation_rate(Z, D, dx, dy):
    """
    Compute scalar dissipation rate χ = 2D |∇Z|².
    """
    dZdx = np.gradient(Z, axis=0) / dx
    dZdy = np.gradient(Z, axis=1) / dy
    chi = 2.0 * D * (dZdx**2 + dZdy**2)
    return chi


def turbulent_kinetic_energy_spectrum(u, v, dx, dy):
    """
    Compute 1D energy spectrum E(k) from velocity fields via FFT.
    E(k) = 0.5 * (|û(k)|² + |v̂(k)|²) / (dx * dy)
    """
    u_hat = np.fft.fftn(u)
    v_hat = np.fft.fftn(v)
    nx, ny = u.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    k_mag = np.sqrt(KX**2 + KY**2)

    energy = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2) / (nx * ny)

    # Bin by wavenumber magnitude
    k_max = np.max(k_mag)
    n_bins = max(nx, ny) // 2
    bins = np.linspace(0, k_max, n_bins + 1)
    E_k = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (k_mag >= bins[i]) & (k_mag < bins[i + 1])
        if np.any(mask):
            E_k[i] = np.sum(energy[mask])

    return bins[:-1], E_k
