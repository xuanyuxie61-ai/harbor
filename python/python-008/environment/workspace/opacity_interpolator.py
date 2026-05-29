"""
Opacity Interpolation Module
============================
Based on seed project 927_pwl_interp_2d:
- pwl_interp_2d.m  →  piecewise linear 2D interpolation

Physics:
--------
In the GRB afterglow, the Rosseland mean opacity κ_R and the
Planck mean opacity κ_P depend on the local mass density ρ and
temperature T:

    κ_R = κ_R(ρ, T)   [cm² g⁻¹]
    κ_P = κ_P(ρ, T)   [cm² g⁻¹]

The radiative diffusion flux is:

    F_rad = - (c / (3 κ_R ρ)) ∇(a T⁴)

where a is the radiation density constant.  For a relativistic
plasma, the electron-scattering (Thomson) opacity is:

    κ_es = σ_T / m_p ≈ 0.34 cm² g⁻¹

and the free-free (Kramers) opacity scales as:

    κ_ff ∝ ρ T^{-7/2}

This module provides piecewise-linear interpolation on a structured
2D grid of (log ρ, log T), splitting each rectangular cell into
two triangles for barycentric interpolation.
"""

import numpy as np


def r8vec_bracket5(nxd, xd, xq):
    """
    Locate the interval [xd[i], xd[i+1]] containing xq.

    Returns
    -------
    i : int
        Interval index, or -1 if out of bounds.
    """
    xd = np.asarray(xd, dtype=float)
    if xq < xd[0] or xq > xd[-1]:
        return -1
    i = int(np.searchsorted(xd, xq, side='right')) - 1
    i = max(0, min(i, nxd - 2))
    return i


def pwl_interp_2d_scalar(nxd, nyd, xd, yd, zd, xi, yi):
    """
    Piecewise-linear interpolation at a single point (xi, yi).

    The rectangle [xd[i], xd[i+1]] × [yd[j], yd[j+1]] is split into
    two triangles:

       (i,j+1)---(i+1,j+1)
          | \       |
          |  \  T2  |
          |   \     |
          | T1 \    |
       (i,j)-----(i+1,j)

    Barycentric coordinates (α, β, γ) within each triangle give:

        z = α z_a + β z_b + γ z_c

    Parameters
    ----------
    nxd, nyd : int
        Grid dimensions.
    xd, yd : ndarray
        1D sorted coordinate arrays.
    zd : ndarray, shape (nxd, nyd)
        Data values.
    xi, yi : float
        Query point.

    Returns
    -------
    zi : float
        Interpolated value, or np.inf if out of bounds.
    """
    i = r8vec_bracket5(nxd, xd, xi)
    if i == -1:
        return np.inf
    j = r8vec_bracket5(nyd, yd, yi)
    if j == -1:
        return np.inf

    # Diagonal splitting
    y_diag = yd[j + 1] + (yd[j] - yd[j + 1]) * (xi - xd[i]) / (xd[i + 1] - xd[i])

    if yi < y_diag:
        # Lower-left triangle: vertices (i+1,j), (i,j+1), (i,j)
        dxa = xd[i + 1] - xd[i]
        dya = yd[j] - yd[j]
        dxb = xd[i] - xd[i]
        dyb = yd[j + 1] - yd[j]
        dxi = xi - xd[i]
        dyi = yi - yd[j]

        det = dxa * dyb - dya * dxb
        if abs(det) < 1e-15:
            return np.inf

        alpha = (dxi * dyb - dyi * dxb) / det
        beta = (dxa * dyi - dya * dxi) / det
        gamma = 1.0 - alpha - beta

        zi = alpha * zd[i + 1, j] + beta * zd[i, j + 1] + gamma * zd[i, j]
    else:
        # Upper-right triangle: vertices (i,j+1), (i+1,j), (i+1,j+1)
        dxa = xd[i] - xd[i + 1]
        dya = yd[j + 1] - yd[j + 1]
        dxb = xd[i + 1] - xd[i + 1]
        dyb = yd[j] - yd[j + 1]
        dxi = xi - xd[i + 1]
        dyi = yi - yd[j + 1]

        det = dxa * dyb - dya * dxb
        if abs(det) < 1e-15:
            return np.inf

        alpha = (dxi * dyb - dyi * dxb) / det
        beta = (dxa * dyi - dya * dxi) / det
        gamma = 1.0 - alpha - beta

        zi = alpha * zd[i, j + 1] + beta * zd[i + 1, j] + gamma * zd[i + 1, j + 1]

    return zi


def pwl_interp_2d(nxd, nyd, xd, yd, zd, ni, xi, yi):
    """
    Vectorized piecewise-linear 2D interpolation.

    Parameters
    ----------
    nxd, nyd : int
        Grid dimensions.
    xd, yd : ndarray
        Sorted 1D coordinate arrays.
    zd : ndarray, shape (nxd, nyd)
        Data values.
    ni : int
        Number of interpolation points.
    xi, yi : ndarray, shape (ni,)
        Query coordinates.

    Returns
    -------
    zi : ndarray, shape (ni,)
        Interpolated values.
    """
    zi = np.full(ni, np.inf, dtype=float)
    for k in range(ni):
        zi[k] = pwl_interp_2d_scalar(nxd, nyd, xd, yd, zd, xi[k], yi[k])
    return zi


def build_opacity_table(n_rho=16, n_T=16):
    """
    Build a synthetic opacity table κ(ρ, T) for a relativistic
    GRB afterglow plasma.

    Combines electron scattering and free-free opacities:

        κ = κ_es + κ_ff
        κ_es = σ_T / m_p  (constant)
        κ_ff = 0.64 × 10²³ (ρ [g/cm³]) (T [K])^{-7/2} Z² g_ff

    Returns
    -------
    log_rho : ndarray
        log10 density bins.
    log_T : ndarray
        log10 temperature bins.
    kappa : ndarray, shape (n_rho, n_T)
        Opacity in cm² g⁻¹.
    """
    log_rho = np.linspace(-20, -10, n_rho)
    log_T = np.linspace(4, 9, n_T)

    sigma_T = 6.6524587158e-25  # cm²
    m_p = 1.6726219e-24         # g
    kappa_es = sigma_T / m_p    # ≈ 0.398 cm²/g

    rho = 10.0 ** log_rho.reshape(-1, 1)
    T = 10.0 ** log_T.reshape(1, -1)

    # Kramers free-free (approximate)
    kappa_ff = 0.64e23 * rho * T ** (-3.5)
    kappa_ff = np.clip(kappa_ff, 0.0, 1e4)

    kappa = kappa_es + kappa_ff
    kappa = np.clip(kappa, 1e-4, 1e4)
    return log_rho, log_T, kappa


def interpolate_opacity(rho_query, T_query, log_rho, log_T, kappa_table):
    """
    Interpolate opacity at query points (rho, T).
    """
    log_rho_q = np.log10(np.clip(rho_query, 1e-30, None))
    log_T_q = np.log10(np.clip(T_query, 1.0, None))

    n = log_rho_q.size
    kappa = pwl_interp_2d(log_rho.size, log_T.size,
                          log_rho, log_T, kappa_table,
                          n, log_rho_q, log_T_q)

    # Replace infinities with nearest edge values
    kappa = np.where(np.isinf(kappa), kappa_table.mean(), kappa)
    return kappa
