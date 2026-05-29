"""
Spectrum Interpolation Module
=============================
Based on seed project 590_interp:
- interp_lagrange.m  →  Lagrange polynomial interpolation
- lagrange_value.m   →  Lagrange basis evaluation
- interp_linear.m    →  linear interpolation
- interp_nearest.m   →  nearest-neighbor interpolation

Physics:
--------
In GRB afterglow modeling, observed spectra are discretized into
energy bins.  To evaluate the specific intensity I_ν at arbitrary
frequencies, we employ multi-method interpolation:

    Lagrange interpolation:
        L_i(ν) = ∏_{j≠i} (ν - ν_j) / (ν_i - ν_j)
        I_ν = Σ_i I_i · L_i(ν)

    Piecewise-linear interpolation (monotonicity-preserving):
        I_ν = I_k + (I_{k+1} - I_k) · (ν - ν_k) / (ν_{k+1} - ν_k)

The Compton-y parameter for inverse-Compton scattering is:

    y = ∫ (4kT_e / (m_e c²)) · σ_T n_e dl

and the interpolated photon occupation number n(ν) enters the
Kompaneets equation:

    ∂n/∂y = (1/x²) ∂/∂x [ x⁴ (n + n² + ∂n/∂x) ]

where x = hν / (kT_e).
"""

import numpy as np


def lagrange_value(data_num, t_data, interp_num, t_interp):
    """
    Evaluates Lagrange polynomials L_i(T) for i = 1..data_num at
    interpolation points t_interp.

        L_i(T) = ∏_{j≠i} (T - T_j) / (T_i - T_j)

    Parameters
    ----------
    data_num : int
        Number of data points.
    t_data : ndarray, shape (data_num,)
        Abscissas (must be distinct).
    interp_num : int
        Number of interpolation points.
    t_interp : ndarray, shape (interp_num,)
        Points at which to evaluate.

    Returns
    -------
    l_interp : ndarray, shape (data_num, interp_num)
        Lagrange basis values.
    """
    t_data = np.asarray(t_data, dtype=float)
    t_interp = np.asarray(t_interp, dtype=float)

    l_interp = np.ones((data_num, interp_num), dtype=float)

    for i in range(data_num):
        for j in range(data_num):
            if j != i:
                denom = t_data[i] - t_data[j]
                if abs(denom) < 1e-15:
                    denom = 1e-15
                l_interp[i, :] *= (t_interp - t_data[j]) / denom

    return l_interp


def interp_lagrange(m, data_num, t_data, p_data, interp_num, t_interp):
    """
    Lagrange polynomial interpolation for M-dimensional curve data.

    Parameters
    ----------
    m : int
        Spatial dimension of dependent variable.
    data_num : int
        Number of data points.
    t_data : ndarray, shape (data_num,)
        Independent variable samples.
    p_data : ndarray, shape (m, data_num)
        Dependent variable samples.
    interp_num : int
        Number of interpolation points.
    t_interp : ndarray, shape (interp_num,)
        Independent variable at which to interpolate.

    Returns
    -------
    p_interp : ndarray, shape (m, interp_num)
        Interpolated values.
    """
    l_interp = lagrange_value(data_num, t_data, interp_num, t_interp)
    p_interp = p_data @ l_interp
    return p_interp


def interp_linear(t_data, p_data, t_interp):
    """
    Piecewise-linear interpolation with robust boundary handling.

    Parameters
    ----------
    t_data : ndarray, shape (n,)
        Must be strictly increasing.
    p_data : ndarray, shape (m, n) or (n,)
        Dependent data.
    t_interp : ndarray, shape (k,)
        Query points.

    Returns
    -------
    p_interp : ndarray
        Interpolated values, same leading dims as p_data.
    """
    t_data = np.asarray(t_data, dtype=float)
    p_data = np.asarray(p_data, dtype=float)
    t_interp = np.asarray(t_interp, dtype=float)

    single_dim = (p_data.ndim == 1)
    if single_dim:
        p_data = p_data.reshape(1, -1)

    m, n = p_data.shape
    k = t_interp.size
    p_interp = np.zeros((m, k), dtype=float)

    for idx in range(k):
        tq = t_interp[idx]
        # Bracket search
        if tq <= t_data[0]:
            p_interp[:, idx] = p_data[:, 0]
        elif tq >= t_data[-1]:
            p_interp[:, idx] = p_data[:, -1]
        else:
            i = np.searchsorted(t_data, tq, side='right') - 1
            i = max(0, min(i, n - 2))
            dt = t_data[i + 1] - t_data[i]
            if abs(dt) < 1e-15:
                p_interp[:, idx] = p_data[:, i]
            else:
                w = (tq - t_data[i]) / dt
                p_interp[:, idx] = (1.0 - w) * p_data[:, i] + w * p_data[:, i + 1]

    if single_dim:
        return p_interp.reshape(-1)
    return p_interp


def interp_nearest(t_data, p_data, t_interp):
    """
    Nearest-neighbor interpolation.
    """
    t_data = np.asarray(t_data, dtype=float)
    p_data = np.asarray(p_data, dtype=float)
    t_interp = np.asarray(t_interp, dtype=float)

    single_dim = (p_data.ndim == 1)
    if single_dim:
        p_data = p_data.reshape(1, -1)

    m, n = p_data.shape
    k = t_interp.size
    p_interp = np.zeros((m, k), dtype=float)

    for idx in range(k):
        tq = t_interp[idx]
        if tq <= t_data[0]:
            j = 0
        elif tq >= t_data[-1]:
            j = n - 1
        else:
            i = np.searchsorted(t_data, tq, side='right') - 1
            if i < 0:
                j = 0
            elif i >= n - 1:
                j = n - 1
            else:
                if abs(tq - t_data[i]) <= abs(tq - t_data[i + 1]):
                    j = i
                else:
                    j = i + 1
        p_interp[:, idx] = p_data[:, j]

    if single_dim:
        return p_interp.reshape(-1)
    return p_interp


def interpolate_spectrum(nu_bins, flux_bins, nu_query, method='linear'):
    """
    Interpolate a GRB spectral energy distribution.

    Parameters
    ----------
    nu_bins : ndarray
        Frequency bins in Hz.
    flux_bins : ndarray
        νF_ν in erg cm⁻² s⁻¹.
    nu_query : ndarray
        Frequencies at which to evaluate.
    method : str
        'lagrange', 'linear', or 'nearest'.

    Returns
    -------
    flux_query : ndarray
        Interpolated νF_ν.
    """
    if method == 'lagrange':
        data_num = nu_bins.size
        if data_num > 8:
            # Lagrange unstable for many points; fall back to linear
            return interp_linear(nu_bins, flux_bins, nu_query)
        return interp_lagrange(1, data_num, nu_bins,
                               flux_bins.reshape(1, -1),
                               nu_query.size, nu_query).reshape(-1)
    elif method == 'linear':
        return interp_linear(nu_bins, flux_bins, nu_query)
    elif method == 'nearest':
        return interp_nearest(nu_bins, flux_bins, nu_query)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
