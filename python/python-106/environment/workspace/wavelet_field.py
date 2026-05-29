"""
wavelet_field.py
================
Haar wavelet decomposition of 2D plasmonic near-field distributions.

The Haar wavelet transform provides a multiresolution analysis (MRA) of
spatial field maps.  For a scalar field f(x,y) sampled on a uniform grid,
the 2D Haar transform separates the data into:

    Approximation coefficients  A_j[f]
    Horizontal detail           H_j[f]
    Vertical detail             V_j[f]
    Diagonal detail             D_j[f]

at each scale j.  The transform is defined by the scaling function
φ(x) = 1_{[0,1)}(x) and mother wavelet ψ(x) = 1_{[0,1/2)}(x) − 1_{[1/2,1)}(x).

In 2D, the basis functions are tensor products:
    φ_j,m(x) φ_j,n(y)   → approximation
    φ_j,m(x) ψ_j,n(y)   → horizontal detail
    ψ_j,m(x) φ_j,n(y)   → vertical detail
    ψ_j,m(x) ψ_j,n(y)   → diagonal detail

The Haar transform helps identify localized plasmonic hotspots at different
spatial scales, enabling scale-adaptive numerical refinement.
"""

import numpy as np


def haar_1d_transform(u):
    """
    1D Haar wavelet transform of a real vector.

    Parameters
    ----------
    u : ndarray

    Returns
    -------
    v : ndarray
        Transformed coefficients.
    """
    n = u.size
    if n < 1:
        raise ValueError("Input vector must be non-empty.")
    v = u.astype(float).copy()
    s = np.sqrt(2.0)
    w = np.zeros(n)

    k = 1
    while k * 2 <= n:
        k *= 2

    while k > 1:
        k = k // 2
        w[:k] = (v[:2 * k:2] + v[1:2 * k:2]) / s
        w[k:2 * k] = (v[:2 * k:2] - v[1:2 * k:2]) / s
        v[:2 * k] = w[:2 * k]
    return v


def haar_1d_inverse(v):
    """
    1D inverse Haar wavelet transform.

    Parameters
    ----------
    v : ndarray

    Returns
    -------
    u : ndarray
    """
    n = v.size
    if n < 1:
        raise ValueError("Input vector must be non-empty.")
    u = v.astype(float).copy()
    s = np.sqrt(2.0)
    w = np.zeros(n)

    # Determine the smallest power of 2 used in the transform
    k = 1
    while k * 2 <= n:
        k *= 2

    # Inverse: rebuild from smallest scale to largest
    scale = 1
    while scale < k:
        w[:2 * scale] = u[:2 * scale]
        u[:2 * scale:2] = (w[:scale] + w[scale:2 * scale]) / s
        u[1:2 * scale:2] = (w[:scale] - w[scale:2 * scale]) / s
        scale *= 2
    return u


def haar_2d_transform(u):
    """
    2D Haar wavelet transform: apply 1D Haar to all columns, then all rows.

    Parameters
    ----------
    u : ndarray, shape (m, n)

    Returns
    -------
    v : ndarray, shape (m, n)
    """
    m, n = u.shape
    if m < 1 or n < 1:
        raise ValueError("Input array must be non-empty.")
    v = u.astype(float).copy()

    # Column transform (along axis 0)
    for j in range(n):
        v[:, j] = haar_1d_transform(v[:, j])

    # Row transform (along axis 1)
    for i in range(m):
        v[i, :] = haar_1d_transform(v[i, :])

    return v


def haar_2d_inverse(v):
    """
    2D inverse Haar wavelet transform.

    Parameters
    ----------
    v : ndarray, shape (m, n)

    Returns
    -------
    u : ndarray, shape (m, n)
    """
    m, n = v.shape
    if m < 1 or n < 1:
        raise ValueError("Input array must be non-empty.")
    u = v.astype(float).copy()

    # Inverse row transform first
    for i in range(m):
        u[i, :] = haar_1d_inverse(u[i, :])

    # Inverse column transform
    for j in range(n):
        u[:, j] = haar_1d_inverse(u[:, j])

    return u


def extract_multiresolution_hotspots(field, threshold_factor=2.0):
    """
    Use Haar wavelet detail coefficients to identify spatial scales
    at which plasmonic hotspots dominate.

    Parameters
    ----------
    field : ndarray, shape (m, n)
        2D near-field intensity or amplitude map.
    threshold_factor : float
        Detail coefficients exceeding threshold_factor * std(detail)
        are flagged as hotspots.

    Returns
    -------
    scales : list of int
        Characteristic scales (in pixels) where hotspots were detected.
    coefficients : dict
        Dictionary of wavelet coefficients at each decomposition level.
    """
    m, n = field.shape
    if m < 2 or n < 2:
        raise ValueError("Field must be at least 2×2.")

    v = haar_2d_transform(field)
    scales = []
    coefficients = {}

    # Determine number of decomposition levels
    num_levels = int(np.floor(np.log2(min(m, n))))

    for level in range(1, num_levels + 1):
        block = min(m, n) // (2 ** level)
        if block < 1:
            break

        # In a standard 2D Haar, after both passes the layout is:
        # top-left: approximation at this level
        # The detail blocks are mixed; for simplicity we examine
        # the high-frequency tail of the transformed array.
        detail_region = v[block:2 * block, block:2 * block]
        if detail_region.size == 0:
            continue
        std_val = np.std(detail_region)
        mean_val = np.mean(np.abs(detail_region))
        coefficients[level] = {
            'std': std_val,
            'mean_abs': mean_val,
            'max_abs': np.max(np.abs(detail_region))
        }
        if std_val > 0 and mean_val > threshold_factor * std_val:
            scales.append(block)

    return scales, coefficients
