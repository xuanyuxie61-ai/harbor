"""
wavelet_decomposition.py
------------------------
Multi-resolution Haar wavelet analysis of reflected electromagnetic
signals from plasma coatings.  Used to detect structural non-uniformities
and resonance peaks in the reflection spectrum.

Incorporates core ideas from:
  - 496_haar_transform  (1D and 2D Haar wavelet transforms)
"""

import numpy as np
import math


def haar_1d_transform(u: np.ndarray) -> np.ndarray:
    """
    Perform a 1-D Haar wavelet transform on vector u.

    For dyadic length N = 2^m:
        v_{2j}   = (u_{2j} + u_{2j+1}) / sqrt(2)
        v_{2j+1} = (u_{2j} - u_{2j+1}) / sqrt(2)

    For non-dyadic lengths, the transform is applied to the largest
    power-of-two prefix.

    Parameters
    ----------
    u : (N,) ndarray
        Input real vector.

    Returns
    -------
    v : (N,) ndarray
        Haar-transformed vector.  Untransformed tail entries are copied.
    """
    u = np.asarray(u, dtype=float)
    N = u.size
    if N == 0:
        return u.copy()

    # Find largest power of two <= N
    n_transform = 2 ** int(math.floor(math.log2(N)))
    if n_transform < 2:
        return u.copy()

    v = u.copy()
    n = n_transform
    while n > 1:
        n_half = n // 2
        temp = v.copy()
        for j in range(n_half):
            a = temp[2 * j]
            b = temp[2 * j + 1]
            v[j] = (a + b) / math.sqrt(2.0)
            v[n_half + j] = (a - b) / math.sqrt(2.0)
        n = n_half

    return v


def haar_1d_inverse(v: np.ndarray) -> np.ndarray:
    """
    Inverse 1-D Haar wavelet transform.
    """
    v = np.asarray(v, dtype=float)
    N = v.size
    if N == 0:
        return v.copy()

    n_transform = 2 ** int(math.floor(math.log2(N)))
    if n_transform < 2:
        return v.copy()

    u = v.copy()
    n = 2
    while n <= n_transform:
        n_half = n // 2
        temp = u.copy()
        for j in range(n_half):
            a = temp[j]
            b = temp[n_half + j]
            u[2 * j] = (a + b) / math.sqrt(2.0)
            u[2 * j + 1] = (a - b) / math.sqrt(2.0)
        n = n * 2

    return u


def haar_2d_transform(U: np.ndarray) -> np.ndarray:
    """
    Perform a 2-D Haar wavelet transform on matrix U by applying
    the 1-D transform to rows then columns.

    Parameters
    ----------
    U : (M, N) ndarray

    Returns
    -------
    V : (M, N) ndarray
    """
    U = np.asarray(U, dtype=float)
    M, N = U.shape
    if M == 0 or N == 0:
        return U.copy()

    V = U.copy()
    # Rows
    for i in range(M):
        V[i, :] = haar_1d_transform(V[i, :])
    # Columns
    for j in range(N):
        V[:, j] = haar_1d_transform(V[:, j])
    return V


def haar_2d_inverse(V: np.ndarray) -> np.ndarray:
    """Inverse 2-D Haar wavelet transform."""
    V = np.asarray(V, dtype=float)
    M, N = V.shape
    if M == 0 or N == 0:
        return V.copy()

    U = V.copy()
    # Columns inverse
    for j in range(N):
        U[:, j] = haar_1d_inverse(U[:, j])
    # Rows inverse
    for i in range(M):
        U[i, :] = haar_1d_inverse(U[i, :])
    return U


def detect_reflection_peaks_haar(
    signal: np.ndarray,
    threshold_factor: float = 2.0,
) -> np.ndarray:
    """
    Detect peaks in a 1-D reflection signal using Haar wavelet detail coefficients.

    The detail coefficients at the finest scale capture rapid changes.
    Large magnitudes indicate sudden changes in reflection (peaks/dips).

    Parameters
    ----------
    signal : (N,) ndarray
        Reflection spectrum or time-domain signal.
    threshold_factor : float
        Peaks are declared where |detail| > factor * std(detail).

    Returns
    -------
    peak_indices : (P,) ndarray of int
        Indices of detected peaks.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.size < 4:
        return np.array([], dtype=int)

    v = haar_1d_transform(signal)
    n_transform = 2 ** int(math.floor(math.log2(signal.size)))
    if n_transform < 4:
        return np.array([], dtype=int)

    # Detail coefficients at the finest scale occupy positions n_transform/2 .. n_transform-1
    n_half = n_transform // 2
    details = v[n_half:n_transform]
    std_d = float(np.std(details))
    if std_d < 1e-15:
        return np.array([], dtype=int)

    threshold = threshold_factor * std_d
    local_peaks = np.where(np.abs(details) > threshold)[0]
    # Map back to original indices (each detail corresponds to a pair)
    peak_indices = np.unique(np.clip(2 * local_peaks, 0, signal.size - 1))
    return peak_indices


def multiscale_energy_distribution(signal: np.ndarray) -> np.ndarray:
    """
    Compute the energy (sum of squared coefficients) at each Haar wavelet scale.

    For a signal of dyadic length N = 2^m, there are m+1 scales:
        scale 0 : approximation coefficient (single value)
        scale 1..m : detail coefficients of size 2^{j-1}

    Returns
    -------
    energies : (m+1,) ndarray
    """
    signal = np.asarray(signal, dtype=float)
    N = signal.size
    if N < 2:
        return np.array([np.sum(signal ** 2)], dtype=float)

    n_transform = 2 ** int(math.floor(math.log2(N)))
    v = haar_1d_transform(signal)

    energies = []
    # Approximation at the coarsest scale is v[0]
    energies.append(float(v[0] ** 2))

    n = n_transform
    pos = 1
    while n > 1:
        n_half = n // 2
        detail_block = v[pos:pos + n_half]
        energies.append(float(np.sum(detail_block ** 2)))
        pos += n_half
        n = n_half

    return np.array(energies, dtype=float)
