# -*- coding: utf-8 -*-
"""
================================================================================
Neural Field Spectral Analysis Module
================================================================================

This module provides spectral methods for analyzing local field potentials (LFP),
population firing rates, and synaptic transfer functions using:
1. Fast Fourier Transform (FFT)
2. Trigonometric interpolation for periodic signals
3. Chebyshev interpolation for non-periodic transfer functions

Mathematical Background:
------------------------
For a neural field φ(x,t) defined on a periodic domain [0,L], the spatial
Fourier transform is:

    φ̃(k,t) = ∫_0^L φ(x,t) · exp(-i·2π·k·x/L) dx

Discretized with N points x_n = n·L/N:

    φ̃_k = Σ_{n=0}^{N-1} φ_n · exp(-i·2π·k·n/N)

This is computed efficiently via FFT in O(N log N) operations.

Power Spectral Density:
-----------------------
The PSD quantifies the contribution of each frequency:

    S(k) = |φ̃_k|² / N

For neural oscillations, characteristic peaks appear at:
    - Delta: 0.5-4 Hz
    - Theta: 4-8 Hz
    - Alpha: 8-13 Hz
    - Beta: 13-30 Hz
    - Gamma: 30-100 Hz

Trigonometric Interpolation:
----------------------------
For N equispaced points on [0,2π), the trigonometric interpolant is:

    P(x) = Σ_{j=0}^{N-1} y_j · C_j(x)

where C_j(x) are cardinal functions:

    C_j(x) = sin(N·(x-x_j)/2) / [N·sin((x-x_j)/2)] · exp(-i·(N-1)·(x-x_j)/2)

Chebyshev Interpolation:
------------------------
For non-periodic synaptic transfer functions, Chebyshev nodes on [a,b] are:

    x_j = [(b-a)·cos(jπ/(N-1)) + (a+b)] / 2,   j=0,...,N-1

The interpolating polynomial in Newton form uses divided differences.

================================================================================
"""

import numpy as np
from typing import Tuple, Optional


def compute_fft_spectrum(
    signal: np.ndarray,
    dt: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the FFT power spectrum of a neural signal.

    Parameters
    ----------
    signal : np.ndarray
        Time series data.
    dt : float
        Sampling interval [ms]. Must be positive.

    Returns
    -------
    freqs : np.ndarray
        Frequency array [Hz].
    spectrum : np.ndarray
        Complex FFT coefficients.
    psd : np.ndarray
        Power spectral density.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive.")

    n = signal.shape[0]
    if n < 2:
        raise ValueError("Signal must have at least 2 points.")

    # FFT
    spectrum = np.fft.fft(signal)

    # Frequencies
    freqs = np.fft.fftfreq(n, d=dt / 1000.0)  # Convert ms to s for Hz

    # Power spectral density
    psd = np.abs(spectrum) ** 2 / n

    return freqs, spectrum, psd


def compute_band_power(
    freqs: np.ndarray,
    psd: np.ndarray,
    band: Tuple[float, float],
) -> float:
    """
    Compute the power in a specific frequency band.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency array [Hz].
    psd : np.ndarray
        Power spectral density.
    band : tuple
        (f_min, f_max) frequency band.

    Returns
    -------
    power : float
        Integrated power in the band.
    """
    f_min, f_max = band
    mask = (freqs >= f_min) & (freqs <= f_max)
    power = np.sum(psd[mask])
    return power


def trig_cardinal(
    x: np.ndarray,
    xj: float,
    n: int,
    h: float,
) -> np.ndarray:
    """
    Evaluate the trigonometric cardinal function at points x.

    For n equispaced points with spacing h on a periodic domain:

        C_j(x) = sin(n·(x-x_j)/(2h)) / [n·sin((x-x_j)/(2h))]

    Parameters
    ----------
    x : np.ndarray
        Evaluation points.
    xj : float
        Node position.
    n : int
        Number of nodes.
    h : float
        Node spacing.

    Returns
    -------
    C : np.ndarray
        Cardinal function values.
    """
    if h <= 0.0:
        raise ValueError("h must be positive.")
    if n < 1:
        raise ValueError("n must be >= 1.")

    dx = x - xj
    # Handle periodicity
    dx = np.mod(dx + np.pi * h * n, 2.0 * np.pi * h * n) - np.pi * h * n

    denom = n * np.sin(dx / (2.0 * h))

    # Avoid division by zero at node
    C = np.ones_like(x, dtype=float)
    nonzero = np.abs(denom) > 1e-12
    C[nonzero] = np.sin(n * dx[nonzero] / (2.0 * h)) / denom[nonzero]

    return C


def trig_interpolate(
    xd: np.ndarray,
    yd: np.ndarray,
    xi: np.ndarray,
) -> np.ndarray:
    """
    Trigonometric interpolation of periodic data.

    Parameters
    ----------
    xd : np.ndarray
        Equispaced data nodes (must be periodic).
    yd : np.ndarray
        Data values at nodes.
    xi : np.ndarray
        Interpolation points.

    Returns
    -------
    yi : np.ndarray
        Interpolated values.
    """
    nd = xd.shape[0]
    if nd < 2:
        raise ValueError("Need at least 2 data points.")
    if yd.shape[0] != nd:
        raise ValueError("xd and yd must have same length.")

    h = xd[1] - xd[0]

    yi = np.zeros_like(xi, dtype=float)
    for j in range(nd):
        yi += yd[j] * trig_cardinal(xi, xd[j], nd, h)

    return yi


def chebyspace(a: float, b: float, n: int) -> np.ndarray:
    """
    Generate Chebyshev-spaced nodes on the interval [a, b].

    Nodes are:
        x_j = [(b-a)·cos(j·π/(n-1)) + (a+b)] / 2,   j=0,...,n-1

    For n=1: x_0 = (a+b)/2.

    Parameters
    ----------
    a, b : float
        Interval endpoints.
    n : int
        Number of nodes. Must be >= 1.

    Returns
    -------
    x : np.ndarray
        Chebyshev nodes.
    """
    if n < 1:
        raise ValueError("n must be >= 1.")
    if n == 1:
        return np.array([(a + b) / 2.0])

    theta = np.pi * np.arange(n) / (n - 1)
    c = np.cos(theta)

    # Ensure middle point is exactly 0 for odd n
    if n % 2 == 1:
        c[(n - 1) // 2] = 0.0

    x = ((1.0 - c) * a + (1.0 + c) * b) / 2.0
    return x


def divided_differences(
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
    Compute divided differences for Newton interpolation.

    dd[j] = f[x_0, ..., x_j]

    Parameters
    ----------
    x : np.ndarray
        Node coordinates.
    y : np.ndarray
        Node values.

    Returns
    -------
    dd : np.ndarray
        Divided difference table.
    """
    n = x.shape[0]
    dd = y.copy().astype(float)

    for i in range(1, n):
        for j in range(n - 1, i - 1, -1):
            dd[j] = (dd[j] - dd[j - 1]) / (x[j] - x[j - i])

    return dd


def newton_interpolate(
    xd: np.ndarray,
    dd: np.ndarray,
    xi: np.ndarray,
) -> np.ndarray:
    """
    Evaluate the Newton interpolating polynomial at points xi.

    P(x) = dd_0 + dd_1·(x-x_0) + dd_2·(x-x_0)·(x-x_1) + ...

    Parameters
    ----------
    xd : np.ndarray
        Node coordinates.
    dd : np.ndarray
        Divided differences.
    xi : np.ndarray
        Evaluation points.

    Returns
    -------
    yi : np.ndarray
        Interpolated values.
    """
    nd = xd.shape[0]
    yi = dd[nd - 1] * np.ones_like(xi, dtype=float)

    for i in range(nd - 2, -1, -1):
        yi = dd[i] + (xi - xd[i]) * yi

    return yi


def chebyshev_interpolate(
    f: callable,
    a: float,
    b: float,
    n: int,
    xi: np.ndarray,
) -> np.ndarray:
    """
    Chebyshev interpolation of a function on [a, b].

    Parameters
    ----------
    f : callable
        Function to interpolate.
    a, b : float
        Interval endpoints.
    n : int
        Number of Chebyshev nodes.
    xi : np.ndarray
        Evaluation points.

    Returns
    -------
    yi : np.ndarray
        Interpolated values.
    """
    xd = chebyspace(a, b, n)
    yd = f(xd)
    dd = divided_differences(xd, yd)
    yi = newton_interpolate(xd, dd, xi)
    return yi


def analyze_neural_field_spectrum(
    n_points: int = 512,
    t_max: float = 1000.0,
) -> dict:
    """
    Generate a synthetic neural field signal and analyze its spectrum.

    The signal contains theta (6 Hz), beta (20 Hz), and gamma (40 Hz)
    oscillations with additive noise.

    Parameters
    ----------
    n_points : int
        Number of time points.
    t_max : float
        Total time [ms].

    Returns
    -------
    results : dict
        Signal and spectrum data.
    """
    if n_points < 2:
        raise ValueError("n_points must be >= 2.")
    if t_max <= 0.0:
        raise ValueError("t_max must be positive.")

    dt = t_max / n_points
    t = np.linspace(0.0, t_max, n_points)

    # Synthetic LFP signal with multiple frequency components
    rng = np.random.default_rng(130)
    signal = (
        0.5 * np.sin(2.0 * np.pi * 6.0 * t / 1000.0)      # Theta: 6 Hz
        + 0.3 * np.sin(2.0 * np.pi * 20.0 * t / 1000.0)   # Beta: 20 Hz
        + 0.2 * np.sin(2.0 * np.pi * 40.0 * t / 1000.0)   # Gamma: 40 Hz
        + 0.1 * rng.standard_normal(n_points)              # Noise
    )

    freqs, spectrum, psd = compute_fft_spectrum(signal, dt)

    # Keep only positive frequencies
    pos_mask = freqs >= 0
    freqs_pos = freqs[pos_mask]
    psd_pos = psd[pos_mask]

    # Band powers
    bands = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 100.0),
    }
    band_powers = {name: compute_band_power(freqs_pos, psd_pos, band) for name, band in bands.items()}

    # Dominant frequency
    dominant_freq = freqs_pos[np.argmax(psd_pos)]

    return {
        "t": t,
        "signal": signal,
        "freqs": freqs_pos,
        "psd": psd_pos,
        "band_powers": band_powers,
        "dominant_freq": dominant_freq,
    }


def test_interpolation_accuracy(
    n_test: int = 100,
) -> dict:
    """
    Test trigonometric and Chebyshev interpolation accuracy.

    Parameters
    ----------
    n_test : int
        Number of test points.

    Returns
    -------
    results : dict
        Error metrics.
    """
    # Test function: synaptic transfer function (sigmoid-like)
    def transfer_func(x):
        return 1.0 / (1.0 + np.exp(-5.0 * (x - 0.5)))

    # Trigonometric interpolation on periodic domain
    n_trig = 16
    xd_trig = np.linspace(0.0, 2.0 * np.pi, n_trig, endpoint=False)
    yd_trig = transfer_func(np.sin(xd_trig) * 0.5 + 0.5)
    xi_trig = np.linspace(0.0, 2.0 * np.pi, n_test, endpoint=False)
    yi_trig = trig_interpolate(xd_trig, yd_trig, xi_trig)
    y_true_trig = transfer_func(np.sin(xi_trig) * 0.5 + 0.5)
    err_trig = np.linalg.norm(yi_trig - y_true_trig) / np.linalg.norm(y_true_trig)

    # Chebyshev interpolation
    n_cheb = 16
    a, b = 0.0, 1.0
    xi_cheb = np.linspace(a, b, n_test)
    yi_cheb = chebyshev_interpolate(transfer_func, a, b, n_cheb, xi_cheb)
    y_true_cheb = transfer_func(xi_cheb)
    err_cheb = np.linalg.norm(yi_cheb - y_true_cheb) / np.linalg.norm(y_true_cheb)

    return {
        "trig_error": err_trig,
        "cheb_error": err_cheb,
    }


if __name__ == "__main__":
    results = analyze_neural_field_spectrum()
    print(f"Dominant frequency: {results['dominant_freq']:.2f} Hz")
    for band, power in results["band_powers"].items():
        print(f"  {band} power: {power:.4f}")

    interp = test_interpolation_accuracy()
    print(f"Trig interpolation error: {interp['trig_error']:.6e}")
    print(f"Chebyshev interpolation error: {interp['cheb_error']:.6e}")
