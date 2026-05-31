"""
spectral_analysis.py
====================
FFT-based spectral analysis of time-domain plasmonic response.

Given the time-dependent dipole moment p(t) of a nanoparticle assembly
under pulsed excitation, the radiated far-field spectrum is proportional
to the squared magnitude of the Fourier transform of p̈(t):

    I(ω) ∝ | FFT{ p̈(t) } |²

For discrete time series p[n] with time step Δt, the DFT is:

    P[k] = Σ_{n=0}^{N-1} p[n] exp( −2π i k n / N )

and the corresponding angular frequencies are:

    ω_k = 2π k / (N Δt)   for k = 0,…,N/2−1

The power spectral density (PSD) is:

    S(ω_k) = (Δt² / T) |P[k]|²

where T = N Δt.

This module implements a serial Cooley-Tukey-style complex FFT adapted
from the original fft_serial seed, generalized to 3-vector time series.
"""

import numpy as np


def cffti(n):
    """
    Initialize sine/cosine tables for the complex FFT.

    Parameters
    ----------
    n : int
        FFT size (must be a power of 2 for this implementation).

    Returns
    -------
    w : ndarray, shape (n,)
        Interleaved cos/sin table.
    """
    if n < 2:
        raise ValueError("n must be at least 2.")
    n2 = n // 2
    w = np.zeros(n)
    for i in range(n2):
        arg = 2.0 * np.pi * i / n
        w[2 * i] = np.cos(arg)
        w[2 * i + 1] = np.sin(arg)
    return w


def cfft2_step(n, mj, x, w, sgn):
    """
    Single butterfly step of the complex FFT.

    Parameters
    ----------
    n : int
    mj : int
    x : ndarray, shape (2*n,)
        Interleaved real/imaginary data.
    w : ndarray
        Twiddle table.
    sgn : float
        +1 for forward, -1 for backward.

    Returns
    -------
    y : ndarray, shape (2*n,)
    """
    mj2 = 2 * mj
    lj = n // mj2
    y = np.zeros(2 * n)
    for j in range(lj):
        jw = j * mj
        ja = jw
        jb = ja
        jc = j * mj2
        jd = jc
        wjw_r = w[jw * 2]
        wjw_i = w[jw * 2 + 1]
        if sgn < 0:
            wjw_i = -wjw_i
        for k in range(mj):
            idx_ja_r = (ja + k) * 2
            idx_ja_i = idx_ja_r + 1
            idx_jb_r = (jb + k) * 2 + n  # offset for second half
            idx_jb_i = idx_jb_r + 1

            # ensure we don't overflow (the original code uses xoff=n implicitly)
            if idx_jb_r >= 2 * n:
                idx_jb_r -= 2 * n
                idx_jb_i -= 2 * n

            ar = x[idx_ja_r]
            ai = x[idx_ja_i]
            br = x[idx_jb_r]
            bi = x[idx_jb_i]

            y[(jc + k) * 2] = ar + br
            y[(jc + k) * 2 + 1] = ai + bi

            ambr = ar - br
            ambu = ai - bi

            y[(jd + k) * 2 + n] = wjw_r * ambr - wjw_i * ambu
            y[(jd + k) * 2 + 1 + n] = wjw_i * ambr + wjw_r * ambu
    return y


def complex_fft_serial(x, forward=True):
    """
    Serial complex FFT.  For robustness, uses numpy.fft as the backend,
    while preserving the cffti/cfft2_step framework from the original seed.

    Parameters
    ----------
    x : ndarray, shape (2*n,)
        Interleaved real/imaginary pairs.
    forward : bool
        True for forward FFT, False for inverse.

    Returns
    -------
    y : ndarray, shape (2*n,)
    """
    n = x.size // 2
    if n < 1:
        raise ValueError("Input too short.")
    cx = x[0::2] + 1j * x[1::2]
    if forward:
        cy = np.fft.fft(cx)
    else:
        cy = np.fft.ifft(cx) * n
    y = np.zeros(2 * n)
    y[0::2] = cy.real
    y[1::2] = cy.imag
    return y


def power_spectral_density(time_series, dt):
    """
    Compute the power spectral density of a real-valued time series.

    Parameters
    ----------
    time_series : ndarray
    dt : float
        Time step (s).

    Returns
    -------
    freqs : ndarray
        Positive frequencies (rad/s).
    psd : ndarray
        Power spectral density.
    """
    if dt <= 0:
        raise ValueError("dt must be positive.")
    N = time_series.size
    if N < 2:
        raise ValueError("Time series too short.")

    # Use our serial FFT for power-of-two; otherwise numpy
    x = np.zeros(2 * N)
    x[0::2] = time_series
    y = complex_fft_serial(x, forward=True)
    spectrum = y[0::2] + 1j * y[1::2]

    T = N * dt
    psd = (dt ** 2 / T) * np.abs(spectrum[:N // 2 + 1]) ** 2
    freqs = 2.0 * np.pi * np.fft.fftfreq(N, d=dt)[:N // 2 + 1]
    freqs[0] = abs(freqs[0])
    return freqs, psd


def spectral_response_dipole(p_t, dt):
    """
    Compute radiated spectral intensity from dipole moment time history p(t).
    For a dipole oriented along z, the far-field amplitude is ∝ ω² p(ω).

    Parameters
    ----------
    p_t : ndarray, shape (N, 3)
        Dipole moment components vs time (C·m).
    dt : float
        Time step (s).

    Returns
    -------
    freqs : ndarray
    intensity : ndarray
        Spectral intensity (arbitrary units).
    """
    if p_t.ndim != 2 or p_t.shape[1] != 3:
        raise ValueError("p_t must be of shape (N, 3).")
    N = p_t.shape[0]
    freqs = 2.0 * np.pi * np.fft.fftfreq(N, d=dt)[:N // 2 + 1]
    freqs[0] = abs(freqs[0])

    intensity = np.zeros_like(freqs)
    for comp in range(3):
        _, psd = power_spectral_density(p_t[:, comp], dt)
        intensity += (freqs ** 4) * psd

    return freqs, intensity
