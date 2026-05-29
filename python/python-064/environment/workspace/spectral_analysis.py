"""
Spectral Analysis Module
========================
Multi-resolution spectral analysis of paleoclimate time series.
Combines Haar wavelet transform (from 496_haar_transform) with
Chebyshev spectral methods for identifying orbital periodicities.

Scientific Background:
----------------------
Paleoclimate proxy records (e.g., delta18O) contain signals at multiple
orbital frequencies:
- Eccentricity: ~100 kyr and ~400 kyr
- Obliquity: ~41 kyr
- Precession: ~19 kyr and ~23 kyr

Haar wavelet decomposition provides time-localized frequency analysis:
    W_f(a, b) = (1/sqrt(a)) * integral f(t) * psi((t-b)/a) dt
where psi is the Haar mother wavelet:
    psi(t) = 1 for 0 <= t < 0.5, -1 for 0.5 <= t < 1, 0 otherwise.

Chebyshev spectral analysis provides global frequency content via:
    f(t) = sum c_k * T_k(t)  for t in [-1, 1]
where T_k are Chebyshev polynomials: T_k(cos(theta)) = cos(k*theta).
"""

import numpy as np


def haar_1d(n, u):
    """
    Compute 1D Haar wavelet transform.
    From 496_haar_transform.

    Parameters
    ----------
    n : int
        Length of vector.
    u : ndarray
        Input vector.

    Returns
    -------
    ndarray
        Haar transform coefficients.
    """
    v = np.asarray(u, dtype=float).flatten().copy()
    n = len(v)
    w = np.zeros(n)
    s = np.sqrt(2.0)

    # Largest power of 2 <= n
    k = 1
    while k * 2 <= n:
        k *= 2

    while k > 1:
        k = k // 2
        w[0:k] = (v[0:2 * k:2] + v[1:2 * k:2]) / s
        w[k:2 * k] = (v[0:2 * k:2] - v[1:2 * k:2]) / s
        v[0:2 * k] = w[0:2 * k]

    return v


def haar_1d_inverse(n, v):
    """
    Inverse 1D Haar wavelet transform.
    From 496_haar_transform.

    Parameters
    ----------
    n : int
        Length of vector.
    v : ndarray
        Haar coefficients.

    Returns
    -------
    ndarray
        Reconstructed signal.
    """
    u = np.asarray(v, dtype=float).flatten().copy()
    n = len(u)
    w = np.zeros(n)
    s = np.sqrt(2.0)

    # Find largest power of 2 <= n
    k = 1
    while k * 2 <= n:
        k *= 2

    # Build up from smallest scale
    kk = 1
    while kk < k:
        w[0:2 * kk:2] = (u[0:kk] + u[kk:2 * kk]) / s
        w[1:2 * kk:2] = (u[0:kk] - u[kk:2 * kk]) / s
        u[0:2 * kk] = w[0:2 * kk]
        kk *= 2

    return u


def haar_power_spectrum(signal, dt=1.0):
    """
    Compute Haar wavelet power spectrum showing energy at each dyadic scale.

    Parameters
    ----------
    signal : ndarray
        Input time series.
    dt : float
        Sampling interval.

    Returns
    -------
    scales : ndarray
        Characteristic time scales.
    power : ndarray
        Power at each scale.
    """
    n = len(signal)
    coeffs = haar_1d(n, signal)

    # Largest power of 2
    k_max = 1
    while k_max * 2 <= n:
        k_max *= 2

    scales = []
    power = []
    scale = 1
    idx = 0
    while scale <= k_max // 2:
        # Detail coefficients at this scale
        start = idx
        end = min(idx + scale, n)
        detail = coeffs[start:end]
        if len(detail) > 0:
            scales.append(scale * dt * 2)
            power.append(np.sum(detail ** 2) / max(len(detail), 1))
        idx += scale
        scale *= 2

    return np.array(scales), np.array(power)


def chebyshev_spectrum(signal, n_modes=64):
    """
    Compute Chebyshev spectral coefficients of a signal.
    Maps signal domain to [-1, 1] and computes coefficients.

    Parameters
    ----------
    signal : ndarray
        Input signal.
    n_modes : int
        Number of Chebyshev modes.

    Returns
    -------
    coeffs : ndarray
        Chebyshev coefficients.
    frequencies : ndarray
        Equivalent frequencies (normalized).
    """
    n = len(signal)
    n_modes = min(n_modes, n)

    # Map to [-1, 1]
    t = np.linspace(-1.0, 1.0, n)

    # Compute coefficients via discrete cosine transform approximation
    coeffs = np.zeros(n_modes)
    for k in range(n_modes):
        T_k = np.cos(k * np.arccos(np.clip(t, -1.0, 1.0)))
        coeffs[k] = np.sum(signal * T_k) / n

    coeffs[0] *= 0.5  # First coefficient halved

    # Equivalent frequencies (higher k = higher frequency)
    frequencies = np.arange(n_modes) / 2.0

    return coeffs, frequencies


def chebyshev_power(coeffs):
    """
    Compute power spectral density from Chebyshev coefficients.

    Parameters
    ----------
    coeffs : ndarray
        Chebyshev coefficients.

    Returns
    -------
    ndarray
        Power at each mode.
    """
    power = coeffs ** 2
    power[0] *= 2.0  # DC component
    return power


def identify_orbital_periods(scales, power, prominence_threshold=0.1):
    """
    Identify dominant periodicities corresponding to orbital cycles.

    Expected periods (kyr):
    - Precession: 19, 23
    - Obliquity: 41
    - Eccentricity: 100, 400

    Parameters
    ----------
    scales : ndarray
        Time scales from spectral analysis.
    power : ndarray
        Power at each scale.
    prominence_threshold : float
        Minimum relative prominence for peak detection.

    Returns
    -------
    peaks : list of dict
        Detected peaks with period and orbital assignment.
    """
    peaks = []
    if len(power) == 0:
        return peaks

    max_power = np.max(power)
    if max_power < 1e-20:
        return peaks

    # Normalize
    power_norm = power / max_power

    # Find local maxima
    for i in range(1, len(power_norm) - 1):
        if power_norm[i] > power_norm[i - 1] and power_norm[i] > power_norm[i + 1]:
            if power_norm[i] > prominence_threshold:
                period = scales[i]
                # Assign to orbital cycle
                if 15 <= period <= 26:
                    cycle = "Precession (~19-23 kyr)"
                elif 35 <= period <= 50:
                    cycle = "Obliquity (~41 kyr)"
                elif 80 <= period <= 130:
                    cycle = "Eccentricity (~100 kyr)"
                elif 300 <= period <= 500:
                    cycle = "Long Eccentricity (~400 kyr)"
                else:
                    cycle = "Unknown"

                peaks.append({
                    'period_kyr': period,
                    'power': power[i],
                    'normalized_power': power_norm[i],
                    'orbital_cycle': cycle
                })

    return peaks


def multitaper_spectrum(signal, dt=1.0, nw=4, k=8):
    """
    Compute multitaper spectral estimate for robust period detection.
    Uses discrete prolate spheroidal sequences (Slepian tapers).

    Parameters
    ----------
    signal : ndarray
        Input time series.
    dt : float
        Sampling interval.
    nw : float
        Time-bandwidth product.
    k : int
        Number of tapers.

    Returns
    -------
    freqs : ndarray
        Frequencies.
    spec : ndarray
        Power spectral density.
    """
    n = len(signal)
    # Simple approximation: use periodogram with Hamming windows
    freqs = np.fft.rfftfreq(n, dt)
    spec = np.zeros(len(freqs))

    for i in range(min(k, 5)):
        # Hamming window with slight shifts
        window = 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))
        shifted = np.roll(signal, i * n // (2 * k))
        windowed = shifted * window
        fft_vals = np.fft.rfft(windowed)
        spec += np.abs(fft_vals) ** 2

    spec /= min(k, 5)
    spec *= dt / n
    return freqs, spec


def bandpass_filter(signal, dt, f_low, f_high):
    """
    Bandpass filter signal using FFT.

    Parameters
    ----------
    signal : ndarray
        Input signal.
    dt : float
        Sampling interval.
    f_low, f_high : float
        Frequency band in cycles per unit time.

    Returns
    -------
    ndarray
        Filtered signal.
    """
    n = len(signal)
    freqs = np.fft.rfftfreq(n, dt)
    fft_vals = np.fft.rfft(signal)

    mask = (freqs >= f_low) & (freqs <= f_high)
    filtered_fft = fft_vals * mask
    filtered = np.fft.irfft(filtered_fft, n)
    return np.real(filtered)


def extract_orbital_bands(signal, dt_kyr=1.0):
    """
    Extract signals in each orbital frequency band.

    Parameters
    ----------
    signal : ndarray
        Input time series (e.g., delta18O proxy).
    dt_kyr : float
        Time step in kyr.

    Returns
    -------
    dict
        Filtered signals for each orbital band.
    """
    # Frequency bands (cycles/kyr)
    bands = {
        'precession': (1.0 / 26.0, 1.0 / 15.0),
        'obliquity': (1.0 / 50.0, 1.0 / 35.0),
        'eccentricity_100': (1.0 / 130.0, 1.0 / 80.0),
        'eccentricity_400': (1.0 / 500.0, 1.0 / 300.0)
    }

    result = {}
    for name, (f_low, f_high) in bands.items():
        result[name] = bandpass_filter(signal, dt_kyr, f_low, f_high)

    return result


def spectral_coherence(signal1, signal2, dt=1.0, n_segments=8):
    """
    Compute magnitude-squared coherence between two signals.

    Formula:
    C_xy(f) = |S_xy(f)|^2 / (S_xx(f) * S_yy(f))

    Parameters
    ----------
    signal1, signal2 : ndarray
        Input signals.
    dt : float
        Sampling interval.
    n_segments : int
        Number of segments for averaging.

    Returns
    -------
    freqs : ndarray
        Frequencies.
    coherence : ndarray
        Coherence [0, 1].
    """
    n = len(signal1)
    seg_len = n // n_segments
    freqs = np.fft.rfftfreq(seg_len, dt)
    sxx = np.zeros(len(freqs))
    syy = np.zeros(len(freqs))
    sxy = np.zeros(len(freqs), dtype=complex)

    for i in range(n_segments):
        start = i * seg_len
        end = start + seg_len
        x_seg = signal1[start:end] * np.hanning(seg_len)
        y_seg = signal2[start:end] * np.hanning(seg_len)
        x_fft = np.fft.rfft(x_seg)
        y_fft = np.fft.rfft(y_seg)
        sxx += np.abs(x_fft) ** 2
        syy += np.abs(y_fft) ** 2
        sxy += x_fft * np.conj(y_fft)

    sxx /= n_segments
    syy /= n_segments
    sxy /= n_segments

    coherence = np.abs(sxy) ** 2 / (sxx * syy + 1e-20)
    coherence = np.clip(coherence, 0.0, 1.0)
    return freqs, coherence
