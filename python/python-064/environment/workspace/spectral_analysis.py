
import numpy as np


def haar_1d(n, u):
    v = np.asarray(u, dtype=float).flatten().copy()
    n = len(v)
    w = np.zeros(n)
    s = np.sqrt(2.0)


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
    u = np.asarray(v, dtype=float).flatten().copy()
    n = len(u)
    w = np.zeros(n)
    s = np.sqrt(2.0)


    k = 1
    while k * 2 <= n:
        k *= 2


    kk = 1
    while kk < k:
        w[0:2 * kk:2] = (u[0:kk] + u[kk:2 * kk]) / s
        w[1:2 * kk:2] = (u[0:kk] - u[kk:2 * kk]) / s
        u[0:2 * kk] = w[0:2 * kk]
        kk *= 2

    return u


def haar_power_spectrum(signal, dt=1.0):
    n = len(signal)
    coeffs = haar_1d(n, signal)


    k_max = 1
    while k_max * 2 <= n:
        k_max *= 2

    scales = []
    power = []
    scale = 1
    idx = 0
    while scale <= k_max // 2:

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
    n = len(signal)
    n_modes = min(n_modes, n)


    t = np.linspace(-1.0, 1.0, n)


    coeffs = np.zeros(n_modes)
    for k in range(n_modes):
        T_k = np.cos(k * np.arccos(np.clip(t, -1.0, 1.0)))
        coeffs[k] = np.sum(signal * T_k) / n

    coeffs[0] *= 0.5


    frequencies = np.arange(n_modes) / 2.0

    return coeffs, frequencies


def chebyshev_power(coeffs):
    power = coeffs ** 2
    power[0] *= 2.0
    return power


def identify_orbital_periods(scales, power, prominence_threshold=0.1):
    peaks = []
    if len(power) == 0:
        return peaks

    max_power = np.max(power)
    if max_power < 1e-20:
        return peaks


    power_norm = power / max_power


    for i in range(1, len(power_norm) - 1):
        if power_norm[i] > power_norm[i - 1] and power_norm[i] > power_norm[i + 1]:
            if power_norm[i] > prominence_threshold:
                period = scales[i]

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
    n = len(signal)

    freqs = np.fft.rfftfreq(n, dt)
    spec = np.zeros(len(freqs))

    for i in range(min(k, 5)):

        window = 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))
        shifted = np.roll(signal, i * n // (2 * k))
        windowed = shifted * window
        fft_vals = np.fft.rfft(windowed)
        spec += np.abs(fft_vals) ** 2

    spec /= min(k, 5)
    spec *= dt / n
    return freqs, spec


def bandpass_filter(signal, dt, f_low, f_high):
    n = len(signal)
    freqs = np.fft.rfftfreq(n, dt)
    fft_vals = np.fft.rfft(signal)

    mask = (freqs >= f_low) & (freqs <= f_high)
    filtered_fft = fft_vals * mask
    filtered = np.fft.irfft(filtered_fft, n)
    return np.real(filtered)


def extract_orbital_bands(signal, dt_kyr=1.0):

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
