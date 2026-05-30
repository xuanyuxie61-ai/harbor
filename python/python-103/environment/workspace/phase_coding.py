
import numpy as np


def caesar_shift_phase(spectrum, shift_amount):
    spectrum = np.asarray(spectrum)
    if spectrum.size == 0:
        return spectrum
    shift_amount = int(shift_amount) % spectrum.size
    if shift_amount == 0:
        return spectrum.copy()
    return np.roll(spectrum, shift_amount)


def magic_matrix(n):
    if n % 2 != 1:
        raise ValueError("magic_matrix: n must be odd")
    if n < 1:
        raise ValueError("magic_matrix: n must be positive")

    A = np.zeros((n, n), dtype=int)
    k = 1
    i = 0
    j = n // 2
    A[i, j] = k

    while k < n * n:
        k += 1
        im1 = (i - 1) % n
        jp1 = (j + 1) % n
        if A[im1, jp1] != 0:
            im1 = (i + 1) % n
            jp1 = j
        A[im1, jp1] = k
        i = im1
        j = jp1

    return A


def magic_phase_mask(n):
    M = magic_matrix(n)
    n2 = n * n
    phase = 2.0 * np.pi * (M - 1) / (n2 - 1)
    return phase


def apply_phase_mask_to_pulse(t, A, mask_size=5):
    if t.size < 2 or A.size != t.size:
        return A.copy()

    n = A.size

    window_size = min(mask_size * 2, n)
    hop = max(1, window_size // 4)
    n_frames = max(1, (n - window_size) // hop + 1)


    if mask_size % 2 == 0:
        mask_size += 1
    mask = magic_phase_mask(mask_size)


    spectrum = np.fft.fft(A)
    n_freq = spectrum.size


    phase_mod = np.zeros(n_freq)
    for idx in range(n_freq):
        band = idx % mask_size
        phase_mod[idx] = mask[band, band]

    modulated = spectrum * np.exp(1j * phase_mod)
    return np.fft.ifft(modulated)


def vector_sumlex_next(w):
    m = w.size
    for i in range(m - 1, -1, -1):
        if w[i] > 0:
            w[i] -= 1
            if i < m - 1:
                w[i + 1] += 1
            return w
    return w


def four_fifths_search(n_max, exponent=5):
    if n_max < 2:
        return None

    fifths = np.arange(1, n_max + 1, dtype=np.float64) ** exponent


    best_error = np.inf
    best_tuple = None


    search_limit = min(n_max, 30)
    for a in range(1, search_limit):
        for b in range(a, search_limit):
            for c in range(b, search_limit):
                for d in range(c, search_limit):
                    s = a ** exponent + b ** exponent + c ** exponent + d ** exponent

                    e_float = s ** (1.0 / exponent)
                    e_int = int(round(e_float))
                    if e_int > 0 and e_int <= n_max:
                        error = abs(e_int ** exponent - s)
                        if error < best_error:
                            best_error = error
                            best_tuple = (a, b, c, d, e_int)
                        if error == 0:
                            return best_tuple

    return best_tuple


def wdm_channel_search(center_wavelength_nm=1550, channel_spacing_nm=0.8, n_channels=4, target_fwm_efficiency=0.01):
    best_channels = None
    best_fwm = np.inf


    for base in range(1, 20):
        channels = [center_wavelength_nm + base * channel_spacing_nm * i for i in range(n_channels)]

        fwm_total = 0.0
        count = 0
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                for k in range(j + 1, n_channels):

                    mismatch = abs(channels[i] + channels[j] - 2.0 * channels[k])
                    fwm_total += 1.0 / (1.0 + mismatch)
                    count += 1
        if count > 0:
            avg_fwm = fwm_total / count
            if avg_fwm < best_fwm:
                best_fwm = avg_fwm
                best_channels = channels

    return best_channels, best_fwm
