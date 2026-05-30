
import numpy as np


def prime_factors(n):
    if n != int(n):
        raise ValueError("输入必须是整数")
    n = int(n)
    if n < 1:
        raise ValueError("输入必须 >= 1")

    i = 2
    factors = []
    while i * i <= n:
        if n % i != 0:
            i += 1
        else:
            n = n // i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


def optimal_fft_length(min_length, max_search=1000):
    for length in range(min_length, min_length + max_search):
        factors = prime_factors(length)
        if all(f in (2, 3, 5) for f in factors):
            return length
    return min_length


def mixed_radix_fft_complexity(n):
    factors = prime_factors(n)
    if not factors:
        return 0
    from collections import Counter
    cnt = Counter(factors)
    complexity = n * sum(e * p for p, e in cnt.items())
    return complexity


def power_spectrum_1d(signal, sample_rate=1.0):
    n = len(signal)
    n_fft = optimal_fft_length(n)

    padded = np.zeros(n_fft)
    padded[:n] = signal
    fft_vals = np.fft.rfft(padded)
    power = np.abs(fft_vals) ** 2 / n_fft
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    return freqs, power


def dominant_periods(signal, sample_rate=1.0, n_peaks=3):
    freqs, power = power_spectrum_1d(signal, sample_rate)

    power[0] = 0.0
    peak_indices = np.argsort(power)[-n_peaks:][::-1]
    periods = []
    for idx in peak_indices:
        if freqs[idx] > 1e-14:
            periods.append(1.0 / freqs[idx])
    return periods, freqs, power


def spectral_coherence(series1, series2, sample_rate=1.0):
    n = max(len(series1), len(series2))
    n_fft = optimal_fft_length(n)
    s1 = np.zeros(n_fft)
    s2 = np.zeros(n_fft)
    s1[:len(series1)] = series1
    s2[:len(series2)] = series2

    f1 = np.fft.rfft(s1)
    f2 = np.fft.rfft(s2)

    sxx = np.abs(f1) ** 2
    syy = np.abs(f2) ** 2
    sxy = f1 * np.conj(f2)

    coherence = np.abs(sxy) ** 2 / (sxx * syy + 1e-14)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    return freqs, np.real(coherence)


def test_spectral():

    assert prime_factors(75) == [3, 5, 5]

    opt = optimal_fft_length(100)
    factors = prime_factors(opt)
    assert all(f in (2, 3, 5) for f in factors)

    t = np.linspace(0, 10, 256)
    signal = np.sin(2 * np.pi * t) + 0.5 * np.sin(6 * np.pi * t)
    freqs, power = power_spectrum_1d(signal)
    assert len(freqs) == len(power)
    print("spectral_analysis 自测试通过")


if __name__ == "__main__":
    test_spectral()
