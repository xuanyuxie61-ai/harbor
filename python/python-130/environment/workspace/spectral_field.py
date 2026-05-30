# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple, Optional


def compute_fft_spectrum(
    signal: np.ndarray,
    dt: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if dt <= 0.0:
        raise ValueError("dt must be positive.")

    n = signal.shape[0]
    if n < 2:
        raise ValueError("Signal must have at least 2 points.")


    spectrum = np.fft.fft(signal)


    freqs = np.fft.fftfreq(n, d=dt / 1000.0)


    psd = np.abs(spectrum) ** 2 / n

    return freqs, spectrum, psd


def compute_band_power(
    freqs: np.ndarray,
    psd: np.ndarray,
    band: Tuple[float, float],
) -> float:
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
    if h <= 0.0:
        raise ValueError("h must be positive.")
    if n < 1:
        raise ValueError("n must be >= 1.")

    dx = x - xj

    dx = np.mod(dx + np.pi * h * n, 2.0 * np.pi * h * n) - np.pi * h * n

    denom = n * np.sin(dx / (2.0 * h))


    C = np.ones_like(x, dtype=float)
    nonzero = np.abs(denom) > 1e-12
    C[nonzero] = np.sin(n * dx[nonzero] / (2.0 * h)) / denom[nonzero]

    return C


def trig_interpolate(
    xd: np.ndarray,
    yd: np.ndarray,
    xi: np.ndarray,
) -> np.ndarray:
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
    if n < 1:
        raise ValueError("n must be >= 1.")
    if n == 1:
        return np.array([(a + b) / 2.0])

    theta = np.pi * np.arange(n) / (n - 1)
    c = np.cos(theta)


    if n % 2 == 1:
        c[(n - 1) // 2] = 0.0

    x = ((1.0 - c) * a + (1.0 + c) * b) / 2.0
    return x


def divided_differences(
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
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
    xd = chebyspace(a, b, n)
    yd = f(xd)
    dd = divided_differences(xd, yd)
    yi = newton_interpolate(xd, dd, xi)
    return yi


def analyze_neural_field_spectrum(
    n_points: int = 512,
    t_max: float = 1000.0,
) -> dict:
    if n_points < 2:
        raise ValueError("n_points must be >= 2.")
    if t_max <= 0.0:
        raise ValueError("t_max must be positive.")

    dt = t_max / n_points
    t = np.linspace(0.0, t_max, n_points)


    rng = np.random.default_rng(130)
    signal = (
        0.5 * np.sin(2.0 * np.pi * 6.0 * t / 1000.0)
        + 0.3 * np.sin(2.0 * np.pi * 20.0 * t / 1000.0)
        + 0.2 * np.sin(2.0 * np.pi * 40.0 * t / 1000.0)
        + 0.1 * rng.standard_normal(n_points)
    )

    freqs, spectrum, psd = compute_fft_spectrum(signal, dt)


    pos_mask = freqs >= 0
    freqs_pos = freqs[pos_mask]
    psd_pos = psd[pos_mask]


    bands = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 100.0),
    }
    band_powers = {name: compute_band_power(freqs_pos, psd_pos, band) for name, band in bands.items()}


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

    def transfer_func(x):
        return 1.0 / (1.0 + np.exp(-5.0 * (x - 0.5)))


    n_trig = 16
    xd_trig = np.linspace(0.0, 2.0 * np.pi, n_trig, endpoint=False)
    yd_trig = transfer_func(np.sin(xd_trig) * 0.5 + 0.5)
    xi_trig = np.linspace(0.0, 2.0 * np.pi, n_test, endpoint=False)
    yi_trig = trig_interpolate(xd_trig, yd_trig, xi_trig)
    y_true_trig = transfer_func(np.sin(xi_trig) * 0.5 + 0.5)
    err_trig = np.linalg.norm(yi_trig - y_true_trig) / np.linalg.norm(y_true_trig)


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
