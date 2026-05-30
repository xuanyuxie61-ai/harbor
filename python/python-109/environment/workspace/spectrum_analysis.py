
import numpy as np
from typing import Tuple


def spectral_bandwidth(omega: np.ndarray, power_spectrum: np.ndarray,
                       method: str = "fwhm") -> float:
    if len(omega) != len(power_spectrum):
        raise ValueError("spectral_bandwidth: length mismatch")
    p = np.maximum(power_spectrum, 0.0)
    p_max = np.max(p)
    if p_max <= 0.0:
        return 0.0
    if method == "fwhm":
        threshold = p_max * 0.5
        indices = np.where(p > threshold)[0]
        if len(indices) == 0:
            return 0.0
        return float(omega[indices[-1]] - omega[indices[0]])
    elif method == "twenty_db":
        threshold = p_max * 0.01
        indices = np.where(p > threshold)[0]
        if len(indices) == 0:
            return 0.0
        return float(omega[indices[-1]] - omega[indices[0]])
    elif method == "rms":
        p_norm = p / (np.trapz(p, omega) + 1e-20)
        mean_o = np.trapz(omega * p_norm, omega)
        mean_o2 = np.trapz(omega ** 2 * p_norm, omega)
        return float(np.sqrt(max(mean_o2 - mean_o ** 2, 0.0)))
    else:
        raise ValueError(f"spectral_bandwidth: unknown method {method}")


def spectral_flatness(power_spectrum: np.ndarray) -> float:
    p = np.maximum(power_spectrum, 1e-20)
    geo_mean = np.exp(np.mean(np.log(p)))
    arith_mean = np.mean(p)
    if arith_mean <= 0.0:
        return 0.0
    return float(geo_mean / arith_mean)


def coherence_degree(spectrum_ensemble: np.ndarray) -> np.ndarray:
    if spectrum_ensemble.ndim != 2:
        raise ValueError("coherence_degree: expected 2D array")
    mean_field = np.mean(spectrum_ensemble, axis=0)
    mean_power = np.mean(np.abs(spectrum_ensemble) ** 2, axis=0)
    coherence = np.abs(mean_field) / np.sqrt(mean_power + 1e-20)
    coherence = np.clip(coherence, 0.0, 1.0)
    return coherence


def dispersion_length(T0: float, beta2: float) -> float:
    if abs(beta2) < 1e-30:
        return 1e20
    return float(T0 ** 2 / abs(beta2))


def nonlinear_length(gamma: float, P0: float) -> float:
    if gamma <= 0.0 or P0 <= 0.0:
        return 1e20
    return float(1.0 / (gamma * P0))


def soliton_order(beta2: float, gamma: float, T0: float, P0: float) -> float:
    ld = dispersion_length(T0, beta2)
    lnl = nonlinear_length(gamma, P0)
    return float(np.sqrt(ld / lnl))


def fourier_limit_duration(bandwidth_hz: float, pulse_shape: str = "sech") -> float:
    if bandwidth_hz <= 0.0:
        return 0.0
    if pulse_shape == "sech":
        return 0.315 / bandwidth_hz
    elif pulse_shape == "gaussian":
        return 0.441 / bandwidth_hz
    else:
        return 0.4 / bandwidth_hz


def spectral_snr(power_spectrum: np.ndarray, signal_band: Tuple[int, int]) -> float:
    start, end = signal_band
    p = np.maximum(power_spectrum, 1e-20)
    signal_power = np.mean(p[start:end])

    noise_indices = list(range(0, start)) + list(range(end, len(p)))
    if len(noise_indices) == 0:
        noise_power = 1e-20
    else:
        noise_power = np.mean(p[noise_indices])
    snr_db = 10.0 * np.log10(signal_power / noise_power)
    return float(snr_db)
