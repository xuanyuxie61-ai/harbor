
import numpy as np
from typing import Tuple, List


def morlet_wavelet(t: np.ndarray, scale: float, omega0: float = 6.0) -> np.ndarray:
    norm = np.pi ** (-0.25)
    psi = norm * np.exp(1j * omega0 * t / scale) * np.exp(-0.5 * (t / scale) ** 2)
    return psi


def cwt_1d(signal: np.ndarray, dt: float,
           scales: np.ndarray, omega0: float = 6.0) -> Tuple[np.ndarray, np.ndarray]:
    if signal.ndim != 1:
        raise ValueError("signal must be 1D")

    N = signal.shape[0]
    n_scales = scales.shape[0]
    W = np.zeros((n_scales, N), dtype=complex)


    signal_fft = np.fft.fft(signal)
    freqs_fft = np.fft.fftfreq(N, dt)

    for i, a in enumerate(scales):

        psi_hat = np.zeros(N, dtype=complex)
        for k in range(N):
            s = 2.0 * np.pi * freqs_fft[k] * a

            psi_hat[k] = np.pi ** (-0.25) * np.exp(-0.5 * (s - omega0) ** 2)

            if freqs_fft[k] <= 0:
                psi_hat[k] = 0.0


        conv = np.fft.ifft(signal_fft * np.conj(psi_hat))
        W[i, :] = conv * np.sqrt(dt / a)

    freqs = omega0 / (2.0 * np.pi * scales)
    return W, freqs


def global_wavelet_spectrum(W: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(W) ** 2, axis=1)


def red_noise_spectrum(scales: np.ndarray, dt: float,
                       rho: float, sigma: float) -> np.ndarray:
    if abs(rho) >= 1.0:
        rho = 0.99
    return (sigma ** 2 / (1.0 - rho ** 2)) * (1.0 - rho ** (2.0 * scales))


def find_scale_peaks(P_global: np.ndarray, scales: np.ndarray,
                     min_prominence: float = 0.1) -> List[Tuple[float, float]]:
    peaks = []
    max_p = np.max(P_global)
    if max_p < 1e-14:
        return peaks

    for i in range(1, P_global.shape[0] - 1):
        if P_global[i] > P_global[i - 1] and P_global[i] > P_global[i + 1]:
            if P_global[i] / max_p > min_prominence:
                peaks.append((float(scales[i]), float(P_global[i])))

    return peaks


def nested_multiscale_analysis(signal: np.ndarray, dt: float,
                               n_levels: int = 5) -> dict:

    s0 = 2.0 * dt
    s_max = len(signal) * dt / 4.0
    scales = s0 * (2.0 ** np.linspace(0, np.log2(s_max / s0), n_levels * 10))

    W, freqs = cwt_1d(signal, dt, scales)
    P_global = global_wavelet_spectrum(W)


    rho_est = np.corrcoef(signal[:-1], signal[1:])[0, 1]
    sigma_est = np.std(signal)
    P_red = red_noise_spectrum(scales, dt, rho_est, sigma_est)


    peaks = find_scale_peaks(P_global, scales, min_prominence=0.05)
    significant_peaks = []
    for s, p in peaks:
        idx = np.argmin(np.abs(scales - s))
        if P_global[idx] > 1.5 * P_red[idx]:
            period_months = s / dt
            significant_peaks.append({
                "scale": s,
                "period_months": period_months,
                "power": p,
            })

    if not significant_peaks and len(peaks) > 0:
        best_peak = max(peaks, key=lambda x: x[1])
        idx = np.argmin(np.abs(scales - best_peak[0]))
        significant_peaks.append({
            "scale": best_peak[0],
            "period_months": best_peak[0] / dt,
            "power": best_peak[1],
        })

    return {
        "scales": scales,
        "frequencies": freqs,
        "global_spectrum": P_global,
        "red_noise_spectrum": P_red,
        "ar1_rho": float(rho_est),
        "ar1_sigma": float(sigma_est),
        "significant_peaks": significant_peaks,
    }


def cross_scale_energy_flux(W: np.ndarray, scales: np.ndarray,
                            threshold: float = 0.5) -> np.ndarray:
    n_scales = scales.shape[0]
    flux = np.zeros(n_scales - 1)

    P = np.abs(W) ** 2
    for i in range(n_scales - 1):

        ratio = P[i + 1] / (P[i] + 1e-14)
        flux[i] = np.mean(np.log(ratio + 1e-14))

    return flux
