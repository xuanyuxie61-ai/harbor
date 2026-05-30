
import numpy as np
from parameters import N_FFT


def compute_fft_spectrum(signal, dt=1.0e-3):
    signal = np.asarray(signal, dtype=float)
    N = len(signal)
    if N < 2:
        return np.array([0.0]), np.array([0.0])


    N_fft = max(N, N_FFT)
    N_fft = 1 << (N_fft - 1).bit_length()

    signal_pad = np.zeros(N_fft)
    signal_pad[:N] = signal


    X = np.fft.fft(signal_pad)
    power = np.abs(X) ** 2 / N_fft
    freqs = np.fft.fftfreq(N_fft, d=dt)


    positive = freqs >= 0
    return freqs[positive], power[positive]


def compute_wavenumber_spectrum(phi_signal, m_max=32, n_max=16):
    phi = np.asarray(phi_signal, dtype=float)
    if phi.ndim != 3:
        raise ValueError("phi_signal 必须为 3 维数组 (theta, phi, t)")
    n_theta, n_phi, n_t = phi.shape

    P_mn = np.zeros((m_max + 1, n_max + 1))
    gamma_mn = np.zeros((m_max + 1, n_max + 1))

    for m in range(m_max + 1):
        for n in range(n_max + 1):

            A = np.zeros(n_t, dtype=complex)
            for it in range(n_t):
                phase = np.outer(np.arange(n_theta), m) * 2j * np.pi / n_theta
                phase += np.outer(np.arange(n_phi), n) * 2j * np.pi / n_phi

                fft2 = np.fft.fft2(phi[:, :, it])
                if m < fft2.shape[0] and n < fft2.shape[1]:
                    A[it] = fft2[m, n]

            power_t = np.abs(A) ** 2
            P_mn[m, n] = np.mean(power_t)


            if n_t > 10 and np.mean(power_t) > 1e-30:
                logP = np.log(power_t + 1e-30)
                t_idx = np.arange(n_t)

                slope = np.cov(t_idx, logP)[0, 1] / np.var(t_idx)
                gamma_mn[m, n] = 0.5 * slope

    return P_mn, gamma_mn


def alfvén_dispersion(k_parallel, k_perp, B, rho_m):
    from parameters import MU0
    v_A = B / np.sqrt(MU0 * rho_m + 1e-30)
    k_par = np.asarray(k_parallel)
    k_perp = np.asarray(k_perp)
    rho_s = 1.0e-3
    omega = np.abs(k_par) * v_A / np.sqrt(1.0 + (k_perp * rho_s) ** 2)
    return omega, v_A


def compute_growth_rate_from_spectrum(power_history, dt=1.0e-3):
    P = np.asarray(power_history, dtype=float)
    P = np.maximum(P, 1e-30)
    logP = np.log(P)
    t = np.arange(len(P)) * dt

    if len(t) < 2:
        return 0.0, 0.0


    slope, intercept = np.polyfit(t, logP, 1)
    gamma = 0.5 * slope


    y_mean = np.mean(logP)
    ss_tot = np.sum((logP - y_mean) ** 2)
    ss_res = np.sum((logP - (slope * t + intercept)) ** 2)
    r_squared = 1.0 - ss_res / (ss_tot + 1e-30)

    return gamma, r_squared


def detect_unstable_modes(P_mn, gamma_mn, power_threshold=1.0e-3, gamma_threshold=0.0):
    P_max = np.max(P_mn)
    if P_max < 1e-30:
        return []

    unstable = []
    m_max, n_max = P_mn.shape[0] - 1, P_mn.shape[1] - 1
    for m in range(m_max + 1):
        for n in range(n_max + 1):
            if P_mn[m, n] / P_max > power_threshold and gamma_mn[m, n] > gamma_threshold:
                unstable.append((m, n, float(gamma_mn[m, n]), float(P_mn[m, n])))


    unstable.sort(key=lambda x: x[2], reverse=True)
    return unstable


def generate_turbulent_signal(n_t=2048, dt=1.0e-4, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n_t) * dt


    modes = [
        {"A": 1.0, "f": 5.0e3, "gamma": 1.0e2, "phi": 0.0},
        {"A": 0.6, "f": 12.0e3, "gamma": 5.0e1, "phi": 1.2},
        {"A": 0.3, "f": 25.0e3, "gamma": -2.0e1, "phi": 2.5},
    ]

    signal = np.zeros(n_t)
    for mode in modes:
        envelope = mode["A"] * np.exp(mode["gamma"] * t)
        signal += envelope * np.sin(2.0 * np.pi * mode["f"] * t + mode["phi"])


    noise = rng.normal(0, 0.1, n_t)
    signal += noise

    true_params = {"modes": modes, "dt": dt, "noise_std": 0.1}
    return signal, true_params
