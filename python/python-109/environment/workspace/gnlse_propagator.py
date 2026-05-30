
import numpy as np
from typing import Callable, Optional, Tuple


def dispersion_operator_fft(omega: np.ndarray, alpha: float,
                             beta_coeffs: np.ndarray) -> np.ndarray:





    raise NotImplementedError("Hole 1: 请实现 dispersion_operator_fft 的色散算子构建")


def linear_step_fft(A_freq: np.ndarray, dz: float,
                    D_omega: np.ndarray) -> np.ndarray:
    return A_freq * np.exp(D_omega * dz)


def nonlinear_step_rk4(A_time: np.ndarray, dz: float, gamma: float,
                       omega0: float, dt: float,
                       raman_conv: np.ndarray,
                       use_shock: bool = True) -> np.ndarray:
    def rhs(A):
        pwr = np.abs(A) ** 2

        pwr = np.clip(pwr, 0.0, 1e12)
        raman_local = np.clip(raman_conv, -1e12, 1e12)
        nonlinear_term = A * ((1.0 - 0.18) * pwr + raman_local)
        if use_shock and omega0 > 0.0:

            n = len(A)
            dAdt = np.zeros_like(A, dtype=complex)
            if n >= 3:
                dAdt[1:-1] = (A[2:] - A[:-2]) / (2.0 * dt)
                dAdt[0] = (A[1] - A[0]) / dt
                dAdt[-1] = (A[-1] - A[-2]) / dt
            nonlinear_term = nonlinear_term + (1j / omega0) * dAdt * ((1.0 - 0.18) * pwr + raman_local)
        result = 1j * gamma * nonlinear_term

        result = np.where(np.isfinite(result), result, 0.0)
        return result

    k1 = rhs(A_time)
    k2 = rhs(A_time + 0.5 * k1 * dz)
    k3 = rhs(A_time + 0.5 * k2 * dz)
    k4 = rhs(A_time + k3 * dz)
    A_new = A_time + dz * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return A_new


def arclength_parameterization(y: np.ndarray, t: np.ndarray) -> Tuple[float, np.ndarray]:
    n = len(t)
    if n < 2:
        return 0.0, np.zeros_like(t)
    dt = t[1] - t[0]

    dydt = np.zeros_like(y, dtype=complex)
    if n >= 3:
        dydt[1:-1] = (y[2:] - y[:-2]) / (2.0 * dt)
        dydt[0] = (y[1] - y[0]) / dt
        dydt[-1] = (y[-1] - y[-2]) / dt
    else:
        dydt[0] = (y[1] - y[0]) / dt
    fx = np.abs(dydt)

    s = dt * (np.sum(fx) - 0.5 * fx[0] - 0.5 * fx[-1])

    cum = np.zeros(n)
    for i in range(1, n):
        cum[i] = cum[i - 1] + 0.5 * (fx[i - 1] + fx[i]) * dt
    s_param = cum / (s + 1e-20)
    return float(s), s_param


def adaptive_step_size_estimate(A_time: np.ndarray, t: np.ndarray,
                                 dz_current: float,
                                 z: float, z_target: float,
                                 min_dz: float = 1e-6,
                                 max_dz: float = 1e-2) -> float:
    S, _ = arclength_parameterization(A_time, t)
    S_target = 1.0
    ratio = S_target / (S + 1e-10)
    dz_new = dz_current * np.sqrt(np.clip(ratio, 0.1, 10.0))

    dz_new = np.clip(dz_new, min_dz, max_dz)

    remaining = z_target - z
    if remaining > 0:
        dz_new = min(dz_new, remaining)
    return float(dz_new)


def ssfm_propagate(A0_time: np.ndarray, t: np.ndarray, z_target: float,
                   alpha: float, gamma: float, beta_coeffs: np.ndarray,
                   omega0: float, f_R: float = 0.18,
                   tau1: float = 12.2e-15, tau2: float = 32.0e-15,
                   dz_initial: float = 1e-4,
                   n_z_records: int = 100,
                   use_symmetrized: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_t = len(t)
    dt = t[1] - t[0]

    omega = 2.0 * np.pi * np.fft.fftfreq(n_t, dt)

    D_omega = dispersion_operator_fft(omega, alpha, beta_coeffs)


    z_out = np.linspace(0.0, z_target, n_z_records)
    A_z = np.zeros((n_z_records, n_t), dtype=complex)
    spec_z = np.zeros((n_z_records, n_t), dtype=float)

    A = A0_time.copy()
    z = 0.0
    dz = dz_initial
    record_idx = 0


    A_z[0, :] = A
    spec_z[0, :] = np.abs(np.fft.fft(A)) ** 2

    while z < z_target and dz > 1e-12:

        if z + dz > z_target:
            dz = z_target - z


        if use_symmetrized:
            A_freq = np.fft.fft(A)
            A_freq = A_freq * np.exp(D_omega * dz * 0.5)
            A = np.fft.ifft(A_freq)


        pwr = np.abs(A) ** 2
        if f_R > 0.0:
            from nonlinear_response import raman_response_convolution
            raman_conv = raman_response_convolution(pwr, dt, f_R, tau1, tau2)
        else:
            raman_conv = np.zeros_like(pwr)


        A = nonlinear_step_rk4(A, dz, gamma, omega0, dt, raman_conv, use_shock=True)


        if use_symmetrized:
            A_freq = np.fft.fft(A)
            A_freq = A_freq * np.exp(D_omega * dz * 0.5)
            A = np.fft.ifft(A_freq)
        else:

            A_freq = np.fft.fft(A)
            A_freq = A_freq * np.exp(D_omega * dz)
            A = np.fft.ifft(A_freq)

        z += dz


        if record_idx + 1 < n_z_records and z >= z_out[record_idx + 1]:
            record_idx += 1
            A_z[record_idx, :] = A
            spec_z[record_idx, :] = np.abs(np.fft.fft(A)) ** 2


        dz = adaptive_step_size_estimate(A, t, dz, z, z_target)


    while record_idx + 1 < n_z_records:
        record_idx += 1
        A_z[record_idx, :] = A
        spec_z[record_idx, :] = np.abs(np.fft.fft(A)) ** 2

    return z_out, A_z, spec_z


def sech_pulse(t: np.ndarray, T0: float, P0: float,
               C: float = 0.0, omega_shift: float = 0.0) -> np.ndarray:
    if T0 <= 0.0:
        raise ValueError("sech_pulse: T0 must be > 0")
    envelope = np.sqrt(P0) / np.cosh(t / T0)
    phase = -0.5 * C * (t / T0) ** 2 + omega_shift * t
    return envelope * np.exp(1j * phase)


def gaussian_pulse(t: np.ndarray, T0: float, P0: float,
                   C: float = 0.0) -> np.ndarray:
    if T0 <= 0.0:
        raise ValueError("gaussian_pulse: T0 must be > 0")
    return np.sqrt(P0) * np.exp(-(1.0 + 1j * C) * (t ** 2) / (2.0 * T0 ** 2))
