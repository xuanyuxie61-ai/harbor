
import numpy as np
from typing import Tuple, Callable, Optional


def etdrk4_coefficients(nx: int, dt: float, vis: float, m: int = 64) -> Tuple:
    if nx % 2 != 0:
        raise ValueError("nx must be even for FFT")
    if vis < 0:
        raise ValueError("viscosity must be non-negative")
    if dt <= 0:
        raise ValueError("dt must be positive")


    k = np.concatenate([
        np.arange(0, nx // 2),
        np.array([0]),
        np.arange(-nx // 2 + 1, 0)
    ])


    L = 1j * vis * k ** 2

    E = np.exp(dt * L)
    E2 = np.exp(dt * L / 2.0)


    r = np.exp(2.0j * np.pi * (np.arange(1, m + 1) - 0.5) / m)


    LR = dt * L[:, np.newaxis] + r[np.newaxis, :]


    Q = dt * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1))
    f1 = dt * np.real(np.mean(
        (-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR ** 2)) / LR ** 3,
        axis=1
    ))
    f2 = dt * np.real(np.mean(
        (2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR ** 3,
        axis=1
    ))
    f3 = dt * np.real(np.mean(
        (-4.0 - 3.0 * LR - LR ** 2 + np.exp(LR) * (4.0 - LR)) / LR ** 3,
        axis=1
    ))


    g = -0.5j * k

    return k, L, E, E2, Q, f1, f2, f3, g


def solve_burgers_etdrk4(nx: int, nt: int, vis: float,
                         tmax: float = 1.0,
                         forcing: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if nx < 4:
        raise ValueError("nx must be at least 4")
    if nt < 2:
        raise ValueError("nt must be at least 2")


    x = np.linspace(-np.pi, np.pi, nx + 1)[:-1]


    u = np.exp(-10.0 * np.sin(0.5 * x) ** 2)
    v = np.fft.fft(u)


    dt = 0.4 / nx ** 2
    if vis > 0.01:
        dt = min(dt, 0.4 / (vis * nx ** 2))

    nmax = max(1, int(np.ceil(tmax / dt)))
    jstep = max(1, nmax // (nt - 1)) if nt > 1 else 1


    _, _, E, E2, Q, f1, f2, f3, g = etdrk4_coefficients(nx, dt, vis)


    uu = np.zeros((nx, nt), dtype=float)
    tt = np.zeros(nt, dtype=float)
    uu[:, 0] = u
    tt[0] = 0.0

    out_idx = 1

    for i in range(1, nmax + 1):
        t = i * dt


        u_phys = np.real(np.fft.ifft(v))
        Nv = g * np.fft.fft(u_phys ** 2)

        if forcing is not None:
            f_vec = forcing(x, t)
            Nv += dt * np.fft.fft(f_vec) / nx

        a = E2 * v + Q * Nv
        Na = g * np.fft.fft(np.real(np.fft.ifft(a)) ** 2)

        b = E2 * v + Q * Na
        Nb = g * np.fft.fft(np.real(np.fft.ifft(b)) ** 2)

        c = E2 * a + Q * (2.0 * Nb - Nv)
        Nc = g * np.fft.fft(np.real(np.fft.ifft(c)) ** 2)

        v = E * v + Nv * f1 + 2.0 * (Na + Nb) * f2 + Nc * f3


        if out_idx < nt and i % jstep == 0:
            u_out = np.real(np.fft.ifft(v))
            uu[:, out_idx] = u_out
            tt[out_idx] = t
            out_idx += 1


    while out_idx < nt:
        uu[:, out_idx] = np.real(np.fft.ifft(v))
        tt[out_idx] = tmax
        out_idx += 1

    return x, tt, uu


def kelvin_wave_amplitude(x: np.ndarray, t: float,
                          c_k: float = 2.5,
                          decay: float = 0.1,
                          width: float = 0.5) -> np.ndarray:
    return np.exp(-((x - c_k * t) ** 2) / (2.0 * width ** 2)) * np.exp(-decay * t)


def rossby_wave_amplitude(x: np.ndarray, t: float,
                          c_r: float = -0.8,
                          decay: float = 0.05,
                          width: float = 0.8) -> np.ndarray:
    return np.exp(-((x - c_r * t) ** 2) / (2.0 * width ** 2)) * np.exp(-decay * t)


def solve_coupled_wave_envelope(nx: int, nt: int,
                                vis: float = 0.03,
                                tmax: float = 5.0,
                                coupling_strength: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    def forcing(x_arr, t_val):
        k = kelvin_wave_amplitude(x_arr, t_val)
        r = rossby_wave_amplitude(x_arr, t_val)
        return coupling_strength * (k + r)

    return solve_burgers_etdrk4(nx, nt, vis, tmax, forcing)


def wave_energy(uu: np.ndarray, dx: float) -> np.ndarray:
    return 0.5 * np.sum(uu ** 2, axis=0) * dx


def recharge_discharge_timescale(c_k: float, c_r: float, basin_width: float) -> float:
    if c_k <= 0 or abs(c_r) <= 0:
        raise ValueError("Wave speeds must be non-zero")
    tau_r = basin_width / c_k + 2.0 * basin_width / abs(c_r)
    return tau_r
