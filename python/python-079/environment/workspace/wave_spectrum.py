
import numpy as np
from typing import Tuple, Optional
from utils import cliff_rng_sequence, arc_cosine_safe, normalize_vector


def jonswap_spectrum(
    f: np.ndarray,
    fp: float,
    Hs: float,
    gamma_peak: float = 3.3,
) -> np.ndarray:
    f = np.asarray(f, dtype=float)
    if fp <= 0:
        raise ValueError("谱峰频率 fp 必须为正")
    if Hs <= 0:
        raise ValueError("有效波高 Hs 必须为正")
    g = 9.80665
    alpha = 0.0081
    sigma = np.where(f <= fp, 0.07, 0.09)
    A = alpha * g * g * (2.0 * np.pi) ** (-4.0)
    term1 = f ** (-5.0)
    term2 = np.exp(-1.25 * (f / fp) ** (-4.0))
    term3 = gamma_peak ** (np.exp(-0.5 * ((f - fp) / (sigma * fp)) ** 2.0))
    S = A * term1 * term2 * term3


    df = np.diff(f)
    if len(df) > 0:
        m0 = np.trapezoid(S, f)
        if m0 > 1e-12:
            target_m0 = (Hs / 4.0) ** 2.0
            S *= target_m0 / m0
    S = np.where(f <= 0, 0.0, S)
    return S


def dispersion_relation_deep_water(f: np.ndarray) -> np.ndarray:
    f = np.asarray(f, dtype=float)
    g = 9.80665
    omega = 2.0 * np.pi * f
    k = omega * omega / g
    return np.where(f > 0, k, 0.0)


def dispersion_relation_finite_depth(
    f: np.ndarray, h: float, tol: float = 1e-10, max_iter: int = 100
) -> np.ndarray:
    f = np.asarray(f, dtype=float)
    if h <= 0:
        raise ValueError("水深 h 必须为正")
    g = 9.80665
    omega = 2.0 * np.pi * f
    k = omega * omega / g
    k = np.where(f <= 0, 0.0, k)
    for _ in range(max_iter):
        mask = f > 0
        if not np.any(mask):
            break
        km = k[mask]
        om = omega[mask]
        fk = g * km * np.tanh(km * h) - om * om
        dfk = g * np.tanh(km * h) + g * km * h / (np.cosh(km * h) ** 2.0)
        delta = fk / dfk
        k[mask] = km - delta
        if np.max(np.abs(delta)) < tol:
            break
    return k


def directional_spreading_gaussian(
    theta: np.ndarray,
    theta_mean: float,
    n_spread: float,
) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    if n_spread < 0:
        raise ValueError("方向集中度 n_spread 必须非负")
    dtheta = theta - theta_mean

    dtheta = np.mod(dtheta + np.pi, 2.0 * np.pi) - np.pi
    cos_half = np.cos(dtheta * 0.5)
    cos_half = np.where(cos_half < 0, -cos_half, cos_half)
    D = cos_half ** (2.0 * n_spread)

    integral = np.trapezoid(D, theta)
    if integral > 1e-15:
        D /= integral
    else:
        D = np.ones_like(D) / len(D)
    return D


def anisotropic_gaussian_2d(
    X: np.ndarray,
    Y: np.ndarray,
    xmu: float,
    ymu: float,
    xsigma: float,
    ysigma: float,
    A: np.ndarray,
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    A = np.asarray(A, dtype=float)
    if A.shape != (2, 2):
        raise ValueError("A 必须是 2×2 矩阵")
    if xsigma <= 0 or ysigma <= 0:
        raise ValueError("标准差必须为正")
    vx = (X - xmu) / xsigma
    vy = (Y - ymu) / ysigma

    q = A[0, 0] * vx * vx + (A[0, 1] + A[1, 0]) * vx * vy + A[1, 1] * vy * vy
    q = np.maximum(q, 0.0)
    Z = np.exp(-0.5 * q)
    return Z


def hypersphere_uniform_sample(m: int, n: int, seed: float = 0.5) -> np.ndarray:
    if m < 1:
        raise ValueError("维度 m 至少为 1")
    if n < 1:
        raise ValueError("采样数 n 至少为 1")

    seq = cliff_rng_sequence(seed, m * n * 2)

    samples = np.zeros((m, n), dtype=float)
    idx = 0
    for j in range(n):
        for i in range(m):
            if idx + 1 >= len(seq):
                seq = cliff_rng_sequence(seq[-1] if not np.isnan(seq[-1]) else 0.3, m * n * 2)
                idx = 0
            u1 = max(seq[idx], 1e-10)
            u2 = seq[idx + 1]
            idx += 2

            z = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
            samples[i, j] = z

        norm = np.linalg.norm(samples[:, j])
        if norm > 1e-15:
            samples[:, j] /= norm
    return samples


def directional_spectrum_integrate_montecarlo(
    S_f: np.ndarray,
    f_arr: np.ndarray,
    theta_mean: float,
    n_spread: float,
    n_samples: int = 256,
) -> Tuple[float, float]:
    if len(S_f) != len(f_arr):
        raise ValueError("S_f 与 f_arr 长度不一致")

    dirs = hypersphere_uniform_sample(2, n_samples, seed=0.42)
    thetas = np.arctan2(dirs[1, :], dirs[0, :])
    D_vals = directional_spreading_gaussian(thetas, theta_mean, n_spread)

    df = np.mean(np.diff(f_arr)) if len(f_arr) > 1 else 1.0
    dtheta = 2.0 * np.pi / n_samples
    m0 = np.sum(S_f) * df

    dir_mean = np.arctan2(
        np.sum(D_vals * np.sin(thetas)) / n_samples,
        np.sum(D_vals * np.cos(thetas)) / n_samples,
    )
    return float(m0), float(dir_mean)


def synthesize_wave_elevation_1d(
    x: np.ndarray,
    t: float,
    fp: float,
    Hs: float,
    theta_mean: float,
    n_spread: float,
    h: float = 100.0,
    n_freq: int = 64,
    n_dir: int = 36,
    seed: float = 0.37,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if fp <= 0 or Hs <= 0:
        raise ValueError("fp 和 Hs 必须为正")
    f_min = max(0.05, fp * 0.2)
    f_max = min(2.0, fp * 5.0)
    f_arr = np.linspace(f_min, f_max, n_freq)
    df = f_arr[1] - f_arr[0] if n_freq > 1 else 1.0
    theta_arr = np.linspace(-np.pi, np.pi, n_dir, endpoint=False)
    dtheta = theta_arr[1] - theta_arr[0] if n_dir > 1 else 2.0 * np.pi

    S_f = jonswap_spectrum(f_arr, fp, Hs)
    D_t = directional_spreading_gaussian(theta_arr, theta_mean, n_spread)
    k_arr = dispersion_relation_finite_depth(f_arr, h)
    omega_arr = 2.0 * np.pi * f_arr

    eta = np.zeros_like(x)
    rng = cliff_rng_sequence(seed, n_freq * n_dir)
    idx = 0
    for i in range(n_freq):
        for j in range(n_dir):
            amp = np.sqrt(2.0 * S_f[i] * D_t[j] * df * dtheta)
            phase = 2.0 * np.pi * rng[idx]
            idx += 1
            if idx >= len(rng):
                rng = cliff_rng_sequence(rng[-1] if not np.isnan(rng[-1]) else 0.3, n_freq * n_dir)
                idx = 0
            eta += amp * np.cos(k_arr[i] * x * np.cos(theta_arr[j]) - omega_arr[i] * t + phase)
    return eta


def wave_group_velocity(f: np.ndarray, h: float) -> np.ndarray:
    f = np.asarray(f, dtype=float)
    g = 9.80665
    k = dispersion_relation_finite_depth(f, h)
    kh = k * h
    kh = np.where(kh > 100, 100.0, kh)
    cg = np.zeros_like(k)
    mask = k > 1e-12
    cg[mask] = 0.5 * (g / k[mask]) * (1.0 + 2.0 * kh[mask] / np.sinh(2.0 * kh[mask]))
    return cg
