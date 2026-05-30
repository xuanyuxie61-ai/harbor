# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple, Optional





def envelope_ari(t: np.ndarray, t_rise: float = 2.0, t_flat: float = 8.0, t_decay: float = 4.0) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    env = np.zeros_like(t)
    t_total = t_rise + t_flat + t_decay


    mask_rise = (t >= 0) & (t <= t_rise)
    env[mask_rise] = (t[mask_rise] / t_rise) ** 2


    mask_flat = (t > t_rise) & (t <= t_rise + t_flat)
    env[mask_flat] = 1.0


    mask_decay = t > t_rise + t_flat
    c = np.log(100.0) / t_decay
    env[mask_decay] = np.exp(-c * (t[mask_decay] - t_rise - t_flat))


    env[t < 0] = 0.0
    return env





def kanai_tajimi_psd(omega: np.ndarray, omega_g: float = 15.0, zeta_g: float = 0.6, S0: float = 0.01) -> np.ndarray:
    omega = np.asarray(omega, dtype=float)

    omega_safe = np.where(omega == 0, 1e-12, omega)

    num = S0 * (omega_g ** 4 + 4.0 * zeta_g ** 2 * omega_g ** 2 * omega_safe ** 2)
    den = (omega_g ** 2 - omega_safe ** 2) ** 2 + 4.0 * zeta_g ** 2 * omega_g ** 2 * omega_safe ** 2

    psd = np.where(den > 0, num / den, 0.0)
    return psd





def chirikov_map_perturbation(phases: np.ndarray, K: float = 0.8, iterations: int = 3) -> np.ndarray:
    theta = np.asarray(phases, dtype=float).copy()
    N = len(theta)
    I = np.zeros(N, dtype=float)

    for _ in range(iterations):
        I = I + K * np.sin(theta)
        theta = theta + I
        theta = np.mod(theta, 2.0 * np.pi)

    return theta





def constrained_random_phases(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n)


    n_bins = 8
    max_iter = 100
    for _iter in range(max_iter):
        hist, _ = np.histogram(phases, bins=n_bins, range=(0.0, 2.0 * np.pi))
        expected = n / n_bins
        max_dev = np.max(np.abs(hist - expected)) / expected
        if max_dev < 0.20:
            break

        max_bin = np.argmax(hist)
        bin_mask = (
            (phases >= max_bin * 2.0 * np.pi / n_bins)
            & (phases < (max_bin + 1) * 2.0 * np.pi / n_bins)
        )
        over_count = int(hist[max_bin] - expected)
        if over_count > 0:
            idx_in_bin = np.where(bin_mask)[0]
            replace_idx = rng.choice(idx_in_bin, size=over_count, replace=False)

            min_bin = np.argmin(hist)
            new_low = min_bin * 2.0 * np.pi / n_bins
            new_high = (min_bin + 1) * 2.0 * np.pi / n_bins
            phases[replace_idx] = rng.uniform(new_low, new_high, size=over_count)

    return phases





def synthesize_ground_motion(
    dt: float = 0.01,
    t_max: float = 20.0,
    omega_g: float = 15.0,
    zeta_g: float = 0.6,
    S0: float = 0.02,
    K_chirikov: float = 0.5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    t = np.arange(0.0, t_max + dt, dt)
    N = len(t)

    omega_max = np.pi / dt

    domega = 2.0 * np.pi / t_max
    n_freq = int(omega_max / domega)
    omega_k = np.arange(1, n_freq + 1) * domega


    S_omega = kanai_tajimi_psd(omega_k, omega_g, zeta_g, S0)
    A_k = np.sqrt(2.0 * S_omega * domega)


    phi_k = constrained_random_phases(len(omega_k), seed=seed)
    phi_k = chirikov_map_perturbation(phi_k, K=K_chirikov, iterations=2)


    env = envelope_ari(t, t_rise=2.0, t_flat=8.0, t_decay=6.0)


    a_g = np.zeros(N, dtype=float)
    for k in range(len(omega_k)):
        a_g += A_k[k] * np.cos(omega_k[k] * t + phi_k[k])


    a_g = env * a_g


    a_g = a_g - np.mean(a_g)


    pga_target = 0.3 * 9.81
    pga_current = np.max(np.abs(a_g))
    if pga_current > 1e-12:
        a_g = a_g * (pga_target / pga_current)

    return t, a_g





def fuse_components(
    a_x: np.ndarray,
    a_y: np.ndarray,
    a_z: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    if weights is None:
        weights = np.array([0.70, 0.20, 0.10], dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)

    return weights[0] * a_x + weights[1] * a_y + weights[2] * a_z





class SeismicWaveGenerator:

    def __init__(self, dt: float = 0.01, t_max: float = 20.0, seed: int = 84):
        self.dt = dt
        self.t_max = t_max
        self.seed = seed
        self.t = np.arange(0.0, t_max + dt, dt)

    def generate(self, n_components: int = 3) -> np.ndarray:
        acc = np.zeros((len(self.t), n_components), dtype=float)
        for i in range(n_components):
            _, a = synthesize_ground_motion(
                dt=self.dt,
                t_max=self.t_max,
                seed=self.seed + i * 1000,
                omega_g=15.0 + i * 2.0,
            )
            acc[:, i] = a
        return acc

    def get_fused_record(self, weights: Optional[np.ndarray] = None) -> np.ndarray:
        acc = self.generate(n_components=3)
        return fuse_components(acc[:, 0], acc[:, 1], acc[:, 2], weights=weights)
