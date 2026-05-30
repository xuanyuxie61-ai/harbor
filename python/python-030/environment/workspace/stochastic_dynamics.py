# -*- coding: utf-8 -*-

import numpy as np


def langevin_step(r, force_func, gamma, temperature, dt, dim=3):
    r = np.asarray(r, dtype=float)
    F = np.asarray(force_func(r), dtype=float)

    sigma = np.sqrt(2.0 * temperature * dt / gamma)
    noise = sigma * np.random.randn(dim)
    drift = (dt / gamma) * F
    return r + drift + noise


def run_langevin_trajectory(r0, force_func, gamma, temperature, dt,
                            n_steps, dim=3, seed=None):
    if seed is not None:
        np.random.seed(seed)
    traj = np.zeros((n_steps + 1, dim))
    traj[0, :] = r0
    for n in range(n_steps):
        traj[n + 1, :] = langevin_step(traj[n, :], force_func, gamma,
                                       temperature, dt, dim)
    return traj


def ensemble_langevin(r0_list, force_func, gamma, temperature, dt,
                      n_steps, dim=3, seed=None):
    if seed is not None:
        np.random.seed(seed)
    n_ens = r0_list.shape[0]
    all_traj = np.zeros((n_ens, n_steps + 1, dim))
    for i in range(n_ens):
        all_traj[i, :, :] = run_langevin_trajectory(
            r0_list[i, :], force_func, gamma, temperature, dt, n_steps, dim)

    msd = np.mean(np.sum((all_traj - all_traj[:, 0:1, :]) ** 2, axis=2), axis=0)
    mean_traj = np.mean(all_traj, axis=0)
    return msd, mean_traj


def diffusion_coefficient_from_msd(msd, dt):
    t = np.arange(msd.size) * dt

    start = msd.size // 2
    if start < 2:
        start = 1
    coef = np.polyfit(t[start:], msd[start:], 1)
    slope = coef[0]

    return slope / 6.0


def nuclear_temperature(excitation_energy, A):
    a = A / 8.0
    if excitation_energy <= 0 or a <= 0:
        return 0.0
    return np.sqrt(excitation_energy / a)


def evaporative_decay_rate(A, Z, T, separation_energy):
    if T <= 0 or separation_energy <= 0:
        return 0.0

    g = 2.0
    sigma = np.pi * (1.2 * (A ** (1.0 / 3.0))) ** 2

    prefactor = 1.0e-4
    rate = prefactor * g * sigma * (T ** 2) * np.exp(-separation_energy / T)
    return rate
