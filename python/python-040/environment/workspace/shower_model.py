#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple


def lambert_w_approx(z: float, max_iter: int = 50) -> float:
    if z < -1.0 / np.e + 1e-10:
        return -1.0
    if z == 0.0:
        return 0.0


    if z > np.e:
        w = np.log(z) - np.log(np.log(z))
    else:
        w = z / np.e

    for _ in range(max_iter):
        ew = np.exp(w)
        f = w * ew - z
        if abs(f) < 1e-12:
            break
        df = ew * (w + 1.0)
        ddf = ew * (w + 2.0)

        w = w - f / (df - f * ddf / (2.0 * df))

    return w


def flame_ode_solve(
    t_span: Tuple[float, float],
    y0: float,
    delta: float = 0.0001,
    n_steps: int = 10000
) -> Tuple[np.ndarray, np.ndarray]:
    t0, tf = t_span
    if tf <= t0:
        raise ValueError("t_span 必须满足 t0 < tf")
    if delta <= 0.0:
        raise ValueError("delta 必须为正")
    if y0 <= 0.0:
        y0 = delta

    dt = (tf - t0) / n_steps
    t = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros(n_steps + 1)
    y[0] = y0


    A_param = 1.0 / delta - 1.0

    for i in range(n_steps):

        y_pred = y[i] + dt * (y[i] ** 2 * (1.0 - y[i]))

        y_pred = np.clip(y_pred, 0.0, 1.0)

        f_i = y[i] ** 2 * (1.0 - y[i])
        f_pred = y_pred ** 2 * (1.0 - y_pred)
        y[i + 1] = y[i] + 0.5 * dt * (f_i + f_pred)
        y[i + 1] = np.clip(y[i + 1], 0.0, 1.0)

    return t, y


def electromagnetic_shower_profile(
    depth_x0: np.ndarray,
    E0: float = 100.0,
    Ec: float = 0.008
) -> np.ndarray:
    depth_x0 = np.atleast_1d(depth_x0)
    if E0 <= 0.0 or Ec <= 0.0:
        return np.zeros_like(depth_x0)



    delta = min(Ec / E0, 0.99)
    tf = np.max(depth_x0)

    t_ode, y_ode = flame_ode_solve((0.0, tf), delta, delta, n_steps=5000)


    profile = np.interp(depth_x0, t_ode, y_ode, left=0.0, right=0.0)


    profile *= (E0 / Ec)

    return profile


def burgers_hadronization_pde(
    nx: int = 256,
    nt: int = 200,
    viscosity: float = 0.03,
    t_max: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if nx < 4:
        raise ValueError("nx 必须 >= 4")


    x = np.linspace(-np.pi, np.pi, nx + 1)[:-1]


    u = np.exp(-10.0 * (np.sin(0.5 * x) ** 2))
    v = np.fft.fft(u)


    dt = 0.4 / nx ** 2
    nmax = max(int(t_max / dt), nt * 10)
    jstep = max(nmax // (nt - 1), 1)


    k = np.concatenate([np.arange(nx // 2), [0], np.arange(-nx // 2 + 1, 0)])


    L = 1.0j * viscosity * (k ** 2)
    E = np.exp(dt * L)
    E2 = np.exp(dt * L / 2.0)


    m = 64
    r = np.exp(2.0j * np.pi * (np.arange(m) + 0.5) / m)

    LR = dt * L[:, None] + r[None, :]


    LR_safe = np.where(np.abs(LR) < 1e-14, 1e-14, LR)

    Q = dt * np.real(np.mean((np.exp(LR_safe / 2.0) - 1.0) / LR_safe, axis=1))
    f1 = dt * np.real(np.mean(
        (-4.0 - LR_safe + np.exp(LR_safe) * (4.0 - 3.0 * LR_safe + LR_safe ** 2))
        / LR_safe ** 3, axis=1))
    f2 = dt * np.real(np.mean(
        (2.0 + LR_safe + np.exp(LR_safe) * (-2.0 + LR_safe))
        / LR_safe ** 3, axis=1))
    f3 = dt * np.real(np.mean(
        (-4.0 - 3.0 * LR_safe - LR_safe ** 2 + np.exp(LR_safe) * (4.0 - LR_safe))
        / LR_safe ** 3, axis=1))


    g = -0.5j * k

    uu = [u.copy()]
    tt = [0.0]

    for i in range(1, nmax + 1):
        t = i * dt

        Nv = g * np.fft.fft(np.real(np.fft.ifft(v)) ** 2)
        a = E2 * v + Q * Nv
        Na = g * np.fft.fft(np.real(np.fft.ifft(a)) ** 2)
        b = E2 * v + Q * Na
        Nb = g * np.fft.fft(np.real(np.fft.ifft(b)) ** 2)
        c = E2 * a + Q * (2.0 * Nb - Nv)
        Nc = g * np.fft.fft(np.real(np.fft.ifft(c)) ** 2)

        v = E * v + Nv * f1 + 2.0 * (Na + Nb) * f2 + Nc * f3

        if i % jstep == 0 and len(tt) < nt:
            u = np.real(np.fft.ifft(v))
            uu.append(u.copy())
            tt.append(t)

    uu = np.array(uu)
    tt = np.array(tt)

    return x, tt, uu


def hadronization_energy_spectrum(
    parton_energy: float,
    n_particles: int = 100,
    fragmentation_func: str = 'lund'
) -> np.ndarray:
    if parton_energy <= 0.0:
        return np.zeros(n_particles)


    alpha = 0.3
    beta = 1.5



    try:
        from numpy.random import default_rng
        rng = default_rng()
        z = rng.beta(alpha, beta, size=n_particles)
    except Exception:

        u = np.random.uniform(0.0, 1.0, n_particles)
        z = u ** (1.0 / alpha) * (1.0 - u ** (1.0 / beta))
        z = np.clip(z, 0.01, 0.99)

    energies = z * parton_energy


    total = np.sum(energies)
    if total > parton_energy:
        energies *= parton_energy / total

    return energies
