# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple
from numerical_integrator import rk1_integrate, rk4_integrate


def fisher_exact_solution(
    x: np.ndarray,
    t: float,
    a: float = 2.0,
    c: float = None,
    k: float = None,
    D: float = 1.0,
    r: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if c is None:
        c = 5.0 / np.sqrt(6.0)
    if k is None:
        k = 1.0 / np.sqrt(6.0)

    if D <= 0.0 or r <= 0.0:
        raise ValueError("D and r must be positive.")
    if a <= 0.0:
        raise ValueError("a must be positive.")

    z = x - c * t
    exp_kz = np.exp(k * z)
    denom = 1.0 + a * exp_kz
    denom2 = denom ** 2
    denom3 = denom ** 3
    denom4 = denom ** 4

    u = 1.0 / denom2
    ut = 2.0 * c * a * k * exp_kz / denom3
    ux = -2.0 * a * k * exp_kz / denom3
    uxx = 6.0 * (a ** 2) * (k ** 2) * np.exp(2.0 * k * z) / denom4 \
          - 2.0 * a * (k ** 2) * exp_kz / denom3


    u = np.clip(u, 0.0, 1.0)

    return u, ut, ux, uxx


def fisher_wave_speed(D: float, r: float) -> float:
    if D <= 0.0 or r <= 0.0:
        raise ValueError("D and r must be positive.")
    return 2.0 * np.sqrt(D * r)


def build_fisher_rhs(
    n: int,
    h: float,
    D: float,
    r: float,
    bc: str = "NN",
) -> callable:
    if n < 3:
        raise ValueError("n must be >= 3.")
    if h <= 0.0:
        raise ValueError("h must be positive.")
    if D <= 0.0 or r <= 0.0:
        raise ValueError("D and r must be positive.")

    inv_h2 = 1.0 / (h * h)

    def rhs(t, u):







        pass

    return rhs


def simulate_ltp_wave(
    n: int = 128,
    length: float = 200.0,
    D: float = 1.0,
    r: float = 1.0,
    t_final: float = 20.0,
    n_steps: int = 2000,
    bc: str = "NN",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if n < 3:
        raise ValueError("n must be >= 3.")
    if length <= 0.0:
        raise ValueError("length must be positive.")
    if D <= 0.0 or r <= 0.0:
        raise ValueError("D and r must be positive.")

    h = length / n
    x = np.linspace(0.0, length, n)


    u0 = np.zeros(n)
    center_idx = n // 2
    width = n // 10
    u0[center_idx - width:center_idx + width] = 0.8

    for i in range(n):
        u0[i] = 0.8 / (1.0 + np.exp(0.5 * (abs(i - center_idx) - width)))

    c_min = fisher_wave_speed(D, r)


    dt = t_final / n_steps
    dx2 = h * h
    if D * dt / dx2 > 0.5:
        n_steps = int(np.ceil(2.0 * D * t_final / dx2))
        dt = t_final / n_steps
        print(f"[plasticity_wave] n_steps adjusted to {n_steps} for stability")

    rhs = build_fisher_rhs(n, h, D, r, bc)
    t, u_history = rk4_integrate(rhs, (0.0, t_final), u0, n_steps)

    return x, t, u_history, c_min


def verify_fisher_exact(
    n: int = 64,
    length: float = 20.0,
    t_test: float = 2.0,
) -> float:
    h = length / n
    x = np.linspace(0.0, length, n)


    u_exact_0, _, _, _ = fisher_exact_solution(x, 0.0)


    D = 1.0
    r = 1.0
    rhs = build_fisher_rhs(n, h, D, r, bc="NN")
    n_steps = max(100, int(t_test * 500))
    _, u_history = rk4_integrate(rhs, (0.0, t_test), u_exact_0, n_steps)
    u_num = u_history[-1, :]

    u_exact, _, _, _ = fisher_exact_solution(x, t_test)

    denom = np.linalg.norm(u_exact)
    if denom < 1e-15:
        return np.linalg.norm(u_num - u_exact)
    return np.linalg.norm(u_num - u_exact) / denom


if __name__ == "__main__":
    x, t, u_hist, c = simulate_ltp_wave()
    print(f"LTP wave speed (theoretical min): {c:.4f} μm/ms")
    print(f"Wave front position at t={t[-1]}: ~{np.argmax(u_hist[-1] > 0.5) * (x[1]-x[0]):.2f} μm")
    err = verify_fisher_exact()
    print(f"Exact solution verification error: {err:.6e}")
