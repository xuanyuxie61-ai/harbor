# -*- coding: utf-8 -*-

import numpy as np
from typing import Callable, Tuple, Optional


def rk1_integrate(
    f: Callable[[float, np.ndarray], np.ndarray],
    tspan: Tuple[float, float],
    y0: np.ndarray,
    n_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if n_steps < 1:
        raise ValueError(f"n_steps={n_steps} must be >= 1.")
    t0, t_stop = tspan
    if t_stop <= t0:
        raise ValueError(f"t_stop={t_stop} must be > t0={t0}.")

    dt = (t_stop - t0) / n_steps
    m = y0.shape[0]

    t = np.zeros(n_steps + 1)
    y = np.zeros((n_steps + 1, m))

    t[0] = t0
    y[0, :] = y0.flatten()

    for i in range(n_steps):
        t[i + 1] = t[i] + dt
        dydt = f(t[i], y[i, :])
        if dydt is None or not np.isfinite(dydt).all():
            raise RuntimeError(f"ODE rhs returned invalid values at t={t[i]}")
        y[i + 1, :] = y[i, :] + dt * dydt

    return t, y


def rk4_integrate(
    f: Callable[[float, np.ndarray], np.ndarray],
    tspan: Tuple[float, float],
    y0: np.ndarray,
    n_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1.")
    t0, t_stop = tspan
    if t_stop <= t0:
        raise ValueError("t_stop must be > t0.")

    dt = (t_stop - t0) / n_steps
    m = y0.shape[0]

    t = np.zeros(n_steps + 1)
    y = np.zeros((n_steps + 1, m))

    t[0] = t0
    y[0, :] = y0.flatten()

    for i in range(n_steps):









        pass

    return t, y


def estimate_stability_jacobian(
    f: Callable[[float, np.ndarray], np.ndarray],
    t: float,
    y: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    m = y.shape[0]
    J = np.zeros((m, m))
    f0 = f(t, y)

    for j in range(m):
        y_plus = y.copy()
        y_plus[j] += eps
        f_plus = f(t, y_plus)

        y_minus = y.copy()
        y_minus[j] -= eps
        f_minus = f(t, y_minus)

        J[:, j] = (f_plus - f_minus) / (2.0 * eps)

    return J


def compute_stiffness_ratio(
    J: np.ndarray,
) -> Tuple[float, complex, complex]:
    eigvals = np.linalg.eigvals(J)
    re_parts = np.real(eigvals)
    abs_re = np.abs(re_parts)

    idx_max = np.argmax(abs_re)
    idx_min = np.argmin(abs_re + 1e-20)

    S = abs_re[idx_max] / (abs_re[idx_min] + 1e-20)
    return S, eigvals[idx_max], eigvals[idx_min]


def adaptive_rk12(
    f: Callable[[float, np.ndarray], np.ndarray],
    tspan: Tuple[float, float],
    y0: np.ndarray,
    tol: float = 1e-6,
    h0: float = 0.01,
    h_min: float = 1e-10,
    h_max: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t0, t_stop = tspan
    if t_stop <= t0:
        raise ValueError("t_stop must be > t0.")
    if tol <= 0.0 or h0 <= 0.0:
        raise ValueError("tol and h0 must be positive.")

    t_list = [t0]
    y_list = [y0.copy()]
    h_list = []

    t_curr = t0
    y_curr = y0.copy()
    h = h0

    max_steps = 100000
    step_count = 0

    while t_curr < t_stop and step_count < max_steps:
        h = min(h, h_max)
        h = max(h, h_min)
        if t_curr + h > t_stop:
            h = t_stop - t_curr


        k1 = f(t_curr, y_curr)
        y_rk1 = y_curr + h * k1


        k2 = f(t_curr + h, y_rk1)
        y_rk2 = y_curr + 0.5 * h * (k1 + k2)


        e = np.linalg.norm(y_rk2 - y_rk1)

        if e <= tol or h <= h_min:

            t_curr = t_curr + h
            y_curr = y_rk2.copy()
            t_list.append(t_curr)
            y_list.append(y_curr.copy())
            h_list.append(h)
            step_count += 1


            if e > 1e-20:
                h = h * min(5.0, max(0.2, 0.9 * np.sqrt(tol / e)))
            else:
                h = h * 2.0
        else:

            h = h * max(0.2, 0.9 * np.sqrt(tol / e))
            if h < h_min:
                h = h_min

                t_curr = t_curr + h
                y_curr = y_rk2.copy()
                t_list.append(t_curr)
                y_list.append(y_curr.copy())
                h_list.append(h)
                step_count += 1

    t_arr = np.array(t_list)
    y_arr = np.array(y_list)
    h_arr = np.array(h_list)
    return t_arr, y_arr, h_arr


if __name__ == "__main__":

    def harmonic(t, y):
        return np.array([y[1], -y[0]])

    t, y = rk1_integrate(harmonic, (0.0, 10.0), np.array([1.0, 0.0]), 1000)
    print(f"RK1 final state: {y[-1]}")
