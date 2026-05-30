
import numpy as np
from typing import Callable, Tuple, Optional


def exponential_integrator_linear(
    A: np.ndarray, b: np.ndarray, y0: np.ndarray,
    t_span: Tuple[float, float], n_steps: int
) -> Tuple[np.ndarray, np.ndarray]:
    t0, tf = t_span
    dt = (tf - t0) / n_steps
    t_array = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros((n_steps + 1, len(y0)))
    y[0] = y0


    from scipy.linalg import expm
    exp_Adt = expm(A * dt)


    try:
        A_inv = np.linalg.inv(A)
        phi = A_inv @ (exp_Adt - np.eye(len(y0)))
    except np.linalg.LinAlgError:
        phi = dt * np.eye(len(y0))

    for n in range(n_steps):
        y[n + 1] = exp_Adt @ y[n] + phi @ b

    return t_array, y


def backward_euler_viscoelastic(
    M: np.ndarray, K: np.ndarray, F: np.ndarray,
    u0: np.ndarray, v0: np.ndarray,
    t_span: Tuple[float, float], n_steps: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t0, tf = t_span
    dt = (tf - t0) / n_steps
    t_array = np.linspace(t0, tf, n_steps + 1)
    n_dof = len(u0)

    u = np.zeros((n_steps + 1, n_dof))
    v = np.zeros((n_steps + 1, n_dof))
    u[0] = u0
    v[0] = v0


    if F.ndim == 1:
        F_hist = np.tile(F, (n_steps + 1, 1))
    else:
        F_hist = F


    for n in range(n_steps):


        try:
            u[n + 1] = np.linalg.solve(K, F_hist[n + 1])
        except np.linalg.LinAlgError:
            u[n + 1] = np.linalg.lstsq(K, F_hist[n + 1], rcond=None)[0]

        v[n + 1] = (u[n + 1] - u[n]) / dt

    return t_array, u, v


def imex_time_integration(
    f_explicit: Callable, f_implicit: Callable,
    y0: np.ndarray, t_span: Tuple[float, float], n_steps: int
) -> Tuple[np.ndarray, np.ndarray]:
    t0, tf = t_span
    dt = (tf - t0) / n_steps
    t_array = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros((n_steps + 1, len(y0)))
    y[0] = y0

    for n in range(n_steps):
        f_expl_n = f_explicit(y[n])

        if callable(f_implicit):

            y_next = y[n] + dt * f_expl_n
            for _ in range(10):
                y_next_new = y[n] + dt * (f_implicit(y_next) + f_expl_n)
                if np.linalg.norm(y_next_new - y_next) < 1e-12:
                    break
                y_next = y_next_new
            y[n + 1] = y_next
        else:

            A = f_implicit
            lhs = np.eye(len(y0)) - dt * A
            rhs = y[n] + dt * f_expl_n
            try:
                y[n + 1] = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                y[n + 1] = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    return t_array, y


def adaptive_time_stepping(
    f: Callable, y0: np.ndarray, t_span: Tuple[float, float],
    dt_init: float, tol: float = 1e-6, dt_min: float = 1e-6, dt_max: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    t0, tf = t_span
    t = t0
    y = y0.copy().astype(float)
    dt = dt_init

    t_list = [t]
    y_list = [y.copy()]

    while t < tf:
        dt = min(dt, tf - t)


        k1 = f(y)
        y_euler = y + dt * k1


        k2 = f(y + dt * k1)
        y_rk2 = y + 0.5 * dt * (k1 + k2)


        e = np.linalg.norm(y_rk2 - y_euler)

        if e < tol or dt <= dt_min:

            y = y_rk2
            t += dt
            t_list.append(t)
            y_list.append(y.copy())


            if e > 0:
                dt = min(dt_max, dt * min(2.0, max(0.5, 0.9 * np.sqrt(tol / e))))
            else:
                dt = min(dt_max, dt * 2.0)
        else:

            dt = max(dt_min, dt * max(0.5, 0.9 * np.sqrt(tol / e)))

    return np.array(t_list), np.array(y_list)


def hereditary_integral_discrete(
    kernel: Callable, f_history: np.ndarray,
    t_history: np.ndarray
) -> np.ndarray:
    n = len(t_history)
    y = np.zeros(n)

    for j in range(n):
        integral = 0.0
        for i in range(j):
            dt_i = t_history[i + 1] - t_history[i]
            k1 = kernel(t_history[j] - t_history[i])
            k2 = kernel(t_history[j] - t_history[i + 1])
            integral += 0.5 * dt_i * (k1 * f_history[i] + k2 * f_history[i + 1])
        y[j] = integral

    return y


def power_law_creep_kernel(
    tau: float, E0: float, n_power: float, A_c: float
) -> float:
    if tau <= 0:
        return 0.0
    return (A_c / E0) * (tau ** (-n_power))


def log_double_power_law(
    t: float, t_prime: float, q1: float, q2: float, m: float, n: float
) -> float:
    if t <= t_prime:
        return q1
    dt = t - t_prime
    lambda0 = 1.0
    J = q1 + q2 * np.log1p((dt / lambda0) ** m)
    if t / t_prime > 1.0:
        J += q2 * 0.5 * (np.log(t / t_prime) ** n)
    return J


def viscoelastic_relaxation_spectrum(
    E_t: Callable, times: np.ndarray, n_maxwell: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    E_vals = np.array([E_t(t) for t in times])


    t_min, t_max = times[0], times[-1]
    tau_i = np.logspace(np.log10(t_min + 1e-6), np.log10(t_max), n_maxwell)


    A = np.zeros((len(times), n_maxwell))
    for i, tau in enumerate(tau_i):
        A[:, i] = np.exp(-times / tau)


    E_inf = E_vals[-1]
    b = E_vals - E_inf

    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    E_i = np.maximum(coeffs, 0)

    return E_inf, E_i, tau_i


def effective_time_for_aging_creep(
    t: float, t_prime: float, alpha_h: float = 1.0
) -> float:
    return alpha_h * (t - t_prime)
