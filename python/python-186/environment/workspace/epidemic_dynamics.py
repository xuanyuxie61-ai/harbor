
import numpy as np
from typing import Tuple, List, Optional, Callable


def seaihr_ode_rhs(t: float, y: np.ndarray, params: dict) -> np.ndarray:



    raise NotImplementedError("Hole_1: seaihr_ode_rhs 尚未实现")


def coupled_info_epidemic_rhs(t: float, y: np.ndarray,
                              history_func: Callable,
                              params: dict) -> np.ndarray:
    n_epi = 7
    y_epi = y[:n_epi]
    I_info = max(y[n_epi], 0.0)


    tau = params.get('tau', 5.0)
    t_delayed = max(0.0, t - tau)
    y_delayed = history_func(t_delayed)

    I_past = max(y_delayed[3] + y_delayed[2], 0.0)


    beta_0 = params['beta_0']
    k_beta = params.get('k_beta', 0.5)
    beta_eff = beta_0 * np.exp(-k_beta * I_info)


    params_local = params.copy()
    params_local['beta'] = max(beta_eff, 0.0)


    dydt_epi = seaihr_ode_rhs(t, y_epi, params_local)


    beta_info = params.get('beta_info', 2.0)
    n_info = params.get('n_info', 9.65)
    gamma_info = params.get('gamma_info', 1.0)


    denom = 1.0 + I_past**n_info
    if denom < 1e-10:
        denom = 1e-10

    dI_info_dt = beta_info * I_past / denom - gamma_info * I_info

    return np.concatenate([dydt_epi, [dI_info_dt]])


def rk4_integrate(rhs_func: Callable,
                  y0: np.ndarray,
                  t_span: Tuple[float, float],
                  dt: float,
                  params: dict,
                  history_func: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray]:
    t0, tf = t_span
    n_steps = max(1, int(np.ceil((tf - t0) / dt)))
    dt_actual = (tf - t0) / n_steps

    t_history = np.linspace(t0, tf, n_steps + 1)
    y_history = np.zeros((n_steps + 1, len(y0)), dtype=np.float64)
    y_history[0, :] = y0

    y = y0.copy()

    for n in range(n_steps):
        t_n = t_history[n]

        if history_func is not None:
            def wrapped_rhs(t, y):
                return rhs_func(t, y, history_func, params)
        else:
            def wrapped_rhs(t, y):
                return rhs_func(t, y, params)

        k1 = wrapped_rhs(t_n, y)
        k2 = wrapped_rhs(t_n + 0.5 * dt_actual, y + 0.5 * dt_actual * k1)
        k3 = wrapped_rhs(t_n + 0.5 * dt_actual, y + 0.5 * dt_actual * k2)
        k4 = wrapped_rhs(t_n + dt_actual, y + dt_actual * k3)

        y = y + (dt_actual / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


        y = np.maximum(y, 0.0)

        y_history[n + 1, :] = y

    return t_history, y_history


def dde_rk4_integrate(rhs_func: Callable,
                      y0: np.ndarray,
                      t_span: Tuple[float, float],
                      dt: float,
                      params: dict,
                      history_func: Callable) -> Tuple[np.ndarray, np.ndarray]:
    t0, tf = t_span
    n_steps = max(1, int(np.ceil((tf - t0) / dt)))
    dt_actual = (tf - t0) / n_steps

    t_history = np.linspace(t0, tf, n_steps + 1)
    y_history = np.zeros((n_steps + 1, len(y0)), dtype=np.float64)
    y_history[0, :] = y0

    y = y0.copy()

    for n in range(n_steps):
        t_n = t_history[n]


        def current_history(t_query):
            if t_query <= t0:
                return history_func(t_query)

            idx = int((t_query - t0) / dt_actual)
            idx = min(idx, n)
            frac = (t_query - t0) / dt_actual - idx
            frac = max(0.0, min(1.0, frac))
            if idx + 1 <= n:
                return (1 - frac) * y_history[idx, :] + frac * y_history[idx + 1, :]
            return y_history[idx, :]

        def wrapped_rhs(t, y_state):
            return rhs_func(t, y_state, current_history, params)

        k1 = wrapped_rhs(t_n, y)
        k2 = wrapped_rhs(t_n + 0.5 * dt_actual, y + 0.5 * dt_actual * k1)
        k3 = wrapped_rhs(t_n + 0.5 * dt_actual, y + 0.5 * dt_actual * k2)
        k4 = wrapped_rhs(t_n + dt_actual, y + dt_actual * k3)

        y = y + (dt_actual / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        y = np.maximum(y, 0.0)
        y_history[n + 1, :] = y

    return t_history, y_history


def compute_reproduction_number(params: dict) -> float:



    raise NotImplementedError("Hole_2: compute_reproduction_number 尚未实现")
