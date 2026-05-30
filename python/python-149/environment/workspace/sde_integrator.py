
import numpy as np
from typing import Callable, Optional, Tuple


def euler_maruyama(
    f: Callable[[float, np.ndarray], np.ndarray],
    g: Callable[[float, np.ndarray], np.ndarray],
    tspan: Tuple[float, float],
    y0: np.ndarray,
    n_steps: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng(seed=42)

    t0, tstop = tspan
    dt = (tstop - t0) / n_steps
    if dt <= 0:
        raise ValueError("时间步长必须为正")

    dim = len(np.atleast_1d(y0))
    t = np.linspace(t0, tstop, n_steps + 1)
    y = np.zeros((n_steps + 1, dim))
    y[0, :] = np.atleast_1d(y0)

    sqrt_dt = np.sqrt(dt)

    for j in range(1, n_steps + 1):
        y_prev = y[j - 1, :]
        t_prev = t[j - 1]


        drift = np.atleast_1d(f(t_prev, y_prev))

        diffusion = np.atleast_1d(g(t_prev, y_prev))


        dW = sqrt_dt * rng.standard_normal(dim)


        if not np.all(np.isfinite(drift)):
            drift = np.zeros(dim)
        if not np.all(np.isfinite(diffusion)):
            diffusion = np.zeros(dim)

        y[j, :] = y_prev + drift * dt + diffusion * dW


        y[j, :] = np.clip(y[j, :], -0.1, 1.1)

    return t, y


def milstein_method(
    f: Callable[[float, np.ndarray], np.ndarray],
    g: Callable[[float, np.ndarray], np.ndarray],
    dg: Callable[[float, np.ndarray], np.ndarray],
    tspan: Tuple[float, float],
    y0: np.ndarray,
    n_steps: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng(seed=42)

    t0, tstop = tspan
    dt = (tstop - t0) / n_steps
    if dt <= 0:
        raise ValueError("时间步长必须为正")

    dim = len(np.atleast_1d(y0))
    t = np.linspace(t0, tstop, n_steps + 1)
    y = np.zeros((n_steps + 1, dim))
    y[0, :] = np.atleast_1d(y0)

    sqrt_dt = np.sqrt(dt)

    for j in range(1, n_steps + 1):
        y_prev = y[j - 1, :]
        t_prev = t[j - 1]

        drift = np.atleast_1d(f(t_prev, y_prev))
        diff = np.atleast_1d(g(t_prev, y_prev))
        diff_deriv = np.atleast_1d(dg(t_prev, y_prev))

        dW = sqrt_dt * rng.standard_normal(dim)

        if not np.all(np.isfinite(drift)):
            drift = np.zeros(dim)
        if not np.all(np.isfinite(diff)):
            diff = np.zeros(dim)
        if not np.all(np.isfinite(diff_deriv)):
            diff_deriv = np.zeros(dim)




        pass

    return t, y


def stochastic_explicit_midpoint(
    f: Callable[[float, np.ndarray], np.ndarray],
    g: Callable[[float, np.ndarray], np.ndarray],
    tspan: Tuple[float, float],
    y0: np.ndarray,
    n_steps: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng(seed=42)

    t0, tstop = tspan
    dt = (tstop - t0) / n_steps
    if dt <= 0:
        raise ValueError("时间步长必须为正")

    dim = len(np.atleast_1d(y0))
    t = np.linspace(t0, tstop, n_steps + 1)
    y = np.zeros((n_steps + 1, dim))
    y[0, :] = np.atleast_1d(y0)

    sqrt_dt = np.sqrt(dt)

    for j in range(1, n_steps + 1):
        y_prev = y[j - 1, :]
        t_prev = t[j - 1]

        drift = np.atleast_1d(f(t_prev, y_prev))
        diff = np.atleast_1d(g(t_prev, y_prev))
        dW = sqrt_dt * rng.standard_normal(dim)

        if not np.all(np.isfinite(drift)):
            drift = np.zeros(dim)
        if not np.all(np.isfinite(diff)):
            diff = np.zeros(dim)


        y_mid = y_prev + 0.5 * drift * dt + 0.5 * diff * dW
        y_mid = np.clip(y_mid, -0.1, 1.1)


        t_mid = t_prev + 0.5 * dt
        drift_mid = np.atleast_1d(f(t_mid, y_mid))
        diff_mid = np.atleast_1d(g(t_mid, y_mid))

        if not np.all(np.isfinite(drift_mid)):
            drift_mid = np.zeros(dim)
        if not np.all(np.isfinite(diff_mid)):
            diff_mid = np.zeros(dim)

        y[j, :] = y_prev + drift_mid * dt + diff_mid * dW
        y[j, :] = np.clip(y[j, :], -0.1, 1.1)

    return t, y


def generate_brownian_path(
    tspan: Tuple[float, float],
    n_steps: int,
    dim: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng(seed=42)

    t0, tstop = tspan
    dt = (tstop - t0) / n_steps
    t = np.linspace(t0, tstop, n_steps + 1)

    dW = np.sqrt(dt) * rng.standard_normal((n_steps, dim))
    W = np.zeros((n_steps + 1, dim))
    W[1:, :] = np.cumsum(dW, axis=0)

    return t, W


def mean_square_stability_check(
    lambda_val: float,
    mu_val: float,
    dt: float,
) -> bool:
    lhs = (1.0 + lambda_val * dt) ** 2 + (mu_val ** 2) * dt
    return lhs < 1.0


def compute_strong_error(
    f: Callable,
    g: Callable,
    y0: float,
    tspan: Tuple[float, float],
    n_ref: int,
    n_coarse_list: list,
    n_paths: int = 500,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng(seed=42)

    t0, tstop = tspan
    dt_ref = (tstop - t0) / n_ref


    dW_ref = np.sqrt(dt_ref) * rng.standard_normal((n_paths, n_ref))


    X_ref = np.zeros(n_paths)
    for p in range(n_paths):
        xtemp = y0
        for j in range(n_ref):
            xtemp = xtemp + f(0.0, xtemp) * dt_ref + g(0.0, xtemp) * dW_ref[p, j]
        X_ref[p] = xtemp

    dt_vals = []
    errors = []

    for n_coarse in n_coarse_list:
        if n_ref % n_coarse != 0:
            continue
        ratio = n_ref // n_coarse
        dt_coarse = ratio * dt_ref

        X_coarse = np.zeros(n_paths)
        for p in range(n_paths):
            xtemp = y0
            for j in range(n_coarse):
                winc = np.sum(dW_ref[p, j * ratio:(j + 1) * ratio])
                xtemp = xtemp + f(0.0, xtemp) * dt_coarse + g(0.0, xtemp) * winc
            X_coarse[p] = xtemp

        err = np.mean(np.abs(X_coarse - X_ref))
        dt_vals.append(dt_coarse)
        errors.append(err)

    return np.array(dt_vals), np.array(errors)
