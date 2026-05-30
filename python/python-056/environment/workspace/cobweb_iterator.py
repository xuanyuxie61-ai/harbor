
import numpy as np
from typing import Callable, Tuple


def cobweb_iterate(
    f: Callable[[float], float],
    x0: float,
    n_iter: int,
    tol: float = 1e-10,
) -> Tuple[float, np.ndarray, bool]:
    history = np.zeros(n_iter + 1)
    history[0] = x0
    for k in range(n_iter):
        history[k + 1] = f(history[k])
        if abs(history[k + 1] - history[k]) < tol:
            return history[k + 1], history[:k + 2], True
    return history[-1], history, False


def aitken_acceleration(
    f: Callable[[float], float],
    x0: float,
    n_iter: int,
    tol: float = 1e-10,
) -> Tuple[float, np.ndarray, bool]:
    history = np.zeros(n_iter + 1)
    history[0] = x0
    k = 0
    while k + 2 <= n_iter:
        xk = history[k]
        xk1 = f(xk)
        xk2 = f(xk1)
        denom = xk2 - 2.0 * xk1 + xk
        if abs(denom) < 1e-14:
            history[k + 1] = xk1
            k += 1
            continue
        x_star = xk - (xk1 - xk) ** 2 / denom
        history[k + 1] = x_star
        if abs(x_star - xk) < tol:
            return x_star, history[:k + 2], True
        k += 1
    return history[k], history[:k + 1], False


def fsi_fixed_point_solver(
    fluid_solver: Callable[[np.ndarray], np.ndarray],
    structure_solver: Callable[[np.ndarray], np.ndarray],
    initial_guess: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-6,
    relaxation: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray, bool, int]:
    delta = np.asarray(initial_guess, dtype=float).copy()
    history = np.zeros(max_iter + 1)
    history[0] = np.linalg.norm(delta)

    for it in range(max_iter):
        load = fluid_solver(delta)
        delta_new = structure_solver(load)
        delta = relaxation * delta_new + (1.0 - relaxation) * delta
        history[it + 1] = np.linalg.norm(delta)
        residual = np.linalg.norm(delta_new - delta) / (np.linalg.norm(delta) + 1e-12)
        if residual < tol:
            return delta, load, True, it + 1

    return delta, load, False, max_iter


def contraction_factor_estimate(history: np.ndarray) -> float:
    if len(history) < 3:
        return 1.0
    diffs = np.abs(np.diff(history))
    valid = diffs[1:] / (diffs[:-1] + 1e-14)
    return float(np.median(valid[valid < 1.0])) if np.any(valid < 1.0) else 1.0
