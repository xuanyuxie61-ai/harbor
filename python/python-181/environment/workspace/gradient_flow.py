
import numpy as np
from typing import Tuple, Callable


def centered_difference(f: Callable, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    D = len(x)
    grad = np.zeros(D, dtype=np.float64)
    for d in range(D):
        x_plus = x.copy()
        x_minus = x.copy()

        step = max(abs(x[d]) * h, h)
        x_plus[d] += step
        x_minus[d] -= step
        grad[d] = (f(x_plus) - f(x_minus)) / (2.0 * step)
    return grad


def hessian_approximation(f: Callable, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    D = len(x)
    H = np.zeros((D, D), dtype=np.float64)
    for i in range(D):
        for j in range(i, D):
            x_pp = x.copy()
            x_pm = x.copy()
            x_mp = x.copy()
            x_mm = x.copy()
            hi = max(abs(x[i]) * h, h)
            hj = max(abs(x[j]) * h, h)
            x_pp[i] += hi; x_pp[j] += hj
            x_pm[i] += hi; x_pm[j] -= hj
            x_mp[i] -= hi; x_mp[j] += hj
            x_mm[i] -= hi; x_mm[j] -= hj
            H[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4.0 * hi * hj)
            H[j, i] = H[i, j]
    return H


def gradient_flow_descent(f: Callable, x0: np.ndarray, dt: float = 0.01,
                          max_steps: int = 1000, tol: float = 1e-8) -> np.ndarray:
    x = x0.copy()
    for step in range(max_steps):
        grad = centered_difference(f, x)
        x_new = x - dt * grad
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x


def predator_prey_dynamics(x0: np.ndarray, r: np.ndarray,
                            A: np.ndarray, t_span: Tuple[float, float],
                            n_steps: int = 5000) -> np.ndarray:
    n = len(x0)
    t_start, t_stop = t_span
    dt = (t_stop - t_start) / n_steps
    x = x0.copy()
    trajectory = np.zeros((n_steps + 1, n))
    trajectory[0] = x
    for i in range(n_steps):

        dx = x * (r - A @ x)
        x = x + dt * dx
        x = np.maximum(x, 0.0)
        trajectory[i + 1] = x
    return trajectory


def feature_selection_by_competition(feature_scores: np.ndarray,
                                      interaction_matrix: np.ndarray = None,
                                      n_selected: int = 5) -> np.ndarray:
    n = len(feature_scores)
    if interaction_matrix is None:

        A = np.ones((n, n)) * 0.5
        np.fill_diagonal(A, 1.0)
    else:
        A = interaction_matrix

    r = feature_scores / (np.max(feature_scores) + 1e-15)
    x0 = feature_scores / (np.sum(feature_scores) + 1e-15)

    trajectory = predator_prey_dynamics(x0, r, A, (0.0, 10.0), n_steps=2000)
    steady_state = trajectory[-1]

    idx = np.argsort(steady_state)[::-1][:n_selected]
    return idx


def diffusion_map_gradient(data: np.ndarray, embedding: np.ndarray,
                            target_point: int, sigma: float = 1.0) -> np.ndarray:
    N = len(data)

    dists = np.linalg.norm(data - data[target_point], axis=1)
    weights = np.exp(-dists ** 2 / (2.0 * sigma ** 2))
    weights[target_point] = 0.0
    weights = weights / (np.sum(weights) + 1e-15)

    grad = np.zeros(embedding.shape[1])
    for i in range(N):
        grad += weights[i] * (embedding[i] - embedding[target_point])
    return grad


def conserved_quantity_prey_predator(prey: float, predator: float) -> float:
    c1 = 0.003
    c2 = 0.004
    a1 = 10.0
    a2 = 2.0
    if prey <= 0 or predator <= 0:
        return np.inf
    return c1 * prey + c2 * predator - a1 * np.log(prey) - a2 * np.log(predator)
