
import numpy as np
from typing import Callable, Tuple


def backward_euler_step(y0: np.ndarray, h: float,
                        f: Callable[[np.ndarray], np.ndarray],
                        df: Callable[[np.ndarray], np.ndarray],
                        max_iter: int = 20, tol: float = 1e-8) -> np.ndarray:
    y1 = y0.copy()
    dim = y0.size
    I = np.eye(dim, dtype=np.float64)

    for _ in range(max_iter):
        fy = f(y1)
        g = y1 - y0 - h * fy
        norm_g = np.linalg.norm(g)
        if norm_g < tol:
            break
        Jf = df(y1)
        Jg = I - h * Jf

        try:
            dy = np.linalg.solve(Jg, -g)
        except np.linalg.LinAlgError:

            dy = -np.linalg.lstsq(Jg, g, rcond=None)[0]
        y1 = y1 + dy

        if np.linalg.norm(dy) > 10.0:
            y1 = y1 - 0.5 * dy
    return y1


def damped_gradient_flow(relax_coords: np.ndarray,
                        energy_func: Callable[[np.ndarray], float],
                        grad_func: Callable[[np.ndarray], np.ndarray],
                        hess_func: Callable[[np.ndarray], np.ndarray],
                        n_steps: int = 50,
                        h: float = 0.01,
                        tol: float = 1e-5) -> Tuple[np.ndarray, float]:
    y = relax_coords.flatten().astype(np.float64)
    dim = y.size

    def f(y_vec):
        g = grad_func(y_vec.reshape(-1, 3))
        return -g.flatten()

    def df(y_vec):
        H = hess_func(y_vec.reshape(-1, 3))
        return -H

    for step in range(n_steps):
        y = backward_euler_step(y, h, f, df, max_iter=15, tol=1e-7)
        g_norm = np.linalg.norm(grad_func(y.reshape(-1, 3)))
        if g_norm < tol:
            break

    coords_opt = y.reshape(-1, 3)
    energy_opt = energy_func(coords_opt)
    return coords_opt, energy_opt


def lennard_jones_potential(coords: np.ndarray,
                            epsilon: float = 1.0,
                            sigma: float = 1.0) -> float:
    n = coords.shape[0]
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(coords[i] - coords[j])
            r = max(r, 0.8 * sigma)
            sr6 = (sigma / r) ** 6
            sr12 = sr6 ** 2
            energy += 4.0 * epsilon * (sr12 - sr6)
    return energy


def lennard_jones_gradient(coords: np.ndarray,
                           epsilon: float = 1.0,
                           sigma: float = 1.0) -> np.ndarray:
    n = coords.shape[0]
    grad = np.zeros_like(coords)
    for i in range(n):
        for j in range(i + 1, n):
            dr = coords[i] - coords[j]
            r = np.linalg.norm(dr)
            r = max(r, 0.8 * sigma)
            sr6 = (sigma / r) ** 6
            sr12 = sr6 ** 2
            dVdr = 4.0 * epsilon * (-12.0 * sr12 / r + 6.0 * sr6 / r)
            g_vec = (dVdr / r) * dr
            grad[i] += g_vec
            grad[j] -= g_vec
    return grad


def lennard_jones_hessian(coords: np.ndarray,
                          epsilon: float = 1.0,
                          sigma: float = 1.0) -> np.ndarray:
    n = coords.shape[0]
    dim = 3 * n
    H = np.zeros((dim, dim), dtype=np.float64)
    delta = 1e-5
    for idx in range(dim):
        coords_plus = coords.copy().flatten()
        coords_minus = coords.copy().flatten()
        coords_plus[idx] += delta
        coords_minus[idx] -= delta
        g_plus = lennard_jones_gradient(coords_plus.reshape(n, 3), epsilon, sigma).flatten()
        g_minus = lennard_jones_gradient(coords_minus.reshape(n, 3), epsilon, sigma).flatten()
        H[:, idx] = (g_plus - g_minus) / (2.0 * delta)

    H = 0.5 * (H + H.T)
    return H
