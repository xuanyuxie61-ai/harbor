
import numpy as np
from typing import Callable, Optional, Tuple


def discrete_laplacian_2d(I: np.ndarray, h: float = 1.0) -> np.ndarray:
    I = np.asarray(I, dtype=float)
    if I.ndim != 2:
        raise ValueError("输入必须是二维图像")

    lap = np.zeros_like(I)
    lap[1:-1, 1:-1] = (I[2:, 1:-1] + I[:-2, 1:-1] +
                       I[1:-1, 2:] + I[1:-1, :-2] - 4.0 * I[1:-1, 1:-1]) / (h ** 2)


    lap[0, :] = lap[1, :]
    lap[-1, :] = lap[-2, :]
    lap[:, 0] = lap[:, 1]
    lap[:, -1] = lap[:, -2]

    return lap


def diffusion_rhs(t: float, I: np.ndarray, D: float = 0.1,
                  alpha: float = 0.01) -> np.ndarray:
    return D * discrete_laplacian_2d(I) - alpha * I


def midpoint_fixed_step(f: Callable, t0: float, I0: np.ndarray,
                        dt: float, it_max: int = 10,
                        theta: float = 0.5) -> np.ndarray:
    I0 = np.asarray(I0, dtype=float)
    xm = t0 + theta * dt
    ym = I0.copy()

    for _ in range(it_max):
        ym = I0 + theta * dt * f(xm, ym)

    I1 = (1.0 / theta) * ym + (1.0 - 1.0 / theta) * I0
    return I1


def midpoint_implicit_step(f: Callable, t0: float, I0: np.ndarray,
                           dt: float, tol: float = 1e-8,
                           max_iter: int = 20) -> np.ndarray:
    I0 = np.asarray(I0, dtype=float)
    tm = t0 + 0.5 * dt
    I_mid = I0.copy()

    for _ in range(max_iter):
        I_mid_new = I0 + 0.5 * dt * f(tm, I_mid)
        if np.linalg.norm(I_mid_new - I_mid) < tol * max(1.0, np.linalg.norm(I_mid)):
            I_mid = I_mid_new
            break
        I_mid = I_mid_new

    I1 = 2.0 * I_mid - I0
    return I1


def solve_dynamic_diffusion(I0: np.ndarray, tspan: Tuple[float, float],
                            n_steps: int, D: float = 0.1, alpha: float = 0.01,
                            method: str = 'implicit') -> Tuple[np.ndarray, np.ndarray]:
    I0 = np.asarray(I0, dtype=float)
    if I0.ndim != 2:
        raise ValueError("初始图像必须是二维")

    t_start, t_end = tspan
    if t_end <= t_start:
        raise ValueError("t_end 必须大于 t_start")
    if n_steps <= 0:
        raise ValueError("n_steps 必须为正")

    dt = (t_end - t_start) / n_steps
    t_array = np.linspace(t_start, t_end, n_steps + 1)

    I_series = np.zeros((n_steps + 1, I0.shape[0], I0.shape[1]), dtype=float)
    I_series[0] = I0

    def rhs(t, I):
        return diffusion_rhs(t, I, D, alpha)

    for n in range(n_steps):
        if method == 'implicit':
            I_series[n + 1] = midpoint_implicit_step(rhs, t_array[n], I_series[n], dt)
        elif method == 'fixed':
            I_series[n + 1] = midpoint_fixed_step(rhs, t_array[n], I_series[n], dt)
        else:
            raise ValueError(f"未知方法: {method}")

    return t_array, I_series


def dynamic_cs_reconstruction(measurements: np.ndarray, Phi: np.ndarray,
                              Psi: np.ndarray, lambda_reg: float,
                              temporal_smoothness: float = 0.1) -> np.ndarray:
    from cs_detector import fista_reconstruction

    A = Phi @ Psi
    y = np.asarray(measurements, dtype=float).ravel()


    c = fista_reconstruction(A, y, lambda_reg, max_iter=500, tol=1e-5)
    return c
