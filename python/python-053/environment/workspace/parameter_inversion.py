
import numpy as np
from typing import Tuple, Callable, Optional


def solve_steady_heat_2d(nx: int, ny: int, dx: float, dy: float,
                         diffusivity: float, tau: float,
                         heat_source: np.ndarray,
                         boundary_value: float = 0.0) -> np.ndarray:
    if heat_source.shape != (nx, ny):
        raise ValueError("heat_source shape mismatch")
    if diffusivity <= 0 or tau <= 0:
        raise ValueError("diffusivity and tau must be positive")

    n = nx * ny

    A = np.zeros((n, n), dtype=float)
    b = heat_source.ravel().copy()

    cx = diffusivity / (dx * dx)
    cy = diffusivity / (dy * dy)
    c_inv_tau = 1.0 / tau

    for j in range(ny):
        for i in range(nx):
            row = j * nx + i
            diag = 0.0


            if i > 0:
                A[row, row - 1] = -cx
                diag += cx
            else:
                b[row] += cx * boundary_value
                diag += cx


            if i < nx - 1:
                A[row, row + 1] = -cx
                diag += cx
            else:
                b[row] += cx * boundary_value
                diag += cx


            if j > 0:
                A[row, row - nx] = -cy
                diag += cy
            else:
                b[row] += cy * boundary_value
                diag += cy


            if j < ny - 1:
                A[row, row + nx] = -cy
                diag += cy
            else:
                b[row] += cy * boundary_value
                diag += cy


            A[row, row] = diag + c_inv_tau


    T = np.linalg.solve(A, b)
    return T.reshape((nx, ny))


def piecewise_diffusivity(nx: int, ny: int,
                          param_blocks: np.ndarray,
                          x_breaks: np.ndarray,
                          y_breaks: np.ndarray) -> np.ndarray:
    nxc, nyc = param_blocks.shape
    if x_breaks.shape[0] != nxc + 1 or y_breaks.shape[0] != nyc + 1:
        raise ValueError("Breaks array dimension mismatch")

    D = np.zeros((nx, ny), dtype=float)
    x_grid = np.linspace(x_breaks[0], x_breaks[-1], nx)
    y_grid = np.linspace(y_breaks[0], y_breaks[-1], ny)

    for j in range(ny):
        for i in range(nx):
            x, y = x_grid[i], y_grid[j]

            ix = min(nxc - 1, max(0, np.searchsorted(x_breaks[1:], x)))
            iy = min(nyc - 1, max(0, np.searchsorted(y_breaks[1:], y)))
            D[i, j] = param_blocks[ix, iy]

    return D


def objective_function(theta: np.ndarray,
                       T_obs: np.ndarray,
                       nx: int, ny: int, dx: float, dy: float,
                       heat_source: np.ndarray,
                       theta_prior: np.ndarray,
                       lam: float = 0.01) -> float:
    D_h, tau, coupling = theta[0], theta[1], theta[2]


    D_h = max(D_h, 1e-6)
    tau = max(tau, 1e-6)
    coupling = np.clip(coupling, -10.0, 10.0)


    Q_eff = heat_source + coupling * T_obs

    T_model = solve_steady_heat_2d(nx, ny, dx, dy, D_h, tau, Q_eff)

    residual = T_model - T_obs
    data_misfit = 0.5 * np.sum(residual ** 2)
    regularization = 0.5 * lam * np.sum((theta - theta_prior) ** 2)

    return data_misfit + regularization


def gradient_descent_inversion(T_obs: np.ndarray,
                               nx: int, ny: int, dx: float, dy: float,
                               heat_source: np.ndarray,
                               theta_init: np.ndarray,
                               theta_prior: np.ndarray,
                               lr: float = 0.01,
                               n_iter: int = 100,
                               lam: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    theta = theta_init.copy().astype(float)
    history = np.zeros(n_iter)


    eps_vals = np.array([10.0, 1.0 * 24 * 3600, 0.001])

    scales = np.array([1000.0, 20.0 * 24 * 3600, 0.01])

    for it in range(n_iter):
        J0 = objective_function(theta, T_obs, nx, ny, dx, dy,
                                heat_source, theta_prior, lam)
        history[it] = J0

        if not np.isfinite(J0):
            break

        grad = np.zeros(3)
        for k in range(3):
            theta_plus = theta.copy()
            theta_plus[k] += eps_vals[k]
            J_plus = objective_function(theta_plus, T_obs, nx, ny, dx, dy,
                                        heat_source, theta_prior, lam)
            if np.isfinite(J_plus):
                grad[k] = (J_plus - J0) / eps_vals[k]
            else:
                grad[k] = 0.0


        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-14:
            grad = grad / grad_norm


        step = lr * grad * scales
        theta = theta - step


        theta[0] = max(theta[0], 100.0)
        theta[1] = max(theta[1], 1.0 * 24 * 3600)
        theta[2] = np.clip(theta[2], -1.0, 1.0)


        if it > 0 and history[it] > history[it - 1]:
            lr *= 0.5
            if lr < 1e-6:
                break

    return theta, history
    """
    使用梯度下降法反演热力学参数。

    参数
    ----
    T_obs : np.ndarray
        观测温度场。
    theta_init : np.ndarray
        初始参数猜测。
    theta_prior : np.ndarray
        先验参数。
    lr : float
        学习率。
    n_iter : int
        迭代次数。

    返回
    ----
    theta_opt : np.ndarray
        优化后的参数。
    history : np.ndarray
        目标函数历史。
    """
    theta = theta_init.copy()
    history = np.zeros(n_iter)


    eps = 1e-5

    for it in range(n_iter):
        J0 = objective_function(theta, T_obs, nx, ny, dx, dy,
                                heat_source, theta_prior, lam)
        history[it] = J0

        grad = np.zeros(3)
        for k in range(3):
            theta_plus = theta.copy()
            theta_plus[k] += eps
            J_plus = objective_function(theta_plus, T_obs, nx, ny, dx, dy,
                                        heat_source, theta_prior, lam)
            grad[k] = (J_plus - J0) / eps


        theta = theta - lr * grad


        theta[0] = max(theta[0], 1e-6)
        theta[1] = max(theta[1], 1e-6)
        theta[2] = np.clip(theta[2], -10.0, 10.0)


        if it > 0 and history[it] > history[it - 1]:
            lr *= 0.5

    return theta, history


def sensitivity_analysis(theta: np.ndarray,
                         T_obs: np.ndarray,
                         nx: int, ny: int, dx: float, dy: float,
                         heat_source: np.ndarray,
                         perturbation: float = 0.1) -> dict:
    base_model = solve_steady_heat_2d(nx, ny, dx, dy,
                                      max(theta[0], 1e-6),
                                      max(theta[1], 1e-6),
                                      heat_source + theta[2] * T_obs)

    sens = {}
    param_names = ["diffusivity", "tau", "coupling"]
    for k in range(3):
        theta_plus = theta.copy()
        theta_plus[k] *= (1.0 + perturbation)
        model_plus = solve_steady_heat_2d(nx, ny, dx, dy,
                                          max(theta_plus[0], 1e-6),
                                          max(theta_plus[1], 1e-6),
                                          heat_source + theta_plus[2] * T_obs)
        rel_change = np.linalg.norm(model_plus - base_model) / (np.linalg.norm(base_model) + 1e-14)
        sens[param_names[k]] = float(rel_change / perturbation)

    return sens
