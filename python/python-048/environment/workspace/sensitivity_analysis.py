
import numpy as np
from typing import Tuple, Callable
from pore_pressure_solver import euler_integrate


def sensitive_deriv(t: float, y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.size != 2:
        raise ValueError("状态向量维度必须为 2")
    dydt = np.zeros(2)
    dydt[0] = y[1]
    dydt[1] = y[0]
    return dydt


def sensitive_exact(t: float, y0: np.ndarray) -> np.ndarray:
    y0 = np.asarray(y0, dtype=float)
    return np.array([
        y0[0] * np.cosh(t) + y0[1] * np.sinh(t),
        y0[0] * np.sinh(t) + y0[1] * np.cosh(t)
    ])


def lyapunov_exponent_euler(y0: np.ndarray, delta_y0: np.ndarray,
                            tspan: Tuple[float, float], n_steps: int) -> float:
    t, y_base = euler_integrate(sensitive_deriv, tspan, y0, n_steps)
    t, y_pert = euler_integrate(sensitive_deriv, tspan, y0 + delta_y0, n_steps)

    delta_y_final = y_pert[-1, :] - y_base[-1, :]
    norm_final = np.linalg.norm(delta_y_final)
    norm_init = np.linalg.norm(delta_y0)

    if norm_init < 1.0e-15 or norm_final < 1.0e-15:
        return 0.0
    T = tspan[1] - tspan[0]
    return np.log(norm_final / norm_init) / T


class FracturePropagationSensitivity:

    def __init__(self, kappa: float = 1.0, sigma_noise: float = 0.05):
        self.kappa = kappa
        self.sigma_noise = sigma_noise

    def front_angle_ode(self, t: float, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=float)
        if state.size != 2:
            raise ValueError("状态维度必须为 2")
        dstate = np.zeros(2)
        dstate[0] = state[1]
        dstate[1] = self.kappa ** 2 * state[0]
        return dstate

    def simulate_front(self, y0: np.ndarray, tspan: Tuple[float, float],
                       n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        return euler_integrate(self.front_angle_ode, tspan, y0, n_steps)

    def adjoint_sensitivity(self, y0: np.ndarray, tspan: Tuple[float, float],
                            n_steps: int, parameter_index: int = 0) -> float:

        t_fwd, y_fwd = self.simulate_front(y0, tspan, n_steps)
        dt = t_fwd[1] - t_fwd[0]


        lambda_final = np.zeros(2)
        lambdas = np.zeros((n_steps + 1, 2))
        lambdas[-1, :] = lambda_final


        for k in range(n_steps, 0, -1):
            theta_k = y_fwd[k, 0]



            A = np.array([[0.0, self.kappa ** 2],
                          [1.0, 0.0]])
            rhs = -np.array([2.0 * theta_k, 0.0]) - A.T @ lambdas[k, :]
            lambdas[k - 1, :] = lambdas[k, :] + dt * rhs


        dJ_dkappa = 0.0
        for k in range(n_steps + 1):
            dJ_dkappa += 2.0 * self.kappa * y_fwd[k, 0] * lambdas[k, 1] * dt

        return float(dJ_dkappa)
