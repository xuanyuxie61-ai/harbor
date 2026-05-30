
import numpy as np
from typing import Tuple, Callable, Optional


def rdo_derivatives(t: float, u: np.ndarray,
                    r: float = 0.25,
                    alpha: float = 0.5,
                    R: float = 1.0,
                    epsilon: float = 0.3,
                    gamma: float = 0.4,
                    seasonal_forcing: Optional[Callable] = None) -> np.ndarray:
    h_w, t_e = u[0], u[1]


    dh_dt = -r * h_w - alpha * t_e
    dT_dt = R * h_w - epsilon * (t_e ** 3) + gamma * t_e

    if seasonal_forcing is not None:
        forcing = seasonal_forcing(t)
        dT_dt += forcing

    return np.array([dh_dt, dT_dt])


def seasonal_cycle(t: float, amplitude: float = 0.1, phase: float = 0.0) -> float:
    return amplitude * np.cos(2.0 * np.pi * t - phase)


def rk4_integrate(f: Callable, u0: np.ndarray, t0: float, tf: float,
                  n_steps: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")

    dt = (tf - t0) / n_steps
    dim = u0.shape[0]
    t = np.linspace(t0, tf, n_steps + 1)
    u = np.zeros((n_steps + 1, dim), dtype=float)
    u[0] = u0

    for i in range(n_steps):
        k1 = f(t[i], u[i], **kwargs)
        k2 = f(t[i] + dt / 2.0, u[i] + dt / 2.0 * k1, **kwargs)
        k3 = f(t[i] + dt / 2.0, u[i] + dt / 2.0 * k2, **kwargs)
        k4 = f(t[i] + dt, u[i] + dt * k3, **kwargs)

        u[i + 1] = u[i] + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


        if np.any(np.isnan(u[i + 1])) or np.any(np.isinf(u[i + 1])):
            u[i + 1] = u[i]

    return t, u


def solve_rdo(years: float = 20.0,
              n_steps: int = 20000,
              h_w0: float = 0.5,
              t_e0: float = 0.3,
              r: float = 0.25,
              alpha: float = 0.5,
              R: float = 1.0,
              epsilon: float = 0.3,
              gamma: float = 0.4,
              seasonal_amp: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    u0 = np.array([h_w0, t_e0])

    def forcing(t):
        return seasonal_cycle(t, amplitude=seasonal_amp)

    def f(t, u):
        return rdo_derivatives(t, u, r=r, alpha=alpha, R=R,
                               epsilon=epsilon, gamma=gamma,
                               seasonal_forcing=forcing)

    t, u = rk4_integrate(f, u0, 0.0, years, n_steps)
    return t, u[:, 1]


def find_equilibrium(r: float, alpha: float, R: float,
                     epsilon: float, gamma: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    trivial = np.array([0.0, 0.0])



    discriminant = gamma - R * alpha / r

    if discriminant <= 0:
        return trivial, None

    t_star = np.sqrt(discriminant / epsilon)
    h_star = -alpha * t_star / r

    nontrivial = np.array([
        [h_star, t_star],
        [h_star, -t_star]
    ])

    return trivial, nontrivial


def oscillation_period_approx(r: float, alpha: float, R: float,
                              epsilon: float, gamma: float) -> float:
    omega_sq = R * alpha / r - gamma ** 2 / 4.0
    if omega_sq <= 0:
        return float('inf')
    return 2.0 * np.pi / np.sqrt(omega_sq)


def classify_dynamics(r: float, alpha: float, R: float,
                      epsilon: float, gamma: float) -> str:
    ratio = R * alpha / r
    if gamma < ratio:
        if gamma ** 2 / 4.0 > ratio:
            return "stable_node"
        else:
            return "damped_oscillation"
    else:
        if epsilon > 0.5:
            return "strongly_nonlinear_limit_cycle"
        return "limit_cycle"
