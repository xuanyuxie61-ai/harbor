
import numpy as np
from typing import Tuple


def raman_response_blow_wood(t: np.ndarray, tau1: float = 12.2e-15,
                              tau2: float = 32.0e-15) -> np.ndarray:
    if tau1 <= 0.0 or tau2 <= 0.0:
        raise ValueError("raman_response_blow_wood: tau1, tau2 must be > 0")
    A = (tau1 ** 2 + tau2 ** 2) / (tau1 * tau2 ** 2)
    h = np.zeros_like(t, dtype=float)
    mask = t >= 0.0
    h[mask] = A * np.exp(-t[mask] / tau2) * np.sin(t[mask] / tau1)
    return h


def raman_response_lin_agrawal(t: np.ndarray, tau1: float = 12.2e-15,
                                tau2: float = 32.0e-15) -> np.ndarray:
    return raman_response_blow_wood(t, tau1, tau2)


def nonlinear_response_full(t: np.ndarray, f_R: float = 0.18,
                            tau1: float = 12.2e-15,
                            tau2: float = 32.0e-15) -> np.ndarray:
    if not (0.0 <= f_R <= 1.0):
        raise ValueError("nonlinear_response_full: f_R must be in [0, 1]")
    h = raman_response_blow_wood(t, tau1, tau2)

    return h


def raman_auxiliary_ode_rhs(y: np.ndarray, t: float,
                             tau1: float, tau2: float,
                             pump_term: float) -> np.ndarray:
    A = (tau1 ** 2 + tau2 ** 2) / (tau1 * tau2 ** 2)
    u, v = y[0], y[1]
    omega0_sq = 1.0 / (tau1 ** 2) + 1.0 / (tau2 ** 2)
    damping = 2.0 / tau2
    dudt = v
    dvdt = -damping * v - omega0_sq * u + A * pump_term
    return np.array([dudt, dvdt], dtype=float)


def raman_response_convolution(A_power: np.ndarray, dt: float,
                                f_R: float = 0.18,
                                tau1: float = 12.2e-15,
                                tau2: float = 32.0e-15) -> np.ndarray:
    n = len(A_power)
    if n < 2:
        raise ValueError("raman_response_convolution: need at least 2 points")
    t = np.arange(n) * dt
    h = raman_response_blow_wood(t, tau1, tau2)





    raise NotImplementedError("Hole 2: 请实现 raman_response_convolution 的 FFT 卷积")


def self_steepening_factor(omega: np.ndarray, omega0: float) -> np.ndarray:
    if omega0 <= 0.0:
        raise ValueError("self_steepening_factor: omega0 must be > 0")
    return 1.0 + omega / omega0


def shock_operator_time(A: np.ndarray, dt: float, omega0: float) -> np.ndarray:
    n = len(A)
    dAdt = np.zeros_like(A, dtype=complex)
    if n >= 3:
        dAdt[1:-1] = (A[2:] - A[:-2]) / (2.0 * dt)

        dAdt[0] = (A[1] - A[0]) / dt
        dAdt[-1] = (A[-1] - A[-2]) / dt
    elif n == 2:
        dAdt[0] = (A[1] - A[0]) / dt
        dAdt[1] = (A[1] - A[0]) / dt
    return A + (1j / omega0) * dAdt
