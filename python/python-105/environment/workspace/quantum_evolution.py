
import numpy as np
from linear_solver import gauss_elimination_partial_pivot


def spdc_derivative(t: float, y: np.ndarray,
                    gamma: np.ndarray, kappa_func: callable,
                    f_noise: callable) -> np.ndarray:
    y = np.asarray(y, dtype=np.complex128)
    if y.shape != (3,):
        raise ValueError("y 必须为长度 3 的向量。")
    if np.any(gamma < 0.0):
        raise ValueError("损耗率 gamma 必须非负。")

    kappa = kappa_func(t)
    f = f_noise(t)

    dydt = np.zeros(3, dtype=np.complex128)
    dydt[0] = -0.5 * gamma[0] * y[0] + np.conj(kappa) * np.conj(y[1]) * y[2] + f[0]
    dydt[1] = -0.5 * gamma[1] * y[1] + np.conj(kappa) * np.conj(y[0]) * y[2] + f[1]
    dydt[2] = -0.5 * gamma[2] * y[2] - kappa * y[0] * y[1] + f[2]
    return dydt


def spdc_jacobian(t: float, y: np.ndarray,
                  gamma: np.ndarray, kappa_func: callable) -> np.ndarray:
    kappa = kappa_func(t)
    J = np.zeros((3, 3), dtype=np.complex128)
    J[0, 0] = -0.5 * gamma[0]
    J[0, 1] = np.conj(kappa) * np.conj(y[2])
    J[0, 2] = np.conj(kappa) * np.conj(y[1])

    J[1, 0] = np.conj(kappa) * np.conj(y[2])
    J[1, 1] = -0.5 * gamma[1]
    J[1, 2] = np.conj(kappa) * np.conj(y[0])

    J[2, 0] = -kappa * y[1]
    J[2, 1] = -kappa * y[0]
    J[2, 2] = -0.5 * gamma[2]
    return J


def backward_euler_spdc(y0: np.ndarray, t_span: tuple, n_steps: int,
                        gamma: np.ndarray, kappa_func: callable,
                        f_noise: callable,
                        newton_tol: float = 1e-10,
                        max_newton: int = 20) -> tuple:
    y0 = np.asarray(y0, dtype=np.complex128)
    if y0.shape != (3,):
        raise ValueError("y0 必须为长度 3 的向量。")
    if n_steps <= 0:
        raise ValueError("n_steps 必须为正。")

    t0, tf = t_span
    h = (tf - t0) / n_steps
    if h <= 0.0:
        raise ValueError("t_span 必须满足 tf > t0。")

    t = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros((n_steps + 1, 3), dtype=np.complex128)
    y[0, :] = y0

    for n in range(n_steps):
        tp = t[n + 1]
        y_old = y[n, :]
        yp = y_old + h * spdc_derivative(t[n], y_old, gamma, kappa_func, f_noise)


        for _ in range(max_newton):
            f_tp = spdc_derivative(tp, yp, gamma, kappa_func, f_noise)
            Jf = spdc_jacobian(tp, yp, gamma, kappa_func)
            R = yp - y_old - h * f_tp


            J_R = np.eye(6, dtype=np.float64)
            J_R[:3, :3] -= h * Jf.real
            J_R[:3, 3:] += h * Jf.imag
            J_R[3:, :3] -= h * Jf.imag
            J_R[3:, 3:] -= h * Jf.real

            R_real = np.hstack([R.real, R.imag])
            try:
                delta = gauss_elimination_partial_pivot(J_R, -R_real)
            except ValueError:

                delta = np.linalg.lstsq(J_R, -R_real, rcond=None)[0]

            yp += delta[:3] + 1j * delta[3:]
            if np.linalg.norm(R) < newton_tol:
                break

        y[n + 1, :] = yp

    return t, y


def robertson_like_conservation(y: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(y) ** 2, axis=1)
