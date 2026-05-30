
import numpy as np
from scipy.special import gamma, factorial, comb
from typing import Tuple


def associated_legendre_normalized(l_max: int, m: int, x: np.ndarray) -> np.ndarray:
    if l_max < m:
        raise ValueError("l_max 必须不小于 |m|")
    x = np.atleast_1d(x)
    if np.any(np.abs(x) > 1.0 + 1e-12):
        raise ValueError("x 必须位于 [-1, 1] 区间。")
    x = np.clip(x, -1.0, 1.0)

    n_points = x.size
    P = np.zeros((n_points, l_max + 1), dtype=np.float64)


    if m == 0:
        P[:, 0] = np.sqrt(0.5)
    else:
        pmm = np.ones(n_points, dtype=np.float64)
        somx2 = np.sqrt(np.maximum(0.0, 1.0 - x * x))
        fact = 1.0
        for i in range(1, m + 1):
            pmm *= (-fact) * somx2
            fact += 2.0

        norm_pm = np.sqrt((2.0 * m + 1.0) / 2.0 * factorial(2 * m) /
                          (2.0 ** (2 * m) * factorial(m) ** 2))
        P[:, m] = norm_pm * pmm


    if m < l_max:
        if m == 0:
            P[:, 1] = np.sqrt(3.0 / 2.0) * x
        else:
            pll = x * (2.0 * m + 1.0) * P[:, m] / np.sqrt((2.0 * m + 1.0) * (2.0 * m + 3.0))



            p_unnorm = np.zeros((n_points, l_max + 1), dtype=np.float64)
            p_unnorm[:, m] = 1.0
            for mm in range(1, m + 1):
                p_unnorm[:, m] *= (-1.0) * (2.0 * mm - 1.0) * somx2
            if m < l_max:
                p_unnorm[:, m + 1] = x * (2.0 * m + 1.0) * p_unnorm[:, m]
            for ell in range(m + 2, l_max + 1):
                p_unnorm[:, ell] = (x * (2.0 * ell - 1.0) * p_unnorm[:, ell - 1]
                                    - (ell + m - 1.0) * p_unnorm[:, ell - 2]) / (ell - m)

            for ell in range(m, l_max + 1):
                norm = np.sqrt((2.0 * ell + 1.0) / 2.0 * factorial(ell - m) / factorial(ell + m))
                P[:, ell] = norm * p_unnorm[:, ell]
            return P

    for ell in range(m + 1, l_max + 1):

        a1 = np.sqrt((4.0 * ell ** 2 - 1.0) / (ell ** 2 - m ** 2))
        a2 = np.sqrt(((2.0 * ell + 1.0) * (ell + m - 1.0) * (ell - m - 1.0)) /
                     ((2.0 * ell - 3.0) * (ell ** 2 - m ** 2)))
        P[:, ell] = a1 * x * P[:, ell - 1] - a2 * P[:, ell - 2]

    return P


def spherical_harmonic_basis(l_max: int, theta: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    theta = np.atleast_1d(theta)
    phi = np.atleast_1d(phi)
    n_points = theta.size
    n_modes = (l_max + 1) ** 2
    Y_real = np.zeros((n_points, n_modes), dtype=np.float64)
    Y_imag = np.zeros((n_points, n_modes), dtype=np.float64)

    idx = 0
    for l in range(l_max + 1):
        x = np.cos(theta)
        Plm = associated_legendre_normalized(l, l, x)
        for m in range(-l, l + 1):
            m_abs = abs(m)

            plm_val = Plm[:, m_abs] if m_abs <= l else np.zeros(n_points)
            if m < 0:

                phase = (-1.0) ** m_abs
                Y_real[:, idx] = phase * plm_val * np.cos(m_abs * phi)
                Y_imag[:, idx] = -phase * plm_val * np.sin(m_abs * phi)
            else:
                Y_real[:, idx] = plm_val * np.cos(m * phi)
                Y_imag[:, idx] = plm_val * np.sin(m * phi)
            idx += 1

    return Y_real, Y_imag


def jacobi_polynomial(n: int, alpha: float, beta: float, x: np.ndarray) -> np.ndarray:
    x = np.atleast_1d(x)
    if n < 0:
        raise ValueError("n 必须为非负整数。")
    if n == 0:
        return np.ones_like(x, dtype=np.float64)
    if n == 1:
        return 0.5 * (alpha - beta) + 0.5 * (alpha + beta + 2.0) * x

    P_prev2 = np.ones_like(x, dtype=np.float64)
    P_prev1 = 0.5 * (alpha - beta) + 0.5 * (alpha + beta + 2.0) * x

    for nn in range(2, n + 1):
        a_n = 2.0 * nn * (nn + alpha + beta) * (2.0 * nn + alpha + beta - 2.0)
        b_n = (2.0 * nn + alpha + beta - 1.0) * (alpha ** 2 - beta ** 2)
        c_n = (2.0 * nn + alpha + beta - 2.0) * (2.0 * nn + alpha + beta - 1.0) * (2.0 * nn + alpha + beta)
        d_n = 2.0 * (nn + alpha - 1.0) * (nn + beta - 1.0) * (2.0 * nn + alpha + beta)

        if abs(a_n) < 1e-15:
            raise RuntimeError(f"Jacobi 递推系数 a_{nn} 过小。")
        P_curr = ((b_n + c_n * x) * P_prev1 - d_n * P_prev2) / a_n
        P_prev2 = P_prev1
        P_prev1 = P_curr

    return P_prev1


def bernstein_basis(n: int, x: np.ndarray) -> np.ndarray:
    x = np.atleast_1d(x)
    if np.any((x < -1e-12) | (x > 1.0 + 1e-12)):
        raise ValueError("Bernstein 基仅定义在 [0,1] 上。")
    x = np.clip(x, 0.0, 1.0)

    n_points = x.size
    B = np.zeros((n_points, n + 1), dtype=np.float64)
    if n == 0:
        B[:, 0] = 1.0
        return B


    B[:, 0] = 1.0 - x
    B[:, 1] = x
    for j in range(2, n + 1):
        B[:, j] = x * B[:, j - 1]
        for k in range(j - 1, 0, -1):
            B[:, k] = x * B[:, k - 1] + (1.0 - x) * B[:, k]
        B[:, 0] = (1.0 - x) * B[:, 0]

    return B


def bernstein_approximate(f_values: np.ndarray, a: float, b: float,
                          n: int, x_eval: np.ndarray) -> np.ndarray:
    if len(f_values) != n + 1:
        raise ValueError("f_values 长度必须等于 n+1。")
    if abs(b - a) < 1e-15:
        raise ValueError("区间长度必须为正。")

    t = (x_eval - a) / (b - a)
    B = bernstein_basis(n, t)
    y_eval = B @ f_values
    return y_eval
