
import numpy as np
from typing import Tuple


def legendre_gauss_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    if n < 1:
        raise ValueError("Order must be >= 1")
    if n == 1:
        return np.array([0.0]), np.array([2.0])




    j = np.arange(1.0, n)
    beta = np.sqrt(j**2 / (4.0 * j**2 - 1.0))


    J = np.diag(beta, k=1) + np.diag(beta, k=-1)


    eigenvalues, eigenvectors = np.linalg.eigh(J)


    x = eigenvalues
    w = 2.0 * eigenvectors[0, :]**2

    return x, w


def gauss_legendre_quadrature(f, a: float, b: float, n: int = 64) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or a >= b:
        return 0.0
    x, w = legendre_gauss_nodes_weights(n)

    t = 0.5 * (b - a) * x + 0.5 * (b + a)
    jac = 0.5 * (b - a)
    ft = np.array([f(ti) for ti in t])

    ft = np.where(np.isfinite(ft), ft, 0.0)
    return float(jac * np.sum(w * ft))


def precipitable_water(pressure_levels: np.ndarray, qv: np.ndarray,
                       p_sfc: float, T_sfc: float) -> float:
    g = 9.80665
    if len(pressure_levels) < 2:
        return 0.0
    p_top = max(100.0, pressure_levels[-1])
    p_bot = min(p_sfc, pressure_levels[0])


    def qv_of_p(p: float) -> float:
        if p <= pressure_levels[-1]:
            return float(qv[-1])
        if p >= pressure_levels[0]:
            return float(qv[0])

        idx = np.searchsorted(pressure_levels[::-1], p)
        idx = len(pressure_levels) - 1 - idx
        idx = max(0, min(idx, len(pressure_levels) - 2))
        dp = pressure_levels[idx] - pressure_levels[idx+1]
        if abs(dp) < 1e-6:
            return float(qv[idx])
        w = (p - pressure_levels[idx+1]) / dp
        return float(qv[idx+1] + w * (qv[idx] - qv[idx+1]))


    pw = gauss_legendre_quadrature(qv_of_p, p_top, p_bot, n=32) / g
    return max(0.0, pw)


def mass_weighted_integral(pressure_levels: np.ndarray, field: np.ndarray,
                           p_sfc: float) -> float:
    g = 9.80665
    if len(pressure_levels) < 2:
        return 0.0
    p_top = max(100.0, pressure_levels[-1])
    p_bot = min(p_sfc, pressure_levels[0])

    def field_of_p(p: float) -> float:
        if p <= pressure_levels[-1]:
            return float(field[-1])
        if p >= pressure_levels[0]:
            return float(field[0])
        idx = np.searchsorted(pressure_levels[::-1], p)
        idx = len(pressure_levels) - 1 - idx
        idx = max(0, min(idx, len(pressure_levels) - 2))
        dp = pressure_levels[idx] - pressure_levels[idx+1]
        if abs(dp) < 1e-6:
            return float(field[idx])
        w = (p - pressure_levels[idx+1]) / dp
        return float(field[idx+1] + w * (field[idx] - field[idx+1]))

    return gauss_legendre_quadrature(field_of_p, p_top, p_bot, n=24) / g


def convective_inhibition_integral(pressure_levels: np.ndarray,
                                   buoyancy: np.ndarray,
                                   p_sfc: float) -> float:
    g = 9.80665
    Rd = 287.05

    if len(pressure_levels) < 2:
        return 0.0

    p_top = max(100.0, pressure_levels[-1])
    p_bot = min(p_sfc, pressure_levels[0])

    def neg_buoyancy(p: float) -> float:
        if p <= pressure_levels[-1]:
            return min(0.0, float(buoyancy[-1]))
        if p >= pressure_levels[0]:
            return min(0.0, float(buoyancy[0]))
        idx = np.searchsorted(pressure_levels[::-1], p)
        idx = len(pressure_levels) - 1 - idx
        idx = max(0, min(idx, len(pressure_levels) - 2))
        dp = pressure_levels[idx] - pressure_levels[idx+1]
        if abs(dp) < 1e-6:
            return min(0.0, float(buoyancy[idx]))
        w = (p - pressure_levels[idx+1]) / dp
        b = float(buoyancy[idx+1] + w * (buoyancy[idx] - buoyancy[idx+1]))
        return min(0.0, b)

    return gauss_legendre_quadrature(neg_buoyancy, p_top, p_bot, n=48)
