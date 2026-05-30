
import numpy as np
from typing import Tuple, Optional


def deformation_gradient(dN_dX: np.ndarray, u_e: np.ndarray) -> np.ndarray:
    F = np.eye(3, dtype=np.float64)
    for i in range(4):
        F += np.outer(u_e[i], dN_dX[i])
    return F


def right_cauchy_green(F: np.ndarray) -> np.ndarray:
    return F.T @ F


def compute_invariants(C: np.ndarray) -> Tuple[float, float, float]:
    I1 = float(np.trace(C))
    I2 = 0.5 * (I1 ** 2 - float(np.trace(C @ C)))
    I3 = float(np.linalg.det(C))
    return I1, I2, I3


def neo_hookean_pk2_stress(C: np.ndarray, mu: float, lam: float) -> np.ndarray:

    raise NotImplementedError("Hole 1: 请实现neo_hookean_pk2_stress")


def neo_hookean_material_tangent(C: np.ndarray, mu: float, lam: float) -> np.ndarray:

    raise NotImplementedError("Hole 1: 请实现neo_hookean_material_tangent")


def green_lagrange_strain(C: np.ndarray) -> np.ndarray:
    return 0.5 * (C - np.eye(3))


def voigt_strain(E: np.ndarray) -> np.ndarray:
    return np.array([
        E[0, 0], E[1, 1], E[2, 2],
        2.0 * E[0, 1], 2.0 * E[0, 2], 2.0 * E[1, 2]
    ], dtype=np.float64)


def voigt_stress(S: np.ndarray) -> np.ndarray:
    return np.array([
        S[0, 0], S[1, 1], S[2, 2],
        S[0, 1], S[0, 2], S[1, 2]
    ], dtype=np.float64)


def cauchy_stress_from_pk2(F: np.ndarray, S: np.ndarray) -> np.ndarray:
    J = float(np.linalg.det(F))
    if abs(J) < 1e-14:
        raise ValueError("Cauchy应力: J 接近零")
    return (F @ S @ F.T) / J


def von_mises_cauchy(sigma: np.ndarray) -> float:
    tr = float(np.trace(sigma))
    s = sigma - (tr / 3.0) * np.eye(3)
    return float(np.sqrt(1.5 * np.sum(s * s)))






def _effective_shear_residual(mu_eff: float, mu0: float, gamma: float,
                               alpha: float = 0.1) -> float:
    return mu_eff - mu0 * np.exp(-alpha * gamma * mu_eff)


def brent_method(f, a: float, b: float, tol: float = 1e-12,
                 max_iter: int = 100) -> float:
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        raise ValueError("Brent法要求 f(a)*f(b) < 0")

    c = a
    fc = fa
    s = b
    d = e = b - a

    for _ in range(max_iter):
        if fb * fc > 0:
            c = a
            fc = fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, b = b, c
            c = a
            fa, fb = fb, fc
            fc = fa

        tol_act = 2.0 * tol * abs(b) + 0.5 * tol
        m = 0.5 * (c - b)
        if abs(m) <= tol_act or abs(fb) < tol:
            return b

        if abs(e) < tol_act or abs(fa) <= abs(fb):
            d = e = m
        else:
            s = fb / fa
            if a == c:

                p = 2.0 * m * s
                q = 1.0 - s
            else:

                q = fa / fc
                r = fb / fc
                p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0:
                q = -q
            p = abs(p)
            min1 = 3.0 * m * q - abs(tol_act * q)
            min2 = abs(e * q)
            if 2.0 * p < (min1 if min1 < min2 else min2):
                e = d
                d = p / q
            else:
                d = e = m
        a = b
        fa = fb
        if abs(d) > tol_act:
            b += d
        else:
            b += tol_act if m > 0 else -tol_act
        fb = f(b)
    return b


def solve_effective_shear_modulus(mu0: float, gamma: float,
                                   alpha: float = 0.1) -> float:
    if gamma < 0:
        raise ValueError("等效剪应变 γ 必须非负")
    if mu0 <= 0:
        raise ValueError("初始剪切模量 μ0 必须为正")


    f = lambda m: _effective_shear_residual(m, mu0, gamma, alpha)
    a = 1e-8
    b = mu0 * 5.0

    for _ in range(20):
        if f(a) * f(b) <= 0:
            break
        b *= 2.0
        if b > mu0 * 1e6:

            return mu0 * np.exp(-alpha * gamma * mu0)
    if f(a) * f(b) > 0:
        return mu0 * np.exp(-alpha * gamma * mu0)
    return brent_method(f, a, b, tol=1e-14)
