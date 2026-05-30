
import numpy as np
from typing import Tuple, Callable, Optional


def golden_section_search(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-8,
    max_iter: int = 100
) -> Tuple[float, float]:
    if a >= b:
        a, b = min(a, b), max(a, b)
        if a == b:
            return a, f(a)

    phi = (np.sqrt(5.0) - 1.0) / 2.0
    resphi = 1.0 - phi

    c = a + resphi * (b - a)
    d = a + phi * (b - a)
    fc = f(c)
    fd = f(d)

    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = a + resphi * (b - a)
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + phi * (b - a)
            fd = f(d)

    x_min = 0.5 * (a + b)
    f_min = f(x_min)
    return x_min, f_min


def backtracking_line_search(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    p: np.ndarray,
    alpha_init: float = 1.0,
    c1: float = 1e-4,
    rho: float = 0.5,
    max_iter: int = 20
) -> float:
    fx = f(x)
    gp = float(np.dot(grad_f(x), p))
    if gp >= 0.0:

        return 0.0

    alpha = alpha_init
    for _ in range(max_iter):
        x_new = x + alpha * p
        fx_new = f(x_new)
        if fx_new <= fx + c1 * alpha * gp:
            return alpha
        alpha *= rho
        if alpha < 1e-15:
            return 0.0
    return alpha


def unimodal_test_suite():
    tests = {}

    tests["sharp_power"] = (
        lambda x: 12.0 + 1000.0 * abs(x - 2.8) ** 8.4,
        2.0, 3.5, 2.8
    )

    tests["cubic"] = (
        lambda x: x ** 3 - 3.0 * x ** 2 - 5.0 * x + 8.0,
        -2.0, 5.0, (3.0 + np.sqrt(24.0)) / 3.0
    )

    tests["trig"] = (
        lambda x: 1.2 + x - 5.0 * np.sin(2.0 * x),
        -2.0, 5.0, None
    )
    return tests


def optimize_subdomain_overlap(
    residual_func: Callable[[float], float],
    overlap_range: Tuple[float, float] = (0.0, 0.5),
    tol: float = 1e-4
) -> float:
    a, b = overlap_range
    if a < 0:
        a = 0.0
    if b > 1.0:
        b = 1.0
    x_min, f_min = golden_section_search(residual_func, a, b, tol=tol, max_iter=50)
    return x_min


def power_iteration_estimate(
    A_matvec: Callable[[np.ndarray], np.ndarray],
    n: int,
    max_iter: int = 30,
    tol: float = 1e-6
) -> float:
    v = np.random.randn(n)
    v = v / (np.linalg.norm(v) + 1e-15)
    lam_old = 0.0
    for _ in range(max_iter):
        Av = A_matvec(v)
        lam = float(np.dot(v, Av))
        v = Av / (np.linalg.norm(Av) + 1e-15)
        if abs(lam - lam_old) < tol * abs(lam + 1e-15):
            break
        lam_old = lam
    return abs(lam)
