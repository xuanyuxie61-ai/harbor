
import numpy as np
from typing import Callable


def quad_serial(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int) -> float:
    if n < 2:
        raise ValueError("quad_serial: n 必须至少为 2")
    x = np.linspace(a, b, n)
    fx = f(x)
    h = (b - a) / (n - 1)
    return h * (0.5 * fx[0] + np.sum(fx[1:-1]) + 0.5 * fx[-1])


def quad_simpson(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int) -> float:
    if n < 3 or n % 2 == 0:
        n = n + 1 if n % 2 == 0 else n
        if n < 3:
            n = 3
    x = np.linspace(a, b, n)
    fx = f(x)
    h = (b - a) / (n - 1)
    return h / 3.0 * (fx[0] + 4.0 * np.sum(fx[1:-1:2]) + 2.0 * np.sum(fx[2:-1:2]) + fx[-1])


def quad_adaptive(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-8,
    max_depth: int = 20,
) -> float:
    def _recurse(left: float, right: float, fl: float, fm: float, fr: float, depth: int) -> float:
        mid = 0.5 * (left + right)
        m1 = 0.5 * (left + mid)
        m2 = 0.5 * (mid + right)
        f1 = f(m1)
        f2 = f(m2)

        h = right - left
        whole = h / 6.0 * (fl + 4.0 * fm + fr)
        left_part = h / 12.0 * (fl + 4.0 * f1 + fm)
        right_part = h / 12.0 * (fm + 4.0 * f2 + fr)
        total = left_part + right_part

        if depth >= max_depth or abs(total - whole) <= 15.0 * tol:
            return total + (total - whole) / 15.0

        return (_recurse(left, mid, fl, f1, fm, depth + 1)
                + _recurse(mid, right, fm, f2, fr, depth + 1))

    mid = 0.5 * (a + b)
    return _recurse(a, b, f(a), f(mid), f(b), 0)


def integrate_power_density(velocity: np.ndarray, rho: float = 1025.0) -> float:
    u = np.asarray(velocity, dtype=float)
    if u.size == 0:
        return 0.0
    if u.size == 1:
        return 0.5 * rho * abs(u[0]) ** 3
    power = 0.5 * rho * np.abs(u) ** 3

    return np.trapezoid(power) / (power.size - 1) if power.size > 1 else power[0]


def integrate_structural_work(force: np.ndarray, displacement: np.ndarray) -> float:
    return np.trapezoid(force, displacement)
