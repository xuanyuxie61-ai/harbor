
import numpy as np
from scipy.special import jv, yv, gamma, gammaln
from scipy.optimize import newton
import warnings


class RandomState:

    def __init__(self, seed=None):
        if seed is None:
            seed = 123456789
        self._state = int(seed) & 0xFFFFFFFF
        self._a = 1103515245
        self._b = 12345
        self._mod = 2 ** 31

    def _lcg_next(self):
        self._state = (self._a * self._state + self._b) % self._mod
        return self._state

    def uniform_ab(self, n, a, b):
        if n <= 0:
            return np.array([])
        vals = np.array([self._lcg_next() for _ in range(n)], dtype=np.float64)
        vals = vals / self._mod
        return a + (b - a) * vals

    def maxwell_boltzmann(self, n, T, m):
        if n <= 0 or T <= 0 or m <= 0:
            raise ValueError("n, T, m must be positive for Maxwell-Boltzmann sampling.")
        sigma = np.sqrt(T / m)
        u1 = self.uniform_ab(n, 1e-12, 1.0)
        u2 = self.uniform_ab(n, 0.0, 2.0 * np.pi)
        u3 = self.uniform_ab(n, 0.0, 1.0)
        r = np.sqrt(-2.0 * np.log(u1)) * sigma
        theta = np.arccos(2.0 * u3 - 1.0)
        vx = r * np.sin(theta) * np.cos(u2)
        vy = r * np.sin(theta) * np.sin(u2)
        vz = r * np.cos(theta)
        return np.stack([vx, vy, vz], axis=1)


def bessel_zero_newton(n, k, kind=1, tol=1e-14, max_iter=100):
    n = abs(n)
    if k <= 0:
        raise ValueError("k must be positive integer.")


    if kind == 1:
        if k == 1:
            x0 = 0.411557 + 0.999987 * n + 0.698029 * (n + 1) ** 0.335300 + 1.069775 * (n + 1) ** 0.339671
        elif k == 2:
            x0 = 1.933951 + 1.000077 * n - 0.805720 * (n + 1) ** 0.456215 + 3.387646 * (n + 1) ** 0.388380
        elif k == 3:
            x0 = 5.407708 + 1.000939 * n + 2.669262 * (n + 1) ** 0.429702 - 0.174926 * (n + 1) ** 0.633480
        else:

            z2 = bessel_zero_newton(n, 2, kind)
            z3 = bessel_zero_newton(n, 3, kind)
            spacing = z3 - z2
            x0 = z3 + (k - 3) * spacing
    else:
        if k == 1:
            x0 = 0.079505 + 0.999998 * n + 0.890381 * (n + 1) ** 0.335377 + 0.027060 * (n + 1) ** 0.308720
        elif k == 2:
            x0 = 1.045025 + 1.000021 * n - 0.437921 * (n + 1) ** 0.434823 + 2.701131 * (n + 1) ** 0.366245
        elif k == 3:
            x0 = 3.727779 + 1.000353 * n + 2.685667 * (n + 1) ** 0.398248 - 0.112980 * (n + 1) ** 0.604770
        else:
            z2 = bessel_zero_newton(n, 2, kind)
            z3 = bessel_zero_newton(n, 3, kind)
            spacing = z3 - z2
            x0 = z3 + (k - 3) * spacing

    def f(x):
        if kind == 1:
            return float(jv(n, x))
        else:
            return float(yv(n, x))

    try:
        zero = newton(f, x0, tol=tol, maxiter=max_iter)
    except RuntimeError:
        zero = x0
    return zero


def laguerre_polynomial_alpha(x, n, alpha=0.0):
    x = np.atleast_1d(x)
    if alpha <= -1.0:
        raise ValueError("alpha must be > -1 for Laguerre polynomials.")
    if n < 0:
        return np.zeros_like(x)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 1.0 + alpha - x

    L_prev2 = np.ones_like(x)
    L_prev1 = 1.0 + alpha - x
    for i in range(2, n + 1):
        L_curr = ((2.0 * i - 1.0 + alpha - x) * L_prev1 - (i - 1.0 + alpha) * L_prev2) / i
        L_prev2 = L_prev1
        L_prev1 = L_curr
    return L_prev1


def gegenbauer_polynomial(x, n, lambda_):
    x = np.atleast_1d(x)
    if lambda_ <= -0.5:
        raise ValueError("lambda must be > -0.5 for Gegenbauer polynomials.")
    if n < 0:
        return np.zeros_like(x)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * lambda_ * x

    C_prev2 = np.ones_like(x)
    C_prev1 = 2.0 * lambda_ * x
    for i in range(1, n):
        C_curr = (2.0 * (i + lambda_) * x * C_prev1 - (i + 2.0 * lambda_ - 1.0) * C_prev2) / (i + 1.0)
        C_prev2 = C_prev1
        C_prev1 = C_curr
    return C_prev1


def check_bounds(x, lower, upper, name="variable"):
    x = np.atleast_1d(x)
    if np.any(x < lower) or np.any(x > upper):
        warnings.warn(f"{name} out of bounds [{lower}, {upper}], clipping applied.")
        x = np.clip(x, lower, upper)
    return x


def relative_convergence_check(val_new, val_old, rtol=1e-6, atol=1e-12):
    diff = np.abs(val_new - val_old)
    scale = 0.5 * (np.abs(val_new) + np.abs(val_old)) + atol
    return np.all(diff < rtol * scale)


def safe_sqrt(x, eps=1e-30):
    return np.sqrt(np.maximum(x, eps))


def compute_radial_grid(r_min, r_max, n_r, grid_type="legendre"):
    if grid_type == "uniform":
        return np.linspace(r_min, r_max, n_r)
    elif grid_type == "legendre":

        from numpy.polynomial.legendre import leggauss
        xi, wi = leggauss(n_r)
        r = 0.5 * (r_max - r_min) * (xi + 1.0) + r_min
        w = 0.5 * (r_max - r_min) * wi
        return r, w
    else:
        raise ValueError(f"Unknown grid_type: {grid_type}")
