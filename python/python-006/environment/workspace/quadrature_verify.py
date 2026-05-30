
import numpy as np
import math
from typing import Callable, Tuple





def hermite_integral(p: int) -> float:
    if p < 0:
        raise ValueError("Exponent p must be non-negative.")
    if p % 2 == 1:
        return 0.0


    double_fact = 1.0
    for k in range(p - 1, 0, -2):
        double_fact *= k

    return double_fact * math.sqrt(math.pi) / (2.0**(p / 2.0))


def hermite_quadrature_exactness(n: int, x: np.ndarray, w: np.ndarray,
                                 p_max: int = 10) -> np.ndarray:
    x = np.asarray(x).reshape(-1)
    w = np.asarray(w).reshape(-1)

    if x.size != n or w.size != n:
        raise ValueError("x and w must have length n.")

    errors = np.zeros(p_max + 1)

    for p in range(p_max + 1):
        exact = hermite_integral(p)
        values = x**p
        quad = np.dot(w, values)

        if abs(exact) < 1e-30:
            err = abs(quad - exact)
        else:
            err = abs((quad - exact) / exact)

        errors[p] = err

    return errors





def chebyshev1_integral(p: int) -> float:
    if p < 0:
        raise ValueError("p must be non-negative.")
    if p % 2 == 1:
        return 0.0


    ratio = 1.0
    for k in range(2, p + 1, 2):
        ratio *= (k - 1.0) / k

    return math.pi * ratio


def chebyshev_quadrature_exactness(n: int, x: np.ndarray, w: np.ndarray,
                                   p_max: int = 10, kind: int = 1) -> np.ndarray:
    x = np.asarray(x).reshape(-1)
    w = np.asarray(w).reshape(-1)
    errors = np.zeros(p_max + 1)

    for p in range(p_max + 1):
        if kind == 1:
            exact = chebyshev1_integral(p)
        else:
            exact = 0.0

        values = x**p
        quad = np.dot(w, values)

        if abs(exact) < 1e-30:
            err = abs(quad - exact)
        else:
            err = abs((quad - exact) / exact)

        errors[p] = err

    return errors





def laguerre_integral(p: int) -> float:
    if p < 0:
        raise ValueError("p must be non-negative.")
    return math.factorial(p)


def laguerre_quadrature_exactness(n: int, x: np.ndarray, w: np.ndarray,
                                  p_max: int = 10) -> np.ndarray:
    x = np.asarray(x).reshape(-1)
    w = np.asarray(w).reshape(-1)
    errors = np.zeros(p_max + 1)

    for p in range(p_max + 1):
        exact = laguerre_integral(p)
        values = x**p
        quad = np.dot(w, values)

        if abs(exact) < 1e-30:
            err = abs(quad - exact)
        else:
            err = abs((quad - exact) / exact)

        errors[p] = err

    return errors





def gauss_hermite_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    if n <= 0:
        raise ValueError("n must be positive.")


    diag = np.zeros(n)
    off_diag = np.zeros(n - 1)
    for i in range(n - 1):
        off_diag[i] = math.sqrt((i + 1) / 2.0)

    J = np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
    eigvals, eigvecs = np.linalg.eigh(J)

    x = eigvals

    w = math.sqrt(math.pi) * eigvecs[0, :]**2

    return x, w


def gauss_laguerre_nodes_weights(n: int, alpha: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    if n <= 0:
        raise ValueError("n must be positive.")

    diag = np.zeros(n)
    off_diag = np.zeros(n - 1)
    for i in range(n):
        diag[i] = 2.0 * i + 1.0 + alpha
    for i in range(n - 1):
        off_diag[i] = -math.sqrt((i + 1.0) * (i + 1.0 + alpha))

    J = np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
    eigvals, eigvecs = np.linalg.eigh(J)

    x = eigvals
    mu0 = math.gamma(1.0 + alpha)
    w = mu0 * eigvecs[0, :]**2

    return x, w





def verify_eos_integrals() -> dict:
    results = {}


    n = 16
    x_lag, w_lag = gauss_laguerre_nodes_weights(n)
    errors_lag = laguerre_quadrature_exactness(n, x_lag, w_lag, p_max=2 * n - 1)
    results['laguerre_max_error'] = float(np.max(errors_lag))


    x_her, w_her = gauss_hermite_nodes_weights(n)
    errors_her = hermite_quadrature_exactness(n, x_her, w_her, p_max=2 * n - 1)
    results['hermite_max_error'] = float(np.max(errors_her))




    def fermi_dirac_integrand(t):
        return t**3 / (math.exp(t) + 1.0) * math.exp(t)

    quad_val = np.sum(w_lag * np.array([fermi_dirac_integrand(xi) for xi in x_lag]))
    exact_fd = 7.0 * math.pi**4 / 120.0
    results['fermi_dirac_relative_error'] = abs((quad_val - exact_fd) / exact_fd)

    return results
