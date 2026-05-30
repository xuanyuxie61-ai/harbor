
import numpy as np
from scipy.special import gamma as scipy_gamma
from utils import safe_exp


def generalized_hermite_integral(expon, alpha):
    if expon % 2 == 1:
        return 0.0
    a = alpha + expon
    if a <= -1.0:
        return -np.inf
    return scipy_gamma((a + 1.0) / 2.0)


def gauss_hermite_nodes_weights(n, alpha=0.0):
    from numpy.polynomial.hermite import hermgauss
    x, w = hermgauss(n)

    if alpha != 0.0:
        w = w * (np.abs(x) ** alpha)
    return x.astype(np.float64), w.astype(np.float64)


def integrate_hermite_quadrature(f, n, alpha=0.0):
    x, w = gauss_hermite_nodes_weights(n, alpha)
    fx = np.array([f(xi) for xi in x], dtype=np.float64)
    return np.sum(w * fx)


def integrate_daem_activation_energy(E, sigma, T, n_quad=16):
    if T < 1e-6:
        return 0.0
    R = 8.314
    x, w = gauss_hermite_nodes_weights(n_quad, alpha=0.0)

    integrand = safe_exp(-(E + np.sqrt(2.0) * sigma * x) / (R * T))
    return np.sum(w * integrand) / np.sqrt(np.pi)


def square01_sample(n):
    return np.random.rand(2, n).astype(np.float64)


def square01_monte_carlo_integrate(f, n_samples):
    points = square01_sample(n_samples)
    values = np.array([f(points[0, i], points[1, i]) for i in range(n_samples)], dtype=np.float64)
    return np.mean(values), np.std(values) / np.sqrt(n_samples)


def reactor_cross_section_average(f, radius=1.0, n_samples=10000):
    u1 = np.random.rand(n_samples)
    u2 = np.random.rand(n_samples)
    r = radius * np.sqrt(u1)
    theta = 2.0 * np.pi * u2
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    values = np.array([f(x[i], y[i]) for i in range(n_samples)], dtype=np.float64)
    return np.mean(values), np.std(values) / np.sqrt(n_samples)


def quadrature_error_analysis(f, f_exact, max_degree=10, alpha=0.0):
    errors = []
    for degree in range(max_degree + 1):
        exact = generalized_hermite_integral(degree, alpha)
        x, w = gauss_hermite_nodes_weights(degree + 2, alpha)
        quad = np.sum(w * (x ** degree))
        if abs(exact) < 1e-15:
            err = abs(quad)
        else:
            err = abs((quad - exact) / exact)
        errors.append((degree, exact, quad, err))
    return errors
