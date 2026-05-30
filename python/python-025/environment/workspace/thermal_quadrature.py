
import numpy as np
from scipy.special import gamma as gamma_func


def generalized_hermite_integral(expon, alpha):
    if expon < 0:
        return 0.0
    if expon % 2 == 1:
        return 0.0
    return gamma_func((alpha + expon + 1.0) / 2.0)


def wedge_monomial_integral(exponents):
    from math import factorial
    e1, e2, e3 = exponents
    
    if e3 % 2 == 1:
        return 0.0
    
    xy_int = factorial(e1) * factorial(e2) / factorial(e1 + e2 + 2)
    z_int = 2.0 / (e3 + 1.0)
    return xy_int * z_int


def thermal_average_displacement(omega, T, m_d=1.0):
    hbar = 1.054571817e-34
    k_B = 1.380649e-23
    
    if T < 1e-12:

        return hbar / (2.0 * m_d * max(omega, 1e-20))
    
    x = hbar * omega / (k_B * T)
    if x > 50.0:

        n_B = 0.0
    else:
        n_B = 1.0 / (np.exp(x) - 1.0)
    
    return (hbar / (m_d * omega)) * (n_B + 0.5)


def partition_function_harmonic(omegas, T):
    hbar = 1.054571817e-34
    k_B = 1.380649e-23
    
    if T < 1e-12:
        return 0.0
    
    log_Z = 0.0
    for w in omegas:
        if w > 1e-15:
            x = hbar * w / (2.0 * k_B * T)
            if x > 50.0:
                log_Z -= x
            else:
                log_Z -= np.log(2.0 * np.sinh(x))
    return np.exp(log_Z)


def specific_heat_harmonic(omegas, T):
    hbar = 1.054571817e-34
    k_B = 1.380649e-23
    
    if T < 1e-12:
        return 0.0
    
    C = 0.0
    for w in omegas:
        if w > 1e-15:
            x = hbar * w / (k_B * T)
            if x > 50.0:
                C += k_B * x**2 * np.exp(-x)
            else:
                ex = np.exp(x)
                C += k_B * x**2 * ex / (ex - 1.0)**2
    return C


def gauss_hermite_quadrature(f, n_nodes, alpha=0.0):
    from numpy.polynomial.hermite import hermgauss
    x, w = hermgauss(n_nodes)
    if alpha != 0.0:
        w = w * np.abs(x)**alpha
    return np.sum(w * f(x))
