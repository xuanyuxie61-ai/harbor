"""
thermal_quadrature.py
=====================
Thermal averages and high-dimensional quadrature synthesized from seed projects:
  - 464_gen_hermite_exactness (generalized Gauss-Hermite quadrature exactness)
  - 1406_wedge_exactness (wedge monomial integral exactness)

Core algorithms:
  - Exact generalized Gauss-Hermite monomial integrals
  - Exact unit wedge monomial integrals for Voronoi cell geometry
  - Thermal average displacement via Bose-Einstein statistics
  - Partition function for harmonic crystal
  - Quadrature-based thermodynamic integration
"""

import numpy as np
from scipy.special import gamma as gamma_func


def generalized_hermite_integral(expon, alpha):
    """
    Exact integral of generalized Gauss-Hermite monomial:
      H(n, alpha) = integral_{-inf}^{+inf} x^n * |x|^alpha * exp(-x^2) dx
    
    Based on seed 464_gen_hermite_exactness.
    
    Analytical result:
      H(n, alpha) = 0                              if n is odd
                  = Gamma((n + alpha + 1) / 2)     if n is even
    
    This is the exact integral used to validate quadrature rules for
    thermal distribution moments.
    """
    if expon < 0:
        return 0.0
    if expon % 2 == 1:
        return 0.0
    return gamma_func((alpha + expon + 1.0) / 2.0)


def wedge_monomial_integral(exponents):
    """
    Exact integral of monomial x^e1 * y^e2 * z^e3 over the unit wedge.
    
    Based on seed 1406_wedge_exactness.
    
    Unit wedge domain:
      0 <= x,  0 <= y,  x + y <= 1,  -1 <= z <= 1
    
    Analytical result:
      I = [e1! * e2! / (e1 + e2 + 2)!] * [2 / (e3 + 1)]   if e3 is even
        = 0                                                if e3 is odd
    
    Used for computing charge integrals over wedge-shaped Voronoi cells
    in the plasma sheath region.
    """
    from math import factorial
    e1, e2, e3 = exponents
    
    if e3 % 2 == 1:
        return 0.0
    
    xy_int = factorial(e1) * factorial(e2) / factorial(e1 + e2 + 2)
    z_int = 2.0 / (e3 + 1.0)
    return xy_int * z_int


def thermal_average_displacement(omega, T, m_d=1.0):
    """
    Compute mean-square displacement <u^2> for a quantum harmonic oscillator
    of mass m_d at temperature T.
    
    For a phonon mode with frequency omega:
      <u^2> = (hbar / (m_d * omega)) * [ n_B(omega) + 1/2 ]
    
    where n_B is the Bose-Einstein occupation number:
      n_B = 1 / (exp(hbar * omega / (k_B * T)) - 1)
    
    In the classical limit (k_B*T >> hbar*omega):
      <u^2> -> k_B * T / (m_d * omega^2)
    """
    hbar = 1.054571817e-34  # J*s
    k_B = 1.380649e-23
    
    if T < 1e-12:
        # Zero-point motion only
        return hbar / (2.0 * m_d * max(omega, 1e-20))
    
    x = hbar * omega / (k_B * T)
    if x > 50.0:
        # Low-temperature limit: exp(x) is huge, n_B -> 0
        n_B = 0.0
    else:
        n_B = 1.0 / (np.exp(x) - 1.0)
    
    return (hbar / (m_d * omega)) * (n_B + 0.5)


def partition_function_harmonic(omegas, T):
    """
    Compute partition function for a system of independent harmonic oscillators.
    
    Z = product_k [ 1 / (2 * sinh(hbar * omega_k / (2 * k_B * T))) ]
    
    In the classical limit:
      Z -> product_k [ k_B * T / (hbar * omega_k) ]
    """
    hbar = 1.054571817e-34
    k_B = 1.380649e-23
    
    if T < 1e-12:
        return 0.0
    
    log_Z = 0.0
    for w in omegas:
        if w > 1e-15:
            x = hbar * w / (2.0 * k_B * T)
            if x > 50.0:
                log_Z -= x  # sinh(x) -> 0.5*exp(x), so 1/sinh -> 2*exp(-x)
            else:
                log_Z -= np.log(2.0 * np.sinh(x))
    return np.exp(log_Z)


def specific_heat_harmonic(omegas, T):
    """
    Compute specific heat per particle for harmonic oscillators.
    
    C_V = k_B * sum_k (x_k^2 * exp(x_k) / (exp(x_k) - 1)^2 )
    where x_k = hbar * omega_k / (k_B * T).
    
    In the classical limit: C_V -> 3 * k_B (Dulong-Petit law).
    """
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
    """
    Evaluate integral of f(x) * |x|^alpha * exp(-x^2) using Gauss-Hermite quadrature.
    
    Uses numpy's hermite_gauss for standard nodes and weights, then modifies
    weights for the generalized weight |x|^alpha.
    
    integral ~ sum_{i=1}^n w_i * f(x_i)
    """
    from numpy.polynomial.hermite import hermgauss
    x, w = hermgauss(n_nodes)
    if alpha != 0.0:
        w = w * np.abs(x)**alpha
    return np.sum(w * f(x))
