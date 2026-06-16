"""
quadrature_validator.py
=======================
Validation of the quadrature rules used in the CMB power-spectrum
pipeline, via Chebyshev-type exactness testing
(adapted from chebyshev1_exactness.m and quad_monte_carlo.m).

Two quadrature rules are tested:
  1. Gauss-Legendre on [-1, 1]  (for theta integration)
  2. Trapezoidal on [0, 2pi]    (for phi integration, exact for bandlimited)

For each rule, we integrate monomials x^n  (or  exp(i m phi)) up to
degree N and compare to the known exact value.  The "exactness degree"
is the largest n for which the relative error is below a tolerance.

Also provides:
  - Monomial integral on the unit cube (cube01_monomial_integral.m)
  - Monte-Carlo quadrature (quad_monte_carlo.m)
"""

from __future__ import annotations
import math
import random
from typing import List, Tuple, Callable, Dict


# ---------------------------------------------------------------------------
# Gauss-Legendre nodes and weights
# ---------------------------------------------------------------------------
def gauss_legendre_nodes_weights(n: int) -> Tuple[List[float], List[float]]:
    """
    Compute n Gauss-Legendre nodes and weights on [-1, 1]
    by Newton iteration on Legendre polynomials.
    """
    nodes = []
    weights = []
    for i in range(n):
        # Initial guess (Chebyshev-like)
        x = math.cos(math.pi * (i + 0.75) / (n + 0.5))
        for _ in range(100):
            p0 = 1.0
            p1 = x
            for j in range(2, n + 1):
                p2 = ((2.0 * j - 1.0) * x * p1 - (j - 1.0) * p0) / j
                p0 = p1
                p1 = p2
            # Derivative  P_n'(x) = n (x P_n - P_{n-1}) / (x^2 - 1)
            dp = n * (x * p1 - p0) / (x * x - 1.0) if abs(x * x - 1.0) > 1e-30 else 0.0
            if abs(dp) < 1e-30:
                break
            dx = -p1 / dp
            x += dx
            if abs(dx) < 1e-15:
                break
        nodes.append(x)
        weights.append(2.0 / ((1.0 - x * x) * dp * dp) if abs(dp) > 1e-30 else 0.0)
    return nodes, weights


# ---------------------------------------------------------------------------
# Monomial exact value on [-1, 1] with weight (1 - x^2)^{-1/2}
# (Chebyshev type 1)
# ---------------------------------------------------------------------------
def chebyshev1_integral(expon: int) -> float:
    """
    Exact integral  int_{-1}^{1} x^n / sqrt(1 - x^2) dx.
    = 0 if n is odd,  pi * (n-1)!! / n!!  if n is even.
    """
    if expon % 2 == 1:
        return 0.0
    top = 1.0
    bot = 1.0
    for i in range(2, expon + 1, 2):
        top *= (i - 1)
        bot *= i
    return math.pi * top / bot


# ---------------------------------------------------------------------------
# Chebyshev exactness test
# ---------------------------------------------------------------------------
def chebyshev1_exactness(nodes: List[float], weights: List[float],
                           degree_max: int) -> List[Tuple[int, float, float]]:
    """
    For each degree 0..degree_max, compute the relative error
    of the quadrature on  int x^n / sqrt(1-x^2) dx.
    Returns list of (degree, exact, relative_error).
    """
    results = []
    for n in range(degree_max + 1):
        exact = chebyshev1_integral(n)
        # Quadrature approximation
        quad = sum(weights[i] * (nodes[i] ** n) for i in range(len(nodes)))
        if abs(exact) < 1e-30:
            err = abs(quad - exact)
        else:
            err = abs((quad - exact) / exact)
        results.append((n, exact, err))
    return results


# ---------------------------------------------------------------------------
# Trapezoidal rule for periodic functions
# ---------------------------------------------------------------------------
def trapezoidal_periodic(f: Callable[[float], complex], n: int,
                            period: float = 2.0 * math.pi) -> complex:
    """
    Trapezoidal rule on [0, period] with n equally spaced points.
    For bandlimited functions (max frequency < n/2) this is exact.
    """
    dx = period / n
    s = complex(0.0, 0.0)
    for k in range(n):
        x = k * dx
        s += f(x)
    return s * dx


# ---------------------------------------------------------------------------
# Monomial exactness on unit cube (cube01)
# ---------------------------------------------------------------------------
def cube01_monomial_integral(e: List[int]) -> float:
    """
    Exact integral of x^e[0] y^e[1] z^e[2] over [0,1]^3.
    = prod_{i} 1/(e_i + 1).
    """
    if any(ei < 0 for ei in e):
        raise ValueError("All exponents must be nonnegative.")
    result = 1.0
    for ei in e:
        result /= (ei + 1)
    return result


def cube01_monomial_mc(e: List[int], n_samples: int, seed: int = 42) -> float:
    """
    Monte-Carlo estimate of the same integral.
    """
    rng = random.Random(seed)
    s = 0.0
    for _ in range(n_samples):
        pt = [rng.random() for _ in range(3)]
        v = 1.0
        for i, ei in enumerate(e):
            v *= pt[i] ** ei
        s += v
    return s / n_samples


# ---------------------------------------------------------------------------
# Monomial value (monomial_value.m)
# ---------------------------------------------------------------------------
def monomial_value(m: int, e: List[int], x: List[float]) -> float:
    """
    Evaluate  prod_{i=1}^m x_i^{e_i}  at a single point.
    """
    v = 1.0
    for i in range(m):
        if e[i] != 0:
            v *= x[i] ** e[i]
    return v


# ---------------------------------------------------------------------------
# 1D Monte-Carlo quadrature (quad_monte_carlo.m)
# ---------------------------------------------------------------------------
def quad_monte_carlo(f: Callable[[float], float], a: float, b: float,
                       n: int, seed: int = 42) -> float:
    """
    Monte-Carlo estimate of  int_a^b f(x) dx.
    q = (b - a) * mean(f(x_i))   with x_i ~ U(a, b).
    """
    rng = random.Random(seed)
    s = 0.0
    for _ in range(n):
        u = rng.random()
        x = a * (1.0 - u) + b * u
        s += f(x)
    return (b - a) * s / n


# ---------------------------------------------------------------------------
# Validate spherical harmonic quadrature
# ---------------------------------------------------------------------------
def validate_spherical_quadrature(l_max: int, n_theta: int, n_phi: int
                                    ) -> List[Tuple[int, float]]:
    """
    Check that Gauss-Legendre x trapezoidal rule exactly integrates
    |Y_{lm}|^2 over the sphere for l <= l_max.

    The exact value is 1 (normalisation).
    """
    # Gauss-Legendre nodes on [-1, 1]  -> cos(theta)
    gl_nodes, gl_weights = gauss_legendre_nodes_weights(n_theta)
    results = []
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            # |Y_{lm}(theta, phi)|^2  =  |N_{lm} P_l^|m|(cos theta)|^2 * {cos^2/sin^2}(m phi)
            # Integral over phi of cos^2(m phi) = pi, sin^2 = pi for m != 0; = 2pi for m=0
            phi_factor = 2.0 * math.pi if m == 0 else math.pi
            # Integral over theta
            theta_integral = 0.0
            for i in range(n_theta):
                ct = gl_nodes[i]
                st = math.sqrt(max(0.0, 1.0 - ct * ct))
                P = _assoc_legendre(l, abs(m), ct)
                N = _Y_norm(l, m)
                theta_integral += gl_weights[i] * (N * P) ** 2
            value = theta_integral * phi_factor
            error = abs(value - 1.0)
            results.append(((l, m), error))
    return results


def _assoc_legendre(l: int, m: int, x: float) -> float:
    if m < 0 or m > l:
        return 0.0
    pmm = 1.0
    if m > 0:
        somx2 = math.sqrt(max(0.0, (1.0 - x) * (1.0 + x)))
        fact = 1.0
        for i in range(1, m + 1):
            pmm *= -fact * somx2
            fact += 2.0
    if l == m:
        return pmm
    pmm1 = x * (2.0 * m + 1.0) * pmm
    if l == m + 1:
        return pmm1
    pll = 0.0
    for ll in range(m + 2, l + 1):
        pll = ((2.0 * ll - 1.0) * x * pmm1 - (ll + m - 1.0) * pmm) / (ll - m)
        pmm = pmm1
        pmm1 = pll
    return pll


def _Y_norm(l: int, m: int) -> float:
    m_abs = abs(m)
    num = 1.0
    den = 1.0
    for i in range(1, l - m_abs + 1):
        den *= i
    for i in range(1, l + m_abs + 1):
        num *= i
    return math.sqrt((2.0 * l + 1.0) / (4.0 * math.pi) * den / num)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Test 1: Gauss-Legendre nodes
    x, w = gauss_legendre_nodes_weights(5)
    print("5-point Gauss-Legendre nodes:", [f"{v:.6f}" for v in x])
    print("Weights:", [f"{v:.6f}" for v in w])

    # Test 2: Chebyshev exactness
    x, w = gauss_legendre_nodes_weights(8)
    res = chebyshev1_exactness(x, w, 20)
    print("\nChebyshev exactness:")
    for n, exact, err in res[:15]:
        print(f"  degree {n:2d}: exact = {exact:+.6e}, err = {err:.4e}")

    # Test 3: MC quadrature
    q = quad_monte_carlo(lambda t: t * t, 0.0, 1.0, 10000)
    print(f"\nMC integral of x^2 on [0,1] = {q:.6f} (exact = 0.333333)")
