"""
circle_quadrature.py
====================

Angular/spectral quadrature rules on the unit circle S^1, used to
integrate periodic uncertainty over the azimuthal direction of the
protoplanetary disk.

In the multi-fidelity UQ framework we encounter integrals of the form

    I[f] = (1 / 2pi) * integral_0^{2pi} f(cos theta, sin theta) d theta

where f is the forward map evaluated along an angular parameterization of
the input uncertainty ellipse.  Trigonometric interpolation on equispaced
nodes is exponentially convergent for smooth periodic integrands, making
circle quadrature the natural choice.

We provide:
  - `circle_rule(nt)`: trapezoidal rule on nt equispaced nodes (exact for
    trigonometric polynomials of degree < nt).
  - `clenshaw_curtis_circle(n)`: Clenshaw-Curtis quadrature on S^1 with
    n nodes, with weights derived from the DCT-I.
  - `fejer2_circle(n)`: Fejer-2 rule (no endpoints), exponentially convergent.
  - `circle_monomial_integral(p, q)`: exact integral of cos^p sin^q on S^1
    (used for verification).
  - `angular_integral(f, rule)`: dispatch the integration of f(theta)
    under a chosen rule.

The quadrature rules also serve as the inner loop of the multi-index
stochastic collocation operator: we integrate each fidelity level's
output over the azimuthal direction before building the GP surrogate.
"""

from __future__ import annotations

import math
from typing import Callable, List, Tuple


# ----------------------------------------------------------------------
# Exact circle monomial integrals (for verification).
#
# For integers p, q >= 0,
#   I_{p,q} = (1/2pi) int_0^{2pi} cos^p(theta) sin^q(theta) d theta
#           = 0  if p or q is odd
#           = [(p-1)!! (q-1)!!] / [(p+q)!!]   if p and q both even
# where !! is the double factorial.
# ----------------------------------------------------------------------
def double_factorial(n: int) -> int:
    """n!! for n >= -1, with the conventions (-1)!! = 0!! = 1."""
    if n < -1:
        raise ValueError("double_factorial: n must be >= -1.")
    if n <= 0:
        return 1
    out = 1
    k = n
    while k > 0:
        out *= k
        k -= 2
    return out


def circle_monomial_integral(p: int, q: int) -> float:
    """Exact value of (1/2pi) int_0^{2pi} cos^p(theta) sin^q(theta) d theta."""
    if p < 0 or q < 0:
        raise ValueError("circle_monomial_integral: p, q must be >= 0.")
    if p % 2 == 1 or q % 2 == 1:
        return 0.0
    num = double_factorial(p - 1) * double_factorial(q - 1)
    den = double_factorial(p + q)
    return float(num) / float(den)


# ----------------------------------------------------------------------
# Basic circle rule (equispaced trapezoidal, exact for deg < nt).
# ----------------------------------------------------------------------
def circle_rule(nt: int) -> Tuple[List[float], List[float]]:
    """Trapezoidal quadrature on the unit circle.

    Returns (weights w, angles t) with w_i = 1/nt, t_i = 2 pi i / nt.
    The integral is approximated as:
        I[f] ~ 2 pi * sum_{i=1}^{nt} w_i f(cos t_i, sin t_i)
    """
    if nt < 1:
        raise ValueError("circle_rule: nt must be >= 1.")
    w = [1.0 / nt] * nt
    t = [2.0 * math.pi * i / nt for i in range(nt)]
    return w, t


# ----------------------------------------------------------------------
# Clenshaw-Curtis on the circle.
# ----------------------------------------------------------------------
def clenshaw_curtis_circle(n: int) -> Tuple[List[float], List[float]]:
    """Clenshaw-Curtis quadrature on [0, 2pi] wrapped to S^1.

    Nodes:   theta_j = pi j / (n-1),  j = 0..n-1
    Weights: derived from DCT-I of the even cosine coefficients; for the
    periodic wrapping we identify theta=0 and theta=2pi, so we effectively
    drop the duplicate endpoint.
    """
    if n < 2:
        raise ValueError("clenshaw_curtis_circle: n must be >= 2.")
    # Build CC weights on [0, pi], then double to [0, 2pi] by symmetry.
    m = n - 1
    theta = [math.pi * j / m for j in range(n)]
    w = [0.0] * n
    for j in range(n):
        s = 1.0
        for k in range(1, m // 2 + 1):
            b = 2.0 if 2 * k < m else 1.0
            s -= b * math.cos(2.0 * k * theta[j]) / (4.0 * k * k - 1.0)
        w[j] = 2.0 * s / m
    # Rescale to [0, 2pi]: multiply by pi (the half-period weight).
    for j in range(n):
        w[j] *= math.pi
    return w, theta


# ----------------------------------------------------------------------
# Fejer-2 rule on the circle.
# ----------------------------------------------------------------------
def fejer2_circle(n: int) -> Tuple[List[float], List[float]]:
    """Fejer-2 quadrature on [0, 2pi).

    Nodes:   theta_j = (2j - 1) pi / (2n),  j = 1..n   (no endpoints)
    Weights: w_j = (2/n) * sum_{k=1}^{n/2} sin((2k-1) theta_j) / (2k-1)
    """
    if n < 1:
        raise ValueError("fejer2_circle: n must be >= 1.")
    theta = [(2 * j - 1) * math.pi / (2 * n) for j in range(1, n + 1)]
    w = [0.0] * n
    for j in range(n):
        s = 0.0
        for k in range(1, n // 2 + 1):
            s += math.sin((2 * k - 1) * theta[j]) / (2 * k - 1)
        w[j] = (2.0 / n) * s * 2.0  # factor 2 for [0, 2pi] wrapping
    return w, theta


# ----------------------------------------------------------------------
# Generic angular integral dispatcher.
# ----------------------------------------------------------------------
def angular_integral(
    f: Callable[[float], float],
    rule: str = "trapezoidal",
    n: int = 16,
) -> float:
    """Approximate (1/2pi) int_0^{2pi} f(theta) d theta.

    For the trapezoidal rule on the circle with w_i = 1/n, the
    normalized integral is simply sum_i w_i f(theta_i) (no further
    division by 2pi is needed).  For CC / Fejer2 we normalize by
    dividing the raw weighted sum by 2pi.
    """
    rule = rule.lower()
    if rule == "trapezoidal":
        w, t = circle_rule(n)
        total = sum(wi * f(ti) for wi, ti in zip(w, t))
        return total
    if rule in ("clenshaw-curtis", "cc"):
        w, t = clenshaw_curtis_circle(n)
    elif rule in ("fejer2", "fejer-2"):
        w, t = fejer2_circle(n)
    else:
        raise ValueError(f"angular_integral: unknown rule '{rule}'.")
    total = sum(wi * f(ti) for wi, ti in zip(w, t))
    return total / (2.0 * math.pi)


# ----------------------------------------------------------------------
# Tensor-product rule for (radius, angle) integration.
# ----------------------------------------------------------------------
def circle_tensor_radial(
    f_radial: Callable[[float], float],
    r_vals: List[float],
    w_r: List[float],
    f_angular: Callable[[float], float],
    n_theta: int = 16,
) -> float:
    """Separate (radial x circle) integral:
        int_r w_r(r) f_radial(r) dr  *  (1/2pi) int_0^{2pi} f_angular(theta) d theta
    """
    Ir = 0.0
    for i in range(len(r_vals) - 1):
        dr = r_vals[i + 1] - r_vals[i]
        Ir += 0.5 * (w_r[i] * f_radial(r_vals[i])
                     + w_r[i + 1] * f_radial(r_vals[i + 1])) * dr
    Ia = angular_integral(f_angular, "trapezoidal", n_theta)
    return Ir * Ia
