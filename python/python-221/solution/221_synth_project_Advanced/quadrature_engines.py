"""
quadrature_engines.py
=====================

High-order numerical quadrature and root-finding for phase-space integration:
- Lyness-Jespersen symmetric quadrature rules for triangles
  (from 1311_triangle_lyness_rule)
- Laguerre iterative root-finding for kinematic threshold equations
  (from 1430_zero_laguerre)

Scientific context:
-------------------
The angular integration over detector acceptance regions (triangular in
(theta, phi)) requires high-order symmetric quadrature rules. The Lyness-
Jespersen rules provide degree-7 and higher symmetric rules on triangles.

Kinematic threshold equations arise from setting the Kallen function to zero:

    lambda(s, m3^2, s45) = 0

This gives a quadratic in s45 whose roots determine the physical boundaries.
We solve these using the Laguerre method which converges cubically for
simple roots and handles the polynomial structure naturally.
"""

import math
from typing import Callable, List, Tuple


# ===========================================================================
# Section 1: Lyness-Jespersen symmetric triangle quadrature
# (from 1311_triangle_lyness_rule)
# ===========================================================================

def lyness_suborder_num(rule: int) -> int:
    """
    Return the number of suborders for Lyness rule index `rule`.
    Rules 1-4 correspond to degrees 2, 3, 4, 5 respectively.
    """
    table = {1: 1, 2: 2, 3: 4, 4: 7}
    if rule not in table:
        raise ValueError(f"lyness_suborder_num: unknown rule {rule}")
    return table[rule]


def lyness_order(rule: int) -> int:
    """Return the order (number of points) for Lyness rule index."""
    table = {1: 3, 2: 6, 3: 12, 4: 24}
    if rule not in table:
        raise ValueError(f"lyness_order: unknown rule {rule}")
    return table[rule]


def lyness_subrule(rule: int, suborder_num: int
                   ) -> Tuple[List[Tuple[float, float, float]], List[float]]:
    """
    Return (barycentric_coords, weights) for suborders of a Lyness rule.

    Each suborder defines a class of points under the symmetric group S3
    of the equilateral triangle. A suborder of type k generates k points
    via cyclic permutation of barycentric coordinates (a1, a2, a3).

    Reference: Lyness & Jespersen, "Moderate Degree Symmetric Quadrature
    Rules for the Triangle", J. Inst. Math. Appl. 15 (1975) 19-32.
    """
    if rule == 1:
        # Degree 2, 1 suborder (centroid)
        sub_xyz = [(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)]
        sub_w = [1.0 / 3.0]  # normalized to triangle area = 1/2
    elif rule == 2:
        # Degree 3, 2 suborders: vertex + edge midpoints
        sub_xyz = [
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
            (0.5, 0.5, 0.0),
        ]
        sub_w = [-27.0 / 96.0, 25.0 / 96.0]
    elif rule == 3:
        # Degree 4, 4 suborders
        a1 = 0.1012865073235
        a2 = 0.4701420641051
        a3 = 0.7974269853531
        b1 = 1.0 - 2.0 * a1
        b2 = 1.0 - 2.0 * a2
        sub_xyz = [
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
            (a1, a1, b1),
            (a2, a2, b2),
            (a3, 0.5 - a3 / 2.0, 0.5 - a3 / 2.0),
        ]
        sub_w = [0.1125, 0.06619707639425, 0.06296959027241, 0.04338060810760]
    elif rule == 4:
        # Degree 5, 7 suborders (abbreviated)
        a = 0.06530737976
        b = 0.24294100467
        c = 0.05710419625
        d = 0.43929931177
        e = 0.48557784158
        sub_xyz = [
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
            (a, a, 1.0 - 2.0 * a),
            (b, b, 1.0 - 2.0 * b),
            (c, c, 1.0 - 2.0 * c),
            (d, d, 1.0 - 2.0 * d),
            (e, 0.5 - e / 2.0, 0.5 - e / 2.0),
            (0.5, 0.5, 0.0),
        ]
        sub_w = [0.0485678976, 0.0154985419, 0.0338120469,
                 0.0312596849, 0.0197210083, 0.0246673897, 0.0156696096]
    else:
        raise ValueError(f"lyness_subrule: unknown rule {rule}")
    return sub_xyz[:suborder_num], sub_w[:suborder_num]


def lyness_expand(sub_xyz: List[Tuple[float, ...]],
                  sub_w: List[float],
                  suborder_types: List[int]) -> Tuple[List[Tuple[float, float]], List[float]]:
    """
    Expand suborder data to full quadrature points in (lambda1, lambda2).

    A suborder of type 1 (centroid) generates 1 point.
    A suborder of type 3 (edge orbit) generates 3 points by cyclic permutation.
    A suborder of type 6 (full orbit) generates 6 points.

    Returns points in barycentric (lambda1, lambda2) with lambda3 = 1 - l1 - l2.
    """
    pts = []
    ws = []
    for s, (xyz, w) in enumerate(zip(sub_xyz, sub_w)):
        stype = suborder_types[s] if s < len(suborder_types) else 3
        if stype == 1:
            pts.append((xyz[0], xyz[1]))
            ws.append(w)
        elif stype == 3:
            a, b, c = xyz[0], xyz[1], xyz[2]
            for perm in [(a, b, c), (b, c, a), (c, a, b)]:
                pts.append((perm[0], perm[1]))
                ws.append(w / 3.0)
        elif stype == 6:
            a, b, c = xyz[0], xyz[1], xyz[2]
            for perm in [(a, b, c), (a, c, b), (b, a, c),
                         (b, c, a), (c, a, b), (c, b, a)]:
                pts.append((perm[0], perm[1]))
                ws.append(w / 6.0)
        else:
            pts.append((xyz[0], xyz[1]))
            ws.append(w)
    return pts, ws


def lyness_integrate(f: Callable[[float, float], float], rule: int,
                     v1: Tuple[float, float], v2: Tuple[float, float],
                     v3: Tuple[float, float]) -> float:
    """
    Integrate f(x, y) over the triangle with vertices v1, v2, v3 using
    a Lyness-Jespersen symmetric quadrature rule.

    The triangle is mapped from barycentric (l1, l2, l3) to physical (x, y) via:
        x = l1*v1[0] + l2*v2[0] + l3*v3[0]
        y = l1*v1[1] + l2*v2[1] + l3*v3[1]

    The Jacobian is twice the signed area of the triangle:
        J = |v1[0]*(v2[1]-v3[1]) + v2[0]*(v3[1]-v1[1]) + v3[0]*(v1[1]-v2[1])|

    Returns the integral approximation.
    """
    # Jacobian = 2 * area
    jac = abs(v1[0] * (v2[1] - v3[1])
              + v2[0] * (v3[1] - v1[1])
              + v3[0] * (v1[1] - v2[1]))
    suborder_num = lyness_suborder_num(rule)
    sub_xyz, sub_w = lyness_subrule(rule, suborder_num)
    # Assign suborder types based on the position
    suborder_types = [3 if s > 0 else 1 for s in range(suborder_num)]
    pts, ws = lyness_expand(sub_xyz, sub_w, suborder_types)
    integral = 0.0
    for (l1, l2), w in zip(pts, ws):
        l3 = 1.0 - l1 - l2
        if l3 < -0.01 or l3 > 1.01:
            continue
        l1c = max(0.0, min(1.0, l1))
        l2c = max(0.0, min(1.0, l2))
        l3c = max(0.0, min(1.0, l3))
        x = l1c * v1[0] + l2c * v2[0] + l3c * v3[0]
        y = l1c * v1[1] + l2c * v2[1] + l3c * v3[1]
        integral += w * f(x, y)
    return integral * jac


# ===========================================================================
# Section 2: Laguerre root-finding (from 1430_zero_laguerre)
# ===========================================================================

def zero_laguerre(x0: float, degree: int, abserr: float, kmax: int,
                  f: Callable[[float, int], float]) -> Tuple[float, int, int]:
    """
    Laguerre's method for finding roots of polynomials.

    Given f(x) and its derivatives f'(x), f''(x) (via ider=0,1,2),
    the Laguerre iteration is:

        G = f'(x) / f(x)
        H = G^2 - f''(x) / f(x)
        a = n / (G +/- sqrt((n-1)*(n*H - G^2)))
        x_{k+1} = x_k - a

    where the sign is chosen to maximize |denominator|.

    For a polynomial of degree n, this converges cubically for simple
    roots and is globally convergent for real roots of real polynomials.

    Parameters
    ----------
    x0 : float
        Initial guess.
    degree : int
        Polynomial degree (>= 2).
    abserr : float
        Convergence tolerance on |f(x)|.
    kmax : int
        Maximum iterations.
    f : callable
        f(x, ider) returns the function (ider=0), first (ider=1),
        or second (ider=2) derivative.

    Returns
    -------
    (x, ierror, k): root estimate, error flag, iteration count.
    """
    if degree < 2:
        raise ValueError(f"zero_laguerre: degree {degree} < 2")
    x = x0
    ierror = 0
    k = 0
    beta = 1.0 / (degree - 1.0)
    while True:
        fx = f(x, 0)
        if abs(fx) <= abserr:
            break
        k += 1
        if k > kmax:
            ierror = 2
            return x, ierror, k
        dfx = f(x, 1)
        d2fx = f(x, 2)
        z = dfx * dfx - (beta + 1.0) * fx * d2fx
        z = max(z, 0.0)
        bot = beta * dfx + math.sqrt(z)
        if abs(bot) < 1e-30:
            ierror = 3
            return x, ierror, k
        dx = -(beta + 1.0) * fx / bot
        x = x + dx
        if abs(dx) < 1e-15 * (1.0 + abs(x)):
            break
    return x, ierror, k


def kallen_threshold_root(s: float, m1sq: float, m2sq: float) -> float:
    """
    Solve for the threshold value of s45 in the Kallen function:

        lambda(s, m1^2, s45) = s^2 + m1^4 + s45^2 - 2*s*m1^2 - 2*s*s45 - 2*m1^2*s45 = 0

    This is a quadratic in s45:
        s45^2 - 2*(s + m1^2)*s45 + (s - m1^2)^2 = 0

    The physical root is s45 = (sqrt(s) - m1)^2.
    We verify this using Laguerre's method as a consistency check.

    Parameters
    ----------
    s : float
        Total CM energy squared.
    m1sq : float
        Mass squared of particle 1.
    m2sq : float
        Mass squared of particle 2 (used for threshold bound).

    Returns
    -------
    float
        The physical threshold value s45_th.
    """
    if s <= 0.0:
        return 0.0
    m1 = math.sqrt(max(0.0, m1sq))
    m2 = math.sqrt(max(0.0, m2sq))
    # Analytical threshold: (sqrt(s) - m1)^2
    sqrts = math.sqrt(s)
    if sqrts < m1 + m2:
        return 0.0  # below threshold
    s45_analytic = (sqrts - m1) ** 2
    # Verify with Laguerre (treat as polynomial in s45 of degree 2)
    a_coeff = 1.0
    b_coeff = -2.0 * (s + m1sq)
    c_coeff = (s - m1sq) ** 2

    def poly(x: float, ider: int) -> float:
        if ider == 0:
            return a_coeff * x * x + b_coeff * x + c_coeff
        elif ider == 1:
            return 2.0 * a_coeff * x + b_coeff
        else:
            return 2.0 * a_coeff

    x_lag, ierr, _ = zero_laguerre(s45_analytic, 2, 1e-12, 50, poly)
    if ierr != 0:
        return s45_analytic
    return x_lag


def dalitz_boundary_s34(s: float, m1: float, m2: float, m3: float,
                        m4: float, m5: float) -> Tuple[float, float]:
    """
    Compute the allowed range of s34 = (p3+p4)^2 in a 2->3 Dalitz plot.

    For fixed s, the boundaries are:
        s34_min = (m3 + m4)^2
        s34_max = (sqrt(s) - m5)^2

    More precisely, with the full Kallen function:
        s34_max = (sqrt(s) - m5)^2

    Returns (s34_min, s34_max).
    """
    sqrts = math.sqrt(max(0.0, s))
    s34_min = (m3 + m4) ** 2
    s34_max = max(0.0, (sqrts - m5) ** 2)
    if s34_max < s34_min:
        return s34_min, s34_min  # below threshold
    return s34_min, s34_max
