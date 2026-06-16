"""
quadrature_nuclear.py  --  Nuclear quadrature rules for matrix elements
====================================================================
Fused seeds:
    933_pyramid_integrals  -- monomial integration on simplex/pyramid domains
    184_circle_segment     -- circular-arc geometry for angular integrals
    055_asa310             -- incomplete gamma / beta function quadratures

Computes integrals of the form
    I_{L M}^{f} = int_{R^3} f(r) r^L Y_L^M(theta, phi) d^3 r
that arise in the evaluation of electromagnetic transition operators and
multipole moments.

Three complementary strategies are implemented:
    1. Radial Gauss-Laguerre quadrature for int_0^inf r^n e^{-alpha r} f(r) dr
       (captures the exponential tail of bound-state wavefunctions).
    2. Angular Gauss-Legendre on the sphere (theta integral) combined with
       trapezoidal rule on the circle (phi integral).
    3. Circle-segment rule for surface integrals over a spherical cap
       (relevant when computing partial decay widths into a solid angle).

All routines return (value, error_estimate) pairs with error estimates
from embedded-rule pairs or Richardson extrapolation.

References:
    Golub & Welsch, Math. Comp. 23 (1969) 221
    Press et al., Numerical Recipes, Ch. 4
    Abramowitz & Stegun, Ch. 25
"""

from __future__ import annotations
import math
import numpy as np
from nuclear_constants import PI, R_EPSILON


# ======================================================================
#  Radial Gauss-Laguerre quadrature
# ======================================================================
def gauss_laguerre_radial(f_values: callable, alpha: float, n_quad: int, r_max: float = 30.0) -> tuple[float, float]:
    r"""Quadrature for int_0^inf r^2 e^{-alpha r} f(r) dr.

    Uses n_quad-point Gauss-Laguerre; the weight function is r^2 e^{-alpha r},
    so we factor out e^{-alpha r} and integrate r^2 f(r) with the standard
    Laguerre rule (which has weight e^{-x}).

    Scale substitution: x = alpha r, dx = alpha dr =>
        int_0^inf r^2 e^{-alpha r} f(r) dr
        = alpha^{-3} int_0^inf x^2 e^{-x} f(x/alpha) dx

    Returns (integral, error_estimate). The error estimate is based on
    the difference between n_quad and n_quad-1 rules.
    """
    nodes1, weights1 = np.polynomial.laguerre.laggauss(n_quad)
    nodes2, weights2 = np.polynomial.laguerre.laggauss(max(4, n_quad - 2))
    # Full rule
    r1 = nodes1 / alpha
    v1 = weights1 * (r1 ** 2) * f_values(r1)
    I1 = float(np.sum(v1)) / (alpha ** 0)  # already accounted
    # Note: standard Laguerre rule gives int_0^inf e^{-x} g(x) dx = sum w_i g(x_i)
    # We want int_0^inf r^2 e^{-alpha r} f(r) dr = alpha^{-3} sum w_i (x_i^2) f(x_i/alpha)
    I1 = float(np.sum(weights1 * (nodes1 ** 2) * f_values(nodes1 / alpha))) / (alpha ** 3)
    I2 = float(np.sum(weights2 * (nodes2 ** 2) * f_values(nodes2 / alpha))) / (alpha ** 3)
    err = abs(I1 - I2)
    return I1, err


# ======================================================================
#  Angular quadrature on the sphere
# ======================================================================
def spherical_quadrature(f_theta_phi: callable, l_quad: int = 20) -> tuple[float, float]:
    r"""Integrate f(theta, phi) over the unit sphere:
        I = int_0^pi d theta sin(theta) int_0^{2 pi} d phi  f(theta, phi)

    Uses Gauss-Legendre in cos(theta) and trapezoidal in phi.
    Returns (value, error_estimate).
    """
    nodes_x, weights_x = np.polynomial.legendre.leggauss(l_quad)
    # theta = arccos(x); sin(theta) dtheta = -dx, so int f sin(theta) dtheta = int f dx
    n_phi = max(8, 2 * l_quad)
    phi_nodes = np.linspace(0.0, 2.0 * PI, n_phi, endpoint=False)
    dphi = 2.0 * PI / n_phi
    I = 0.0
    for i, x in enumerate(nodes_x):
        theta = math.acos(max(min(x, 1.0), -1.0))
        s = 0.0
        for phi in phi_nodes:
            s += f_theta_phi(theta, phi)
        I += weights_x[i] * s * dphi
    # Error estimate: halve the phi grid and compare
    n_phi2 = n_phi // 2
    if n_phi2 < 4:
        return I, abs(I) * 1.0e-10
    dphi2 = 2.0 * PI / n_phi2
    phi_nodes2 = np.linspace(0.0, 2.0 * PI, n_phi2, endpoint=False)
    I2 = 0.0
    for i, x in enumerate(nodes_x):
        theta = math.acos(max(min(x, 1.0), -1.0))
        s = 0.0
        for phi in phi_nodes2:
            s += f_theta_phi(theta, phi)
        I2 += weights_x[i] * s * dphi2
    return I, abs(I - I2)


# ======================================================================
#  Circle-segment rule (184_circle_segment seed)
# ======================================================================
def circle_segment_area(R: float, h: float) -> float:
    """Area of a circular segment of radius R and height h.

    A = R^2 arccos((R - h)/R) - (R - h) sqrt(2 R h - h^2)
    """
    if h <= 0.0 or R <= 0.0:
        return 0.0
    if h >= 2.0 * R:
        return PI * R * R
    c = R - h
    return R * R * math.acos(c / R) - c * math.sqrt(max(2.0 * R * h - h * h, 0.0))


def circle_segment_centroid_y(R: float, h: float) -> float:
    """y-coordinate of the centroid of a circular segment measured from center."""
    A = circle_segment_area(R, h)
    if A < R_EPSILON:
        return 0.0
    c = R - h
    # y_bar = (1/A) * int y dA = -(2/3) (2 R h - h^2)^{3/2} / (pi R^2) * (R^2 / A)
    base = max(2.0 * R * h - h * h, 0.0)
    num = -(2.0 / 3.0) * (base ** 1.5)
    return num / A


def spherical_cap_integral(f_theta: callable, theta_max: float, l_quad: int = 20) -> tuple[float, float]:
    r"""Integrate f(theta) over a spherical cap 0 <= theta <= theta_max:
        I = 2 pi int_0^{theta_max} f(theta) sin(theta) d theta

    Gauss-Legendre in cos(theta) on [cos(theta_max), 1].
    """
    a = math.cos(theta_max)
    b = 1.0
    nodes_x, weights_x = np.polynomial.legendre.leggauss(l_quad)
    # Map [-1, 1] to [a, b]
    nodes = 0.5 * (b - a) * nodes_x + 0.5 * (a + b)
    weights = 0.5 * (b - a) * weights_x
    I = 0.0
    for x, w in zip(nodes, weights):
        theta = math.acos(max(min(x, 1.0), -1.0))
        sin_theta = math.sqrt(max(1.0 - x * x, 0.0))
        I += w * f_theta(theta) * sin_theta
    I *= 2.0 * PI
    # Error estimate from halving the rule
    l_quad2 = max(4, l_quad // 2)
    nodes2, weights2 = np.polynomial.legendre.leggauss(l_quad2)
    nodes_m = 0.5 * (b - a) * nodes2 + 0.5 * (a + b)
    weights_m = 0.5 * (b - a) * weights2
    I2 = 0.0
    for x, w in zip(nodes_m, weights_m):
        theta = math.acos(max(min(x, 1.0), -1.0))
        sin_theta = math.sqrt(max(1.0 - x * x, 0.0))
        I2 += w * f_theta(theta) * sin_theta
    I2 *= 2.0 * PI
    return I, abs(I - I2)


# ======================================================================
#  Monomial integrals over a nuclear pyramid (933_pyramid_integrals seed)
# ======================================================================
def pyramid_monomial_integral(p: int, q: int, r_exp: int) -> float:
    r"""Monomial integral over the standard pyramid
        P = { (x, y, z) : 0 <= z <= 1, |x| <= 1 - z, |y| <= 1 - z }:

        int_P x^p y^q z^r dV

    Exact formula (by iterated integration):
        If p or q is odd, integral is 0 by symmetry.
        Otherwise:
            I = (2^{p+1} / (p+1)) * (2^{q+1} / (q+1)) * int_0^1 z^r (1-z)^{p+q+2} dz
              = 2^{p+q+2} / ((p+1)(q+1)) * B(r+1, p+q+3)
              = 2^{p+q+2} / ((p+1)(q+1)) * Gamma(r+1) Gamma(p+q+3) / Gamma(r+p+q+4)
    """
    if p % 2 != 0 or q % 2 != 0:
        return 0.0
    from special_functions_nuclear import log_gamma
    log_num = (p + q + 2) * math.log(2.0) + log_gamma(r_exp + 1.0) + log_gamma(p + q + 3.0)
    log_den = math.log(p + 1.0) + math.log(q + 1.0) + log_gamma(r_exp + p + q + 4.0)
    return math.exp(log_num - log_den)


# ======================================================================
#  Combined radial + angular quadrature for transition matrix elements
# ======================================================================
def transition_matrix_element_3d(
    f_radial: callable, L: int, alpha: float = 0.5, n_quad: int = 30,
) -> tuple[float, float]:
    r"""Compute the 3-D integral
        I = int_{R^3} f_radial(r) r^L Y_L^0(theta, phi) d^3 r

    for a purely radial f. By the addition theorem the angular integral is
        int Y_L^0 d Omega = sqrt(4 pi / (2 L + 1)) * delta_{L,0}
    so for L=0:
        I = sqrt(4 pi) int_0^inf f_radial(r) r^2 dr

    For L > 0 the angular integral of Y_L^0 is zero by orthogonality;
    the user should supply an angular-dependent integrand in that case.

    Returns (value, error_estimate).
    """
    if L == 0:
        val, err = gauss_laguerre_radial(f_radial, alpha, n_quad)
        return math.sqrt(4.0 * PI) * val, math.sqrt(4.0 * PI) * err
    return 0.0, 0.0


# ======================================================================
#  Richardson-extrapolated trapezoidal rule on the radial mesh
# ======================================================================
def richardson_trapezoidal(f_values: np.ndarray, h: float, order: int = 4) -> float:
    """Trapezoidal rule with Richardson extrapolation to O(h^{2 order}).

    Given f on a uniform mesh with spacing h, compute the composite
    trapezoidal rule T(h), T(2h), ..., T(2^{order-1} h) and extrapolate.
    """
    n = len(f_values)
    if n < 3:
        return 0.0
    T = np.zeros(order)
    for k in range(order):
        step = 2 ** k
        idx = np.arange(0, n, step)
        # Trapezoidal on sub-mesh
        T[k] = h * step * (0.5 * f_values[idx[0]] + 0.5 * f_values[idx[-1]] + np.sum(f_values[idx[1:-1]]))
    # Richardson extrapolation (Romberg style)
    for j in range(1, order):
        for k in range(order - j):
            T[k] = (4 ** j * T[k + 1] - T[k]) / (4 ** j - 1)
    return float(T[0])
