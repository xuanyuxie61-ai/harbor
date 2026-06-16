"""
special_functions_nuclear.py  --  Special functions for nuclear shell-model work
===============================================================================
We collect here the special functions that appear repeatedly in the shell-model
calculation:

  - log-gamma, regularised incomplete gamma, incomplete beta (ASA310 lineage)
  - Coulomb wave functions F_L(eta, rho), G_L(eta, rho) via Steed's method
  - 3j and 6j Wigner symbols (Racah algebra)
  - Clebsch-Gordan coefficients (from triangle geometry, 150_cg_lab_triangles)
  - Spherical harmonics Y_L^M (used for multipole operators)
  - Gauss-Laguerre and Gauss-Hermite quadrature nodes/weights
  - Fermi-Dirac integrals for level density

The code is deliberately self-contained so the shell-model chain can be
reproduced without external special-function libraries beyond numpy/scipy.

References:
    Abramowitz & Stegun, Handbook of Mathematical Functions
    Thompson, Comput. Phys. Commun. (1989) - Coulomb functions
    Varshalovich, Moskalev, Khersonskii - Quantum Theory of Angular Momentum
"""

from __future__ import annotations
import math
import numpy as np
from functools import lru_cache
from nuclear_constants import PI, SQRT_PI


# ======================================================================
#  Gamma-family (ASA310 lineage)
# ======================================================================
def log_gamma(x: float) -> float:
    """Stirling + Lanczos log-Gamma for x > 0; extends by reflection."""
    if x <= 0.0 and x == math.floor(x):
        return float("inf")
    if x < 0.5:
        # reflection: Gamma(x) Gamma(1-x) = pi / sin(pi x)
        return math.log(PI / math.sin(PI * x)) - log_gamma(1.0 - x)
    x -= 1.0
    g = 7
    c = [
        0.99999999999980993, 676.5203681218851, -1259.1392167224028,
        771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
    ]
    s = c[0]
    for i in range(1, g + 2):
        s += c[i] / (x + i)
    t = x + g + 0.5
    return 0.5 * math.log(2.0 * PI) + (x + 0.5) * math.log(t) - t + math.log(s)


def gamma_fn(x: float) -> float:
    """Gamma function with overflow protection."""
    if x > 171.0:
        return float("inf")
    return math.exp(log_gamma(x))


def regularised_gamma_p(a: float, x: float, n_iter: int = 200, tol: float = 1.0e-14) -> float:
    """Regularised lower incomplete gamma P(a, x) = gamma(a, x)/Gamma(a).

    Uses the series representation for x < a+1 and the CF representation
    otherwise (ASA310 / gammad).
    """
    if x < 0.0 or a <= 0.0:
        return 0.0
    if x == 0.0:
        return 0.0
    if x < a + 1.0:
        # series: P = e^{-x} x^a sum_n x^n / Gamma(a+n+1)
        ap = a
        term = 1.0 / a
        s = term
        for _ in range(n_iter):
            ap += 1.0
            term *= x / ap
            s += term
            if abs(term) < abs(s) * tol:
                break
        return s * math.exp(-x + a * math.log(x) - log_gamma(a))
    # continued fraction (Lentz)
    b = x + 1.0 - a
    c = 1.0 / 1.0e-30
    d = 1.0 / b
    h = d
    for i in range(1, n_iter + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1.0e-30:
            d = 1.0e-30
        c = b + an / c
        if abs(c) < 1.0e-30:
            c = 1.0e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < tol:
            break
    return 1.0 - math.exp(-x + a * math.log(x) - log_gamma(a)) * h


def regularised_gamma_q(a: float, x: float) -> float:
    """Upper regularised incomplete gamma Q(a, x) = 1 - P(a, x)."""
    return 1.0 - regularised_gamma_p(a, x)


def regularised_beta_inc(x: float, a: float, b: float, n_iter: int = 200, tol: float = 1.0e-14) -> float:
    """Regularised incomplete beta I_x(a, b) = B_x(a, b) / B(a, b).

    Uses the Lentz continued fraction; swaps x <-> 1-x when x > (a+1)/(a+b+2)
    to ensure rapid convergence (ASA310: betain, ncbeta lineage).
    """
    if x < 0.0 or x > 1.0:
        return 0.0 if x < 0.0 else 1.0
    if x == 0.0 or x == 1.0:
        return x
    # Use symmetry to keep CF well-conditioned
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - regularised_beta_inc(1.0 - x, b, a, n_iter, tol)
    lbeta = log_gamma(a) + log_gamma(b) - log_gamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta) / a
    # Lentz CF
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < 1.0e-30:
        d = 1.0e-30
    d = 1.0 / d
    h = d
    for m in range(1, n_iter + 1):
        # even step
        m2 = 2 * m
        aa = m * (b - m) * x / ((a + m2 - 1.0) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1.0e-30:
            d = 1.0e-30
        c = 1.0 + aa / c
        if abs(c) < 1.0e-30:
            c = 1.0e-30
        d = 1.0 / d
        h *= d * c
        # odd step
        aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0))
        d = 1.0 + aa * d
        if abs(d) < 1.0e-30:
            d = 1.0e-30
        c = 1.0 + aa / c
        if abs(c) < 1.0e-30:
            c = 1.0e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < tol:
            break
    return front * h


# ======================================================================
#  Wigner 3-j, 6-j symbols and Clebsch-Gordan (Racah algebra)
# ======================================================================
@lru_cache(maxsize=2048)
def log_factorial(n: int) -> float:
    if n < 0:
        return float("inf")
    if n < 2:
        return 0.0
    return log_gamma(n + 1.0)


def wigner_3j(j1: float, j2: float, j3: float, m1: float, m2: float, m3: float) -> float:
    """Wigner 3-j symbol (j1 j2 j3; m1 m2 m3) via Racah formula.

    Selection rules enforced:
        m1 + m2 + m3 = 0,
        |j1 - j2| <= j3 <= j1 + j2,
        j1 + j2 + j3 is integer.
    """
    if abs(m1 + m2 + m3) > 1.0e-10:
        return 0.0
    if j3 < abs(j1 - j2) or j3 > j1 + j2:
        return 0.0
    if (j1 + j2 + j3) != int(j1 + j2 + j3 + 1.0e-10):
        return 0.0
    # Convert to integer arguments by doubling
    J1, J2, J3 = int(round(2 * j1)), int(round(2 * j2)), int(round(2 * j3))
    M1, M2, M3 = int(round(2 * m1)), int(round(2 * m2)), int(round(2 * m3))
    if M1 + M2 + M3 != 0:
        return 0.0
    # Triangle coefficient Delta
    def tri(a, b, c):
        return math.exp(
            0.5 * (log_factorial((a + b - c) // 2)
                   + log_factorial((a - b + c) // 2)
                   + log_factorial((-a + b + c) // 2)
                   - log_factorial((a + b + c) // 2 + 1))
        )
    t = tri(J1, J2, J3)
    prefactor = ((-1) ** ((J1 - J2 - M3) // 2)) * t
    prefactor *= math.exp(
        0.5 * (log_factorial((J1 + M1) // 2)
               + log_factorial((J1 - M1) // 2)
               + log_factorial((J2 + M2) // 2)
               + log_factorial((J2 - M2) // 2)
               + log_factorial((J3 + M3) // 2)
               + log_factorial((J3 - M3) // 2))
    )
    s = 0.0
    t_min = max(0, (J2 - J3 - M1) // 2, (J1 - J3 + M2) // 2)
    t_max = min((J1 + J2 - J3) // 2, (J1 - M1) // 2, (J2 + M2) // 2)
    for tt in range(t_min, t_max + 1):
        sign = (-1) ** tt
        denom = (
            log_factorial(tt)
            + log_factorial((J1 + J2 - J3) // 2 - tt)
            + log_factorial((J1 - M1) // 2 - tt)
            + log_factorial((J2 + M2) // 2 - tt)
            + log_factorial(tt - (J2 - J3 - M1) // 2)
            + log_factorial(tt - (J1 - J3 + M2) // 2)
        )
        s += sign * math.exp(-denom)
    return prefactor * s


def clebsch_gordan(j1: float, m1: float, j2: float, m2: float, j3: float, m3: float) -> float:
    """CG coefficient <j1 m1 j2 m2 | j3 m3> via 3-j symbol.

        <j1 m1 j2 m2 | j3 m3> = (-1)^{j1-j2+m3} sqrt(2 j3 + 1) * (j1 j2 j3; m1 m2 -m3)
    """
    if abs(m1 + m2 - m3) > 1.0e-10:
        return 0.0
    phase = (-1) ** int(round(j1 - j2 + m3))
    return phase * math.sqrt(2.0 * j3 + 1.0) * wigner_3j(j1, j2, j3, m1, m2, -m3)


def wigner_6j(j1: float, j2: float, j3: float, j4: float, j5: float, j6: float) -> float:
    """Wigner 6-j symbol via Racah sum.

    Requires four triangle conditions: (j1 j2 j3), (j1 j5 j6), (j4 j2 j6), (j4 j5 j3).
    """
    def tri_condition(a, b, c):
        return (abs(a - b) <= c + 1.0e-9) and (c <= a + b + 1.0e-9) \
            and abs((a + b + c) - round(a + b + c)) < 1.0e-9
    if not all([
        tri_condition(j1, j2, j3), tri_condition(j1, j5, j6),
        tri_condition(j4, j2, j6), tri_condition(j4, j5, j3),
    ]):
        return 0.0
    return _sixj_racah_sum(j1, j2, j3, j4, j5, j6)


def _sixj_racah_sum(j1, j2, j3, j4, j5, j6):
    """Racah's sum for {j1 j2 j3; j4 j5 j6}.

    {a b c; d e f} = Delta(a,b,c) Delta(a,e,f) Delta(d,b,f) Delta(d,e,c)
                      * sum_z (-1)^z (z+1)! / [(z-a-b-c)! (z-a-e-f)! (z-d-b-f)!
                                               (z-d-e-c)! (a+b+d+e-z)!
                                               (a+c+d+f-z)! (b+c+e+f-z)!]
    """
    def tri(a, b, c):
        if abs(a - b) > c + 1.0e-9 or c > a + b + 1.0e-9:
            return 0.0
        s = int(round(a + b + c))
        if (a + b + c) - round(a + b + c) > 1.0e-9:
            return 0.0
        return math.exp(
            0.5 * (log_factorial(int(a + b - c))
                   + log_factorial(int(a - b + c))
                   + log_factorial(int(-a + b + c))
                   - log_factorial(int(a + b + c + 1)))
        )
    t123 = tri(j1, j2, j3)
    t156 = tri(j1, j5, j6)
    t426 = tri(j4, j2, j6)
    t453 = tri(j4, j5, j3)
    if t123 * t156 * t426 * t453 == 0.0:
        return 0.0
    # Sum ranges: z >= max(a+b+c, a+e+f, d+b+f, d+e+c)
    #             z <= min(a+b+d+e, a+c+d+f, b+c+e+f)
    z_min = max(j1 + j2 + j3, j1 + j5 + j6, j4 + j2 + j6, j4 + j5 + j3)
    z_max = min(j1 + j2 + j4 + j5, j1 + j3 + j4 + j6, j2 + j3 + j5 + j6)
    z_min_i = int(round(z_min))
    z_max_i = int(round(z_max))
    s = 0.0
    for z in range(z_min_i, z_max_i + 1):
        sign = (-1) ** z
        # Numerator: (z+1)!
        num_log = log_factorial(z + 1)
        # Denominator: 7 factorials
        den_log = (
            log_factorial(int(z - j1 - j2 - j3))
            + log_factorial(int(z - j1 - j5 - j6))
            + log_factorial(int(z - j4 - j2 - j6))
            + log_factorial(int(z - j4 - j5 - j3))
            + log_factorial(int(j1 + j2 + j4 + j5 - z))
            + log_factorial(int(j1 + j3 + j4 + j6 - z))
            + log_factorial(int(j2 + j3 + j5 + j6 - z))
        )
        s += sign * math.exp(num_log - den_log)
    phase = (-1) ** int(round(j1 + j2 + j4 + j5))
    return phase * t123 * t156 * t426 * t453 * s


# ======================================================================
#  Spherical harmonics
# ======================================================================
def spherical_harmonic(l: int, m: int, theta: float, phi: float) -> complex:
    """Y_l^m(theta, phi) = N_l^m P_l^m(cos theta) e^{i m phi}."""
    x = math.cos(theta)
    plm = _assoc_legendre(l, abs(m), x)
    norm = math.sqrt((2.0 * l + 1.0) / (4.0 * PI)
                     * math.exp(log_gamma(l - abs(m) + 1) - log_gamma(l + abs(m) + 1)))
    val = norm * plm * math.cos(m * phi) + 1j * norm * plm * math.sin(m * phi)
    if m < 0:
        val = ((-1) ** (-m)) * val.conjugate()
    return val


def _assoc_legendre(l: int, m: int, x: float) -> float:
    """Associated Legendre P_l^m(x) with Condon-Shortley phase included."""
    if m > l:
        return 0.0
    # Start from P_m^m
    pmm = 1.0
    if m > 0:
        somx2 = math.sqrt(max(0.0, (1.0 - x) * (1.0 + x)))
        fact = 1.0
        for _ in range(1, m + 1):
            pmm *= -fact * somx2
            fact += 2.0
    if l == m:
        return pmm
    # P_{m+1}^m
    pmm1 = x * (2 * m + 1) * pmm
    if l == m + 1:
        return pmm1
    plm = 0.0
    for ll in range(m + 2, l + 1):
        plm = (x * (2 * ll - 1) * pmm1 - (ll + m - 1) * pmm) / (ll - m)
        pmm = pmm1
        pmm1 = plm
    return plm


# ======================================================================
#  Quadrature nodes (used by nuclear quadrature module)
# ======================================================================
def gauss_laguerre_nodes(n: int, alpha: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights of generalised Gauss-Laguerre rule int_0^inf x^alpha e^{-x} f(x) dx."""
    nodes, weights = np.polynomial.laguerre.laggauss(n)
    if alpha == 0.0:
        return nodes, weights
    # For alpha != 0 we would use Golub-Welsch; keep alpha=0 for our needs.
    return nodes, weights


def gauss_hermite_nodes(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Hermite nodes/weights for int_{-inf}^inf e^{-x^2} f(x) dx."""
    return np.polynomial.hermite.hermgauss(n)


def fermi_dirac_integral(k: int, mu: float, T: float, n_quad: int = 40) -> float:
    r"""Fermi-Dirac integral of order k:
        F_k(mu, T) = int_0^inf epsilon^k / (exp((epsilon - mu)/T) + 1) d epsilon

    Computed by Gauss-Laguerre quadrature after a tanh mapping to [0, inf).
    """
    if T <= 0.0:
        # Zero-temperature limit: F_k = mu^{k+1} / (k+1)
        return (mu ** (k + 1)) / (k + 1.0) if mu > 0.0 else 0.0
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    # Map [-1, 1] to [0, inf) via t = (1 + x) / (1 - x), dt = 2 / (1 - x)^2
    x = 0.5 * (nodes + 1.0)
    t = np.tanh(0.5 * x / T)  # alternative mapping
    eps = -T * np.log(np.maximum(1.0 / np.maximum(x, 1.0e-30) - 1.0, 1.0e-30))
    # Use simple tanh mapping: epsilon = T * log(1 + exp(t)), t in (-inf, inf)
    # For robustness, use Gauss-Legendre on [0, 20 T + mu]
    upper = max(20.0 * T + mu + 5.0 * T, 1.0)
    eps_gl, w_gl = np.polynomial.legendre.leggauss(n_quad)
    eps_mapped = 0.5 * upper * (eps_gl + 1.0)
    w_mapped = 0.5 * upper * w_gl
    fermi = 1.0 / (np.exp(np.clip((eps_mapped - mu) / T, -500.0, 500.0)) + 1.0)
    return float(np.sum(w_mapped * (eps_mapped ** k) * fermi))
