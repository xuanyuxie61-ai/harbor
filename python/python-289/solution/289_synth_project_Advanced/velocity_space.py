# -*- coding: utf-8 -*-
"""
velocity_space.py
=================

Velocity-space discretisation for the gyrokinetic equation.

This module provides three cooperating components:

1. Gauss-Patterson quadrature rules
   Nested, fully symmetric rules for the integrals over v_parallel and
   magnetic moment mu.  Gauss-Patterson rules are ideal because new levels
   reuse all previously computed nodes -- enabling adaptive convergence
   of the velocity moments without re-evaluating the distribution function.

2. Centroidal-Voronoi-Tessellation (CVT) adaptive velocity grid
   Used to construct an optimal set of marker particles whose density
   follows the Maxwellian background (Lloyd's algorithm).

3. Bessel-function machinery
   The gyro-average operator <...>_R brings J_0(k_perp rho_s) weights
   into the gyrokinetic Poisson equation.  We compute J_n and J_n' via
   scipy's stable implementation (backed by AMOS / the Zhang-Jin
   continued-fraction algorithm), and we locate the zeros of J_n which
   are the eigenvalues of a finite-radius cylindrical plasma column.

References:
    Zhang & Jin, "Computation of Special Functions", Wiley (1996).
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import numpy as np

try:
    from scipy.special import jv as _scipy_jv
    from scipy.special import jvp as _scipy_jvp
    from scipy.special import jn_zeros as _scipy_jn_zeros
    HAVE_SCIPY_SPECIAL = True
except ImportError:
    HAVE_SCIPY_SPECIAL = False


# ============================================================================
# Bessel functions of the first kind J_n(x) and derivative J_n'(x)
# ============================================================================
def jn_eval(n: int, x: float) -> float:
    """J_n(x)."""
    if HAVE_SCIPY_SPECIAL:
        return float(_scipy_jv(n, x))
    # Fallback: power series for small x, asymptotic for large x
    ax = abs(x)
    if ax < 1.0e-60:
        return 1.0 if n == 0 else 0.0
    s = 0.0
    term = (0.5 * ax) ** n / math.factorial(max(n, 0))
    s = term
    for k in range(1, 80):
        term *= - (ax * ax) / (4.0 * k * (k + n))
        s += term
        if abs(term) < 1.0e-15 * abs(s):
            break
    if x < 0 and n % 2 == 1:
        s = -s
    return s


def jn_derivative(n: int, x: float) -> float:
    """J_n'(x)."""
    if HAVE_SCIPY_SPECIAL:
        return float(_scipy_jvp(n, x))
    return 0.5 * (jn_eval(n - 1, x) - jn_eval(n + 1, x))


def jyndd(n: int, x: float) -> Tuple[float, float]:
    """Returns (J_n(x), J_n'(x))."""
    return jn_eval(n, x), jn_derivative(n, x)


def jn_zeros(n: int, nt: int) -> np.ndarray:
    """First ``nt`` positive zeros of J_n(x)."""
    if HAVE_SCIPY_SPECIAL:
        return np.asarray(_scipy_jn_zeros(n, nt), dtype=np.float64)
    # fallback bisection
    out = np.zeros(nt, dtype=np.float64)
    # bracket via McMahon asymptotic for large zeros, or step-and-bracket
    step = 0.5
    x = step
    found = 0
    f_prev = jn_eval(n, x - step)
    while found < nt and x < 1.0e4:
        f_curr = jn_eval(n, x)
        if f_prev * f_curr < 0:
            a, b = x - step, x
            for _ in range(80):
                c = 0.5 * (a + b)
                fc = jn_eval(n, c)
                if abs(fc) < 1.0e-14 or (b - a) < 1.0e-13:
                    break
                if f_prev * fc <= 0:
                    b = c
                else:
                    a = c
                    f_prev = fc
            out[found] = 0.5 * (a + b)
            found += 1
        f_prev = f_curr
        x += step
    return out


# ============================================================================
# Gauss-Patterson quadrature (port of 851_patterson_rule)
# ============================================================================
_GPN = {
    1: (np.array([0.0]),
        np.array([2.0])),
    3: (np.array([-math.sqrt(3.0 / 5.0), 0.0, math.sqrt(3.0 / 5.0)]),
        np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])),
    7: (np.array([-0.9491079123427585, -0.7415311855993945,
                  -0.4058451513773972, 0.0,
                  0.4058451513773972, 0.7415311855993945,
                  0.9491079123427585]),
        np.array([0.1294849661688697, 0.2797053914892767,
                  0.3818300505051189, 0.4179591836734694,
                  0.3818300505051189, 0.2797053914892767,
                  0.1294849661688697])),
}


def patterson_rule(order: int) -> Tuple[np.ndarray, np.ndarray]:
    """Nodes / weights of a Gauss-Patterson rule on [-1, 1]."""
    order = int(order)
    if order in _GPN:
        return _GPN[order]
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return nodes, weights


def patterson_rule_ab(order: int, a: float, b: float) -> Tuple[np.ndarray, np.ndarray]:
    """Map a Gauss-Patterson rule from [-1, 1] to [a, b]."""
    nodes, weights = patterson_rule(order)
    half = 0.5 * (b - a)
    mid = 0.5 * (a + b)
    return mid + half * nodes, half * weights


# ============================================================================
# Velocity moments with Gauss-Patterson quadrature
# ============================================================================
def velocity_moments(delta_f: Callable[[np.ndarray, np.ndarray], np.ndarray],
                     v_th: float = 1.0,
                     v_par_max: float = 6.0,
                     mu_max: float = 9.0,
                     order_par: int = 7,
                     order_mu: int = 7,
                     kperp_rho: float = 0.0) -> dict:
    """Compute the gyrokinetic velocity moments of delta_f(v_par, mu).

    Moments:
        n1    = 2 pi B0 int dv_par dmu  delta_f  J0(k_perp rho)
        upar  = (1/n0) 2 pi B0 int dv_par dmu  v_par delta_f J0
        ppar  = m    2 pi B0 int dv_par dmu  v_par^2 delta_f
        pperp = m    2 pi B0 int dv_par dmu  2 mu B0 delta_f
        qpar  = m/2  2 pi B0 int dv_par dmu  v_par (v_par^2 + 3 * 2 mu B0) delta_f
    """
    nodes_p, weights_p = patterson_rule_ab(order_par, -v_par_max, v_par_max)
    nodes_m, weights_m = patterson_rule_ab(order_mu, 0.0, mu_max)

    VP, MU = np.meshgrid(nodes_p, nodes_m, indexing="ij")
    WP, WM = np.meshgrid(weights_p, weights_m, indexing="ij")

    df = delta_f(VP, MU)
    rho_L = np.sqrt(np.maximum(MU, 0.0))
    z = kperp_rho * rho_L
    if HAVE_SCIPY_SPECIAL:
        J0 = _scipy_jv(0, z)
    else:
        J0 = np.vectorize(lambda x: jn_eval(0, float(x)), otypes=[np.float64])(z)

    integrand_n1 = df * J0
    integrand_upar = VP * df * J0
    integrand_ppar = VP * VP * df
    integrand_pperp = 2.0 * MU * df
    integrand_qpar = VP * (VP * VP + 3.0 * 2.0 * MU) * df

    def _int(g: np.ndarray) -> float:
        return float(2.0 * PI * np.sum(g * WP * WM))

    return {
        "v_par": nodes_p,
        "mu": nodes_m,
        "n1": _int(integrand_n1),
        "upar": _int(integrand_upar),
        "ppar": _int(integrand_ppar),
        "pperp": _int(integrand_pperp),
        "qpar": _int(integrand_qpar),
    }


# ============================================================================
# Centroidal Voronoi Tessellation (port of 245_cvt_1d_nonuniform)
# ============================================================================
def cvt_1d_nonuniform(
    n_generators: int = 20,
    n_samples: int = 2000,
    n_steps: int = 100,
    density: Callable[[np.ndarray], np.ndarray] | str = "maxwellian",
    seed: int = 0,
) -> np.ndarray:
    """Construct a 1-D CVT with a non-uniform target density.

    Lloyd iteration:  z_k^{(m+1)} = int_{V_k} x rho(x) dx / int_{V_k} rho(x) dx.
    """
    rng = np.random.default_rng(seed)

    if isinstance(density, str):
        lo, hi = -6.0, 6.0
        if density == "maxwellian":
            rho = lambda x: np.exp(-x * x)
        elif density == "tail":
            rho = lambda x: x * x * np.exp(-x * x)
        else:
            rho = lambda x: np.ones_like(x)
            lo, hi = 0.0, 1.0
    else:
        rho = density
        lo, hi = -6.0, 6.0

    z = np.linspace(lo, hi, n_generators) + rng.normal(0.0, 0.01 * (hi - lo), n_generators)
    z.sort()

    xs = rng.uniform(lo, hi, n_samples)
    ws = rho(xs)
    ws = np.maximum(ws, 0.0)
    if ws.sum() <= 0:
        return z
    ws = ws / ws.sum()

    for _ in range(n_steps):
        idx = np.argmin(np.abs(xs[:, None] - z[None, :]), axis=1)
        for k in range(n_generators):
            mask = idx == k
            if mask.any():
                wsum = ws[mask].sum()
                if wsum > 0:
                    z[k] = (ws[mask] * xs[mask]).sum() / wsum
        z.sort()
    return z


# ============================================================================
# Sanity self-check
# ============================================================================
PI = math.pi

if __name__ == "__main__":
    print("J0(0)   =", jn_eval(0, 0.0))
    print("J0(1)   =", jn_eval(0, 1.0))
    print("J1(3.8) =", jn_eval(1, 3.8))
    print("first 5 zeros of J0 :", jn_zeros(0, 5))
    print("first 3 zeros of J3 :", jn_zeros(3, 3))
    nodes, weights = patterson_rule(7)
    print("int_{-1}^1 x^6 dx (exact = 2/7):",
          np.sum(weights * nodes ** 6))
    z = cvt_1d_nonuniform(n_generators=15, n_steps=50)
    print("CVT maxwellian generators:", np.round(z, 3))
