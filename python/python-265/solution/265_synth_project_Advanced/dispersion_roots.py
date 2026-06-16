# -*- coding: utf-8 -*-
"""
dispersion_roots.py
-------------------
Root-finding algorithms applied to the plasma-wave dispersion
relation governing the scattering of cosmic rays off
electromagnetic fluctuations in the heliosphere.

The dominant scattering is resonant: a particle of rigidity R and
pitch-angle mu resonates with a wave of parallel wavenumber k when

    k || v mu = Omega  (cyclotron resonance)

The full hot-plasma dispersion relation for parallel-propagating
electromagnetic waves in a proton-electron plasma is

    D(omega, k) = c^2 k^2 - omega^2 + sum_s (omega_{ps}^2 / omega)
                  * [ (omega - k v_{d,s}) / (omega - k v_{d,s}
                    - s Omega_s) ] Z(zeta_s) = 0

where  Z  is the plasma dispersion function (Faddeeva function).

For the small-scale reproducible experiments here we use a cold-plasma
reduction:

    D(omega) = omega^2 - omega * Omega_p - omega_{pp}^2 = 0

whose solutions are  omega = (Omega_p +/- sqrt(Omega_p^2 +
4 omega_{pp}^2)) / 2.  We solve a transcendental version
incorporating the relativistic cyclotron frequency

    Omega_p(R) = Z e B / (gamma m_p c)

using either Brent's method or the Lambert W function.
"""
from __future__ import annotations
import math
import numpy as np
from typing import Callable, Optional, Tuple

import cosmic_ray_physics as crp


# =====================================================================
# Lambert W function (real branches)
# =====================================================================
def lambert_w0(x: float) -> float:
    """Upper (principal) branch W_0(x) for  x >= -1/e.

    Halley iteration starting from the standard asymptotic.
    """
    if x < -math.exp(-1.0):
        raise ValueError("lambert_w0: x < -1/e is out of range.")
    # initial guess
    if x < 0.0:
        p = math.sqrt(2.0 * (math.e * x + 1.0))
        w = -1.0 + p - p * p / 3.0
    elif x < 1.0:
        w = x * (1.0 - x)
    else:
        w = math.log(x) - math.log(math.log(x)) if x > math.e else x / math.e
    # Halley iterations
    for _ in range(8):
        ew = math.exp(w)
        wew = w * ew
        f = wew - x
        fp = ew * (w + 1.0)
        fpp = ew * (w + 2.0)
        denom = fp - 0.5 * f * fpp / fp
        if abs(denom) < 1.0e-30:
            break
        dw = f / denom
        w -= dw
        if abs(dw) < 1.0e-14 * (1.0 + abs(w)):
            break
    return w


def lambert_wm1(x: float) -> float:
    """Lower branch W_{-1}(x) for  -1/e < x < 0."""
    if x >= 0.0 or x <= -math.exp(-1.0):
        raise ValueError("lambert_wm1: needs -1/e < x < 0.")
    # initial guess
    w = math.log(-x) - math.log(-math.log(-x))
    for _ in range(12):
        ew = math.exp(w)
        wew = w * ew
        f = wew - x
        fp = ew * (w + 1.0)
        fpp = ew * (w + 2.0)
        denom = fp - 0.5 * f * fpp / fp
        if abs(denom) < 1.0e-30:
            break
        dw = f / denom
        w -= dw
        if abs(dw) < 1.0e-14 * (1.0 + abs(w)):
            break
    return w


# =====================================================================
# Brent's method (reverse-communication style, pure-Python)
# =====================================================================
def brent_root(f: Callable[[float], float], a: float, b: float,
               tol: float = 1.0e-10, max_iter: int = 100) -> float:
    """Return a root of  f  on  [a, b]  using Brent's method.

    This is a reimplementation of the classic zero.f algorithm
    (Brent 1973, revised by Burkardt).
    """
    fa = f(a)
    fb = f(b)
    if fa * fb > 0.0:
        raise ValueError(f"brent_root: sign change required, got f({a})="
                         f"{fa}, f({b})={fb}")
    c = a
    fc = fa
    d = b - a
    e = d
    for _ in range(max_iter):
        if fb * fc > 0.0:
            c = a
            fc = fa
            d = b - a
            e = d
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol1 = 2.0 * 1.0e-15 * abs(b) + 0.5 * tol
        m = 0.5 * (c - b)
        if abs(m) <= tol1 or fb == 0.0:
            return b
        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p = 2.0 * m * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0.0:
                q = -q
            else:
                p = -p
            if 2.0 * p < min(3.0 * m * q - abs(tol1 * q), abs(e * q)):
                e = d
                d = p / q
            else:
                d = m
                e = m
        else:
            d = m
            e = m
        a = b
        fa = fb
        if abs(d) > tol1:
            b += d
        else:
            b += tol1 if m > 0 else -tol1
        fb = f(b)
    return b


# =====================================================================
# Plasma dispersion relation
# =====================================================================
def plasma_dispersion_cold(omega: float, k: float, B: float,
                           n_p: float = 5.0e6) -> complex:
    """Cold-plasma dispersion function for parallel whistler /
    ion-cyclotron waves (Stix 1992):

        D(omega) = c^2 k^2 / omega^2 - 1
                   + omega_{pp}^2 / (omega (Omega_p - omega))
                   + omega_{pe}^2 / (omega (omega + Omega_e))

    where  omega_{ps} = sqrt(n e^2 / (epsilon_0 m_s))  and
    Omega_s = e B / m_s.
    """
    eps0 = 1.0 / (crp.mu_0 * crp.c_light ** 2)
    omega_pp = math.sqrt(n_p * crp.q_e ** 2 / (eps0 * crp.m_p))
    omega_pe = math.sqrt(n_p * crp.q_e ** 2 / (eps0 * crp.m_e))
    Omega_p = crp.q_e * B / crp.m_p
    Omega_e = crp.q_e * B / crp.m_e
    if abs(omega) < 1.0e-6 or abs(Omega_p - omega) < 1.0e-6:
        return complex(1.0e30, 0.0)
    term1 = (crp.c_light * k / omega) ** 2 - 1.0
    term2 = omega_pp ** 2 / (omega * (Omega_p - omega))
    term3 = omega_pe ** 2 / (omega * (omega + Omega_e))
    return complex(term1 + term2 + term3, 0.0)


def resonance_condition(k: float, R: float, mu: float,
                        B: float, Z: int = 1, A: int = 1,
                        harmonic: int = -1) -> float:
    """Return  k v mu - s Omega / gamma  for resonance search.

    harmonic = -1 is the fundamental cyclotron (dominant) resonance.
    """
    Ek = crp.rigidity_to_kinetic(R)
    v = crp.velocity_from_Ek(Ek)
    gamma = crp.lorentz_factor(Ek)
    mass = A * crp.m_p
    Omega = Z * crp.q_e * B / mass
    return k * v * mu - harmonic * Omega / gamma


def find_resonant_k(R: float, mu: float, B: float,
                    harmonic: int = -1,
                    k_min: float = 1.0e-9,
                    k_max: float = 1.0e-3) -> float:
    """Find the resonant wavenumber  k  such that

        k v mu = |harmonic| * Omega / gamma.
    """
    Ek = crp.rigidity_to_kinetic(R)
    v = crp.velocity_from_Ek(Ek)
    gamma = crp.lorentz_factor(Ek)
    Omega = abs(crp.q_e * B / crp.m_p)
    k_res = abs(harmonic) * Omega / (gamma * v * abs(mu) + 1.0e-30)
    return k_res


# =====================================================================
# Lambert W inversion of escape-time expression
# =====================================================================
def escape_time_lambert(tau_diff: float, tau_adv: float) -> float:
    """Solve  t + tau_diff * exp(-t / tau_adv) = T_obs  for  t.

    Rearranging yields a Lambert W expression:
        t = T_obs + tau_adv * W( - (tau_diff / tau_adv)
                                  * exp(-T_obs / tau_adv) ).
    This is used to invert the mean escape time of CRs from a
    finite propagation region.
    """
    if tau_adv <= 0.0 or tau_diff <= 0.0:
        return 0.0
    arg = -(tau_diff / tau_adv) * math.exp(-1.0 / tau_adv)
    if arg < -math.exp(-1.0):
        # use W_{-1} branch
        try:
            W = lambert_wm1(max(arg, -math.exp(-1.0) + 1.0e-12))
        except ValueError:
            W = -1.0
    else:
        W = lambert_w0(arg)
    return 1.0 + tau_adv * W


# =====================================================================
# Demo
# =====================================================================
def _demo() -> None:
    print("[dispersion_roots] Lambert W0(1) = "
          f"{lambert_w0(1.0):.8f}  (ref 0.56714329)")
    print("[dispersion_roots] Lambert W0(-0.1) = "
          f"{lambert_w0(-0.1):.8f}")
    print("[dispersion_roots] Lambert Wm1(-0.1) = "
          f"{lambert_wm1(-0.1):.8f}")
    # brent root of sin(x) near 3
    x = brent_root(math.sin, 3.0, 3.5)
    print(f"[dispersion_roots] brent_root(sin, 3, 3.5) = {x:.6f}  "
          f"(ref pi = {math.pi:.6f})")
    # resonant k for 1 GV proton
    k_res = find_resonant_k(1.0e9, 0.5, 5.0e-9)
    print(f"[dispersion_roots] k_res(1 GV, mu=0.5, B=5 nT) = {k_res:.3e} "
          f"1/m")
    print(f"[dispersion_roots] escape time (tau_diff=1 yr, tau_adv=10 yr) "
          f"= {escape_time_lambert(1.0, 10.0):.3f} (units of tau_adv)")


if __name__ == "__main__":
    _demo()
