# -*- coding: utf-8 -*-
"""
special_functions.py
--------------------
High-accuracy evaluation of special functions that appear in the Matsubara-
axis formulation of the isotropic Eliashberg equations:

  * Cosine integral   Ci(x) = gamma + ln x + int_0^x (cos t - 1)/t dt
  * Sine integral     Si(x) = int_0^x sin(t)/t dt
  * Debye function    D_n(x) = (n/x^n) int_0^x t^n / (e^t - 1) dt
  * Digamma function  psi(z) for real z > 0

Scientific origin of the fused algorithms
-----------------------------------------
* Cosine integral routine   (seed project 221_cosine_integral)
    -> full Zhang-Jin algorithm with three regimes:
       (i)   |x| <= 16  : power series
       (ii)  16 < |x| <= 32  : backward recurrence on Bessel-like sequence
       (iii) |x| > 32   : asymptotic expansion via continued fractions

Core physics / mathematics
--------------------------
* The cosine integral enters the *retarded* electron-phonon kernel
      K_{n,m}^{ret}  proportional to  Ci(|omega_n - omega_m| / omega_D)
  when the spectral function alpha^2 F is approximated by a single Einstein
  mode.
* The Debye function D_3 gives the phonon contribution to the specific heat
      C_{ph}(T) = 3 N k_B D_3(theta_D / T).
* The digamma function appears in the linearised gap equation at T_c:
      1 / lambda = psi(1/2) - Re psi(1/2 + i hbar omega_D / (2 pi k_B T_c)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1.  Cosine integral Ci(x)  (Zhang-Jin algorithm, seed project 221)
# ---------------------------------------------------------------------------
def ci(x: float) -> float:
    """Cosine integral Ci(x) for real x."""
    epsilon = 1.0e-15
    xabs = abs(x)
    if xabs == 0.0:
        return float("-inf")
    # Euler-Mascheroni constant
    euler_gamma = 0.5772156649015328606065120900824024

    if xabs <= 16.0:
        # Power series
        xr = -0.25 * x * x
        value = euler_gamma + math.log(xabs) + xr
        for k in range(2, 41):
            xr *= -0.5 * (k - 1) / (k * k * (2 * k - 1)) * x * x
            value += xr
            if abs(xr) < abs(value) * epsilon:
                break
        return value

    if xabs <= 32.0:
        # Backward recurrence (Bessel-like)
        m = int(math.floor(47.2 + 0.82 * xabs))
        bj = np.zeros(m + 1)
        xa1 = 0.0
        xa0 = 1.0e-100
        for k in range(m, 0, -1):
            xa = 4.0 * k * xa0 / xabs - xa1
            bj[k] = xa
            xa1 = xa0
            xa0 = xa
        xs = bj[1]
        for k in range(3, m + 1, 2):
            xs += 2.0 * bj[k]
        bj[1] /= xs
        for k in range(2, m + 1):
            bj[k] /= xs
        xr = 1.0
        f = bj[1]
        for k in range(2, m + 1):
            xr *= -1.0
            f += xr * bj[k]
        f *= 2.0
        value = f * math.sin(xabs / 2.0) ** 2
        # Second part via forward recurrence
        g = bj[1]
        for k in range(2, m + 1):
            g += bj[k]
        value += g * math.sin(xabs) / xabs
        return value

    # Asymptotic expansion
    xi = 1.0 / xabs
    xi2 = xi * xi
    f = 1.0
    g = xi
    term = 1.0
    for k in range(1, 9):
        term *= -(2 * k - 1) * (2 * k) * xi2
        f += term
        term2 = term * (2 * k + 1) * xi
        g += term2
    return (f * math.sin(xabs) - g * math.cos(xabs)) / xabs


# ---------------------------------------------------------------------------
# 2.  Sine integral Si(x)
# ---------------------------------------------------------------------------
def si(x: float) -> float:
    """Sine integral Si(x) for real x."""
    if x == 0.0:
        return 0.0
    sign = 1.0 if x > 0 else -1.0
    x = abs(x)
    if x <= 16.0:
        xr = x
        value = x
        x2 = x * x
        for k in range(1, 41):
            xr *= -0.5 * (2 * k - 1) / (k * (2 * k + 1) * (2 * k)) * x2
            value += xr
            if abs(xr) < abs(value) * 1e-15:
                break
        return sign * value
    # Asymptotic
    xi = 1.0 / x
    f = 1.0
    g = xi
    term = 1.0
    xi2 = xi * xi
    for k in range(1, 9):
        term *= -(2 * k - 1) * (2 * k) * xi2
        f += term
        term2 = term * (2 * k + 1) * xi
        g += term2
    return sign * (math.pi / 2.0 - f * math.cos(x) / x - g * math.sin(x) / x)


# ---------------------------------------------------------------------------
# 3.  Debye function  D_n(x)
# ---------------------------------------------------------------------------
def debye_function(n: int, x: float, n_quad: int = 200) -> float:
    """Debye function D_n(x) = (n/x^n) int_0^x t^n / (e^t - 1) dt.

    For x < 1e-8 returns the small-x limit  D_n(x) -> n/(n+1) - n x/(2(n+2)) + ...
    Uses Gauss-Legendre quadrature with n_quad points on [0, x].
    """
    if n < 1:
        raise ValueError("debye_function: order n must be >= 1")
    if x < 0.0:
        raise ValueError("debye_function: x must be >= 0")
    if x < 1e-8:
        return n / (n + 1.0) - n * x / (2.0 * (n + 2.0))

    nodes, wts = np.polynomial.legendre.leggauss(n_quad)
    # Map [-1,1] -> [0,x]
    t = 0.5 * x * (nodes + 1.0)
    w = 0.5 * x * wts
    integrand = np.where(
        t > 1e-8,
        t ** n / np.expm1(t),
        t ** (n - 1),       # small-t limit of t^n/(e^t-1)
    )
    return float((n / x ** n) * (w * integrand).sum())


# ---------------------------------------------------------------------------
# 4.  Digamma function psi(z) for real z > 0  (asymptotic + recurrence)
# ---------------------------------------------------------------------------
def digamma(z: float) -> float:
    """Digamma function psi(z) = d/dz ln Gamma(z) for real z > 0.

    Uses asymptotic expansion for z > 8 and the recurrence  psi(z+1) = psi(z) + 1/z
    to shift small arguments into the asymptotic regime.
    """
    if z <= 0.0:
        raise ValueError("digamma: argument must be > 0")
    result = 0.0
    while z < 8.0:
        result -= 1.0 / z
        z += 1.0
    # Asymptotic: psi(z) ~ ln z - 1/(2z) - sum B_{2k}/(2k z^{2k})
    inv_z = 1.0 / z
    inv_z2 = inv_z * inv_z
    result += math.log(z) - 0.5 * inv_z
    bern = [
        1.0 / 12.0, -1.0 / 120.0, 1.0 / 252.0, -1.0 / 240.0,
        5.0 / 660.0, -691.0 / 32760.0, 1.0 / 12.0,
    ]
    term = inv_z2
    for b in bern:
        result -= b * term
        term *= inv_z2
    return result


# ---------------------------------------------------------------------------
# 5.  Driver: Matsubara kernel
# ---------------------------------------------------------------------------
@dataclass
class MatsubaraKernel:
    """Precomputed retarded kernel K_{n,m}^{ret} on Matsubara frequencies."""
    omega_n: np.ndarray       # (M,) bosonic or fermionic Matsubara frequencies
    K_ret: np.ndarray         # (M, M) retarded kernel
    lambda_ret: float         # trace of K_ret (a proxy for the EPC strength)


def build_retarded_kernel(
    n_matsubara: int,
    temperature: float,
    omega_E: float,
    g_ep: float = 1.0,
) -> MatsubaraKernel:
    """Build the retarded kernel on fermionic Matsubara frequencies
        omega_n = pi T (2n + 1),    n = 0, ..., M-1
    for a single-Einstein phonon model with coupling g_ep and frequency omega_E.

    The kernel is
        K_{n,m} = g_ep * Ci(|omega_n - omega_m| / omega_E + eps)
    where eps prevents the logarithmic singularity at omega_n = omega_m.
    """
    if temperature <= 0.0:
        raise ValueError("build_retarded_kernel: T must be > 0")
    if omega_E <= 0.0:
        raise ValueError("build_retarded_kernel: omega_E must be > 0")
    M = n_matsubara
    omega = math.pi * temperature * (2.0 * np.arange(M) + 1.0)
    K = np.zeros((M, M))
    eps = 1e-12
    for n in range(M):
        for m in range(M):
            arg = abs(omega[n] - omega[m]) / omega_E + eps
            K[n, m] = g_ep * ci(float(arg))
    lam = float(np.trace(K))
    return MatsubaraKernel(omega_n=omega, K_ret=K, lambda_ret=lam)
