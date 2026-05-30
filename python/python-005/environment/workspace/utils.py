# -*- coding: utf-8 -*-

import numpy as np
from math import factorial, exp, log, sqrt, pi, cos, sin




def gamma_lanczos(z: float) -> float:
    if z <= 0.0:
        raise ValueError("gamma_lanczos 仅适用于 z > 0")
    g = 7.0
    p = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]
    z = z - 1.0
    x = p[0]
    for i in range(1, len(p)):
        x += p[i] / (z + i)
    t = z + g + 0.5
    return sqrt(2.0 * pi) * (t ** (z + 0.5)) * exp(-t) * x


def log_gamma(z: float) -> float:
    if z <= 0.0:
        raise ValueError("log_gamma 仅适用于 z > 0")
    return log(gamma_lanczos(z))





def spherical_bessel_j(l: int, x: float) -> float:
    if abs(x) < 1e-12:
        if l == 0:
            return 1.0
        elif l == 1:
            return x / 3.0
        else:
            return 0.0
    if l == 0:
        return sin(x) / x
    if l == 1:
        return sin(x) / (x * x) - cos(x) / x
    j_lm2 = sin(x) / x
    j_lm1 = sin(x) / (x * x) - cos(x) / x
    for ll in range(2, l + 1):
        j_l = (2.0 * ll - 1.0) / x * j_lm1 - j_lm2
        j_lm2, j_lm1 = j_lm1, j_l
    return j_lm1


def spherical_bessel_j_array(lmax: int, x: np.ndarray) -> np.ndarray:
    nx = len(x)
    out = np.zeros((lmax + 1, nx), dtype=float)
    out[0, :] = np.sinc(x / pi)
    if lmax >= 1:
        out[1, :] = np.sin(x) / (x ** 2) - np.cos(x) / x
    for l in range(2, lmax + 1):
        out[l, :] = (2.0 * l - 1.0) / x * out[l - 1, :] - out[l - 2, :]

    mask = np.abs(x) < 1e-12
    out[0, mask] = 1.0
    if lmax >= 1:
        out[1, mask] = x[mask] / 3.0
    for l in range(2, lmax + 1):
        out[l, mask] = 0.0
    return out





def associated_legendre(l: int, m: int, cos_theta: float) -> float:
    x = cos_theta
    if m < 0 or m > l:
        return 0.0
    if abs(x) > 1.0 + 1e-12:
        raise ValueError("associated_legendre: |cosθ| > 1")
    x = max(-1.0, min(1.0, x))

    pmm = 1.0
    if m > 0:
        somx2 = sqrt(max(0.0, 1.0 - x * x))
        fact = 1.0
        for _ in range(m):
            pmm *= -fact * somx2
            fact += 2.0
    if l == m:
        return pmm

    pmmp1 = x * (2.0 * m + 1.0) * pmm
    if l == m + 1:
        return pmmp1

    pll = 0.0
    for ll in range(m + 2, l + 1):
        pll = (x * (2.0 * ll - 1.0) * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
        pmm, pmmp1 = pmmp1, pll
    return pll


def wigner_3j_000(j1: int, j2: int, j3: int) -> float:
    J = j1 + j2 + j3
    if J % 2 != 0:
        return 0.0
    if not (abs(j1 - j2) <= j3 <= j1 + j2):
        return 0.0
    half = J // 2
    num = factorial(J - 2 * j1) * factorial(J - 2 * j2) * factorial(J - 2 * j3)
    den = factorial(J + 1)
    prefactor = (-1.0) ** half * sqrt(float(num) / float(den))
    fac = factorial(half)
    fac_denom = factorial(half - j1) * factorial(half - j2) * factorial(half - j3)
    return prefactor * fac / fac_denom





def binomial(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    res = 1
    for i in range(k):
        res = res * (n - i) // (i + 1)
    return res





def robust_divide(a: float, b: float, fallback: float = 0.0) -> float:
    if abs(b) < 1e-15:
        return fallback
    return a / b


def clip_to_unit(x):
    if isinstance(x, np.ndarray):
        return np.clip(x, -1.0, 1.0)
    return max(-1.0, min(1.0, x))


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def ensure_positive(x: float, name: str = "variable") -> float:
    if x <= 0.0:
        raise ValueError(f"{name} 必须为正，当前值 = {x}")
    return x
