# -*- coding: utf-8 -*-
"""
utils.py
公共数学工具与数值鲁棒性模块

包含：Gamma 函数近似、球 Bessel 函数、连带 Legendre 函数、
      二项式系数、数值稳定性检查、边界条件验证
"""

import numpy as np
from math import factorial, exp, log, sqrt, pi, cos, sin

# ---------------------------------------------------------------------------
# Gamma 函数近似（Lanczos 近似，保证数值稳定性）
# ---------------------------------------------------------------------------
def gamma_lanczos(z: float) -> float:
    """
    Lanczos 近似计算 Gamma(z)，适用于 z > 0.5。
    公式：
        Γ(z+1) ≈ sqrt(2π) (z+g+0.5)^{z+0.5} exp(-(z+g+0.5)) Σ_{i=0}^{N} p_i / (z+i)
    """
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
    """对数 Gamma 函数，避免大数溢出。"""
    if z <= 0.0:
        raise ValueError("log_gamma 仅适用于 z > 0")
    return log(gamma_lanczos(z))


# ---------------------------------------------------------------------------
# 球 Bessel 函数 j_l(x)
# ---------------------------------------------------------------------------
def spherical_bessel_j(l: int, x: float) -> float:
    """
    计算球 Bessel 函数 j_l(x)。
    递推关系（向上稳定）：
        j_{l+1}(x) = (2l+1)/x * j_l(x) - j_{l-1}(x)
    初值：
        j_0(x) = sin(x)/x
        j_1(x) = sin(x)/x^2 - cos(x)/x
    边界处理：|x| < 1e-12 时返回渐近展开。
    """
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
    """对 x 数组计算 j_0(x) ... j_{lmax}(x)，返回形状 (lmax+1, len(x))。"""
    nx = len(x)
    out = np.zeros((lmax + 1, nx), dtype=float)
    out[0, :] = np.sinc(x / pi)  # sin(x)/x
    if lmax >= 1:
        out[1, :] = np.sin(x) / (x ** 2) - np.cos(x) / x
    for l in range(2, lmax + 1):
        out[l, :] = (2.0 * l - 1.0) / x * out[l - 1, :] - out[l - 2, :]
    # 处理 x≈0
    mask = np.abs(x) < 1e-12
    out[0, mask] = 1.0
    if lmax >= 1:
        out[1, mask] = x[mask] / 3.0
    for l in range(2, lmax + 1):
        out[l, mask] = 0.0
    return out


# ---------------------------------------------------------------------------
# 连带 Legendre 函数 P_l^m(cosθ)
# ---------------------------------------------------------------------------
def associated_legendre(l: int, m: int, cos_theta: float) -> float:
    """
    计算连带 Legendre 函数 P_l^m(cosθ)。
    递推（Ferrers 定义， Condon-Shortley 相位）：
        P_m^m(x) = (-1)^m (2m-1)!! (1-x^2)^{m/2}
        P_{m+1}^m(x) = x (2m+1) P_m^m(x)
        (l-m) P_l^m = x(2l-1) P_{l-1}^m - (l+m-1) P_{l-2}^m
    """
    x = cos_theta
    if m < 0 or m > l:
        return 0.0
    if abs(x) > 1.0 + 1e-12:
        raise ValueError("associated_legendre: |cosθ| > 1")
    x = max(-1.0, min(1.0, x))
    # P_m^m
    pmm = 1.0
    if m > 0:
        somx2 = sqrt(max(0.0, 1.0 - x * x))
        fact = 1.0
        for _ in range(m):
            pmm *= -fact * somx2
            fact += 2.0
    if l == m:
        return pmm
    # P_{m+1}^m
    pmmp1 = x * (2.0 * m + 1.0) * pmm
    if l == m + 1:
        return pmmp1
    # 向上递推
    pll = 0.0
    for ll in range(m + 2, l + 1):
        pll = (x * (2.0 * ll - 1.0) * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
        pmm, pmmp1 = pmmp1, pll
    return pll


def wigner_3j_000(j1: int, j2: int, j3: int) -> float:
    """
    简化 Wigner 3j 符号 (j1 j2 j3; 0 0 0)，用于 Gaunt 积分。
    仅对 j1+j2+j3 为偶数时有非零值。
    公式：
        ( j1 j2 j3 ; 0 0 0 ) = (-1)^{J/2} sqrt{(J-2j1)!(J-2j2)!(J-2j3)! / (J+1)!}
                              × (J/2)! / [(J/2-j1)! (J/2-j2)! (J/2-j3)!]
        其中 J = j1+j2+j3
    """
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


# ---------------------------------------------------------------------------
# 二项式系数
# ---------------------------------------------------------------------------
def binomial(n: int, k: int) -> int:
    """计算 C(n,k)，带边界检查。"""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    res = 1
    for i in range(k):
        res = res * (n - i) // (i + 1)
    return res


# ---------------------------------------------------------------------------
# 数值稳定性与边界检查
# ---------------------------------------------------------------------------
def robust_divide(a: float, b: float, fallback: float = 0.0) -> float:
    """安全除法，分母接近零时返回 fallback。"""
    if abs(b) < 1e-15:
        return fallback
    return a / b


def clip_to_unit(x):
    """将 x 限制在 [-1, 1] 区间内，防止三角函数反函数溢出。支持标量和数组。"""
    if isinstance(x, np.ndarray):
        return np.clip(x, -1.0, 1.0)
    return max(-1.0, min(1.0, x))


def is_power_of_two(n: int) -> bool:
    """检查 n 是否为 2 的整数幂。"""
    return n > 0 and (n & (n - 1)) == 0


def ensure_positive(x: float, name: str = "variable") -> float:
    """确保变量为正，否则抛出异常。"""
    if x <= 0.0:
        raise ValueError(f"{name} 必须为正，当前值 = {x}")
    return x
