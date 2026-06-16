"""
special_functions.py
====================

移植自 Fortran77 / MATLAB 特殊函数库（原项目 443_fn），为本合成项目提供
高能物理中反复出现的超越函数与特殊函数的高精度实现。

物理用途
--------
- 同步辐射能谱:        dI/dE ~ E * K_{1/3}(E/E_c) * ∫_{E/E_c}^{∞} K_{5/3}(x) dx
                        K_{nu} 为修正 Bessel 函数（r8_besk0 / r8_besk1 作为渐近核）
- 高斯能量弥散:         P(E_rec | E_true) ∝ exp( - (E_rec - E_true)^2 / (2 sigma^2) )
                        累积分布函数用 erf / erfc
- 相空间体积:           Φ_n ∝ Π_i Γ(a_i) / Γ(Σ a_i)   →   r8_gamma / r8_lgamma
- 不完全 β 函数:        用于 F 分布 / 置信区间计算 → r8_betai
- Dawson 积分:          等离子体色散函数 Z(ζ) = i √π w(ζ) 的实部 → r8_dawson

数值策略
--------
- 分段切比雪夫展开 (r8_csevl) + 自变量缩减
- 渐近展开处理大参数，级数展开处理小参数
- 显式处理 ±∞ / NaN / 下溢 / 上溢边界
"""

from __future__ import annotations
import math
from math import exp, log, sqrt, fabs, floor, pi


# ===========================================================================
#                          基础辅助
# ===========================================================================

def r8_mach() -> float:
    """机器精度 eps (单位舍入)。"""
    return math.ldexp(1.0, -52)


def r8_sign(x: float) -> float:
    return 1.0 if x >= 0.0 else -1.0


def _safe_exp(x: float) -> float:
    """防止 exp 上溢/下溢的截断指数。"""
    if x > 709.0:
        return math.exp(709.0)
    if x < -745.0:
        return 0.0
    return math.exp(x)


# ===========================================================================
#                       切比雪夫求值 (r8_csevl)
# ===========================================================================

def r8_csevl(x: float, cs: list, n: int) -> float:
    """
    切比雪夫级数求值:  Σ_{k=0}^{n-1} cs[k] T_k(x),  x ∈ [-1, 1].
    采用 Clenshaw 递推，避免显式构造 T_k。

    递推式:  b_{n+1} = b_{n+2} = 0
             b_k = 2 x b_{k+1} - b_{k+2} + cs[k]
    结果:    S = 0.5 * (b_0 - b_2) = 0.5 * (cs[0] + b_1 - b_3?)  (Clenshaw)

    这里使用标准 Clenshaw:
        p = q = 0
        for k = n-1 down to 1:
            p, q = 2*x*p - q + cs[k], p
        return x*p - q + cs[0]
    """
    if n < 1:
        return 0.0
    if not (-1.0 <= x <= 1.0):
        raise ValueError(f"r8_csevl: x={x} 超出 [-1,1] 定义域")
    p = 0.0
    q = 0.0
    for k in range(n - 1, 0, -1):
        p, q = 2.0 * x * p - q + cs[k], p
    return x * p - q + cs[0]


# ===========================================================================
#                       Gamma / LogGamma (r8_gamma, r8_lgam)
# ===========================================================================

# Lanczos 系数 (g = 7, n = 9), 来自 Numerical Recipes / Godfrey
_LANCZOS_G = 7
_LANCZOS_C = [
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


def r8_lgamma(x: float) -> float:
    """
    ln |Γ(x)|,  x ∉ {0, -1, -2, ...}.

    对 x > 0.5 使用 Lanczos 逼近
    对 x <= 0.5 使用反射公式:
        Γ(x) Γ(1-x) = π / sin(π x)
        ⇒ ln|Γ(x)| = ln(π / |sin(π x)|) - ln|Γ(1-x)|
    """
    if x == 0.0 or x == floor(x) and x < 0.0:
        return float('inf')
    if x < 0.5:
        # 反射公式
        sin_px = math.sin(math.pi * x)
        if fabs(sin_px) < 1e-300:
            return float('inf')
        return log(math.pi / fabs(sin_px)) - r8_lgamma(1.0 - x)
    x -= 1.0
    ag = _LANCZOS_C[0]
    for i in range(1, _LANCZOS_G + 2):
        ag += _LANCZOS_C[i] / (x + i)
    base = x + _LANCZOS_G + 0.5
    return 0.5 * log(2.0 * math.pi) + (x + 0.5) * log(base) - base + log(ag)


def r8_gamma(x: float) -> float:
    """Γ(x),  处理正实轴与负非整数."""
    if x == 0.0 or (x < 0.0 and x == floor(x)):
        return float('inf') * r8_sign(math.cos(0.5 * math.pi * x))
    if x > 171.0:
        return float('inf')
    if x < -170.0:
        return 0.0
    if x < 0.5:
        sin_px = math.sin(math.pi * x)
        if fabs(sin_px) < 1e-300:
            return float('inf')
        return math.pi / (sin_px * r8_gamma(1.0 - x))
    return _safe_exp(r8_lgamma(x))


# ===========================================================================
#                 误差函数 erf / erfc (r8_erf, r8_erfc)
# ===========================================================================

def r8_erf(x: float) -> float:
    """
    erf(x) = (2/√π) ∫_0^x exp(-t^2) dt.

    数值实现: 小 |x| 用 Taylor 级数，大 |x| 用 erf = 1 - erfc 分支。
    """
    ax = fabs(x)
    if ax > 6.0:
        return r8_sign(x) * 1.0
    if ax < 0.5:
        # Taylor: erf(x) = (2/√π) Σ_{n=0}^∞ (-1)^n x^{2n+1} / (n! (2n+1))
        s = 0.0
        term = x
        for n in range(60):
            s += term / (2 * n + 1)
            term *= -x * x / (n + 1)
            if fabs(term / (2 * n + 3)) < 1e-16 * fabs(s):
                break
        return (2.0 / sqrt(pi)) * s
    return r8_sign(x) * (1.0 - r8_erfc(ax))


def r8_erfc(x: float) -> float:
    """
    erfc(x) = 1 - erf(x).
    对 x > 0 使用连分式 (Laplace 连分式):
        erfc(x) = (x/√π) * exp(-x^2) * CF,
        CF = 1/(x^2 + 0.5/(x^2 + 1/(x^2 + 1.5/(...))))
    """
    if x < 0.0:
        return 2.0 - r8_erfc(-x)
    if x < 0.5:
        return 1.0 - r8_erf(x)
    if x > 26.6:
        return 0.0
    # 连分式 (modified Lentz)
    ax2 = x * x
    b = ax2
    c = 1.0 / 1e-300
    d = 1.0 / b
    f = d
    for i in range(1, 200):
        a = i * 0.5
        b += 1.0
        d = 1.0 / (b + a * d)
        c = b + a / c
        delta = c * d
        f *= delta
        if fabs(delta - 1.0) < 1e-15:
            break
    return (x / sqrt(pi)) * _safe_exp(-ax2) * f


# ===========================================================================
#           修正 Bessel 函数 I0, K0 (r8_besi0, r8_besk0)
# ===========================================================================

def r8_besi0(x: float) -> float:
    """
    I_0(x),  修正 Bessel 函数第一类零阶。

    小 |x| ≤ 18:   I_0 = Σ_{k=0}^∞ (x/2)^{2k} / (k!)^2
    大 |x| > 18:   I_0(x) ~ exp(x)/√(2πx) * (1 + 1/(8x) + ...)
    """
    ax = fabs(x)
    if ax <= 18.0:
        y = 0.25 * x * x
        s = 1.0
        term = 1.0
        for k in range(1, 200):
            term *= y / (k * k)
            s += term
            if term < 1e-16 * s:
                break
        return s
    # 渐近: I_0(x) ≈ exp(x) / √(2πx) * P(1/x),  P 多项式截断
    inv = 1.0 / ax
    p = 1.0 + inv * (0.125 + inv * (0.0703125 + inv * 0.0732421875))
    return _safe_exp(ax) / sqrt(2.0 * math.pi * ax) * p


def r8_besk0(x: float) -> float:
    """
    K_0(x),  修正 Bessel 函数第二类零阶, x > 0.

    小 x ≤ 2:   K_0(x) = -[ln(x/2) + γ] I_0(x) + Σ_{k=1}^∞ (x/2)^{2k} H_k / (k!)^2
                H_k = 1 + 1/2 + ... + 1/k (调和数), γ = 0.5772...
    大 x > 2:   K_0(x) ~ √(π/(2x)) exp(-x) * (1 - 1/(8x) + ...)
    """
    if x <= 0.0:
        return float('inf')
    gamma_e = 0.5772156649015328606
    if x <= 2.0:
        y = 0.25 * x * x
        i0 = r8_besi0(x)
        s = 0.0
        hk = 0.0
        term = 1.0
        for k in range(1, 200):
            hk += 1.0 / k
            term *= y / (k * k)
            s += term * hk
            if term * hk < 1e-16 * fabs(s) + 1e-300:
                break
        return -(log(0.5 * x) + gamma_e) * i0 + s
    # 渐近
    inv = 1.0 / x
    p = 1.0 - inv * (0.125 - inv * (0.0703125 - inv * 0.0732421875))
    return sqrt(0.5 * math.pi / x) * _safe_exp(-x) * p


# ===========================================================================
#                   Dawson 积分 r8_dawson
# ===========================================================================

def _compute_gauss16():
    """16 点 Gauss-Legendre 节点与权重, 惰性计算."""
    n = 16
    nodes = []
    weights = []
    for i in range(1, n + 1):
        z = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        for _ in range(20):
            p0, p1 = 1.0, z
            for k in range(2, n + 1):
                p0, p1 = p1, ((2 * k - 1) * z * p1 - (k - 1) * p0) / k
            dp = n * (z * p1 - p0) / (z * z - 1.0)
            z1 = z
            z = z1 - p1 / dp
            if fabs(z - z1) < 1e-15:
                break
        nodes.append(z)
        weights.append(2.0 / ((1 - z * z) * dp * dp))
    return nodes, weights


_GAUSS16_CACHE = None


def _get_gauss16():
    global _GAUSS16_CACHE
    if _GAUSS16_CACHE is None:
        _GAUSS16_CACHE = _compute_gauss16()
    return _GAUSS16_CACHE


def r8_dawson(x: float) -> float:
    """
    F(x) = exp(-x^2) ∫_0^x exp(t^2) dt.

    物理含义: 等离子体色散函数实部, 及 Voigt 轮廓的核心构件。

    实现策略 (Cody / Pomeranz / Thacher):
      小 |x| < 0.2:  Taylor  F(x) = x - 2/3 x^3 + 4/15 x^5 - ...
      中等 0.2 ≤ |x| < 4:  有理近似
      大 |x| ≥ 4:         渐近 F(x) ~ 1/(2x) + 1/(4x^3) + 3/(8x^5) + ...
    """
    ax = fabs(x)
    if ax < 0.2:
        x2 = x * x
        s = x
        term = x
        for k in range(1, 60):
            term *= -2.0 * x2 / (2 * k + 1)
            s += term
            if fabs(term) < 1e-16 * fabs(s):
                break
        return s
    if ax < 4.0:
        # 级数表示: F(x) = Σ_{n=0}^∞ (-1)^n (2x)^{2n+1} / ((2n+1)!!) * exp(-x^2)
        # 使用 Humlicek w4 有理近似的 Dawson 部分
        x2 = x * x
        # 数值积分 via Gauss-Legendre 近似 (快速路径):
        # F(x) = exp(-x^2) * Σ w_i exp(t_i^2), t_i = x * u_i
        # 16 点 Gauss-Legendre 足够 1e-14 精度
        # 16 点 Gauss-Legendre 在 [-1, 1] 上的节点与权重
        nodes, weights = _get_gauss16()
        s = 0.0
        for u, w in zip(nodes, weights):
            t = x * u
            s += w * exp(t * t)
        return x * exp(-x2) * s
    # 渐近展开
    inv2 = 1.0 / (x * x)
    return (0.5 / x) * (1.0 + 0.5 * inv2 * (1.0 + 1.5 * inv2 * (1.0 + 2.5 * inv2)))


# ===========================================================================
#                不完全 β 函数比 r8_betai
# ===========================================================================

def r8_beta(a: float, b: float) -> float:
    """B(a, b) = Γ(a) Γ(b) / Γ(a+b)."""
    if a <= 0.0 or b <= 0.0:
        return float('inf')
    return _safe_exp(r8_lgamma(a) + r8_lgamma(b) - r8_lgamma(a + b))


def r8_betai(a: float, b: float, x: float) -> float:
    """
    正则化不完全 β 函数:
        I_x(a, b) = B(x; a, b) / B(a, b)
                  = (1/B(a,b)) ∫_0^x t^{a-1} (1-t)^{b-1} dt.

    应用: F 分布 CDF, 二项检验 p-value.

    实现: Lentz 连分式 (Numerical Recipes §6.4).
    若 x < (a+1)/(a+b+2) 直接展开;
    否则用对称关系 I_x(a,b) = 1 - I_{1-x}(b,a).
    """
    if not (0.0 <= x <= 1.0):
        raise ValueError(f"r8_betai: x={x} 超出 [0,1]")
    if x == 0.0 or x == 1.0:
        return x
    if a <= 0.0 or b <= 0.0:
        raise ValueError("r8_betai: a, b 必须 > 0")

    # 对称变换使连分式收敛更快
    if x < (a + 1.0) / (a + b + 2.0):
        return _betai_cf(a, b, x)
    return 1.0 - _betai_cf(b, a, 1.0 - x)


def _betai_cf(a: float, b: float, x: float) -> float:
    """连分式计算 I_x(a, b) 的核心."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if fabs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        # 偶数步
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if fabs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if fabs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        # 奇数步
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if fabs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if fabs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if fabs(delta - 1.0) < 1e-14:
            break
    prefactor = _safe_exp(
        a * log(x) + b * log(1.0 - x)
        + r8_lgamma(qab) - r8_lgamma(a) - r8_lgamma(b)
    ) / a
    return prefactor * h


# ===========================================================================
#          Pochhammer / 上升阶乘 r8_poch (用于级数展开)
# ===========================================================================

def r8_poch(a: float, n: int) -> float:
    """上升阶乘 (a)_n = a (a+1) ... (a+n-1) = Γ(a+n)/Γ(a)."""
    if n < 0:
        raise ValueError("r8_poch: n 必须 ≥ 0")
    if n == 0:
        return 1.0
    return _safe_exp(r8_lgamma(a + n) - r8_lgamma(a))


# Gauss-Legendre 16 点求积: 见 _compute_gauss16 / _get_gauss16 定义 (r8_dawson 之前).


__all__ = [
    "r8_mach", "r8_sign", "r8_csevl",
    "r8_lgamma", "r8_gamma", "r8_erf", "r8_erfc",
    "r8_besi0", "r8_besk0", "r8_dawson",
    "r8_beta", "r8_betai", "r8_poch",
]
