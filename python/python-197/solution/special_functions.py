"""
special_functions.py
================================================================================
高性能计算检查点容错：核心特殊函数与统计分布模块

融合原项目：
  - 035_asa091 (Gamma / Chi2 / Normal 分布)
  - 036_asa103 (Digamma 函数)
  - 448_fresnel (Fresnel 积分)

科学角色：
  1) 故障到达时间服从 Gamma/Chi2/Weibull 分布，需要高精度的 CDF/逆 CDF；
  2) Digamma 用于故障分布的统计矩与熵分析；
  3) Fresnel 积分用于波动方程检查点精度验证。
================================================================================
"""

import math
import numpy as np


def alnorm(x: float, upper: bool = False) -> float:
    """AS 66: 标准正态尾概率 P(Z > x) 或 P(Z <= x)。"""
    if x < 0.0:
        z = -x
        up = not upper
    else:
        z = x
        up = upper
    if z <= 7.0:
        y = 0.5 * z * z
        if z > 1.28:
            alnorm_result = 0.398942280385 * math.exp(-y) * (
                0.2231664821 / (z + 0.212206591)
                + 0.2788682071 / (z + 1.641345311)
                + 0.1892693916 / (z + 2.802733431)
                + 0.1313086614 / (z + 4.312328407)
                + 0.1667300525 / (z + 6.365332101)
                + 0.4270866103 / (z + 8.898687736)
                + 1.0 / (z + 15.0)
            )
        else:
            alnorm_result = 0.5 - z * (
                0.398942280444 - 0.399903438504 * y
                / (y + 5.75885480458 - 29.8213550787
                   / (y + 2.62433121679 + 48.6959930692
                      / (y + 5.92885724438)))
            )
    else:
        alnorm_result = 0.0
    if up:
        alnorm_result = 1.0 - alnorm_result
    return alnorm_result


def gammad(x: float, p: float):
    """AS 239: 不完全 Gamma P(x, p) = (1/Gamma(p)) int_0^x t^{p-1} e^{-t} dt。"""
    if x < 0.0 or p <= 0.0:
        return 0.0, 1
    if x == 0.0:
        return 0.0, 0
    if p > 1000.0:
        z = (x - p) / math.sqrt(p)
        return alnorm(z, False), 0
    if x <= 1.0 or x < p:
        a = p
        c = 1.0
        value = 1.0 / p
        while True:
            a = a + 1.0
            c = c * x / a
            value = value + c
            if c / value <= 1.0e-15:
                break
        value = value * math.exp(-x + p * math.log(x) - math.lgamma(p))
        return value, 0
    else:
        a = 1.0 - p
        b = a + x + 1.0
        c = 1.0e30
        d = 1.0 / b
        h = d
        i = 1
        while True:
            a = a + 1.0
            b = b + 2.0
            c = b + a / c
            if abs(c) < 1.0e-30:
                c = 1.0e-30
            d = 1.0 / (b + a * d)
            if abs(d) < 1.0e-30:
                d = 1.0e-30
            delta = c * d
            h = h * delta
            i += 1
            if abs(delta - 1.0) <= 1.0e-15 or i > 10000:
                break
        value = 1.0 - h * math.exp(-x + p * math.log(x) - math.lgamma(p))
        return value, 0


def ppchi2(p: float, v: float, g: float = None):
    """AS 91: Chi2 分布的 p 分位数（逆 CDF）。"""
    if p < 0.0 or p > 1.0 or v <= 0.0:
        return 0.0, 1
    if g is None:
        g = math.lgamma(v * 0.5)
    if p == 0.0:
        return 0.0, 0
    if p == 1.0:
        return 1.0e10, 0
    xx = 0.5 * v
    c = xx - 1.0
    if v >= -1.24 * math.log(p):
        if v > 0.32:
            x = alnorm(p, True)
            p1 = 0.222222 / v
            ch = v * ((x * math.sqrt(p1) + 1.0 - p1) ** 3)
            if ch > 2.2 * v + 6.0:
                ch = -2.0 * (math.log(1.0 - p) - c * math.log(0.5 * ch) + g)
        else:
            ch = 0.4
            a = math.log(1.0 - p)
            while True:
                q = ch
                p1 = 1.0 + ch * (4.67 + ch)
                p2 = ch * (6.73 + ch * (6.66 + ch))
                t = -0.5 + (4.67 + 2.0 * ch) / p1 - (6.73 + ch * (13.32 + 3.0 * ch)) / p2
                ch = ch - (1.0 - math.exp(a + g + 0.5 * ch + c * 0.6931471806) * p2 / p1) / t
                if abs(q / ch - 1.0) <= 1.0e-5:
                    break
    else:
        ch = ((p * xx * math.exp(g + xx * 0.6931471806)) ** (1.0 / xx))
        if ch < 1.0e-6:
            return ch, 0
    for _ in range(200):
        q = ch
        p1 = 0.5 * ch
        p2 = p1 - xx
        t = math.exp(-p1 + c * math.log(p1) - g)
        b = t * ch / p1
        r = t - p
        if abs(r) < 1.0e-12:
            break
        t = r / b
        ch = ch - t
        if abs(t / ch) < 1.0e-12:
            break
    return ch, 0


def ppnd(p: float):
    """AS 111: 标准正态逆 CDF。"""
    if p <= 0.0 or p >= 1.0:
        return 0.0, 1
    q = p - 0.5
    if abs(q) <= 0.42:
        r = q * q
        a0 = -25.44106049637
        a1 = 41.39119773534
        a2 = -18.61500062529
        b0 = 3.13082909833
        b1 = -21.06224101826
        b2 = 30.9407534182
        b3 = -15.684081522874
        num = q * ((a0 * r + a1) * r + a2)
        den = ((b0 * r + b1) * r + b2) * r + b3
        x = num / den
    else:
        r = p
        if q > 0.0:
            r = 1.0 - p
        r = math.sqrt(-math.log(r))
        c0 = 2.32121276858
        c1 = 4.85014127135
        c2 = 5.23193377582
        c3 = 3.98399687305
        c4 = 1.00000615302
        d0 = 5.08179191890
        d1 = 9.34579574972
        d2 = 10.7617194817
        d3 = 6.5317760171
        d4 = 1.7399249498
        d5 = 0.319381530
        num = ((((c0 * r + c1) * r + c2) * r + c3) * r + c4)
        den = (((((d0 * r + d1) * r + d2) * r + d3) * r + d4) * r + d5)
        x = num / den
        if q < 0.0:
            x = -x
    return x, 0


def digamma(x: float):
    """AS 103: Digamma 函数 psi(x) = Gamma'(x)/Gamma(x)。"""
    if x <= 0.0:
        return 0.0, 1
    if x <= 1.0e-6:
        value = -0.5772156649015329 - 1.0 / x
        return value, 0
    value = 0.0
    while x < 8.5:
        value = value - 1.0 / x
        x = x + 1.0
    r = 1.0 / x
    value = value + math.log(x) - 0.5 * r
    r = r * r
    value = value - r * (1.0 / 12.0 - r * (1.0 / 120.0 - r * (1.0 / 252.0 - r * (1.0 / 240.0
              - r * (1.0 / 132.0 - r * (691.0 / 32760.0 - r * 1.0 / 12.0))))))
    return value, 0


# =============================================================================
# Fresnel 积分 —— 使用 SciPy 保证精度，同时保留 Zhang & Jin 分段实现
# =============================================================================
try:
    from scipy.special import fresnel as _sp_fresnel

    def fresnel(x: float):
        """C(x), S(x) -- 调用 SciPy 实现。"""
        c, s = _sp_fresnel(x)
        return float(c), float(s)
except ImportError:
    def fresnel(x: float):
        """纯 Python 分段实现（备用）。"""
        if x == 0.0:
            return 0.0, 0.0
        xa = abs(x)
        t = 0.5 * math.pi * xa * xa
        if xa < 2.5:
            csum = 0.0
            term = 1.0
            for k in range(100):
                if k > 0:
                    m = 4 * k - 3
                    term = -term * t * t / (m * (2 * k - 1) * 4 * k)
                csum += term
                if abs(term) < 1.0e-15:
                    break
            ssum = t / 3.0
            term = ssum
            for k in range(1, 100):
                m = 4 * k - 1
                term = -term * t * t / (m * (2 * k + 1) * 4 * k)
                ssum += term
                if abs(term) < 1.0e-15:
                    break
            c = csum * xa
            s = ssum * xa
        elif xa < 4.5:
            m = int(42 + 1.8 * t)
            su = 0.0
            sv = 0.0
            f = 1.0e-35
            f1 = 0.0
            for k in range(m, 0, -1):
                f = (2.0 * k + 3.0) * f1 / t - f
                su = su + (2.0 * k + 1.0) * f1
                sv = sv + f1
                f1 = f
            q = math.sqrt(2.0 / (math.pi * t))
            c = q * (math.sin(t) * su + math.cos(t) * sv)
            s = q * (math.sin(t) * sv - math.cos(t) * su)
        else:
            r = 1.0
            term = 1.0
            for k in range(1, 50):
                term = -0.25 * term * (4.0 * k - 3.0) * (4.0 * k - 1.0) / (t * t)
                r += term
                if abs(term) < 1.0e-15:
                    break
            r1 = 1.0 / (math.pi * xa)
            g = r1 * r
            r = 1.0 / t
            term = r
            for k in range(1, 50):
                term = -0.25 * term * (4.0 * k - 1.0) * (4.0 * k + 1.0) / (t * t)
                r += term
                if abs(term) < 1.0e-15:
                    break
            f = r1 * r
            c = 0.5 + (f * math.sin(t) - g * math.cos(t)) / (math.pi * xa)
            s = 0.5 - (f * math.cos(t) + g * math.sin(t)) / (math.pi * xa)
        if x < 0.0:
            c = -c
            s = -s
        return c, s


def fresnel_cos(x: float) -> float:
    """仅返回 C(x)。"""
    c, _ = fresnel(x)
    return c


def fresnel_sin(x: float) -> float:
    """仅返回 S(x)。"""
    _, s = fresnel(x)
    return s
