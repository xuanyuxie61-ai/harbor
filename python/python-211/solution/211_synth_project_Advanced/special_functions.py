"""
special_functions.py
====================
博士级特殊函数库 —— 无约束非线性优化的数学基础

融合种子项目:
  - 881_polpak: 阶乘、Bernoulli 数、Beta 函数、Chebyshev 多项式、Lerch 超越函数
  - 163_chebyshev_series: Chebyshev 级数求值及高阶导数 (Maess/Clenshaw 算法)

核心公式:
  1. Chebyshev 多项式递推: T_0(x)=1, T_1(x)=x, T_{n+1}(x)=2xT_n(x)-T_{n-1}(x)
  2. Chebyshev 级数: f(x) ≈ Σ_{k=0}^{N} c_k T_k(x), x∈[-1,1]
  3. Clenshaw 递推求值: b_{N+1}=b_N=0, b_k = 2x b_{k+1} - b_{k+2} + c_k
  4. Bernoulli 数递推: B_0=1, Σ_{k=0}^{n} C(n+1,k) B_k = 0
  5. Beta 函数: B(a,b) = Γ(a)Γ(b)/Γ(a+b)
  6. Lerch 超越函数: Φ(z,s,a) = Σ_{n=0}^∞ z^n/(n+a)^s
  7. Pochhammer 符号: (a)_n = a(a+1)...(a+n-1) = Γ(a+n)/Γ(a)
"""

import numpy as np
import math
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# 1. 阶乘与对数阶乘 (源自 881_polpak/r8_factorial_log, i4_factorial2)
# ---------------------------------------------------------------------------

def factorial(n: int) -> int:
    """计算 n! = Γ(n+1), 边界处理: n<0 返回 1."""
    if n < 0:
        return 1
    return math.factorial(n)


def log_factorial(n: int) -> float:
    """计算 ln(n!) = ln Γ(n+1), 使用 Stirling 近似增强大 n 精度.
    Stirling 级数: ln(n!) ≈ n ln(n) - n + 0.5 ln(2πn) + 1/(12n) - 1/(360n³) + ...
    """
    if n <= 0:
        return 0.0
    if n < 20:
        return math.log(float(math.factorial(n)))
    # Stirling 渐近展开 (Poincaré)
    nn = float(n)
    return (nn * math.log(nn) - nn + 0.5 * math.log(2.0 * math.pi * nn)
            + 1.0 / (12.0 * nn)
            - 1.0 / (360.0 * nn ** 3)
            + 1.0 / (1260.0 * nn ** 5)
            - 1.0 / (1680.0 * nn ** 7))


def double_factorial(n: int) -> int:
    """双阶乘 n!! = n·(n-2)·(n-4)·...·(1或2).
    源自 881_polpak/i4_factorial2.
    """
    if n < 1:
        return 1
    val = 1
    while n > 1:
        val *= n
        n -= 2
    return val


# ---------------------------------------------------------------------------
# 2. Bernoulli 数 (源自 881_polpak/bernoulli_number3)
# ---------------------------------------------------------------------------

def bernoulli_numbers(n_max: int) -> np.ndarray:
    """计算 Bernoulli 数 B_0, B_1, ..., B_{n_max}.
    递推公式: B_0 = 1, 对 m ≥ 1:
        Σ_{k=0}^{m} C(m+1, k) B_k = 0
      => B_m = -1/(m+1) Σ_{k=0}^{m-1} C(m+1,k) B_k

    Bernoulli 数在 Euler-Maclaurin 公式、Riemann ζ 函数中核心出现:
        ζ(2n) = (-1)^{n+1} (2π)^{2n} B_{2n} / (2 (2n)!)
    """
    B = np.zeros(n_max + 1)
    B[0] = 1.0
    for m in range(1, n_max + 1):
        s = 0.0
        for k in range(m):
            s += math.comb(m + 1, k) * B[k]
        B[m] = -s / (m + 1.0)
    return B


# ---------------------------------------------------------------------------
# 3. Beta 函数与 Gamma 函数 (源自 881_polpak/r8_beta)
# ---------------------------------------------------------------------------

def beta_function(a: float, b: float) -> float:
    """Beta 函数 B(a,b) = Γ(a)Γ(b)/Γ(a+b).
    利用对数 Gamma 避免溢出:
        B(a,b) = exp(lgamma(a) + lgamma(b) - lgamma(a+b))
    在优化中用于构造 Dirichlet 分布的归一化常数.
    """
    if a <= 0 or b <= 0:
        return float('nan')
    return math.exp(math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))


def log_beta(a: float, b: float) -> float:
    """对数 Beta 函数 ln B(a,b)."""
    if a <= 0 or b <= 0:
        return float('nan')
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


# ---------------------------------------------------------------------------
# 4. Lerch 超越函数 (源自 881_polpak/lerch)
# ---------------------------------------------------------------------------

def lerch_transcendent(z: complex, s: float, a: float, n_terms: int = 200) -> complex:
    """Lerch 超越函数 Φ(z,s,a) = Σ_{n=0}^{∞} z^n / (n+a)^s.
    特殊情形:
      - Φ(1, s, a) = ζ(s, a)  (Hurwitz ζ 函数)
      - Φ(z, 1, 1) = -ln(1-z)/z  (对数生成函数)
      - Φ(z, s, 1) = Li_s(z)/z  (多对数函数)

    收敛条件: |z| < 1, 或 |z|=1 且 Re(s) > 1.
    """
    result = complex(0.0)
    for n in range(n_terms):
        denom = (n + a) ** s
        if abs(denom) < 1e-300:
            break
        term = z ** n / denom
        result += term
        if abs(term) < 1e-15 * abs(result) and n > 10:
            break
    return result


# ---------------------------------------------------------------------------
# 5. Pochhammer 符号 (升阶乘)
# ---------------------------------------------------------------------------

def pochhammer(a: float, n: int) -> float:
    """Pochhammer 符号 (a)_n = a(a+1)...(a+n-1) = Γ(a+n)/Γ(a).
    在超几何级数和高阶优化方法的 Taylor 展开中出现.
    """
    if n == 0:
        return 1.0
    result = 1.0
    for k in range(n):
        result *= (a + k)
    return result


# ---------------------------------------------------------------------------
# 6. Chebyshev 多项式求值 (源自 881_polpak/cheby_t_poly_zero, cheby_t_poly_values)
# ---------------------------------------------------------------------------

def chebyshev_T(n: int, x: float) -> float:
    """第一类 Chebyshev 多项式 T_n(x) 通过递推求值.
    T_0(x) = 1, T_1(x) = x, T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x).
    等价定义: T_n(cos θ) = cos(nθ).
    """
    if n == 0:
        return 1.0
    if n == 1:
        return float(x)
    t_prev2 = 1.0
    t_prev1 = float(x)
    for _ in range(2, n + 1):
        t_curr = 2.0 * x * t_prev1 - t_prev2
        t_prev2 = t_prev1
        t_prev1 = t_curr
    return t_prev1


def chebyshev_U(n: int, x: float) -> float:
    """第二类 Chebyshev 多项式 U_n(x).
    U_0=1, U_1=2x, U_{n+1}=2x U_n - U_{n-1}.
    等价定义: U_n(cos θ) = sin((n+1)θ)/sin(θ).
    """
    if n == 0:
        return 1.0
    if n == 1:
        return 2.0 * x
    u_prev2 = 1.0
    u_prev1 = 2.0 * x
    for _ in range(2, n + 1):
        u_curr = 2.0 * x * u_prev1 - u_prev2
        u_prev2 = u_prev1
        u_prev1 = u_curr
    return u_prev1


# ---------------------------------------------------------------------------
# 7. Chebyshev 级数求值与高阶导数 (源自 163_chebyshev_series)
#    Maess/Clenshaw 算法的推广
# ---------------------------------------------------------------------------

def chebyshev_series_eval(x: float, coef: np.ndarray) -> float:
    """Clenshaw 算法求值 Chebyshev 级数 f(x) = Σ c_k T_k(x).
    源自 163_chebyshev_series/echebser0.

    算法 (Maess 修正):
      b_{N+2} = b_{N+1} = 0
      b_k = 2x b_{k+1} - b_{k+2} + c_k,  k = N, N-1, ..., 1
      f(x) = b_1 x - b_2 + c_0

    数值稳定性: 对于 x ∈ [-1,1], 条件数 O(N).
    """
    nc = len(coef)
    if nc == 0:
        return 0.0
    if nc == 1:
        return float(coef[0])
    b0 = float(coef[nc - 1])
    b1 = 0.0
    b2 = 0.0
    x2 = 2.0 * x
    for i in range(nc - 2, 0, -1):
        b2 = b1
        b1 = b0
        b0 = x2 * b1 - b2 + coef[i]
    return x * b0 - b1 + coef[0]


def chebyshev_series_deriv(x: float, coef: np.ndarray) -> Tuple[float, float]:
    """Clenshaw 算法同时求值 Chebyshev 级数 f(x) 及其一阶导数 f'(x).
    源自 163_chebyshev_series/echebser1.

    递推同时维护:
      - b_k: 原始级数系数
      - d_k: 导数级数系数, d_k = 2 b_{k+1} + d_{k+2} · (适当修正)

    返回 (f(x), f'(x)).
    """
    nc = len(coef)
    if nc == 0:
        return 0.0, 0.0
    if nc == 1:
        return float(coef[0]), 0.0

    # 原始级数
    b0 = float(coef[nc - 1])
    b1 = 0.0
    b2 = 0.0

    # 导数级数: 对 Chebyshev 系数微分
    # 若 f = Σ c_k T_k, 则 f' = Σ d_k T_k
    # 其中 d_N = 0, d_{N-1} = 2N c_N, d_k = d_{k+2} + 2(k+1) c_{k+1}
    d = np.zeros(nc)
    if nc >= 2:
        d[nc - 2] = 2.0 * (nc - 1) * coef[nc - 1]
    for k in range(nc - 3, -1, -1):
        d[k] = d[k + 2] + 2.0 * (k + 1) * coef[k + 1]
    d[0] *= 0.5  # 首项减半修正

    # 同时 Clenshaw 求值
    x2 = 2.0 * x
    # f 部分
    for i in range(nc - 2, 0, -1):
        b2 = b1
        b1 = b0
        b0 = x2 * b1 - b2 + coef[i]
    fx = x * b0 - b1 + coef[0]

    # f' 部分
    d0 = d[nc - 1]
    d1 = 0.0
    d2 = 0.0
    for i in range(nc - 2, 0, -1):
        d2 = d1
        d1 = d0
        d0 = x2 * d1 - d2 + d[i]
    dfx = x * d0 - d1 + d[0]

    return fx, dfx


def chebyshev_series_deriv2(x: float, coef: np.ndarray) -> Tuple[float, float, float]:
    """Clenshaw 算法同时求值 f(x), f'(x), f''(x).
    源自 163_chebyshev_series/echebser2.

    二阶导数系数递推:
      e_N = e_{N-1} = 0
      e_k = e_{k+2} + 2(k+1) d_{k+1}, 其中 d 是一阶导数系数

    返回 (f(x), f'(x), f''(x)).
    """
    nc = len(coef)
    if nc == 0:
        return 0.0, 0.0, 0.0
    if nc == 1:
        return float(coef[0]), 0.0, 0.0

    # 一阶导数系数
    d = np.zeros(nc)
    if nc >= 2:
        d[nc - 2] = 2.0 * (nc - 1) * coef[nc - 1]
    for k in range(nc - 3, -1, -1):
        d[k] = d[k + 2] + 2.0 * (k + 1) * coef[k + 1]
    d[0] *= 0.5

    # 二阶导数系数
    e = np.zeros(nc)
    if nc >= 3:
        e[nc - 3] = 2.0 * (nc - 2) * d[nc - 2]
    for k in range(nc - 4, -1, -1):
        e[k] = e[k + 2] + 2.0 * (k + 1) * d[k + 1]
    e[0] *= 0.5

    # Clenshaw 求值三个级数
    def _clenshaw(coeffs):
        n = len(coeffs)
        if n == 0:
            return 0.0
        if n == 1:
            return float(coeffs[0])
        b0 = float(coeffs[n - 1])
        b1 = 0.0
        b2 = 0.0
        x2 = 2.0 * x
        for i in range(n - 2, 0, -1):
            b2 = b1
            b1 = b0
            b0 = x2 * b1 - b2 + coeffs[i]
        return x * b0 - b1 + coeffs[0]

    return _clenshaw(coef), _clenshaw(d), _clenshaw(e)


# ---------------------------------------------------------------------------
# 8. Chebyshev 系数计算 (离散余弦变换法)
# ---------------------------------------------------------------------------

def chebyshev_coefficients(f, a: float, b: float, n: int) -> np.ndarray:
    """计算函数 f 在 [a,b] 上的 n 阶 Chebyshev 展开系数.
    使用 Chebyshev 节点 x_k = cos(π(k+0.5)/n) 和 DCT-I:
        c_k = (2/n) Σ_{j=0}^{n-1} f(x_j) T_k(x_j),  c_0 减半

    映射: x ∈ [a,b] → t ∈ [-1,1], t = (2x - a - b)/(b - a).
    """
    nodes = np.cos(np.pi * (np.arange(n) + 0.5) / n)
    # 映射到 [a,b]
    x_nodes = 0.5 * (b - a) * nodes + 0.5 * (a + b)
    f_vals = np.array([f(xi) for xi in x_nodes])

    coef = np.zeros(n)
    for k in range(n):
        s = 0.0
        for j in range(n):
            s += f_vals[j] * chebyshev_T(k, nodes[j])
        coef[k] = 2.0 * s / n
    coef[0] *= 0.5
    return coef


# ---------------------------------------------------------------------------
# 9. sigma 除数函数 (源自 881_polpak/sigma_values, moebius)
# ---------------------------------------------------------------------------

def sigma_divisors(n: int) -> int:
    """除数函数 σ(n) = Σ_{d|n} d.
    积性: σ(uv) = σ(u)σ(v) 当 gcd(u,v)=1.
    对素数幂: σ(p^k) = (p^{k+1}-1)/(p-1).
    """
    if n <= 0:
        return 0
    total = 0
    for d in range(1, n + 1):
        if n % d == 0:
            total += d
    return total


def moebius_mu(n: int) -> int:
    """Möbius 函数 μ(n):
      μ(1) = 1
      μ(n) = 0 若 n 有平方因子
      μ(n) = (-1)^k 若 n 是 k 个不同素数之积
    """
    if n <= 0:
        return 0
    if n == 1:
        return 1
    # 质因数分解
    factors = []
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            factors.append(d)
            temp //= d
            if temp % d == 0:
                return 0  # 平方因子
        d += 1
    if temp > 1:
        factors.append(temp)
    return (-1) ** len(factors)


# ---------------------------------------------------------------------------
# 10. AGM 算术几何平均 (源自 881_polpak/r8_agm)
# ---------------------------------------------------------------------------

def agm(a: float, b: float, tol: float = 1e-15) -> float:
    """算术几何平均 AGM(a,b).
    Gauss 迭代: a_{n+1} = (a_n + b_n)/2, b_{n+1} = √(a_n b_n).
    二次收敛, 与椭圆积分关系:
        K(k) = π / (2 · AGM(1, √(1-k²)))
    其中 K 是第一类完全椭圆积分.
    """
    if a < 0 or b < 0:
        return float('nan')
    an, bn = float(a), float(b)
    for _ in range(100):
        an_new = 0.5 * (an + bn)
        bn_new = math.sqrt(max(an * bn, 0.0))
        if abs(an_new - bn_new) < tol * max(abs(an_new), 1e-300):
            break
        an, bn = an_new, bn_new
    return an_new


# ---------------------------------------------------------------------------
# 11. Hermite 多项式 (物理学家约定) —— 用于量子谐振子基函数
# ---------------------------------------------------------------------------

def hermite_phys(n: int, x: float) -> float:
    """物理学家 Hermite 多项式 H_n(x).
    递推: H_0=1, H_1=2x, H_{n+1}=2xH_n - 2nH_{n-1}.
    量子谐振子本征函数: ψ_n(x) = (2^n n! √π)^{-1/2} H_n(x) e^{-x²/2}.
    """
    if n == 0:
        return 1.0
    if n == 1:
        return 2.0 * x
    h0, h1 = 1.0, 2.0 * x
    for k in range(1, n):
        h2 = 2.0 * x * h1 - 2.0 * k * h0
        h0, h1 = h1, h2
    return h1


# ---------------------------------------------------------------------------
# 12. 完全椭圆积分 (通过 AGM)
# ---------------------------------------------------------------------------

def elliptic_K(k: float) -> float:
    """第一类完全椭圆积分 K(k) = ∫_0^{π/2} dθ/√(1-k²sin²θ).
    通过 AGM: K(k) = π/(2·AGM(1, √(1-k²))).
    """
    if abs(k) >= 1.0:
        return float('inf')
    kp = math.sqrt(max(1.0 - k * k, 0.0))
    return math.pi / (2.0 * agm(1.0, kp))


def elliptic_E(k: float) -> float:
    """第二类完全椭圆积分 E(k) = ∫_0^{π/2} √(1-k²sin²θ) dθ.
    数值积分实现 (Gauss-Legendre 4 点).
    """
    if abs(k) >= 1.0:
        return 1.0
    # Gauss-Legendre 5 点
    gl_nodes = np.array([0.0, -0.5384693101056831, 0.5384693101056831,
                         -0.9061798459386640, 0.9061798459386640])
    gl_weights = np.array([0.5688888888888889, 0.4786286704993665, 0.4786286704993665,
                           0.2369268850561891, 0.2369268850561891])
    k2 = k * k
    s = 0.0
    for i in range(5):
        theta = 0.5 * math.pi * (gl_nodes[i] + 1.0)
        integrand = math.sqrt(max(1.0 - k2 * math.sin(theta) ** 2, 0.0))
        s += gl_weights[i] * integrand
    return s * 0.5 * math.pi
