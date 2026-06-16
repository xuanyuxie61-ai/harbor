# -*- coding: utf-8 -*-
"""
scientific_formulas.py
======================

博士级科学计算公式库：随机优化与样本平均近似 (SAA) 在随机亥姆霍兹
特征值优化问题中所用到的全部物理 / 数学 / 统计公式。

本模块集中放置所有常数、特殊函数、误差界与收敛速率的解析表达式，
保证 "公式-算法-代码" 三者的严格一致。
"""

import math
import cmath
from typing import Tuple

# ============================================================
# 1. 物理常数 (SI)
# ============================================================
SPEED_OF_SOUND_AIR = 343.0          # m/s, 20 摄氏度干空气
SPEED_OF_SOUND_WATER = 1482.0       # m/s
AIR_DENSITY = 1.2041                # kg/m^3
BOLTZMANN = 1.380649e-23            # J/K
PLANCK = 6.62607015e-34             # J*s
EPS_NUMERICAL = 2.220446049250313e-16   # IEEE-754 double epsilon


# ============================================================
# 2. 随机场 Karhunen-Loeve 展开的核函数与特征值
# ============================================================
def covariance_kernel(x1: Tuple[float, float],
                      x2: Tuple[float, float],
                      sigma2: float,
                      ell: float) -> float:
    """平方指数协方差核:
        C(x1, x2) = sigma^2 * exp(-||x1-x2||^2 / (2 * ell^2))
    这是随机亥姆霍兹波速场 a(x, omega) 的协方差模型。
    """
    dx = x1[0] - x2[0]
    dy = x1[1] - x2[1]
    r2 = dx * dx + dy * dy
    return sigma2 * math.exp(-r2 / (2.0 * ell * ell))


def kl_eigenvalue_1d(k: int, L: float, ell: float) -> float:
    """一维平方指数核在 [0,L] 上 KL 展开的特征值渐近 (近似):
        lambda_k ~= sigma^2 * ell * sqrt(pi) * exp(- (k*pi*ell/(2L))^2 )
    该式来自Mercer定理与傅里叶渐近。
    """
    arg = (k * math.pi * ell) / (2.0 * L)
    return math.exp(-arg * arg) * ell * math.sqrt(math.pi)


def kl_basis_1d(k: int, x: float, L: float) -> float:
    """一维 KL 基函数:  phi_k(x) = sqrt(2/L) * sin(k*pi*x/L).
    用于随机场展开: a(x,omega) = a0(x) + sum_k sqrt(lambda_k) xi_k phi_k(x).
    """
    return math.sqrt(2.0 / L) * math.sin(k * math.pi * x / L)


# ============================================================
# 3. 亥姆霍兹方程解析解相关的 Bessel 函数与零点
# ============================================================
def bessel_j_small(n: int, x: float, n_terms: int = 40) -> float:
    """第一类 Bessel 函数 J_n(x) 的级数求和:
        J_n(x) = sum_{m=0}^{inf} (-1)^m / (m! Gamma(m+n+1)) * (x/2)^{2m+n}
    对小自变量收敛良好，用于亥姆霍兹圆形膜解析验证。
    """
    s = 0.0
    half_x = 0.5 * x
    for m in range(n_terms):
        try:
            term = ((-1.0) ** m) * (half_x ** (2 * m + n)) / (
                math.factorial(m) * math.gamma(m + n + 1))
        except (OverflowError, ValueError):
            break
        s += term
        if abs(term) < EPS_NUMERICAL * 1e-3 and m > 5:
            break
    return s


def bessel_zero(n: int, m: int, bracket_step: float = 0.05,
                tol: float = 1e-10) -> float:
    """J_n(x) 的第 m 个正零点 rho_{m,n}, 用二分法求根。
    亥姆霍兹圆膜特征值 k_{m,n} = rho_{m,n}/a.
    """
    # 初始 bracket
    a = 0.1
    while bessel_j_small(n, a) * bessel_j_small(n, a + bracket_step) > 0:
        a += bracket_step
        if a > 100.0:
            return float('nan')
    b = a + bracket_step
    # 找到第 m 个零点需要跨过 m-1 个已有的零点
    # 简化: 直接从 a 出发做 m 次符号变化
    fa = bessel_j_small(n, a)
    count = 0
    x0 = a
    x = a + bracket_step * 0.1
    while count < m and x < 200.0:
        fx = bessel_j_small(n, x)
        if fa * fx <= 0:
            count += 1
            if count == m:
                # 二分细化
                lo, hi = x0, x
                for _ in range(80):
                    mid = 0.5 * (lo + hi)
                    fm = bessel_j_small(n, mid)
                    if fa * fm <= 0:
                        hi = mid
                    else:
                        lo = mid
                        fa = fm
                    if hi - lo < tol:
                        break
                return 0.5 * (lo + hi)
            x0 = x
            fa = fx
        x += bracket_step * 0.1
    return float('nan')


def helmholtz_exact_membrane(r: float, theta: float, a: float,
                             m: int, n: int,
                             gamma_amp: float = 1.0,
                             alpha: float = 1.0,
                             beta: float = 0.0) -> float:
    """亥姆霍兹方程圆膜精确解:
        Z(r, theta) = gamma * J_n(rho_{m,n} * r / a)
                      * [alpha * cos(n*theta) + beta * sin(n*theta)]
    用于 manufactured-solution 校验 FD 求解器精度。
    """
    rho = bessel_zero(n, m)
    if not math.isfinite(rho):
        return 0.0
    k = rho / max(a, 1e-12)
    Jn = bessel_j_small(n, k * r)
    angular = alpha * math.cos(n * theta) + beta * math.sin(n * theta)
    return gamma_amp * Jn * angular


# ============================================================
# 4. SAA 统计收敛与置信区间
# ============================================================
def saa_statistical_error_bound(sigma_f: float, N: int,
                                confidence: float = 0.95) -> float:
    """SAA 估计量的统计误差界:
        epsilon_N = z_{alpha/2} * sigma_f / sqrt(N)
    其中 z_{0.975} = 1.96. 这是 SAA 理论的核心大数定律速率 O(1/sqrt(N)).
    """
    z = {0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}.get(confidence, 1.96)
    return z * sigma_f / math.sqrt(max(N, 1))


def kluncer_bounds(L_const: float, mu_strong: float,
                   N_samples: int) -> Tuple[float, float]:
    """K'unneth-type 误差界: 设目标 f(x, xi) 关于 x 是 L-Lipschitz,
    强凸参数 mu, 则 SAA 解 x_N 满足:
        E[f(x_N) - f*] <= (L^2 / (2*mu)) * (1/N)  +  O(1/sqrt(N))
    返回 (偏差项, 方差项) 两项的渐近量级。
    """
    bias = (L_const ** 2) / (2.0 * max(mu_strong, 1e-12)) / max(N_samples, 1)
    variance = L_const / math.sqrt(max(N_samples, 1))
    return bias, variance


# ============================================================
# 5. 指数时间差分 (ETD) 相关的 phi 函数
# ============================================================
def etd_phi0(z: complex) -> complex:
    """phi_0(z) = exp(z), ETD 格式的基石。"""
    return cmath.exp(z)


def etd_phi1(z: complex) -> complex:
    """phi_1(z) = (exp(z) - 1) / z, 在 z->0 处退化为 1.
    ETD-RK4 中的关键整函数。
    """
    if abs(z) < 1e-8:
        return 1.0 + 0.5 * z + z * z / 6.0
    return (cmath.exp(z) - 1.0) / z


def etd_phi2(z: complex) -> complex:
    """phi_2(z) = (phi_1(z) - 1) / z, ETD 二阶校正项。"""
    if abs(z) < 1e-8:
        return 0.5 + z / 6.0 + z * z / 24.0
    return (etd_phi1(z) - 1.0) / z


# ============================================================
# 6. 多项式混沌 / Gram 正交多项式内积
# ============================================================
def gram_inner_product_discrete(f_vals, g_vals, weights) -> float:
    """离散 Gram 内积:  <f, g>_w = sum_i w_i f_i g_i.
    用于多项式混沌展开中随机输入空间的投影。
    """
    s = 0.0
    for fv, gv, w in zip(f_vals, g_vals, weights):
        s += w * fv * gv
    return s


def hermite_probabilist(n: int, x: float) -> float:
    """概率论 Hermite 多项式 He_n(x) ( physicist 版的缩放),
    三项递推: He_{n+1}(x) = x He_n(x) - n He_{n-1}(x).
    用于高斯随机输入下的多项式混沌基。
    """
    if n == 0:
        return 1.0
    if n == 1:
        return x
    h0, h1 = 1.0, x
    for k in range(1, n):
        h2 = x * h1 - k * h0
        h0, h1 = h1, h2
    return h1


# ============================================================
# 7. 强凸 / 光滑性常数相关的收敛速率
# ============================================================
def sgd_convergence_rate(L_smooth: float, mu_strong: float,
                         sigma_noise: float, t: int) -> float:
    """强凸光滑目标下 SGD 的收敛速率:
        E[f(x_t) - f*] <= (2 * L * ||x_0 - x*||^2) / (t^2)
                           + sigma^2 / (mu * t)
    此处返回右侧第二项 (主导的方差项), 体现 O(1/t) 速率。
    """
    return (sigma_noise ** 2) / (max(mu_strong, 1e-14) * max(t, 1))


def variance_reduction_factor(batch_size: int,
                              corr: float) -> float:
    """带相关样本的方差缩减因子:
        VR = (1 + (batch_size - 1) * corr) / batch_size
    corr=0 即 i.i.d. 样本 -> VR = 1/N (标准 SAA 方差缩减).
    """
    bs = max(batch_size, 1)
    return (1.0 + (bs - 1) * max(min(corr, 1.0), -1.0)) / bs
