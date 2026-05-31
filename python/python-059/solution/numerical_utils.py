"""
numerical_utils.py
博士级科学计算工具模块

整合原项目:
  - 094_bisection: 二分法根搜索
  - 1273_toms515: 组合数生成与阶乘计算
  - 029_asa053: 正态分布随机数生成

功能:
  为气溶胶辐射效应计算提供数值基础工具，包括方程求根、
  组合数学（用于粒径分档选择）、以及Wilson-Hilferty近似。
"""

import numpy as np
from math import lgamma, exp, sqrt, cos, sin, log, pi


class NumericalError(Exception):
    pass


def bisection(f, a, b, tol=1e-12, max_iter=100):
    """
    二分法求函数 f 在区间 [a, b] 上的根。

    数学原理:
      若 f(a) * f(b) < 0，则由介值定理可知存在 c ∈ (a, b)
      使得 f(c) = 0。通过不断将区间对半分，收敛速度为
      |b_n - a_n| = |b_0 - a_0| / 2^n。

    参数:
      f: 目标函数
      a, b: 初始区间端点
      tol: 容差
      max_iter: 最大迭代次数

    返回:
      (root, iterations)
    """
    fa = f(a)
    fb = f(b)

    if fa == 0.0:
        return float(a), 0
    if fb == 0.0:
        return float(b), 0

    if fa * fb > 0.0:
        raise NumericalError("bisection: 区间端点函数值同号，无法保证根存在。")

    it = 0
    while abs(b - a) > tol:
        if it >= max_iter:
            raise NumericalError(f"bisection: 超过最大迭代次数 {max_iter}")
        c = (a + b) / 2.0
        fc = f(c)
        it += 1
        if fc == 0.0:
            return float(c), it
        if np.sign(fc) == np.sign(fa):
            a = c
            fa = fc
        else:
            b = c
            fb = fc
    return float((a + b) / 2.0), it


def binomial_coefficient(n, k):
    """
    计算二项式系数 C(n, k) = n! / (k! (n-k)!).
    使用对数伽马函数保证大数稳定性:
      ln C(n,k) = ln Γ(n+1) - ln Γ(k+1) - ln Γ(n-k+1)
    """
    if k < 0 or k > n or n < 0:
        return 0.0
    if k == 0 or k == n:
        return 1.0
    # 使用对称性减少计算量
    k = min(k, n - k)
    val = exp(lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1))
    return val


def comb_lexicographic(n, p, l):
    """
    基于 toms515 的算法，按字典序索引 L 从 n 个元素中选取 p 个元素的组合。

    在气溶胶科学中用于: 从连续粒径分布中按最优字典序选取 p 个代表粒径档，
    以最小化数值离散化误差。

    参数:
      n: 集合大小
      p: 子集大小
      l: 字典序索引 (1 <= l <= C(n,p))

    返回:
      c: 长度为 p 的列表，包含选中的索引 (1-based)
    """
    if p <= 0 or p > n or n <= 0:
        raise NumericalError("comb_lexicographic: 参数非法")
    total = binomial_coefficient(n, p)
    if l < 1 or l > total:
        raise NumericalError(f"comb_lexicographic: 索引 l={l} 超出范围 [1, {total}]")

    c = [0] * p
    if p == 1:
        c[0] = l
        return c

    k = 0
    p1 = p - 1
    c[0] = 0

    for i in range(p1):
        if i > 0:
            c[i] = c[i - 1]
        while True:
            c[i] += 1
            r = binomial_coefficient(n - c[i], p - i - 1)
            k += r
            if l <= k:
                break
        k -= r

    c[p - 1] = c[p1 - 1] + l - k
    return c


def rnorm():
    """
    Box-Muller 变换生成两个独立的标准正态分布 N(0,1) 随机数。
    用于 Wishart 分布生成与蒙特卡洛模拟。

    数学公式:
      U1, U2 ~ Uniform(0,1)
      Z1 = sqrt(-2 ln U1) * cos(2π U2)
      Z2 = sqrt(-2 ln U1) * sin(2π U2)
    """
    u1 = np.random.rand()
    u2 = np.random.rand()
    # 避免 u1 = 0 导致 log(0)
    if u1 < 1e-15:
        u1 = 1e-15
    mag = sqrt(-2.0 * log(u1))
    z1 = mag * cos(2.0 * pi * u2)
    z2 = mag * sin(2.0 * pi * u2)
    return z1, z2


def gamma_log_values(n):
    """
    返回前 n 个整数的 ln Γ 值表，用于加速重复计算。
    """
    return np.array([lgamma(i + 1) for i in range(n + 1)])


def wilson_hilferty_chi_square(df, z):
    """
    Wilson-Hilferty 变换: 将标准正态变量 Z 转换为近似卡方分布的平方根。

    公式:
      χ² ≈ df * (1 - 2/(9df) + Z * sqrt(2/(9df)))^3

    在气溶胶统计中用于: 从多元正态生成样本协方差矩阵时
    近似卡方分布的对角元。
    """
    if df <= 0:
        raise NumericalError("wilson_hilferty_chi_square: df 必须为正")
    u1 = 2.0 / (9.0 * df)
    u2 = 1.0 - u1
    u1_sqrt = sqrt(u1)
    val = df * abs(u2 + z * u1_sqrt) ** 3
    return sqrt(val)


def safe_acos(x):
    """
    安全反余弦，将输入截断到 [-1, 1] 以避免浮点误差导致的 NaN。
    """
    x = max(-1.0, min(1.0, x))
    return np.arccos(x)
