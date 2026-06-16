# -*- coding: utf-8 -*-
"""
grid_integer_lib.py
====================
整数运算库 — 等离子体模拟中的离散数学基础

融合种子项目:
  568_i4lib  —  John Burkardt 的 400+ 整数工具库 (阶乘/组合/位运算)

物理背景
--------
在 Vlasov–Poisson 系统的高阶有限差分离散中，整数运算出现在：
  1. Fornberg 算法计算差分权系数时的组合数 C(n,k)
  2. 位移算子 E^a f(x) = f(x+ah) 的二项式展开
  3. 截断误差分析中 (2k-1)!! 双阶乘
  4. 周期性边界条件中的模运算
  5. FFT 位逆序置换

数学基础
--------
二项式系数:  C(n,k) = n! / (k! (n-k)!)
双阶乘:     (2k-1)!! = 1·3·5·...·(2k-1)
Plasma dispersion function Z(ζ):  Landau 阻尼理论核心函数
"""

import math
import numpy as np
from typing import List


# ===== 阶乘与组合数 ========================================================

def i4_factorial(n: int) -> int:
    """
    计算 n! = 1·2·...·n

    等离子体应用中：
      - Hermite 多项式归一化: H_n(v/v_th) exp(-v²/(2v_th²))
      - 有限差分截断误差常数: C_P = h^P · f^(P+1) / (P+1)!
    """
    if n < 0:
        raise ValueError("i4_factorial: n must be >= 0")
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def i4_choose(n: int, k: int) -> int:
    """
    二项式系数 C(n,k) = n! / (k!(n-k)!)

    用于 Fornberg 有限差分权系数算法中的组合展开。
    """
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def i4_double_factorial(n: int) -> int:
    """
    双阶乘 n!! = n · (n-2) · (n-4) · ...

    在 Maxwellian 速度矩中：
      ∫ v^{2k} exp(-v²/v_th²) dv = √π · v_th^{2k+1} · (2k-1)!! / 2^k
    """
    if n < 0:
        raise ValueError("i4_double_factorial: n must be >= 0")
    if n <= 1:
        return 1
    result = n
    while n > 2:
        n -= 2
        result *= n
    return result


# ===== 等离子体色散函数 ====================================================

def plasma_dispersion_z(zeta: complex) -> complex:
    """
    等离子体色散函数 Z(ζ) — Fried & Conte (1961)

    .. math::
        Z(\\zeta) = \\frac{1}{\\sqrt{\\pi}}
                    \\int_{-\\infty}^{\\infty}
                    \\frac{\\exp(-t^2)}{t - \\zeta} dt,
        \\quad \\mathrm{Im}(\\zeta) > 0

    解析延拓至下半平面：
        Z(ζ) = Z_+(ζ) + 2i√π exp(-ζ²),   Im(ζ) < 0

    物理意义：
      - 线性 Vlasov 理论中色散关系 D(ω,k) = 1 + (1/k²λ_D²)[1 + ζZ(ζ)]
      - D(ω,k) = 0 的根给出 Langmuir 波频率和 Landau 阻尼率
    """
    zeta = complex(zeta)
    x = zeta.real
    y = zeta.imag

    # 大参数渐近展开: Z(ζ) ~ -1/ζ Σ (2n-1)!! / (2ζ²)^n
    if abs(zeta) > 6.0:
        z2 = zeta * zeta
        result = 0.0 + 0.0j
        term = 1.0 / zeta
        for n in range(15):
            coeff = float(i4_double_factorial(max(2 * n - 1, 0)))
            contrib = term * coeff
            result += contrib
            term = term / (2.0 * z2)
            if abs(contrib) < 1.0e-15 * abs(result) + 1e-300:
                break
        return -result

    # 小参数 Taylor 级数:
    # Z(ζ) = i√π exp(-ζ²) - 2ζ[1 - 2ζ²/3 + 4ζ⁴/15 - ...]
    if abs(zeta) < 3.0:
        z2 = zeta * zeta
        series_sum = 0.0 + 0.0j
        term = 1.0 + 0.0j
        for n in range(30):
            series_sum += term
            term *= -2.0 * z2 / (2.0 * n + 3.0)
        result = (1.0j * math.sqrt(math.pi) * np.exp(-z2)
                  - 2.0 * zeta * series_sum)
        if y < 0.0:
            result += 2.0j * math.sqrt(math.pi) * np.exp(-z2)
        return result

    # 中等参数：数值积分
    n_quad = 128
    t_vals = np.linspace(-10.0, 10.0, n_quad)
    dt = t_vals[1] - t_vals[0]
    integrand = np.exp(-t_vals ** 2) / (t_vals - zeta + 0j)
    result = np.sum(integrand) * dt / math.sqrt(math.pi)
    if y < 0.0:
        result += 2.0j * math.sqrt(math.pi) * np.exp(-z2)
    return result


# ===== 位运算 ==============================================================

def i4_bit_hi1(n: int) -> int:
    """
    最高有效 1-位位置 (1-indexed).

    用于 FFT 尺寸确定和二进制树索引。
    """
    if n <= 0:
        return 0
    bit = 0
    i = abs(n)
    while i > 0:
        bit += 1
        i >>= 1
    return bit


def i4_bit_lo0(n: int) -> int:
    """最低有效 0-位位置 (1-indexed)，用于 Gray 码生成."""
    i = abs(n)
    bit = 1
    while (i & 1) == 1:
        i >>= 1
        bit += 1
    return bit


def i4_bit_reverse(n: int, nbits: int) -> int:
    """
    位逆序：Cooley-Tukey FFT 中的基本操作。

    在谱方法 Poisson 求解器中，位逆序置换用于原位 FFT 计算。
    """
    result = 0
    for _ in range(nbits):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result


# ===== 模运算与周期性 ======================================================

def i4_modp(i: int, j: int) -> int:
    """非负模运算，结果在 [0, j)。用于周期性边界条件。"""
    return ((i % j) + j) % j


def i4_wrap(i: int, lo: int, hi: int) -> int:
    """将整数 i 回绕到周期区间 [lo, hi]。用于周期性差分模板索引。"""
    width = hi - lo + 1
    if width <= 0:
        return lo
    return lo + i4_modp(i - lo, width)


def i4_gcd(a: int, b: int) -> int:
    """最大公约数 (Euclidean 算法)，用于网格公倍数分析。"""
    a, b = abs(int(a)), abs(int(b))
    while b:
        a, b = b, a % b
    return a


# ===== 随机数生成器 ========================================================

class IntegerRNG:
    """
    线性同余随机数生成器 (LCG)

    递推: X_{n+1} = (a·X_n + c) mod m
    参数 (Numerical Recipes): a = 1664525, c = 1013904223, m = 2^32

    用于 Maxwellian 初始分布的 Monte Carlo 粒子加载。
    """

    def __init__(self, seed: int = 12345):
        self._state = int(seed) & 0xFFFFFFFF
        self._a = 1664525
        self._c = 1013904223
        self._m = 2 ** 32

    def next_int(self) -> int:
        """生成下一个 [0, 2^32) 整数。"""
        self._state = (self._a * self._state + self._c) % self._m
        return self._state

    def next_uniform(self) -> float:
        """生成 [0, 1) 均匀分布随机数。"""
        return self.next_int() / self._m

    def next_normal(self) -> float:
        """Box-Muller 变换生成 N(0,1) 正态样本。"""
        u1 = max(self.next_uniform(), 1.0e-30)
        u2 = self.next_uniform()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def next_maxwellian(self, v_th: float) -> float:
        """
        Maxwellian 速度分布采样:
          f(v) = (1/(√(2π) v_th)) exp(-v²/(2v_th²))
        """
        return v_th * self.next_normal()

    def get_state(self) -> int:
        """返回当前状态用于检查点。"""
        return self._state

    def set_state(self, state: int) -> None:
        """从检查点恢复状态。"""
        self._state = int(state) & 0xFFFFFFFF
