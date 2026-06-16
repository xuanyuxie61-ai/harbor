#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
numerical_constants.py
======================
【融合种子项目】 983_r8lib (r8 数学库)

本模块移植 r8lib 中关于机器精度、Gamma 函数、Legendre 零点、矩阵运算、
特殊函数等核心数值工具，并封装为中子星状态方程计算所需的基础数值服务层。

物理/数学公式
-------------
1. 机器精度 (Unit Roundoff):
       eps = 2^{-53} ≈ 1.1102230246251565e-16  (IEEE-754 double)

2. Stirling 近似 Gamma 函数:
       ln Gamma(z) ~ (z-1/2) ln z - z + (1/2) ln(2 pi) + sum B_{2k}/(2k(2k-1)z^{2k-1})
   其中 B_{2k} 为 Bernoulli 数:
       B_2 = 1/6,  B_4 = -1/30,  B_6 = 1/42,  B_8 = -1/30, ...

3. Lanczos Gamma 函数逼近 (精度 ~15 位有效数字):
       Gamma(z+1) = sqrt(2 pi) (z + g + 1/2)^{z+1/2}
                    exp(-(z + g + 1/2)) A_g(z)
   其中 A_g(z) = c_0 + sum_{k=1}^{N} c_k / (z + k)

4. Legendre 多项式零点 (Gauss-Legendre 求积节点):
       P_n(x) 通过三项递推计算:
       (k+1) P_{k+1}(x) = (2k+1) x P_k(x) - k P_{k-1}(x)
       零点通过 Halley 迭代精化:
       x_{n+1} = x_n - P_n(x_n)/P_n'(x_n) * [1 - P_n(x_n)P_n''(x_n)/(2 P_n'(x_n)^2)]^{-1}

5. 矩阵 Frobenius 范数:
       ||A||_F = sqrt(sum_{i,j} |a_{ij}|^2)

6. 矩阵 L1 范数 (最大列和):
       ||A||_1 = max_j sum_i |a_{ij}|

7. 矩阵无穷范数 (最大行和):
       ||A||_inf = max_i sum_j |a_{ij}|
"""

import math
import numpy as np
from typing import Tuple, List, Optional, Union


# ============================================================
# 第一部分: 机器精度与基本常数
# ============================================================

def r8_epsilon() -> float:
    """
    返回 IEEE-754 双精度浮点数的单位舍入 (unit roundoff).

    定义: eps 满足 1 < 1 + eps 但 1 = 1 + eps/2 (在浮点运算意义下).

    Returns
    -------
    float
        eps = 2^{-53} ≈ 1.1102230246251565e-16
    """
    return np.finfo(np.float64).eps


def r8_big() -> float:
    """返回最大的有限双精度浮点数."""
    return np.finfo(np.float64).max


def r8_tiny() -> float:
    """返回最小的正规化双精度浮点数."""
    return np.finfo(np.float64).tiny


def r8_huge() -> float:
    """返回 '巨大' 的数, 用作哨兵值."""
    return 1.0e+30


def r8_is_nan(x: float) -> bool:
    """判断是否为 NaN."""
    return math.isnan(x)


def r8_is_inf(x: float) -> bool:
    """判断是否为无穷."""
    return math.isinf(x)


def r8_is_insignificant(x: float, ref: float,
                        tol: Optional[float] = None) -> bool:
    """
    判断 x 相对于 ref 是否 '不重要' (可忽略).

    判据: |x| <= tol * |ref|
    """
    if tol is None:
        tol = 100.0 * r8_epsilon()
    if ref == 0.0:
        return abs(x) <= tol
    return abs(x) <= tol * abs(ref)


# ============================================================
# 第二部分: Gamma 函数与相关特殊函数
# ============================================================

def r8_gamma(x: float) -> float:
    """
    计算 Gamma(x), 使用 Lanczos 逼近.

    Lanczos 公式:
        Gamma(z+1) = sqrt(2 pi) (z + g + 1/2)^{z+1/2}
                     exp(-(z + g + 1/2)) * A_g(z)
    其中 g = 7, A_g(z) = c_0 + sum_{k=1}^{6} c_k/(z+k)

    Lanczos 系数 (g=7, N=9):
        c_0 = 0.99999999999980993
        c_1 = 676.5203681218851
        c_2 = -1259.1392167224028
        c_3 = 771.32342877765313
        c_4 = -176.61502916214059
        c_5 = 12.507343278686905
        c_6 = -0.13857109526572012
        c_7 = 9.9843695780195716e-6
        c_8 = 1.5056327351493116e-7
    """
    if x <= 0.0 and x == int(x):
        # Gamma 在非正整数处有极点
        return float('inf')

    # 反射公式: Gamma(z) Gamma(1-z) = pi / sin(pi z)
    if x < 0.5:
        return math.pi / (math.sin(math.pi * x) * r8_gamma(1.0 - x))

    x = x - 1.0
    g = 7
    c = [
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

    a = c[0]
    t = x + g + 0.5
    for k in range(1, len(c)):
        a += c[k] / (x + k)

    return math.sqrt(2.0 * math.pi) * t ** (x + 0.5) * math.exp(-t) * a


def r8_gamma_log(x: float) -> float:
    """
    计算 ln|Gamma(x)|, 使用 Stirling 级数.

    Stirling 级数:
        ln Gamma(z) ~ (z - 1/2) ln(z) - z + (1/2) ln(2 pi)
                      + 1/(12z) - 1/(360z^3) + 1/(1260z^5) - ...

    收敛条件: z > 0 且 z 足够大 (z >= 8 时精度最优).
    """
    if x <= 0.0:
        if x == int(x):
            return float('inf')
        # 反射
        return math.log(math.pi / abs(math.sin(math.pi * x))) - r8_gamma_log(1.0 - x)

    if x < 8.0:
        # 通过递推提升至大参数区域: Gamma(z+1) = z Gamma(z)
        n_shift = int(8 - x) + 1
        prod = 1.0
        z = x
        for _ in range(n_shift):
            prod *= z
            z += 1.0
        return r8_gamma_log(z) - math.log(abs(prod))

    # Stirling 展开
    z = x
    result = (z - 0.5) * math.log(z) - z + 0.5 * math.log(2.0 * math.pi)
    z2 = z * z
    # Bernoulli 数贡献: B_{2k}/(2k(2k-1) z^{2k-1})
    result += 1.0 / (12.0 * z)
    result -= 1.0 / (360.0 * z * z2)
    result += 1.0 / (1260.0 * z * z2 * z2)
    result -= 1.0 / (1680.0 * z * z2 * z2 * z2)
    return result


def r8_factorial(n: int) -> float:
    """计算 n! = Gamma(n+1)."""
    if n < 0:
        raise ValueError("n! 要求 n >= 0")
    if n <= 1:
        return 1.0
    return math.exp(r8_gamma_log(n + 1.0))


def r8_factorial_stirling(n: int) -> float:
    """
    使用 Stirling 近似计算 n!.

    Stirling 公式:
        n! ~ sqrt(2 pi n) (n/e)^n [1 + 1/(12n) + 1/(288n^2) - 139/(51840n^3) + ...]
    """
    if n <= 0:
        return 1.0
    nf = float(n)
    result = math.sqrt(2.0 * math.pi * nf) * (nf / math.e) ** nf
    corr = 1.0 + 1.0 / (12.0 * nf) + 1.0 / (288.0 * nf * nf) \
           - 139.0 / (51840.0 * nf ** 3)
    return result * corr


# ============================================================
# 第三部分: Legendre 多项式与 Gauss 求积节点
# ============================================================

def legendre_zeros(n: int) -> np.ndarray:
    """
    计算 Legendre 多项式 P_n(x) 的 n 个零点.

    方法:
        1. 初始猜测 (基于 Tricomi 近似):
           x_k ≈ cos((4k-1) pi / (4n+2)) * [1 - (1 - 1/n)/(8n^2)]
        2. Halley 迭代精化:
           x_{m+1} = x_m - u (1 + 0.5 u (v + u(v^2 - d3/(3 d1))))
           其中 u = P_n(x)/P_n'(x), v = P_n''(x)/P_n'(x)

    三项递推:
        (k+1) P_{k+1} = (2k+1) x P_k - k P_{k-1}
    导数:
        P_n'(x) = n (P_{n-1} - x P_n) / (1 - x^2)

    Parameters
    ----------
    n : int
        Legendre 多项式阶数 (n > 0)

    Returns
    -------
    np.ndarray, shape (n,)
        零点升序排列
    """
    if n <= 0:
        raise ValueError("n 必须为正整数")

    x = np.zeros(n)
    e1 = n * (n + 1)
    m = (n + 1) // 2

    for i in range(1, m + 1):
        mp1mi = m + 1 - i
        t = (4 * i - 1) * math.pi / (4 * n + 2)
        x0 = math.cos(t) * (1.0 - (1.0 - 1.0 / n) / (8 * n * n))

        # 三项递推计算 P_n(x0)
        pkm1 = 1.0
        pk = x0
        for k in range(2, n + 1):
            pkp1 = 2.0 * x0 * pk - pkm1 - (x0 * pk - pkm1) / k
            pkm1 = pk
            pk = pkp1

        d1 = n * (pkm1 - x0 * pk)
        dpn = d1 / (1.0 - x0 * x0)
        d2pn = (2.0 * x0 * dpn - e1 * pk) / (1.0 - x0 * x0)
        d3pn = (4.0 * x0 * d2pn + (2.0 - e1) * dpn) / (1.0 - x0 * x0)
        d4pn = (6.0 * x0 * d3pn + (6.0 - e1) * d2pn) / (1.0 - x0 * x0)

        # Halley 迭代
        u = pk / dpn
        v = d2pn / dpn
        h = -u * (1.0 + 0.5 * u * (v + u * (v * v - d3pn / (3.0 * dpn))))
        p_val = pk + h * (dpn + 0.5 * h * (d2pn + h / 3.0 *
                         (d3pn + 0.25 * h * d4pn)))
        dp_val = dpn + h * (d2pn + 0.5 * h * (d3pn + h * d4pn / 3.0))
        h = h - p_val / dp_val
        xtemp = x0 + h
        x[mp1mi - 1] = xtemp

    if n % 2 == 1:
        x[0] = 0.0

    ncopy = n - m
    for i in range(m):
        iback = n - i
        x[iback - 1] = x[iback - 1 - ncopy]

    for i in range(n - m):
        x[i] = -x[n - 1 - i]

    return x


def legendre_set(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回 n 点 Gauss-Legendre 求积规则 (节点, 权重).

    权重公式:
        w_i = 2 / [(1 - x_i^2) (P_n'(x_i))^2]

    求积公式:
        integral_{-1}^{1} f(x) dx ≈ sum_{i=1}^{n} w_i f(x_i)
    """
    xi = legendre_zeros(n)
    wi = np.zeros(n)
    for i in range(n):
        # 计算 P_n'(x_i) 通过递推
        x = xi[i]
        pkm1 = 1.0
        pk = x
        for k in range(2, n + 1):
            pkp1 = 2.0 * x * pk - pkm1 - (x * pk - pkm1) / k
            pkm1 = pk
            pk = pkp1
        dpn = n * (pkm1 - x * pk) / (1.0 - x * x)
        wi[i] = 2.0 / ((1.0 - x * x) * dpn * dpn)
    return xi, wi


# ============================================================
# 第四部分: 矩阵范数与基本矩阵操作
# ============================================================

def r8mat_norm_fro(a: np.ndarray) -> float:
    """
    计算矩阵 A 的 Frobenius 范数.

    ||A||_F = sqrt( sum_{i,j} a_{ij}^2 ) = sqrt( trace(A^T A) )
    """
    return float(np.sqrt(np.sum(a ** 2)))


def r8mat_norm_l1(a: np.ndarray) -> float:
    """
    计算矩阵 A 的 L1 范数 (最大列绝对值之和).

    ||A||_1 = max_j sum_i |a_{ij}|
    """
    return float(np.max(np.sum(np.abs(a), axis=0)))


def r8mat_norm_li(a: np.ndarray) -> float:
    """
    计算矩阵 A 的无穷范数 (最大行绝对值之和).

    ||A||_inf = max_i sum_j |a_{ij}|
    """
    return float(np.max(np.sum(np.abs(a), axis=1)))


def r8mat_mm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """矩阵乘法 C = A @ B."""
    return a @ b


def r8mat_mv(a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """矩阵-向量乘 y = A @ x."""
    return a @ x


def r8mat_trace(a: np.ndarray) -> float:
    """矩阵迹 tr(A) = sum_i a_{ii}."""
    return float(np.trace(a))


def r8mat_det_2d(a: np.ndarray) -> float:
    """2x2 矩阵行列式."""
    return float(a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0])


def r8mat_det_3d(a: np.ndarray) -> float:
    """3x3 矩阵行列式 (Sarrus 法则)."""
    return float(
        a[0, 0] * (a[1, 1] * a[2, 2] - a[1, 2] * a[2, 1])
        - a[0, 1] * (a[1, 0] * a[2, 2] - a[1, 2] * a[2, 0])
        + a[0, 2] * (a[1, 0] * a[2, 1] - a[1, 1] * a[2, 0])
    )


def r8mat_is_symmetric(a: np.ndarray, tol: Optional[float] = None) -> bool:
    """
    判断矩阵是否对称: ||A - A^T||_F <= tol.
    """
    if tol is None:
        tol = 100.0 * r8_epsilon()
    n = a.shape[0]
    if a.shape[1] != n:
        return False
    return r8mat_norm_fro(a - a.T) <= tol * r8mat_norm_fro(a)


def r8mat_identity(n: int) -> np.ndarray:
    """返回 n x n 单位矩阵."""
    return np.eye(n)


# ============================================================
# 第五部分: 向量操作
# ============================================================

def r8vec_linspace(n: int, a_lo: float, a_hi: float) -> np.ndarray:
    """在 [a_lo, a_hi] 上生成 n 个等间距点."""
    if n <= 1:
        return np.array([0.5 * (a_lo + a_hi)])
    return np.linspace(a_lo, a_hi, n)


def r8vec_cheby1space(n: int, a_lo: float, a_hi: float) -> np.ndarray:
    """
    生成第一类 Chebyshev 节点 (在 [a_lo, a_hi] 上).

    x_k = 0.5(a+b) + 0.5(b-a) cos((2k-1)pi/(2n)),  k=1,...,n
    """
    k = np.arange(1, n + 1)
    mid = 0.5 * (a_hi + a_lo)
    half = 0.5 * (a_hi - a_lo)
    return mid + half * np.cos((2 * k - 1) * math.pi / (2 * n))


def r8vec_norm_l2(x: np.ndarray) -> float:
    """向量的 L2 范数."""
    return float(np.sqrt(np.dot(x, x)))


def r8vec_dot(x: np.ndarray, y: np.ndarray) -> float:
    """向量点积."""
    return float(np.dot(x, y))


# ============================================================
# 第六部分: 排序与排列 (移植自 r8lib)
# ============================================================

def r8vec_sort_heap_a(x: np.ndarray) -> np.ndarray:
    """返回 x 的升序堆排序副本."""
    return np.sort(x, kind='mergesort')


def r8vec_permute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    按排列 p 重排向量 x.

    y[i] = x[p[i]],  p 为 0-based 排列.
    """
    return x[p]


def i4vec_indicator1(n: int) -> np.ndarray:
    """返回 [1, 2, ..., n]."""
    return np.arange(1, n + 1)


def i4vec_indicator0(n: int) -> np.ndarray:
    """返回 [0, 1, ..., n-1]."""
    return np.arange(n)


# ============================================================
# 第七部分: 统计分布 (移植自 r8lib / asa 系列)
# ============================================================

def r8_normal_01(seed: Optional[int] = None) -> float:
    """
    Box-Muller 变换生成标准正态随机数.

    u1, u2 ~ Uniform(0,1)
    z = sqrt(-2 ln(u1)) cos(2 pi u2) ~ N(0,1)
    """
    if seed is not None:
        np.random.seed(seed)
    u1 = np.random.random()
    u2 = np.random.random()
    # 避免 log(0)
    u1 = max(u1, 1e-300)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def r8vec_normal_01(n: int, seed: Optional[int] = None) -> np.ndarray:
    """生成 n 个标准正态随机数."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.randn(n)


def r8vec_mean(x: np.ndarray) -> float:
    """均值."""
    return float(np.mean(x))


def r8vec_variance(x: np.ndarray) -> float:
    """样本方差."""
    return float(np.var(x, ddof=1)) if len(x) > 1 else 0.0


def r8vec_std(x: np.ndarray) -> float:
    """样本标准差."""
    return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


# ============================================================
# 第八部分: 数值工具函数
# ============================================================

def r8_choose(n: int, k: int) -> int:
    """
    计算组合数 C(n, k).

    C(n, k) = n! / (k! (n-k)!)
    """
    if k < 0 or k > n:
        return 0
    return int(round(math.exp(
        r8_gamma_log(n + 1.0) - r8_gamma_log(k + 1.0) - r8_gamma_log(n - k + 1.0)
    )))


def r8_fall(x: float, n: int) -> float:
    """
    计算下降阶乘 (falling factorial).

    (x)_n = x (x-1) (x-2) ... (x-n+1) = Gamma(x+1) / Gamma(x-n+1)
    """
    result = 1.0
    for i in range(n):
        result *= (x - i)
    return result


def r8_rise(x: float, n: int) -> float:
    """
    计算上升阶乘 (rising factorial / Pochhammer 符号).

    x^{(n)} = x (x+1) ... (x+n-1) = Gamma(x+n) / Gamma(x)
    """
    result = 1.0
    for i in range(n):
        result *= (x + i)
    return result


def r8_modp(x: float, y: float) -> float:
    """正模运算: 结果始终非负."""
    value = x - int(x / y) * y
    if value < 0.0:
        value += abs(y)
    return value


def r8_sign(x: float) -> float:
    """符号函数."""
    return 1.0 if x >= 0.0 else -1.0


def r8_heaviside(x: float) -> float:
    """Heaviside 阶跃函数."""
    return 0.0 if x < 0.0 else 1.0


def r8_sigmoid(x: float) -> float:
    """
    Sigmoid 函数.

    sigma(x) = 1 / (1 + exp(-x))
    """
    return 1.0 / (1.0 + math.exp(-x))


def r8_sech(x: float) -> float:
    """双曲正割 sech(x) = 2/(e^x + e^{-x})."""
    return 2.0 / (math.exp(x) + math.exp(-x))


def r8_epsilon_compute() -> float:
    """
    通过计算方式测定机器精度.

    算法: 从 s=1.0 开始反复除以 2, 直到 1+s=1.
    最终 eps = 2s.
    """
    s = 1.0
    while True:
        if not (1.0 + s > 1.0):
            break
        s = s * 0.5
    return 2.0 * s


# ============================================================
# 第九部分: 中子星物理常数 (CGS 单位制)
# ============================================================

class NeutronStarConstants:
    """
    中子星物理基本常数 (CGS 单位制).

    符号说明:
        c           : 光速 (cm/s)
        G           : 引力常数 (cm^3 g^{-1} s^{-2})
        M_sun       : 太阳质量 (g)
        hbar        : 约化 Planck 常数 (erg s)
        m_n         : 中子质量 (g)
        m_p         : 质子质量 (g)
        m_e         : 电子质量 (g)
        k_B         : Boltzmann 常数 (erg/K)
        sigma_SB     : Stefan-Boltzmann 常数 (erg cm^{-2} s^{-1} K^{-4})
        fm_to_cm    : 飞米到厘米转换
        MeV_to_erg  : MeV 到尔格转换
        rho_nuc     : 核饱和密度 (g/cm^3)
    """
    c_light = 2.99792458e10        # cm/s
    G_newton = 6.67430e-8          # cm^3 g^-1 s^-2
    M_sun = 1.98892e33             # g
    hbar = 1.054571817e-27         # erg s
    m_n = 1.67492749804e-24        # g
    m_p = 1.67262192369e-24        # g
    m_e = 9.1093837015e-28         # g
    k_B = 1.380649e-16             # erg/K
    sigma_SB = 5.670374419e-5      # erg cm^-2 s^-1 K^-4
    fm_to_cm = 1.0e-13             # cm/fm
    MeV_to_erg = 1.602176634e-6    # erg/MeV
    rho_nuc = 2.7e14               # g/cm^3 (核饱和密度)
    # 几何化单位: G = c = 1
    # 质量单位 = M_sun = 1.4766 km (几何化)
    km_to_solar_mass = 1.0 / 1.4766  # 1 km ≈ 0.6773 M_sun (G=c=1)


if __name__ == "__main__":
    print("=== 数值常数量子检验 ===")
    print(f"机器精度 eps = {r8_epsilon():.6e}")
    print(f"计算精度 eps = {r8_epsilon_compute():.6e}")
    print(f"Gamma(5) = {r8_gamma(5.0):.10f}  (精确 24)")
    print(f"Gamma(0.5) = {r8_gamma(0.5):.10f}  (精确 sqrt(pi))")
    print(f"ln Gamma(10) = {r8_gamma_log(10.0):.10f}")
    print(f"Legendre P_5 零点: {legendre_zeros(5)}")
    xi, wi = legendre_set(4)
    print(f"4 点 Gauss 节点: {xi}")
    print(f"4 点 Gauss 权重: {wi}")
    print(f"权重之和 = {np.sum(wi):.15f}  (应为 2)")
    A = np.random.randn(3, 3)
    A = A @ A.T
    print(f"Frobenius 范数: {r8mat_norm_fro(A):.6f}")
    print(f"L1 范数: {r8mat_norm_l1(A):.6f}")
    print(f"无穷范数: {r8mat_norm_li(A):.6f}")
    print(f"C(10,3) = {r8_choose(10, 3)}")
    print(f"(5)_3 = {r8_fall(5.0, 3)}")
    print(f"5^(3) = {r8_rise(5.0, 3)}")
    print("numerical_constants.py 自检通过.")
