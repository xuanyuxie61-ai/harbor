#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py
水声传播抛物方程模型 — 数值工具与正交多项式基转换

本模块整合多项式基转换、索引管理与数值稳定性工具，来源于：
- 894_polynomial_conversion（正交多项式 ↔ 单项式转换矩阵）
- 1307_triangle_integrals（Pascal 三角索引、三角积分）
- 944_asa152（大数溢出保护、log-gamma 技巧）

核心数学公式：
1. Chebyshev 多项式三项递推：
   T₀(x) = 1,  T₁(x) = x
   Tₙ(x) = 2x·Tₙ₋₁(x) − Tₙ₋₂(x),  n ≥ 2

2. Legendre 多项式三项递推：
   P₀(x) = 1,  P₁(x) = x
   n·Pₙ(x) = (2n−1)x·Pₙ₋₁(x) − (n−1)·Pₙ₋₂(x)

3. Hermite 多项式三项递推：
   H₀(x) = 1,  H₁(x) = 2x
   Hₙ(x) = 2x·Hₙ₋₁(x) − 2(n−1)·Hₙ₋₂(x)

4. Gegenbauer (超球) 多项式：
   C₀^(λ)(x) = 1,  C₁^(λ)(x) = 2λx
   n·Cₙ^(λ) = 2(n+λ−1)x·Cₙ₋₁^(λ) − (n+2λ−2)·Cₙ₋₂^(λ)

5. Laguerre 多项式：
   L₀(x) = 1,  L₁(x) = 1−x
   n·Lₙ(x) = (2n−1−x)Lₙ₋₁(x) − (n−1)Lₙ₋₂(x)

6. Pascal 三角索引（二维多项式 x^i·y^j 的线性存储）：
   线性索引 k 与总次数 d=i+j 的关系：
   k = d(d+1)/2 + j + 1
   逆映射：给定 k，d = floor[(√(8k−7)−1)/2]，j = k − d(d+1)/2 − 1，i = d − j

7. 三角数公式：
   T_d = d(d+1)/2

8. 阶乘与 log-gamma 溢出保护：
   n! = exp(ln Γ(n+1))
   C(n,k) = exp(ln Γ(n+1) − ln Γ(k+1) − ln Γ(n−k+1))
"""

import numpy as np
from special_functions import gammaln_stable

# =============================================================================
# Pascal 索引与三角数
# =============================================================================

def pascal_to_i4(k):
    """
    将线性索引 k (1-based) 映射到二维多项式指数 (i, j)，
    满足 k = d(d+1)/2 + j + 1，其中 d = i+j。
    返回 (i, j) 为 0-based 指数。
    """
    k = int(k)
    if k < 1:
        raise ValueError("Pascal index k must be >= 1")
    d = int((np.sqrt(8 * k - 7) - 1) // 2)
    j = k - d * (d + 1) // 2 - 1
    i = d - j
    return i, j


def i4_to_pascal(i, j):
    """
    将 (i, j) 映射到线性 Pascal 索引 k (1-based)。
    """
    d = i + j
    return d * (d + 1) // 2 + j + 1


def triangle_number(d):
    """三角数 T_d = d(d+1)/2"""
    return d * (d + 1) // 2


# =============================================================================
# 组合数与多项式系数（带溢出保护）
# =============================================================================

def binomial_coefficient(n, k):
    """
    计算二项式系数 C(n,k) = n!/(k!(n−k)!)，使用 log-gamma 防止溢出。
    """
    if k < 0 or k > n or n < 0:
        return 0.0
    if k == 0 or k == n:
        return 1.0
    return float(np.exp(gammaln_stable(n + 1)
                        - gammaln_stable(k + 1)
                        - gammaln_stable(n - k + 1)))


def trinomial_coefficient(i, j, k):
    """
    三项式系数 (i+j+k)! / (i!·j!·k!)。
    """
    n = i + j + k
    if n < 0 or i < 0 or j < 0 or k < 0:
        return 0.0
    return float(np.exp(gammaln_stable(n + 1)
                        - gammaln_stable(i + 1)
                        - gammaln_stable(j + 1)
                        - gammaln_stable(k + 1)))


# =============================================================================
# 正交多项式 ↔ 单项式 转换矩阵
# =============================================================================

def chebyshev_to_monomial_matrix(n):
    """
    构建 Chebyshev T_k(x) → x^k 的 (n+1)×(n+1) 转换矩阵 M，
    满足 [1, x, x², ..., x^n]^T = M · [T₀, T₁, ..., T_n]^T
    实际返回的是单项式系数矩阵：monomial_coeffs = M @ chebyshev_coeffs。
    即 M[j,k] 为 T_k 展开后 x^j 的系数。
    """
    M = np.zeros((n + 1, n + 1), dtype=np.float64)
    M[0, 0] = 1.0
    if n >= 1:
        M[1, 1] = 1.0
    for k in range(2, n + 1):
        # T_k = 2x·T_{k-1} − T_{k-2}
        # x^j 系数：2·M[j-1, k-1] − M[j, k-2]
        for j in range(k, -1, -1):
            val = 0.0
            if j - 1 >= 0:
                val += 2.0 * M[j - 1, k - 1]
            val -= M[j, k - 2]
            M[j, k] = val
    return M


def monomial_to_chebyshev_matrix(n):
    """
    单项式 → Chebyshev 的转换矩阵（利用 Krogh 稳定算法原理）。
    返回矩阵 C 满足 cheb_coeffs = C @ mono_coeffs。
    算法基于递推求逆：利用 T_k 的显式系数递推求逆。
    """
    M = chebyshev_to_monomial_matrix(n)
    # M 是下三角且对角线为 1（T_k 的首项系数为 2^{k-1}，不是 1）
    # 实际上 T_k 首项系数为 2^{k-1} (k≥1)。因此 M 不是单位下三角。
    # 直接数值求逆（n 不大时完全稳定）
    return np.linalg.inv(M)


def legendre_to_monomial_matrix(n):
    """
    Legendre P_k(x) → x^k 的转换矩阵。
    递推：k·P_k = (2k−1)x·P_{k−1} − (k−1)·P_{k−2}
    """
    M = np.zeros((n + 1, n + 1), dtype=np.float64)
    M[0, 0] = 1.0
    if n >= 1:
        M[1, 1] = 1.0
    for k in range(2, n + 1):
        for j in range(k, -1, -1):
            val = 0.0
            if j - 1 >= 0:
                val += (2 * k - 1) * M[j - 1, k - 1]
            val -= (k - 1) * M[j, k - 2]
            M[j, k] = val / k
    return M


def monomial_to_legendre_matrix(n):
    """单项式 → Legendre 转换矩阵。"""
    M = legendre_to_monomial_matrix(n)
    return np.linalg.inv(M)


def hermite_to_monomial_matrix(n):
    """
    Hermite H_k(x) → x^k 的转换矩阵。
    递推：H_k = 2x·H_{k−1} − 2(k−1)·H_{k−2}
    """
    M = np.zeros((n + 1, n + 1), dtype=np.float64)
    M[0, 0] = 1.0
    if n >= 1:
        M[1, 1] = 2.0
    for k in range(2, n + 1):
        for j in range(k, -1, -1):
            val = 0.0
            if j - 1 >= 0:
                val += 2.0 * M[j - 1, k - 1]
            val -= 2.0 * (k - 1) * M[j, k - 2]
            M[j, k] = val
    return M


def gegenbauer_to_monomial_matrix(n, lam=0.5):
    """
    Gegenbauer C_k^{(λ)}(x) → x^k 的转换矩阵。
    递推：k·C_k = 2(k+λ−1)x·C_{k−1} − (k+2λ−2)·C_{k−2}
    默认 λ=0.5 对应 Legendre（相差常数倍）。
    """
    M = np.zeros((n + 1, n + 1), dtype=np.float64)
    M[0, 0] = 1.0
    if n >= 1:
        M[1, 1] = 2.0 * lam
    for k in range(2, n + 1):
        for j in range(k, -1, -1):
            val = 0.0
            if j - 1 >= 0:
                val += 2.0 * (k + lam - 1.0) * M[j - 1, k - 1]
            val -= (k + 2.0 * lam - 2.0) * M[j, k - 2]
            M[j, k] = val / k
    return M


def laguerre_to_monomial_matrix(n):
    """
    Laguerre L_k(x) → x^k 的转换矩阵。
    递推：k·L_k = (2k−1−x)L_{k−1} − (k−1)L_{k−2}
    即 k·L_k = (2k−1)L_{k−1} − x·L_{k−1} − (k−1)L_{k−2}
    """
    M = np.zeros((n + 1, n + 1), dtype=np.float64)
    M[0, 0] = 1.0
    if n >= 1:
        M[0, 1] = -1.0
        M[1, 1] = 1.0
    for k in range(2, n + 1):
        for j in range(k, -1, -1):
            val = (2 * k - 1) * M[j, k - 1]
            if j - 1 >= 0:
                val -= M[j - 1, k - 1]
            val -= (k - 1) * M[j, k - 2]
            M[j, k] = val / k
    return M


# =============================================================================
# 三角单元上的精确积分（参考元 → 物理元）
# =============================================================================

def triangle01_monomial_integral(i, j):
    """
    单位参考三角形 (0,0),(1,0),(0,1) 上 x^i·y^j 的精确积分：
    ∬_Δ r^i s^j dA = i! · j! / (i+j+2)!
    """
    if i < 0 or j < 0:
        return 0.0
    return float(np.exp(gammaln_stable(i + 1)
                        + gammaln_stable(j + 1)
                        - gammaln_stable(i + j + 3)))


def triangle_monomial_integral(i, j, t):
    """
    任意三角形 t = [(x1,y1), (x2,y2), (x3,y3)] 上的 x^i·y^j 积分。
    利用仿射映射：(x,y) = (x1,y1) + J·(r,s)，其中 J = [[x2−x1, x3−x1], [y2−y1, y3−y1]]
    Jacobian 行列式 |det(J)| = 2·Area。
    积分 = |det(J)| · ∬_Δ (x1 + a·r + b·s)^i (y1 + c·r + d·s)^j dr ds
    其中通过 poly_power_linear 与 poly_product 展开。
    """
    t = np.asarray(t, dtype=np.float64)
    if t.shape != (3, 2):
        raise ValueError("Triangle must be 3x2 array")
    x1, y1 = t[0]
    x2, y2 = t[1]
    x3, y3 = t[2]
    # 仿射系数
    a = x2 - x1
    b = x3 - x1
    c = y2 - y1
    d = y3 - y1
    jac = abs(a * d - b * c)
    if jac < 1e-15:
        return 0.0
    # 展开 (x1 + a*r + b*s)^i
    pi = _poly_power_linear_2d(x1, a, b, i)
    # 展开 (y1 + c*r + d*s)^j
    pj = _poly_power_linear_2d(y1, c, d, j)
    # 乘积
    p = _poly_product_2d(pi, pj)
    # 逐项积分
    total = 0.0
    for (exp_r, exp_s), coeff in p.items():
        total += coeff * triangle01_monomial_integral(exp_r, exp_s)
    return jac * total


def _poly_power_linear_2d(const, coef_r, coef_s, n):
    """
    展开 (const + coef_r·r + coef_s·s)^n 为字典 {(i,j): coeff}。
    使用三项式定理：
    (A + B·r + C·s)^n = Σ_{i+j+k=n} n!/(i!j!k!) · A^i · (Br)^j · (Cs)^k
    """
    if n < 0:
        return {}
    result = {}
    for j in range(n + 1):
        for k in range(n - j + 1):
            i = n - j - k
            coeff = trinomial_coefficient(i, j, k)
            coeff *= (const ** i) * (coef_r ** j) * (coef_s ** k)
            result[(j, k)] = result.get((j, k), 0.0) + coeff
    return result


def _poly_product_2d(p1, p2):
    """两个字典表示的二维多项式相乘。"""
    result = {}
    for (i1, j1), c1 in p1.items():
        for (i2, j2), c2 in p2.items():
            key = (i1 + i2, j1 + j2)
            result[key] = result.get(key, 0.0) + c1 * c2
    return result


# =============================================================================
# 数值稳定性工具
# =============================================================================

def log_sum_exp(a, b):
    """
    稳定计算 log(exp(a) + exp(b))，避免溢出：
    log(e^a + e^b) = max(a,b) + log(1 + exp(−|a−b|))
    """
    m = max(a, b)
    return m + np.log1p(np.exp(-abs(a - b)))


def log_sum_exp_array(arr):
    """对数组稳定计算 log(Σ exp(a_i))。"""
    arr = np.asarray(arr, dtype=np.float64)
    m = np.max(arr)
    return m + np.log(np.sum(np.exp(arr - m)))


def safe_divide(a, b, fill_value=0.0):
    """安全除法，b≈0 时返回 fill_value。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    result = np.full_like(a, fill_value, dtype=np.float64)
    mask = np.abs(b) > np.finfo(np.float64).eps * 100
    result[mask] = a[mask] / b[mask]
    return result
