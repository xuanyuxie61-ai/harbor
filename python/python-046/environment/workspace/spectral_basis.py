"""
spectral_basis.py
Legendre-Hermite 混合谱基函数模块。

融合种子项目:
  - 661_legendre_polynomial: Legendre 多项式递推、Gauss-Legendre 节点
  - 524_hermite_product_polynomial: 概率论 Hermite 多项式系数

在 InSAR 形变反演中的应用:
  1. Legendre 多项式用于断层走向-倾向方向的球谐展开，修正地球曲率；
  2. Hermite 多项式用于构造形变随机场的高斯过程基函数，实现随机正则化；
  3. 混合展开用于高效参数化大尺度滑动分布。
"""

import numpy as np
from utils import check_finite


def legendre_polynomial_values(m, n, x):
    """
    计算 Legendre 多项式 P_0(x), ..., P_n(x) 在 m 个样本点上的值。

    递推公式:
        P_0(x) = 1
        P_1(x) = x
        P_n(x) = [(2n-1)*x*P_{n-1}(x) - (n-1)*P_{n-2}(x)] / n

    正交性:
        ∫_{-1}^{1} P_i(x) P_j(x) dx = 2/(2i+1) * δ_{ij}

    参数:
        m: 样本点数
        n: 最高阶数
        x: shape (m,), 评估点，要求在 [-1, 1] 内

    返回:
        v: shape (m, n+1), 各阶多项式值
    """
    x = np.asarray(x).reshape(-1)
    if np.min(x) < -1.0 - 1e-12 or np.max(x) > 1.0 + 1e-12:
        raise ValueError("legendre_polynomial_values: x must be in [-1, 1]")
    v = np.zeros((m, n + 1))
    v[:, 0] = 1.0
    if n >= 1:
        v[:, 1] = x
    for j in range(2, n + 1):
        v[:, j] = ((2.0 * j - 1.0) * x * v[:, j - 1] -
                   (j - 1.0) * v[:, j - 2]) / j
    check_finite(v, "legendre_polynomial_values")
    return v


def legendre_polynomial_derivative(m, n, x):
    """
    计算 Legendre 多项式导数 P'_n(x)。

    递推:
        P'_0(x) = 0
        P'_1(x) = 1
        P'_n(x) = [(2n-1)*(P_{n-1}(x) + x*P'_{n-1}(x)) - (n-1)*P'_{n-2}(x)] / n
    """
    x = np.asarray(x).reshape(-1)
    v = legendre_polynomial_values(m, n, x)
    dp = np.zeros((m, n + 1))
    if n >= 1:
        dp[:, 1] = 1.0
    for j in range(2, n + 1):
        dp[:, j] = ((2.0 * j - 1.0) * (v[:, j - 1] + x * dp[:, j - 1]) -
                    (j - 1.0) * dp[:, j - 2]) / j
    check_finite(dp, "legendre_polynomial_derivative")
    return dp


def hermite_probabilist_coefficients(n):
    """
    计算概率论 Hermite 多项式 He_n(x) 的系数。

    递推关系:
        He_0(x) = 1
        He_1(x) = x
        He_n(x) = x * He_{n-1}(x) - (n-1) * He_{n-2}(x)

    返回:
        c: 系数数组（降幂排列）
    """
    if n < 0:
        raise ValueError("hermite_probabilist_coefficients: n >= 0 required")
    # 使用完整系数表
    ct = np.zeros((n + 1, n + 1))
    ct[0, 0] = 1.0
    if n >= 1:
        ct[1, 1] = 1.0
    for i in range(2, n + 1):
        ct[i, 0] = -(i - 1) * ct[i - 2, 0]
        for k in range(1, i + 1):
            ct[i, k] = ct[i - 1, k - 1] - (i - 1) * ct[i - 2, k]
    c = ct[n, :n + 1]
    return c


def hermite_probabilist_value(n, x):
    """
    计算 He_n(x) 的值（概率论 Hermite 多项式）。

    正交权函数: w(x) = exp(-x^2/2) / sqrt(2π)
    正交性: ∫_{-∞}^{∞} He_i(x) He_j(x) w(x) dx = i! * δ_{ij}
    """
    x = np.asarray(x)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()
    h_prev2 = np.ones_like(x)
    h_prev1 = x.copy()
    for i in range(2, n + 1):
        h_curr = x * h_prev1 - (i - 1) * h_prev2
        h_prev2 = h_prev1
        h_prev1 = h_curr
    return h_prev1


def hermite_probabilist_values_array(max_n, x):
    """
    计算 He_0(x) ... He_max_n(x)，返回 shape (len(x), max_n+1)。
    """
    x = np.asarray(x).reshape(-1)
    m = len(x)
    v = np.zeros((m, max_n + 1))
    v[:, 0] = 1.0
    if max_n >= 1:
        v[:, 1] = x
    for i in range(2, max_n + 1):
        v[:, i] = x * v[:, i - 1] - (i - 1) * v[:, i - 2]
    return v


def mixed_legendre_hermite_basis_2d(x, y, n_leg, n_herm):
    """
    构建二维混合谱基函数：Legendre(x) * Hermite(y)。

    用于将断层滑动分布 m(x, y) 在走向-深度域展开为：
        m(x, y) ≈ Σ_{i=0}^{n_leg} Σ_{j=0}^{n_herm} c_{ij} * P_i(x) * He_j(y)

    参数:
        x: 走向方向坐标，归一化到 [-1, 1]
        y: 深度方向坐标（km），需先标准化
        n_leg: Legendre 最高阶
        n_herm: Hermite 最高阶

    返回:
        B: shape (len(x), (n_leg+1)*(n_herm+1))
    """
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if len(x) != len(y):
        raise ValueError("mixed_legendre_hermite_basis_2d: x and y must have same length")
    m = len(x)
    P = legendre_polynomial_values(m, n_leg, x)
    # 对 y 进行标准化，使 He_j 有合理尺度
    y_std = y / (np.std(y) + 1e-10)
    H = hermite_probabilist_values_array(n_herm, y_std)
    B = np.zeros((m, (n_leg + 1) * (n_herm + 1)))
    idx = 0
    for i in range(n_leg + 1):
        for j in range(n_herm + 1):
            B[:, idx] = P[:, i] * H[:, j]
            idx += 1
    return B


def gauss_legendre_quadrature_weights(n):
    """
    返回 n 点 Gauss-Legendre 的节点和权重，用于谱方法中的内积计算。
    """
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w
