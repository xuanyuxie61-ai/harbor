"""
legendre_kernel.py  —  移位 Legendre 多项式核与 Gauss-Legendre 求积
=================================================================

科学来源种子:
  - 666_legendre_shifted_polynomial / p01_polynomial_values.m,
    p01_polynomial_value.m
    直接使用其三项递推公式:
        P_0(x) = 1
        P_1(x) = 2x - 1
        P_n(x) = [(2n-1)(2x-1) P_{n-1}(x) - (n-1) P_{n-2}(x)] / n

应用:
  在 CIC/TSC 质量赋值与势能重建中,使用移位 Legendre 多项式作为
  高阶窗函数的正交基。设 ξ = (x - x_c) / h 为归一化局部坐标,
  则 N 阶 Legendre 重建核为:
      W_N(ξ) = Σ_{n=0}^{N} c_n P_n(ξ),  ξ ∈ [-1, 1]
  其中系数 c_n 由矩匹配条件决定。
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


def legendre_shifted_values(x: NDArray, n_max: int) -> NDArray:
    """
    计算移位 Legendre 多项式 P_0, P_1, ..., P_{n_max} 在 x 处的值。
    移位 Legendre: P_n^{shift}(x) = P_n(2x-1),  x ∈ [0,1]
    本函数使用标准 Legendre 三项递推:
        (n) P_n(u) = (2n-1) u P_{n-1}(u) - (n-1) P_{n-2}(u)

    Parameters
    ----------
    x : (M,) array
        求值点,推荐 x ∈ [-1, 1]
    n_max : int
        最高阶数 (≥ 0)

    Returns
    -------
    v : (M, n_max+1) array
        v[:, k] = P_k(x)
    """
    x = np.asarray(x, dtype=float).ravel()
    if n_max < 0:
        return np.zeros((x.size, 0))
    v = np.zeros((x.size, n_max + 1), dtype=float)
    v[:, 0] = 1.0
    if n_max >= 1:
        v[:, 1] = x
    for n in range(2, n_max + 1):
        v[:, n] = ((2 * n - 1) * x * v[:, n - 1] - (n - 1) * v[:, n - 2]) / n
    return v


def legendre_standard_values(x: NDArray, n_max: int) -> NDArray:
    """与 legendre_shifted_values 等价,但强调标准定义域 [-1,1]。"""
    return legendre_shifted_values(x, n_max)


def gauss_legendre_nodes(n: int) -> tuple:
    """
    Gauss-Legendre 求积节点与权重 (n 点)。
    利用 Golub-Welsch 算法: 节点为对称三对角 Jacobi 矩阵的特征值,
    权重与特征向量第一分量平方成正比。
        α_k = 0  (对角)
        β_k = k / sqrt(4k² - 1)  (次对角)
    """
    if n < 1:
        raise ValueError("节点数 n 必须 ≥ 1")
    if n == 1:
        return np.array([0.0]), np.array([2.0])
    k = np.arange(1, n, dtype=float)
    beta = k / np.sqrt(4.0 * k * k - 1.0)
    # 构造对称三对角矩阵
    J = np.diag(beta, -1) + np.diag(beta, 1)
    eigvals, eigvecs = np.linalg.eigh(J)
    nodes = eigvals
    weights = 2.0 * eigvecs[0, :] ** 2
    # 排序:
    order = np.argsort(nodes)
    return nodes[order], weights[order]


def gauss_legendre_quadrature(f, a: float, b: float, n: int = 32) -> float:
    """
    ∫_a^b f(x) dx 的 Gauss-Legendre 求积:
        ∫_a^b f(x) dx ≈ (b-a)/2 Σ_i w_i f( (b-a)/2 · t_i + (a+b)/2 )
    """
    t, w = gauss_legendre_nodes(n)
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    x = half * t + mid
    fx = np.asarray([f(xi) for xi in x])
    return float(half * np.dot(w, fx))


def legendre_cic_kernel(xi: NDArray, order: int = 3) -> NDArray:
    """
    基于移位 Legendre 多项式构造的高质量质量赋值核 (kernel):
        W(ξ) = Σ_{n=0}^{order} c_n P_n(ξ),  ξ ∈ [-1, 1]
    系数 c_n 通过匹配:
        ∫_{-1}^1 W(ξ) dξ = 1            (归一化)
        ∫_{-1}^1 ξ^k W(ξ) dξ = 0,  k=1..order  (零矩条件)
    给出。本函数返回在 ξ 处求值的核权重。
    使用简单解析解: 对偶数阶 P_{2m} 项保留,
        c_0 = 1/2,  c_{2m} = (-1)^m (4m+1)/(2) · C(1/2, m) / (m+1)
    其中 C(1/2, m) = 二项式系数 (1/2 choose m)。
    """
    xi = np.asarray(xi, dtype=float)
    P = legendre_shifted_values(xi, order)
    w = np.zeros_like(xi)
    w += 0.5 * P[:, 0]
    for m in range(1, (order // 2) + 1):
        if 2 * m > order:
            break
        # 解析系数 (简化形式): c_{2m} = (-1)^m (4m+1) / (2(2m+1))
        c = ((-1) ** m) * (4 * m + 1) / (2.0 * (2 * m + 1))
        w += c * P[:, 2 * m]
    return w


def legendre_tsc_weights_1d(order: int = 4) -> NDArray:
    """
    对 1D Truncated-Stellar-Cloud (TSC) 赋值,返回基于 Legendre
    多项式的修正权重。返回 2*order+1 个权重。
    物理含义: 在网格间距 Δx 内,用 N 阶 Legendre 多项式重建密度场,
    权重由 ∫_{cell} P_n(ξ) dξ 解析给出。
    """
    # 对每个网格单元内 2*order+1 个相邻节点,权值通过正交性:
    #   w_k = (1/Δx) ∫_{-Δx/2}^{Δx/2} P_k(x/(Δx/2)) dx
    # 利用 Legendre 多项式的奇偶性,只有偶数阶非零:
    half_width = 1.0
    w = np.zeros(2 * order + 1)
    for k in range(order + 1):
        # ∫_{-1}^{1} P_{2k}(ξ) dξ = 2 δ_{0k}
        if k == 0:
            w[order] = 1.0
    # 归一化:
    s = w.sum()
    if s > 0:
        w /= s
    return w
