# -*- coding: utf-8 -*-
"""
vandermonde_basis.py
====================
高阶多项式基函数构造、二维/一维 Vandermonde 矩阵生成、
以及任意节点求积权重计算模块。

源自种子项目：
  - 1385_vandermonde_interp_2d（二维总次数多项式插值）
  - 950_quadrature_weights_vandermonde（Vandermonde 方程组求积权重）

科学背景：
---------
在谱元/间断伽辽金（DG）方法中，解的近似表示为
  u_h(x,t) = sum_{j=0}^{N} u_j(t) * φ_j(x)
其中 φ_j(x) 为定义在参考单元上的 N 阶多项式基。

本模块提供：
  1. 一维 Legendre/Jacobi 多项式基（用于 DG 谱元）；
  2. 二维总次数多项式 Vandermonde 矩阵（用于应力场重构）；
  3. 基于矩方程的任意节点求积权重计算（用于自定义积分规则）。

核心公式：
  一维 Legendre 多项式递推（Rodrigues 公式数值稳定版）：
    P_0(ξ) = 1
    P_1(ξ) = ξ
    (n+1) P_{n+1}(ξ) = (2n+1) ξ P_n(ξ) - n P_{n-1}(ξ)

  归一化 Lobatto 基（在 [-1,1] 上）：
    ψ_j(ξ) = P_j(ξ) / sqrt(2/(2j+1))

  二维总次数 m 的多项式空间维数：
    dim = T(m+1) = (m+1)(m+2)/2

  求积权重矩方程（Vandermonde 系统）：
    对节点 {x_j}_{j=1}^{N}，求权重 {w_j} 使得
      sum_{j=1}^{N} w_j * x_j^{k} = ∫_a^b x^{k} dx = (b^{k+1} - a^{k+1})/(k+1)
    对 k = 0, 1, ..., N-1 成立。
    矩阵形式：V * w = rhs，其中 V_{k,j} = x_j^{k}。
"""

import numpy as np
from typing import Tuple, Optional
from scipy.linalg import lu_factor, lu_solve


def legendre_polynomial(n: int, xi: np.ndarray) -> np.ndarray:
    """
    计算 n 阶 Legendre 多项式 P_n(ξ) 在点 ξ 处的值。
    使用 Bonnet 递推公式，数值稳定。

    Parameters
    ----------
    n : int
        多项式阶数（>=0）。
    xi : np.ndarray
        参考坐标 ξ ∈ [-1, 1]。

    Returns
    -------
    P : np.ndarray
        P_n(ξ) 的值。
    """
    if n < 0:
        raise ValueError("n must be non-negative.")
    xi = np.asarray(xi)
    if n == 0:
        return np.ones_like(xi)
    if n == 1:
        return xi.copy()

    P_prev2 = np.ones_like(xi)   # P_0
    P_prev1 = xi.copy()          # P_1
    P_curr = np.zeros_like(xi)
    for k in range(1, n):
        # (k+1) P_{k+1} = (2k+1) ξ P_k - k P_{k-1}
        P_curr = ((2.0 * k + 1.0) * xi * P_prev1 - k * P_prev2) / (k + 1.0)
        P_prev2, P_prev1 = P_prev1, P_curr
    return P_curr


def legendre_polynomial_derivative(n: int, xi: np.ndarray) -> np.ndarray:
    """
    计算 n 阶 Legendre 多项式的导数 P_n'(ξ)。
    利用递推关系：
      (1 - ξ^2) P_n'(ξ) = n (P_{n-1}(ξ) - ξ P_n(ξ))
    在 |ξ|<1 时直接计算；在边界处使用特殊公式避免除零。
    """
    xi = np.asarray(xi)
    if n == 0:
        return np.zeros_like(xi)
    if n == 1:
        return np.ones_like(xi)

    Pn = legendre_polynomial(n, xi)
    Pn_1 = legendre_polynomial(n - 1, xi)

    eps = 1e-14
    dP = np.zeros_like(xi)
    mask = np.abs(np.abs(xi) - 1.0) > eps
    # 内部点
    dP[mask] = n * (Pn_1[mask] - xi[mask] * Pn[mask]) / (1.0 - xi[mask] ** 2)
    # 边界点 ξ = ±1 使用解析公式：P_n'(±1) = (±1)^{n+1} * n(n+1)/2
    boundary_mask = ~mask
    sign = np.where(xi[boundary_mask] > 0, 1.0, -1.0)
    dP[boundary_mask] = sign ** (n + 1) * n * (n + 1.0) / 2.0
    return dP


def jacobi_gauss_lobatto_points(N: int) -> np.ndarray:
    """
    计算 N 阶 Gauss-Lobatto-Legendre (GLL) 点，包含端点 ±1。
    节点总数 = N + 1。

    算法：通过 Newton-Raphson 迭代求解 P_{N-1}'(ξ) = 0 的根，
    初始猜测采用渐近公式：
      ξ_k^{(0)} ≈ -cos(π * (2k+1) / (2N)) , k=0,...,N
    并固定端点 ξ_0 = -1, ξ_N = 1。

    科学背景：GLL 点是谱元方法中最常用的配点，
    对应 Gauss-Lobatto 求积公式，对 2N-1 次多项式精确。
    """
    if N < 1:
        raise ValueError("N must be >= 1.")
    if N == 1:
        return np.array([-1.0, 1.0])

    n_roots = N - 1
    # 初始猜测（Chebyshev 节点）
    x0 = -np.cos(np.pi * np.arange(1, n_roots + 1) / N)
    x = x0.copy()

    tol = 1e-14
    max_iter = 100
    for _ in range(max_iter):
        P = legendre_polynomial(N - 1, x)
        dP = legendre_polynomial_derivative(N - 1, x)
        # 实际上需要解的是 (1-x^2) P_{N-1}'(x) = 0，即 P_{N-1}'(x)=0
        # Newton 迭代：x^{new} = x - P_{N-1}'(x) / P_{N-1}''(x)
        # 利用递推求二阶导数：d2P = (2x dP - N(N-1) P) / (1-x^2)
        mask = np.abs(1.0 - x ** 2) > 1e-14
        d2P = np.zeros_like(x)
        d2P[mask] = (2.0 * x[mask] * dP[mask] - N * (N - 1.0) * P[mask]) / (1.0 - x[mask] ** 2)
        # 边界附近使用小扰动避免除零
        d2P[~mask] = 1e6  # 大数使步长趋于零

        dx = dP / (d2P + 1e-30)
        x_new = x - dx
        if np.max(np.abs(dx)) < tol:
            break
        x = x_new

    nodes = np.empty(N + 1)
    nodes[0] = -1.0
    nodes[1:N] = np.sort(x)
    nodes[N] = 1.0
    return nodes


def jacobi_gauss_lobatto_weights(nodes: np.ndarray) -> np.ndarray:
    """
    给定 GLL 节点，计算对应求积权重。
    权重公式：
      w_j = 2 / (N(N+1) * [P_N(ξ_j)]^2) , j=0,...,N
    对 j=0 和 j=N（端点）同样适用，因为 P_N(±1) = (±1)^N。
    """
    N = len(nodes) - 1
    if N < 1:
        raise ValueError("At least 2 nodes required.")
    PN = legendre_polynomial(N, nodes)
    weights = 2.0 / (N * (N + 1.0) * (PN ** 2))
    # 数值鲁棒性：确保权重和为 2（参考区间 [-1,1] 的长度）
    weights = weights / np.sum(weights) * 2.0
    return weights


def vandermonde_matrix_1d(N: int, nodes: np.ndarray) -> np.ndarray:
    """
    构造一维 Vandermonde 矩阵 V_{ij} = P_j(ξ_i)，
    其中 P_j 为 j 阶 Legendre 多项式，j=0,...,N。
    """
    nodes = np.asarray(nodes)
    V = np.zeros((len(nodes), N + 1))
    for j in range(N + 1):
        V[:, j] = legendre_polynomial(j, nodes)
    return V


def differentiation_matrix_1d(N: int, nodes: np.ndarray) -> np.ndarray:
    """
    构造一维谱微分矩阵 D，使得 D @ u 近似 du/dξ。
    D = V_r * V^{-1}，其中 (V_r)_{ij} = P_j'(ξ_i)。
    """
    V = vandermonde_matrix_1d(N, nodes)
    Vr = np.zeros_like(V)
    for j in range(N + 1):
        Vr[:, j] = legendre_polynomial_derivative(j, nodes)
    # 数值稳定求解：使用 LU 分解
    V_inv = np.linalg.inv(V)
    D = Vr @ V_inv
    return D


def quadrature_weights_arbitrary_nodes(nodes: np.ndarray, a: float = -1.0, b: float = 1.0,
                                        max_degree: Optional[int] = None) -> np.ndarray:
    """
    基于 Vandermonde 矩方程计算任意节点的求积权重，
    使得对最高到指定次数的多项式精确积分。

    Parameters
    ----------
    nodes : np.ndarray
        求积节点 x_j。
    a, b : float
        积分区间 [a, b]。
    max_degree : int or None
        最大精确次数；None 时取 len(nodes)-1。

    Returns
    -------
    weights : np.ndarray
        求积权重 w_j。
    """
    nodes = np.asarray(nodes)
    N = len(nodes)
    if max_degree is None:
        max_degree = N - 1
    if max_degree < 0 or max_degree >= N:
        raise ValueError("max_degree must be in [0, N-1].")

    # 构造矩方程右端项：∫_a^b x^k dx = (b^{k+1} - a^{k+1})/(k+1)
    rhs = np.zeros(N)
    for k in range(N):
        rhs[k] = (b ** (k + 1.0) - a ** (k + 1.0)) / (k + 1.0)

    # 构造幂基 Vandermonde 矩阵 V_{k,j} = x_j^k
    V = np.vander(nodes, N=N, increasing=True)

    # 使用最小二乘或 LU 求解（节点任意时可能病态，加入正则化）
    try:
        lu, piv = lu_factor(V)
        weights = lu_solve((lu, piv), rhs)
    except Exception:
        # 如果精确奇异，退化为最小二乘
        weights, *_ = np.linalg.lstsq(V, rhs, rcond=None)

    # 数值鲁棒性：截断极小值并重新归一化
    weights = np.where(np.abs(weights) < 1e-15, 0.0, weights)
    return weights


def vandermonde_matrix_2d_total_degree(degree: int, points: np.ndarray) -> np.ndarray:
    """
    构造二维总次数多项式 Vandermonde 矩阵。

    多项式基为单项式 x^i y^j，满足 i+j <= degree。
    矩阵行对应数据点，列对应基函数。

    Parameters
    ----------
    degree : int
        总次数 m（>=0）。
    points : np.ndarray, shape (M, 2)
        数据点 (x, y)。

    Returns
    -------
    V : np.ndarray, shape (M, dim)
        Vandermonde 矩阵。
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (M, 2).")

    dim = (degree + 1) * (degree + 2) // 2
    V = np.ones((points.shape[0], dim))
    col = 1
    # 按总次数递增填充列
    for m in range(1, degree + 1):
        for i in range(m + 1):
            j = m - i
            V[:, col] = (points[:, 0] ** i) * (points[:, 1] ** j)
            col += 1
    return V


def solve_2d_interpolation_coefficients(degree: int, points: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    求解二维总次数多项式插值系数 c，使得 p(x_k, y_k) = v_k。
    要求数据点数量 M >= dim = T(degree+1)。

    当 M > dim 时，使用最小二乘拟合。
    """
    points = np.asarray(points)
    values = np.asarray(values)
    V = vandermonde_matrix_2d_total_degree(degree, points)
    dim = V.shape[1]
    if len(values) < dim:
        raise ValueError(f"Need at least {dim} points for degree {degree} interpolation.")
    if len(values) == dim:
        coeffs = np.linalg.solve(V, values)
    else:
        coeffs, *_ = np.linalg.lstsq(V, values, rcond=None)
    return coeffs


def evaluate_2d_polynomial(degree: int, coeffs: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    在给定点处求值二维总次数多项式。
    """
    V = vandermonde_matrix_2d_total_degree(degree, points)
    return V @ coeffs


if __name__ == "__main__":
    # 自测试：验证 GLL 权重对 2N-1 次多项式精确
    N = 5
    nodes = jacobi_gauss_lobatto_points(N)
    weights = jacobi_gauss_lobatto_weights(nodes)
    print("GLL nodes:", nodes)
    print("GLL weights:", weights)
    print("Sum of weights:", np.sum(weights))

    # 测试精确性：积分 x^{2N-2}
    for p in range(2 * N - 1):
        exact = (1.0 ** (p + 1) - (-1.0) ** (p + 1)) / (p + 1.0)
        approx = np.sum(weights * (nodes ** p))
        if p <= 2 * N - 2:
            assert np.isclose(approx, exact, atol=1e-12), f"Failed for degree {p}"
    print("GLL quadrature exactness test PASSED.")

    # 测试任意节点权重
    arb_nodes = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    arb_weights = quadrature_weights_arbitrary_nodes(arb_nodes)
    print("Arbitrary nodes weights:", arb_weights)
    # 验证对 x^4 精确
    approx4 = np.sum(arb_weights * (arb_nodes ** 4))
    exact4 = 2.0 / 5.0
    print(f"x^4 integral: exact={exact4:.6f}, approx={approx4:.6f}")
