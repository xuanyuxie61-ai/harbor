"""
spectral_time.py
================================================================================
时间方向谱离散与 Gauss-Lobatto 方法模块

本模块融合以下种子项目的核心算法：
  - 693_lobatto_polynomial : Legendre 多项式递推、Lobatto 多项式及其导数

科学背景
--------
在最优控制伴随方程方法中，时间离散精度直接影响状态方程与伴随方程的
相容性，进而影响梯度计算的准确性。传统的一阶隐式欧拉或二阶 Crank-Nicolson
方法在长时间积分或高频振荡问题中精度不足。

Gauss-Lobatto-Legendre (GLL) 谱元方法将时间区间 [0,T] 划分为若干单元，
在每个单元内部使用 GLL 节点进行 Lagrange 插值，可获得谱精度收敛
（误差以 e^{-N} 衰减，N 为单元内节点数）。GLL 节点包含两个端点，
天然保证单元间的连续性，非常适合状态方程的前向积分和伴随方程的后向积分。

关键公式
--------
1. Legendre 多项式递推（三项递推）:
   P₀(x) = 1,  P₁(x) = x
   (n+1) P_{n+1}(x) = (2n+1) x P_n(x) − n P_{n-1}(x)

2. Lobatto 多项式:
   Lo_n(x) = (1 − x²) P'_n(x) = n [P_{n-1}(x) − x P_n(x)]
   性质：Lo_n(±1) = 0，是标准 Sturm-Liouville 问题的特征函数。

3. Gauss-Lobatto 节点：Lo_n(x) = 0 的 n−1 个内根，加上 x=±1，
   共 n+1 个节点 {ξ_j}_{j=0}^n。

4. GLL 质量矩阵（时间方向）:
   M_{ij} = ∫_{-1}^{1} φ_i(ξ) φ_j(ξ) dξ = w_i δ_{ij}
   其中 w_i 为 GLL 积分权重（利用 Legendre-Gauss-Lobatto 求积精确性）。

5. GLL 刚度矩阵（时间方向）:
   S_{ij} = ∫_{-1}^{1} φ'_i(ξ) φ_j(ξ) dξ
   这里 φ_i 是第 i 个 Lagrange 基函数。
"""

import numpy as np


def legendre_polynomial(n, x):
    """
    计算 n 阶 Legendre 多项式 P_n(x) 在所有点 x 上的值。
    使用稳定的三项递推关系。
    """
    x = np.atleast_1d(x)
    if n < 0:
        return np.zeros_like(x)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()

    p0 = np.ones_like(x)
    p1 = x.copy()
    for k in range(1, n):
        p2 = ((2.0 * k + 1.0) * x * p1 - k * p0) / (k + 1.0)
        p0, p1 = p1, p2
    return p1


def legendre_polynomial_derivative(n, x):
    """
    计算 P'_n(x)。使用递推：
    (1 − x²) P'_n(x) = n [P_{n-1}(x) − x P_n(x)]
    对于 |x| ≈ 1，改用递推避免除零：
    P'_{n+1} = (2n+1) P_n + P'_{n-1}
    """
    x = np.atleast_1d(x)
    if n < 0:
        return np.zeros_like(x)
    if n == 0:
        return np.zeros_like(x)
    if n == 1:
        return np.ones_like(x)

    # 递推 P'_n
    dp0 = np.zeros_like(x)
    dp1 = np.ones_like(x)
    for k in range(1, n):
        dp2 = (2.0 * k + 1.0) * legendre_polynomial(k, x) + dp0
        dp0, dp1 = dp1, dp2
    return dp1


def lobatto_polynomial(n, x):
    """
    计算已完成（completed）Lobatto 多项式 Lo_n(x)。
    Lo_n(x) = (1 − x²) P'_n(x) = n [P_{n-1}(x) − x P_n(x)]
    在 x = ±1 处自动为零。
    """
    x = np.atleast_1d(x)
    if n <= 0:
        return np.zeros_like(x)
    val = n * (legendre_polynomial(n - 1, x) - x * legendre_polynomial(n, x))
    return val


def lobatto_polynomial_derivative(n, x):
    """
    计算 Lo'_n(x)。对 Lo_n(x) = n[P_{n-1}(x) − x P_n(x)] 求导：
    Lo'_n = n [P'_{n-1} − P_n − x P'_n]
    """
    x = np.atleast_1d(x)
    if n <= 0:
        return np.zeros_like(x)
    dp_nm1 = legendre_polynomial_derivative(n - 1, x)
    p_n = legendre_polynomial(n, x)
    dp_n = legendre_polynomial_derivative(n, x)
    return n * (dp_nm1 - p_n - x * dp_n)


def gll_nodes_weights(n):
    """
    计算 n+1 个 Gauss-Lobatto-Legendre (GLL) 节点与权重。
    节点包含 x₀ = −1 和 x_n = +1。

    算法：
    1. 用 Newton 法求 Lo_n(x) = 0 的 n−1 个内根。
    2. 权重公式：w_j = 2 / [n(n+1) P_n(x_j)²]
    """
    if n < 1:
        raise ValueError("gll_nodes_weights: 要求 n ≥ 1")

    # 初始猜测：Chebyshev 节点（在内部）
    k = np.arange(1, n)
    x_init = -np.cos(np.pi * k / n)

    x_inner = x_init.copy()
    for _ in range(100):
        f = lobatto_polynomial(n, x_inner)
        df = lobatto_polynomial_derivative(n, x_inner)
        dx = f / (df + 1.0e-30)
        x_inner -= dx
        if np.max(np.abs(dx)) < 1.0e-14:
            break

    nodes = np.concatenate(([-1.0], np.sort(x_inner), [1.0]))

    # 计算权重
    p_n_vals = legendre_polynomial(n, nodes)
    weights = 2.0 / (n * (n + 1.0) * p_n_vals ** 2)

    return nodes, weights


def lagrange_derivative_matrix(nodes):
    """
    构造 Lagrange 插值基函数在 GLL 节点处的导数矩阵 D_{ij} = φ'_j(x_i)。
    使用重心公式（Barycentric formula）计算，具有 O(N²) 复杂度且数值稳定。

    公式：
    D_{ij} = w_j / w_i · 1 / (x_i − x_j)   (i ≠ j)
    D_{ii} = − Σ_{j≠i} D_{ij}
    其中 w_i 为重心权重，对于 Legendre 节点可取 w_i = (−1)^i √(1−x_i²) 的近似。
    更准确地，使用通用的差分公式：
    D_{ij} = Π_{k≠j} (x_i − x_k) / Π_{k≠j} (x_j − x_k)
    """
    n = len(nodes)
    D = np.zeros((n, n), dtype=float)

    # 计算重心权重
    w = np.ones(n, dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j:
                w[i] *= (nodes[i] - nodes[j])
        w[i] = 1.0 / w[i]

    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = w[j] / (w[i] * (nodes[i] - nodes[j]))
    for i in range(n):
        D[i, i] = -np.sum(D[i, :])

    return D


def build_gll_time_operators(n, T=1.0):
    """
    在时间区间 [0, T] 上构造 GLL 谱元离散的时间质量矩阵 M_t 和刚度矩阵 S_t。
    将参考区间 [−1, 1] 通过仿射变换映射到 [0, T]：
        t = T/2 · (ξ + 1),  dt = T/2 · dξ

    返回
    ----
    nodes_t : 物理时间节点
    M_t     : 质量矩阵（对角），M_t[ii] = w_i · T/2
    S_t     : 刚度矩阵，S_t[i,j] = ∫₀^T φ'_i(t) φ_j(t) dt
    D_t     : 导数矩阵，D_t[i,j] = φ'_j(t_i)
    """
    nodes_ref, weights_ref = gll_nodes_weights(n)

    # 仿射变换到 [0, T]
    nodes_t = 0.5 * T * (nodes_ref + 1.0)
    scale = 0.5 * T

    # 质量矩阵（对角）
    M_t = np.diag(weights_ref * scale)

    # 参考区间上的导数矩阵
    D_ref = lagrange_derivative_matrix(nodes_ref)
    # 物理区间上的导数矩阵：d/dt = (1/scale) d/dξ
    D_t = D_ref / scale

    # 刚度矩阵 S_t[i,j] = ∫ φ'_i φ_j dt
    # 在 GLL 节点上，φ_j(t_i) = δ_{ij}，所以积分用 GLL 求积：
    # S_t = diag(weights_ref * scale) @ D_t
    S_t = M_t @ D_t

    return nodes_t, M_t, S_t, D_t
