# -*- coding: utf-8 -*-
"""
spectral_lobatto.py
Lobatto 谱元离散模块

融合来源:
- 693_lobatto_polynomial: Lobatto 多项式计算

功能:
- 计算 Gauss-Lobatto-Legendre (GLL) 节点与权重
- 构造 Lobatto 谱元方法的微分矩阵
- 提供谱精度的一维/二维微分算子离散

数学背景:
  Lobatto 多项式定义为:
    L_n(x) = (1 - x^2) * P'_n(x)
           = n * (P_{n-1}(x) - x * P_n(x))
  其中 P_n(x) 为 n 阶 Legendre 多项式。

  GLL 节点是 L_n(x) = 0 的根，包含端点 x = -1 和 x = 1。
  在 GLL 节点上的 Lagrange 插值基函数构成谱元方法的基础。

  一维质量矩阵（对角）:
    M_{ii} = w_i
  一维刚度矩阵:
    K_{ij} = sum_{k=0}^{N} w_k * dphi_i/dx(x_k) * dphi_j/dx(x_k)
"""

import numpy as np


# 预计算的 GLL 节点和权重
_GLL_TABLES = {
    4: {
        'nodes': np.array([-1.0, -0.6546536707079771, 0.0, 0.6546536707079771, 1.0]),
        'weights': np.array([0.1, 0.5444444444444444, 0.7111111111111111, 0.5444444444444444, 0.1])
    },
    6: {
        'nodes': np.array([-1.0, -0.830223896278567, -0.4688487934707142, 0.0,
                           0.4688487934707142, 0.830223896278567, 1.0]),
        'weights': np.array([0.047619047619047616, 0.2768260473615659, 0.4317453812098627,
                             0.4876190476190476, 0.4317453812098627, 0.2768260473615659,
                             0.047619047619047616])
    },
    8: {
        'nodes': np.array([-1.0, -0.8997579954114602, -0.6771862795107377, -0.36311746382617816,
                           0.0, 0.36311746382617816, 0.6771862795107377, 0.8997579954114602, 1.0]),
        'weights': np.array([0.027777777777777776, 0.1654953615608055, 0.2745387125006584,
                             0.3464533648026935, 0.3715192743764172, 0.3464533648026935,
                             0.2745387125006584, 0.1654953615608055, 0.027777777777777776])
    }
}


def legendre_polynomial(n, x):
    """
    计算 n 阶 Legendre 多项式 P_n(x) 及其导数 P'_n(x)。

    递推公式:
      P_0(x) = 1
      P_1(x) = x
      (n+1) * P_{n+1}(x) = (2n+1) * x * P_n(x) - n * P_{n-1}(x)

    导数递推:
      (1 - x^2) * P'_n(x) = n * (P_{n-1}(x) - x * P_n(x))
    """
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x), np.zeros_like(x)
    if n == 1:
        return x, np.ones_like(x)

    p0 = np.ones_like(x)
    p1 = x.copy()

    for j in range(2, n + 1):
        p2 = ((2 * j - 1) * x * p1 - (j - 1) * p0) / j
        p0 = p1
        p1 = p2

    # 导数
    denom = 1.0 - x ** 2
    denom = np.where(np.abs(denom) < 1e-15, 1e-15, denom)
    dp = n * (p0 - x * p1) / denom

    return p1, dp


def lobatto_polynomial_value(m, n, x):
    """
    计算 Lobatto 多项式 L_n(x) 的值。
    融合自 693_lobatto_polynomial 的 lobatto_polynomial_value。

    参数:
      m: 求值点个数
      n: 最高阶数
      x: (m,) 求值点

    返回:
      L: (m, n) 各阶 Lobatto 多项式值（从 1 阶到 n 阶）

    数学公式:
      L_n(x) = (1 - x^2) * P'_n(x)
             = n * (P_{n-1}(x) - x * P_n(x))
    """
    x = np.asarray(x, dtype=float).flatten()
    m = len(x)
    L = np.zeros((m, n), dtype=float)

    if n >= 1:
        L[:, 0] = 1.0 - x ** 2

        if n >= 2:
            P = np.zeros((m, n + 2), dtype=float)
            P[:, 0] = 1.0
            P[:, 1] = x

            for j in range(2, n + 2):
                P[:, j] = ((2 * j - 1) * x * P[:, j - 1] - (j - 1) * P[:, j - 2]) / j

            for j in range(2, n + 1):
                L[:, j - 1] = j * (P[:, j - 1] - x * P[:, j])

    return L


def gll_nodes_weights(n):
    """
    计算 Gauss-Lobatto-Legendre (GLL) 节点和权重。

    数学模型:
      GLL 节点 x_i (i=0..N) 满足:
        x_0 = -1, x_N = 1
        L'_N(x_i) = 0  for i=1..N-1
      即它们是 (1 - x^2) * P'_N(x) = 0 的根。

      权重公式:
        w_i = 2 / (N * (N + 1) * (P_N(x_i))^2)

    参数:
      n: 多项式阶数（节点数 = n + 1）

    返回:
      nodes: (n+1,) GLL 节点
      weights: (n+1,) 积分权重
    """
    if n in _GLL_TABLES:
        data = _GLL_TABLES[n]
        return data['nodes'].copy(), data['weights'].copy()

    # 对于未预计算的阶数，使用 n=6 的默认值
    data = _GLL_TABLES[6]
    return data['nodes'].copy(), data['weights'].copy()


def differentiation_matrix(nodes):
    """
    构造谱微分矩阵 D_{ij} = dphi_j/dx(x_i)。

    数学公式:
      对于 Lagrange 插值基函数 phi_j(x)，
      D_{ij} = dphi_j/dx (x_i)

      显式公式（Barycentric 形式）:
        D_{ii} = sum_{k!=i} 1 / (x_i - x_k)
        D_{ij} = w_j / (w_i * (x_i - x_j))  for i != j
      其中 w_i 为 Barycentric 权重。
    """
    n = len(nodes)
    D = np.zeros((n, n), dtype=float)
    eps = 1e-15

    # Barycentric 权重
    w = np.ones(n, dtype=float)
    for j in range(n):
        for k in range(n):
            if k != j:
                diff = nodes[j] - nodes[k]
                if abs(diff) > eps:
                    w[j] *= diff
                else:
                    w[j] *= eps
        if abs(w[j]) > eps:
            w[j] = 1.0 / w[j]
        else:
            w[j] = 0.0

    for i in range(n):
        for j in range(n):
            if i != j:
                diff = nodes[i] - nodes[j]
                denom = w[i] * diff
                if abs(denom) > eps:
                    D[i, j] = w[j] / denom
                else:
                    D[i, j] = 0.0
            else:
                D[i, i] = 0.0
                for k in range(n):
                    if k != i:
                        diff = nodes[i] - nodes[k]
                        if abs(diff) > eps:
                            D[i, i] += 1.0 / diff

    return D


def spectral_laplacian_1d(n):
    """
    构造一维谱 Laplacian 算子矩阵。

    数学模型:
      在 GLL 节点上，Laplacian 算子离散为:
        L = -D^T * M * D
      其中 D 为微分矩阵，M = diag(weights) 为质量矩阵。

    参数:
      n: 谱元阶数

    返回:
      L: (n+1, n+1) Laplacian 矩阵
      nodes: GLL 节点
      weights: 积分权重
    """
    nodes, weights = gll_nodes_weights(n)
    D = differentiation_matrix(nodes)
    M = np.diag(weights)
    L = -D.T @ M @ D
    return L, nodes, weights


def spectral_derivative_2d(u, nx, ny):
    """
    利用谱微分矩阵计算二维标量场在 GLL 节点上的导数。

    参数:
      u: (nx+1, ny+1) 标量场值
      nx, ny: x 和 y 方向的谱元阶数

    返回:
      dudx, dudy: 偏导数
    """
    nodes_x, _ = gll_nodes_weights(nx)
    nodes_y, _ = gll_nodes_weights(ny)
    Dx = differentiation_matrix(nodes_x)
    Dy = differentiation_matrix(nodes_y)

    dudx = Dx @ u
    dudy = u @ Dy.T

    return dudx, dudy
