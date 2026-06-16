# -*- coding: utf-8 -*-
"""
vandermonde_fd.py
=================
基于 Vandermonde 矩阵求解的高阶有限差分系数计算.

核心算法 (来自 1381_vandermonde):
--------------------------------
1) 构建 Vandermonde 矩阵 V(i,j) = x_j^(i-1)
2) 求解 V · c = b 获得插值多项式系数
3) Björck-Pereyra 算法高效求解 Vandermonde 系统
4) Fornberg 算法计算任意网格上的有限差分权重

物理应用:
---------
在 fast ignition 模拟中, 日冕-稠密等离子体界面处密度梯度剧烈,
均匀网格 FD 精度不足, 需使用非均匀网格上的高阶 FD 系数.
非均匀网格上的 k 阶导数:
    f^(k)(x_0) ≈ Σ_j w_j · f(x_j)
其中 w_j 由 Vandermonde 系统确定.

理论:
-----
给定节点 {x_0, x_1, ..., x_n}, 求 k 阶导数权重 {w_j}:
    Σ_j w_j · (x_j - x_0)^m / m! = δ_{m,k}   (m = 0, ..., n)

这是 Vandermonde 型系统:
    V^T · w = e_k
其中 V(i,j) = (x_j - x_0)^i / i!
"""

import math


# ============================================================
# Vandermonde 矩阵构建 (来自 1381_vandermonde/vand1)
# ============================================================
def vandermonde_matrix(n, x):
    """
    构建 Vandermonde 矩阵 V(i,j) = x[j]^(i), i=0..n-1, j=0..n-1.

    公式 (来自 1381_vandermonde/vand1):
        V(i,j) = x_j^(i)    (0-indexed)
        det(V) = Π_{i<j} (x_j - x_i)

    参数:
        n: 矩阵阶数
        x: 长度为 n 的节点列表
    返回:
        V: n×n 列表的列表
    """
    V = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == 0 and x[j] == 0.0:
                V[i][j] = 1.0
            else:
                V[i][j] = x[j] ** i
    return V


# ============================================================
# Björck-Pereyra 算法 (来自 1381_vandermonde/pvand, dvand)
# ============================================================
def pvand_solve(n, alpha, b):
    """
    求解 Vandermonde 系统 V · x = b (来自 1381_vandermonde/pvand).
    Björck-Pereyra 算法, O(n^2) 复杂度.

    参数:
        n    : 系统阶数
        alpha: 节点列表 [α_0, ..., α_{n-1}]
        b    : 右端向量
    返回:
        x: 解向量
    """
    x = list(b)
    # 前向消去
    for k in range(n - 1):
        for j in range(n - 1, k, -1):
            x[j] = x[j] - alpha[k] * x[j - 1]
    # 回代
    for k in range(n - 2, -1, -1):
        for j in range(k + 1, n):
            x[j] = x[j] / (alpha[j] - alpha[j - k - 1])
        for j in range(k, n - 1):
            x[j] = x[j] - x[j + 1]
    return x


def dvand_solve(n, alpha, b):
    """
    求解转置 Vandermonde 系统 V^T · x = b (来自 1381_vandermonde/dvand).

    参数:
        n    : 系统阶数
        alpha: 节点列表
        b    : 右端向量
    返回:
        x: 解向量
    """
    x = list(b)
    for k in range(n - 1):
        for j in range(n - 1, k, -1):
            denom = alpha[j] - alpha[j - k - 1]
            if abs(denom) < 1.0e-300:
                x[j] = 0.0
            else:
                x[j] = (x[j] - x[j - 1]) / denom
    for k in range(n - 2, -1, -1):
        for j in range(k, n - 1):
            x[j] = x[j] - alpha[k] * x[j + 1]
    return x


# ============================================================
# Fornberg 算法: 任意网格上的有限差分权重
# ============================================================
def fornberg_fd_weights(x_nodes, x_center, max_derivative_order):
    """
    Fornberg (1988) 算法: 计算任意节点集上各阶导数的有限差分权重.

    算法复杂度: O(n · M) 其中 n = 节点数, M = 最高阶导数.

    参数:
        x_nodes             : 节点坐标列表
        x_center            : 求导点
        max_derivative_order: 最高阶导数阶数
    返回:
        weights[M+1][n]: weights[m][j] 为 m 阶导数在第 j 个节点的权重

    参考:
        B. Fornberg, "Generation of finite difference formulas on
        arbitrarily spaced meshes", Math. Comp. 51, 699 (1988).
    """
    n = len(x_nodes)
    M = max_derivative_order
    # 权重数组
    w = [[0.0] * n for _ in range(M + 1)]
    # 辅助量
    c1 = 1.0
    c4 = x_nodes[0] - x_center
    w[0][0] = 1.0

    for i in range(1, n):
        mn = min(i, M)
        c2 = 1.0
        c5 = c4
        c4 = x_nodes[i] - x_center
        for j in range(i):
            c3 = x_nodes[i] - x_nodes[j]
            c2 = c2 * c3
            if j == i - 1:
                for m in range(mn, 0, -1):
                    w[m][i] = c1 * (m * w[m - 1][i - 1] - c5 * w[m][i - 1]) / c2
                w[0][i] = -c1 * c5 * w[0][i - 1] / c2
            for m in range(mn, 0, -1):
                w[m][j] = (c4 * w[m][j] - m * w[m - 1][j]) / c3
            w[0][j] = c4 * w[0][j] / c3
        c1 = c2
    return w


# ============================================================
# 均匀网格高阶 FD 系数 (用于 fast ignition 能量沉积 PDE)
# ============================================================
def uniform_fd_coefficients(order, half_width, derivative_order):
    """
    均匀网格上 center 差分的 FD 系数.

    对 2p+1 点模板, 求 k 阶导数:
        f^(k)(x_i) ≈ (1/dx^k) Σ_{j=-p}^{p} c_j · f(x_{i+j})

    使用 Vandermonde 方法精确计算.

    参数:
        order          : 精度阶数 (如 2, 4, 6)
        half_width     : 半带宽 p (模板宽度 = 2p+1)
        derivative_order: 求导阶数 (1 或 2)
    返回:
        coeffs: 长度 2p+1 的系数列表, 索引 j 对应节点 x_{i-p+j}
    """
    # 节点: -p, -p+1, ..., 0, ..., p
    p = half_width
    nodes = list(range(-p, p + 1))
    # Fornberg 算法
    w = fornberg_fd_weights([float(x) for x in nodes], 0.0, derivative_order)
    coeffs = w[derivative_order]
    return coeffs


# ============================================================
# Fast Ignition 专用: 非均匀网格 FD 系数
# ============================================================
def nonuniform_second_derivative_weights(grid_x):
    """
    在非均匀一维网格上计算二阶导数 (∂^2/∂x^2) 的 FD 权重.

    对 3 点模板 {x_{i-1}, x_i, x_{i+1}}:
        f''(x_i) ≈ w_{-1} f(x_{i-1}) + w_0 f(x_i) + w_{+1} f(x_{i+1})

    精确公式:
        h_1 = x_i - x_{i-1}
        h_2 = x_{i+1} - x_i
        w_{-1} = 2 / (h_1 (h_1 + h_2))
        w_0    = -2 / (h_1 h_2)
        w_{+1} = 2 / (h_2 (h_1 + h_2))

    参数:
        grid_x: 一维网格节点坐标 (递增)
    返回:
        w_minus, w_center, w_plus: 三组权重列表 (各 len(grid_x))
    """
    n = len(grid_x)
    w_m = [0.0] * n
    w_c = [0.0] * n
    w_p = [0.0] * n

    for i in range(n):
        if i == 0:
            # 前向差分 (边界)
            h1 = grid_x[1] - grid_x[0]
            h2 = grid_x[2] - grid_x[1] if n > 2 else h1
            w_m[i] = 0.0
            w_c[i] = 2.0 / (h1 * (h1 + h2))
            w_p[i] = -2.0 / (h1 * h2) if h2 > 0 else 0.0
        elif i == n - 1:
            # 后向差分 (边界)
            h1 = grid_x[i] - grid_x[i - 1]
            h2 = grid_x[i - 1] - grid_x[i - 2] if i > 1 else h1
            w_m[i] = 2.0 / (h1 * (h1 + h2)) if (h1 + h2) > 0 else 0.0
            w_c[i] = -2.0 / (h1 * h2) if h2 > 0 else 0.0
            w_p[i] = 0.0
        else:
            # 内部点: 精确 3 点非均匀 FD
            h1 = grid_x[i] - grid_x[i - 1]
            h2 = grid_x[i + 1] - grid_x[i]
            if h1 > 0 and h2 > 0:
                w_m[i] = 2.0 / (h1 * (h1 + h2))
                w_c[i] = -2.0 / (h1 * h2)
                w_p[i] = 2.0 / (h2 * (h1 + h2))
            else:
                w_m[i] = w_c[i] = w_p[i] = 0.0

    return w_m, w_c, w_p


def compute_vandermonde_condition_number(nodes):
    """
    计算 Vandermonde 矩阵条件数 (估计).
    条件数大表示 FD 系数对舍入误差敏感.

    参数:
        nodes: 节点坐标列表
    返回:
        估计的条件数 (无穷范数)
    """
    n = len(nodes)
    V = vandermonde_matrix(n, nodes)
    # 简化条件数: ||V||_inf · ||V^{-1}||_inf
    # 使用列主元高斯消去估计
    row_sums = [sum(abs(V[i][j]) for j in range(n)) for i in range(n)]
    norm_inf = max(row_sums) if row_sums else 1.0
    # 逆范数估计 (使用 Vandermonde 行列式)
    det_log = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            diff = abs(nodes[j] - nodes[i])
            if diff > 0:
                det_log += math.log(diff)
    # 粗略估计: cond ~ ||V|| / |det|^(1/n)
    if det_log < -300:
        return float('inf')
    det_est = math.exp(det_log / n)
    if det_est < 1.0e-300:
        return float('inf')
    return norm_inf / max(det_est, 1.0e-300)


def print_fd_summary():
    """打印 FD 系数摘要."""
    print("\n" + "=" * 72)
    print("Vandermonde-based 高阶有限差分系数")
    print("=" * 72)

    # 2阶精度二阶导数 (3点)
    c2 = uniform_fd_coefficients(2, 1, 2)
    print("  2阶精度 ∂²/∂x² (3点中心差分):")
    print("    coeffs = [{}]".format(
        ", ".join("{:+.4f}".format(c) for c in c2)))
    print("    理论: [1, -2, 1]")

    # 4阶精度二阶导数 (5点)
    c4 = uniform_fd_coefficients(4, 2, 2)
    print("  4阶精度 ∂²/∂x² (5点中心差分):")
    print("    coeffs = [{}]".format(
        ", ".join("{:+.6f}".format(c) for c in c4)))
    print("    理论: [-1/12, 4/3, -5/2, 4/3, -1/12]")

    # 4阶精度一阶导数 (5点)
    c4_d1 = uniform_fd_coefficients(4, 2, 1)
    print("  4阶精度 ∂/∂x (5点中心差分):")
    print("    coeffs = [{}]".format(
        ", ".join("{:+.6f}".format(c) for c in c4_d1)))

    # Vandermonde 条件数
    nodes_uniform = [float(i) for i in range(-2, 3)]
    cond = compute_vandermonde_condition_number(nodes_uniform)
    print("  Vandermonde 条件数 (5点均匀): {:.4e}".format(cond))

    # 非均匀网格示例 (fast ignition 界面)
    # 界面附近加密
    grid = []
    for i in range(21):
        x_norm = i / 20.0
        # 双曲正切拉伸: 界面在 x=0.5
        x = 0.5 * (1.0 + math.tanh(3.0 * (x_norm - 0.5)) / math.tanh(1.5))
        grid.append(x)
    wm, wc, wp = nonuniform_second_derivative_weights(grid)
    print("  非均匀网格 (双曲正切拉伸) 二阶导数权重:")
    print("    中心点 (i=10): w-={:+.4f}, w0={:+.4f}, w+={:+.4f}".format(
        wm[10], wc[10], wp[10]))
    print("    边界点 (i=0):  w0={:+.4f}, w+={:+.4f}".format(wc[0], wp[0]))
    print("=" * 72)
