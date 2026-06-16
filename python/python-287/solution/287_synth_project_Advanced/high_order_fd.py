# -*- coding: utf-8 -*-
"""
high_order_fd.py
================

高阶有限差分算子模块.

来源种子项目:
  - 632_lagrange  -> Lagrange 插值基函数及其导数
  - 638_lagrange_nd -> 多维 Lagrange 插值与多项式操作

物理背景:
  在 resistive MHD 方程的离散化中, 我们需要高精度的空间差分格式
  来准确捕捉撕裂模本征函数在共振面 (resonant surface) 附近的
  精细结构. 经典 Furth-Killeen-Rosenbluth 理论指出, 在共振面附近
  存在一个宽度为 delta ~ L * S^{-1/4} 的内层 (inner layer),
  其中 S 为 Lundquist 数. 对于 S ~ 10^6, delta/L ~ 10^{-1.5},
  需要高分辨率格式才能解析.

  我们基于 Lagrange 插值多项式构造高阶差分模板:
    给定 n 个节点 x_0, x_1, ..., x_{n-1},
    Lagrange 基函数:
        L_i(x) = prod_{j != i} (x - x_j) / (x_i - x_j)
    一阶导数:
        L_i'(x_k) = prod_{j != i, j != k} (x_k - x_j) / (x_i - x_j),  k != i
        L_i'(x_i) = sum_{j != i} 1/(x_i - x_j)

  对于均匀网格, 这就是经典的 Fornberg 算法.
  对于非均匀网格 (如 tanh-stretching), 我们直接求解 Vandermonde 系统.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


# -------------------------------------------------------------------
#  Fornberg 算法: 任意阶导数在任意点处的差分系数
# -------------------------------------------------------------------
def fornberg_weights(
    x_nodes: np.ndarray,
    x_eval: float,
    max_deriv: int = 2,
) -> np.ndarray:
    """
    Fornberg (1988) 算法, 计算在点 x_nodes 上的函数值
    对 x_eval 处第 m 阶导数的有限差分权重.

    参数:
        x_nodes:   长度为 n 的节点数组
        x_eval:    求导点
        max_deriv: 最大导数阶数

    返回:
        形状 (max_deriv+1, n) 的数组 weights, 其中
        weights[m, :] 给出第 m 阶导数的差分系数.

    数学原理 (Fornberg 1988, Math. Comp.):
        递推关系:
            delta[i][j][m] = ((x_j - x_{j-1}) * delta[i][j-1][m]
                              - m * delta[i][j-1][m-1]) / (x_j - x_i)
        其中 delta[i][j][m] 表示使用前 j+1 个节点逼近 x_i 处
        第 m 阶导数的系数.
    """
    x_nodes = np.asarray(x_nodes, dtype=float)
    n = x_nodes.size
    if n == 0:
        raise ValueError("节点数组不能为空")

    weights = np.zeros((max_deriv + 1, n))
    c1 = 1.0
    c4 = x_nodes[0] - x_eval
    weights[0, 0] = 1.0

    for i in range(1, n):
        mn = min(i, max_deriv)
        c2 = 1.0
        c5 = c4
        c4 = x_nodes[i] - x_eval
        for j in range(i):
            c3 = x_nodes[i] - x_nodes[j]
            c2 *= c3
            if j == i - 1:
                for m in range(mn, 0, -1):
                    weights[m, i] = c1 * (m * weights[m - 1, i - 1] - c5 * weights[m, i - 1]) / c2
                weights[0, i] = -c1 * c5 * weights[0, i - 1] / c2
            for m in range(mn, 0, -1):
                weights[m, j] = (c4 * weights[m, j] - m * weights[m - 1, j]) / c3
            weights[0, j] = c4 * weights[0, j] / c3
        c1 = c2

    return weights


# -------------------------------------------------------------------
#  一维非均匀网格差分算子 (基于 Lagrange 插值)
# -------------------------------------------------------------------
def build_first_derivative_matrix(
    x: np.ndarray,
    order: int = 4,
    bc: str = "periodic",
) -> np.ndarray:
    """
    构造一维非均匀网格上的高阶一阶导数矩阵.

    参数:
        x:     网格节点 (严格递增)
        order: 差分模板半宽 (总宽度 2*order+1 用于内部点)
        bc:    边界条件 ("periodic" 或 "dirichlet")

    返回:
        n x n 矩阵 D, 使得 D @ f 近似 f'(x).
    """
    n = x.size
    if order < 1:
        order = 1
    if order >= n // 2:
        order = max(1, n // 2 - 1)

    D = np.zeros((n, n))

    # 内部点: 使用 Fornberg 算法
    for i in range(n):
        # 选择模板节点
        if bc == "periodic":
            indices = [(i + k) % n for k in range(-order, order + 1)]
            nodes = np.array([x[idx] for idx in indices])
            # 周期性修正: 保证节点相对于 x[i] 连续
            nodes = x[i] + (nodes - x[i])
            for k in range(len(nodes)):
                diff = nodes[k] - x[i]
                period = x[-1] - x[0] + (x[1] - x[0])
                while diff > period / 2:
                    diff -= period
                while diff < -period / 2:
                    diff += period
                nodes[k] = x[i] + diff
            w = fornberg_weights(nodes, x[i], max_deriv=1)[1]
            for k, idx in enumerate(indices):
                D[i, idx] += w[k]
        else:
            # 非周期: 边界附近缩减模板
            jmin = max(0, i - order)
            jmax = min(n - 1, i + order)
            nodes = x[jmin:jmax + 1]
            w = fornberg_weights(nodes, x[i], max_deriv=1)[1]
            for k, j in enumerate(range(jmin, jmax + 1)):
                D[i, j] = w[k]

    return D


def build_second_derivative_matrix(
    x: np.ndarray,
    order: int = 4,
    bc: str = "periodic",
) -> np.ndarray:
    """
    构造一维非均匀网格上的高阶二阶导数矩阵.

    返回 n x n 矩阵 D2, 使得 D2 @ f 近似 f''(x).
    """
    n = x.size
    if order < 1:
        order = 1
    if order >= n // 2:
        order = max(1, n // 2 - 1)

    D2 = np.zeros((n, n))

    for i in range(n):
        if bc == "periodic":
            indices = [(i + k) % n for k in range(-order, order + 1)]
            nodes = np.array([x[idx] for idx in indices])
            for k in range(len(nodes)):
                diff = nodes[k] - x[i]
                period = x[-1] - x[0] + (x[1] - x[0])
                while diff > period / 2:
                    diff -= period
                while diff < -period / 2:
                    diff += period
                nodes[k] = x[i] + diff
            w = fornberg_weights(nodes, x[i], max_deriv=2)[2]
            for k, idx in enumerate(indices):
                D2[i, idx] += w[k]
        else:
            jmin = max(0, i - order)
            jmax = min(n - 1, i + order)
            nodes = x[jmin:jmax + 1]
            w = fornberg_weights(nodes, x[i], max_deriv=2)[2]
            for k, j in enumerate(range(jmin, jmax + 1)):
                D2[i, j] = w[k]

    return D2


# -------------------------------------------------------------------
#  二维张量积差分算子
# -------------------------------------------------------------------
def build_2d_laplacian(
    x: np.ndarray,
    y: np.ndarray,
    order: int = 4,
    bc_x: str = "periodic",
    bc_y: str = "dirichlet",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    构造二维 Laplace 算子的张量积分解.

    返回:
        (D2x, D2y): 两个 n x n 和 m x m 矩阵,
        使得 Laplacian(f) 的向量化形式为:
            (D2x kron I_y + I_x kron D2y) @ f_flat
    """
    D2x = build_second_derivative_matrix(x, order=order, bc=bc_x)
    D2y = build_second_derivative_matrix(y, order=order, bc=bc_y)
    return D2x, D2y


# -------------------------------------------------------------------
#  Lagrange 插值多项式求值 (从 632_lagrange 移植)
# -------------------------------------------------------------------
def lagrange_basis_value(
    x_data: np.ndarray,
    x_eval: float,
    i_basis: int,
) -> float:
    """
    计算第 i_basis 个 Lagrange 基函数在 x_eval 处的值.

    L_i(x) = prod_{j != i} (x - x_j) / (x_i - x_j)
    """
    n = x_data.size
    val = 1.0
    for j in range(n):
        if j != i_basis:
            denom = x_data[i_basis] - x_data[j]
            if abs(denom) < 1.0e-30:
                raise ValueError(f"节点重合: x[{i_basis}] = x[{j}] = {x_data[i_basis]}")
            val *= (x_eval - x_data[j]) / denom
    return val


def lagrange_interpolant_value(
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_eval: float,
) -> float:
    """
    计算 Lagrange 插值多项式在 x_eval 处的值.

    P(x) = sum_i y_i * L_i(x)
    """
    n = x_data.size
    if y_data.size != n:
        raise ValueError("x_data 和 y_data 长度不一致")
    val = 0.0
    for i in range(n):
        val += y_data[i] * lagrange_basis_value(x_data, x_eval, i)
    return val


# -------------------------------------------------------------------
#  交错网格 (staggered grid) 上的差分
# -------------------------------------------------------------------
def staggered_derivative(
    f_cell: np.ndarray,
    dx: np.ndarray,
) -> np.ndarray:
    """
    交错网格上的一阶导数: 从单元中心到单元面.

    (df/dx)_{i+1/2} = (f_{i+1} - f_i) / dx_i

    参数:
        f_cell: 单元中心值 (长度 n)
        dx:     单元间距 (长度 n-1 或 n)

    返回:
        单元面值 (长度 n-1)
    """
    if f_cell.size < 2:
        raise ValueError("单元数至少为 2")
    return np.diff(f_cell) / dx[: f_cell.size - 1]
