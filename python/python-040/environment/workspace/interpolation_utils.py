#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
interpolation_utils.py
多维插值与参数空间重建模块

融合原项目:
- 928_pwl_interp_2d_scattered: 散乱数据二维分段线性插值（Delaunay三角剖分）
- 1110_sparse_interp_nd: 多维稀疏网格 Lagrange 插值（Smolyak方法）

在BSM信号分析中用于:
- 在探测器非均匀区域对散乱的能量击中点进行二维插值重建
- 在BSM参数空间（质量、耦合、衰变宽度）进行多维稀疏网格扫描
"""

import numpy as np
from typing import Tuple, List, Optional


# ---------------------------------------------------------------------------
# 2D 散乱数据分段线性插值（基于 Delaunay 三角剖分思想）
# ---------------------------------------------------------------------------

def barycentric_interpolate_triangle(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    v1: float,
    v2: float,
    v3: float,
    p: np.ndarray
) -> float:
    """
    使用重心坐标在三角形内进行线性插值。

    对于三角形顶点 p1, p2, p3 及对应值 v1, v2, v3，
    点 p 的重心坐标 (λ1, λ2, λ3) 满足：
        p = λ1 p1 + λ2 p2 + λ3 p3
        λ1 + λ2 + λ3 = 1

    插值公式:
        v(p) = λ1 v1 + λ2 v2 + λ3 v3

    Parameters
    ----------
    p1, p2, p3 : np.ndarray
        三角形顶点坐标，形状 (2,)
    v1, v2, v3 : float
        顶点函数值
    p : np.ndarray
        待插值点，形状 (2,)

    Returns
    -------
    float
        插值结果；若点在三角形外，返回 -1.0
    """
    # 计算重心坐标
    denom = (p2[1] - p3[1]) * (p1[0] - p3[0]) + (p3[0] - p2[0]) * (p1[1] - p3[1])
    if abs(denom) < 1e-14:
        return -1.0

    lam1 = ((p2[1] - p3[1]) * (p[0] - p3[0]) + (p3[0] - p2[0]) * (p[1] - p3[1])) / denom
    lam2 = ((p3[1] - p1[1]) * (p[0] - p3[0]) + (p1[0] - p3[0]) * (p[1] - p3[1])) / denom
    lam3 = 1.0 - lam1 - lam2

    # 检查是否在三角形内（含边界）
    if lam1 < -1e-8 or lam2 < -1e-8 or lam3 < -1e-8:
        return -1.0

    # 边界处理
    lam1 = max(0.0, min(1.0, lam1))
    lam2 = max(0.0, min(1.0, lam2))
    lam3 = max(0.0, min(1.0, lam3))
    total = lam1 + lam2 + lam3
    if total > 1e-15:
        lam1 /= total
        lam2 /= total
        lam3 /= total

    return lam1 * v1 + lam2 * v2 + lam3 * v3


def pwl_interp_2d_scattered(
    data_points: np.ndarray,
    data_values: np.ndarray,
    query_points: np.ndarray
) -> np.ndarray:
    """
    散乱数据的二维分段线性（PWL）插值。

    简化的 Delaunay 三角剖分替代方案：
    对每个查询点，找到最近的三个数据点构成近似三角形，
    使用重心坐标进行插值。

    物理应用: 在探测器非均匀响应区域，根据散乱的校准点
    重建完整的位置-响应映射。

    Parameters
    ----------
    data_points : np.ndarray
        散乱数据点坐标，形状 (N, 2)
    data_values : np.ndarray
        数据点函数值，形状 (N,)
    query_points : np.ndarray
        查询点坐标，形状 (M, 2)

    Returns
    -------
    np.ndarray
        插值结果，形状 (M,)
    """
    n = data_points.shape[0]
    m = query_points.shape[0]
    result = np.zeros(m)

    for i in range(m):
        q = query_points[i]

        # 找到最近的三个数据点
        dists = np.linalg.norm(data_points - q, axis=1)
        idx = np.argsort(dists)[:3]

        if dists[idx[0]] < 1e-12:
            # 精确命中
            result[i] = data_values[idx[0]]
            continue

        p1, p2, p3 = data_points[idx[0]], data_points[idx[1]], data_points[idx[2]]
        v1, v2, v3 = data_values[idx[0]], data_values[idx[1]], data_values[idx[2]]

        val = barycentric_interpolate_triangle(p1, p2, p3, v1, v2, v3, q)
        if val < 0:
            # 若不在三角形内，使用反距离加权
            w = 1.0 / (dists[idx] + 1e-10)
            val = np.sum(w * data_values[idx]) / np.sum(w)

        result[i] = val

    return result


# ---------------------------------------------------------------------------
# 多维稀疏网格 Lagrange 插值（Smolyak 方法）
# ---------------------------------------------------------------------------

def cc_compute_points(n: int) -> np.ndarray:
    """
    计算 Clenshaw-Curtis 积分点（用于稀疏网格）。

    CC 节点:
        x_j = cos(π j / (n-1)),  j = 0, ..., n-1

    Parameters
    ----------
    n : int
        节点数

    Returns
    -------
    np.ndarray
        CC 节点，范围 [-1, 1]
    """
    if n < 1:
        return np.array([0.0])
    if n == 1:
        return np.array([0.0])
    j = np.arange(n)
    return np.cos(np.pi * j / (n - 1))


def order_from_level_135(level: int) -> int:
    """
    将 Smolyak 层级映射到 CC 规则阶数。

    规则: n = 1  若 level = 0
          n = 2  若 level = 1
          n = 2^{level-1} + 1  若 level ≥ 2

    Parameters
    ----------
    level : int
        层级（非负整数）

    Returns
    -------
    int
        CC 节点数
    """
    if level <= 0:
        return 1
    elif level == 1:
        return 2
    else:
        return 2 ** (level - 1) + 1


def smolyak_coefficients(m: int, level_vec: np.ndarray) -> np.ndarray:
    """
    计算 Smolyak 组合系数 c(ℓ)。

    对于每个维度 i，令 ℓ_i 为层级，则组合系数为：
        c(ℓ) = (-1)^{|ℓ|_1 - m} × C(m-1, |ℓ|_1 - m)

    其中 |ℓ|_1 = Σ ℓ_i。

    Parameters
    ----------
    m : int
        空间维度
    level_vec : np.ndarray
        层级向量 ℓ，形状 (m,)

    Returns
    -------
    np.ndarray
        系数数组（这里返回标量扩展）
    """
    l1_norm = np.sum(level_vec)
    k = l1_norm - m
    if k < 0:
        return np.array([0.0])

    # 组合数 C(m-1, k)
    from math import comb
    if k > m - 1:
        return np.array([0.0])

    coeff = ((-1) ** k) * comb(m - 1, k)
    return np.array([float(coeff)])


def lagrange_basis_1d(n: int, x_nodes: np.ndarray, x_query: float) -> np.ndarray:
    """
    一维 Lagrange 基函数在查询点的值。

        L_j(x) = Π_{k≠j} (x - x_k) / (x_j - x_k)

    Parameters
    ----------
    n : int
        节点数
    x_nodes : np.ndarray
        节点坐标，长度 n
    x_query : float
        查询点

    Returns
    -------
    np.ndarray
        基函数值 L_j(x)，长度 n
    """
    basis = np.ones(n)
    for j in range(n):
        for k in range(n):
            if k != j:
                denom = x_nodes[j] - x_nodes[k]
                if abs(denom) < 1e-14:
                    basis[j] = 0.0
                else:
                    basis[j] *= (x_query - x_nodes[k]) / denom
    return basis


def sparse_interp_nd_value(
    m: int,
    ind: np.ndarray,
    a_bounds: np.ndarray,
    b_bounds: np.ndarray,
    nd: int,
    zd: np.ndarray,
    xi: np.ndarray
) -> float:
    """
    使用 Smolyak 稀疏网格进行多维 Lagrange 插值。

    Smolyak 公式:
        A(q,m) = Σ_{|ℓ|_1 ≤ q} c(ℓ) ⊗_{i=1}^m U^{ℓ_i}

    其中 U^{ℓ_i} 是第 i 维的 ℓ_i 层 CC 插值算子。

    物理应用: 在 BSM 参数空间 (M_{Z'}, g_q, g_ℓ, Γ_{Z'}) 中
    稀疏采样信号截面，通过 Smolyak 插值重建连续截面函数。

    Parameters
    ----------
    m : int
        空间维度
    ind : np.ndarray
        每维的层级索引，形状 (m,)
    a_bounds : np.ndarray
        每维下界，形状 (m,)
    b_bounds : np.ndarray
        每维上界，形状 (m,)
    nd : int
        网格总点数
    zd : np.ndarray
        网格点函数值，长度 nd
    xi : np.ndarray
        查询点坐标，形状 (m,)

    Returns
    -------
    float
        插值结果
    """
    # 构建每维的 CC 节点
    cc_nodes = []
    for i in range(m):
        n_1d = order_from_level_135(ind[i])
        x_1d = cc_compute_points(n_1d)
        # 映射到 [a_i, b_i]
        x_1d = 0.5 * ((1.0 - x_1d) * a_bounds[i] + (1.0 + x_1d) * b_bounds[i])
        cc_nodes.append(x_1d)

    # 计算每维的 Lagrange 基
    weights_per_dim = []
    for i in range(m):
        basis = lagrange_basis_1d(cc_nodes[i].size, cc_nodes[i], xi[i])
        weights_per_dim.append(basis)

    # 张量积组合权重
    # 简化为直接遍历所有 nd 个点
    # 实际 Smolyak 有更高效的索引方法，这里用简化版
    w = np.ones(nd)
    idx = 0
    # 由于 nd 可能很大，这里用递归方式构造索引
    # 简化: 假设 nd 是各维节点数的乘积
    strides = [1] * m
    for i in range(m - 2, -1, -1):
        strides[i] = strides[i + 1] * cc_nodes[i + 1].size

    result = 0.0
    for flat_idx in range(nd):
        w = 1.0
        temp_idx = flat_idx
        for i in range(m):
            node_idx = temp_idx // strides[i] if i < m - 1 else temp_idx
            temp_idx = temp_idx % strides[i] if i < m - 1 else 0
            if node_idx < len(weights_per_dim[i]):
                w *= weights_per_dim[i][node_idx]
            else:
                w = 0.0
                break
        if flat_idx < len(zd):
            result += w * zd[flat_idx]

    return result


def bsm_cross_section_interp_2d(
    mass_grid: np.ndarray,
    coupling_grid: np.ndarray,
    cross_section_table: np.ndarray,
    query_mass: float,
    query_coupling: float
) -> float:
    """
    在 BSM 参数空间进行二维双线性插值，查询信号截面。

    参数空间: (M_{Z'}, g_q)

    Parameters
    ----------
    mass_grid : np.ndarray
        质量网格点 [GeV]
    coupling_grid : np.ndarray
        耦合网格点
    cross_section_table : np.ndarray
        截面表，形状 (n_mass, n_coupling) [pb]
    query_mass : float
        查询质量 [GeV]
    query_coupling : float
        查询耦合

    Returns
    -------
    float
        插值截面 [pb]
    """
    # 边界检查
    m_min, m_max = mass_grid[0], mass_grid[-1]
    c_min, c_max = coupling_grid[0], coupling_grid[-1]

    if query_mass < m_min or query_mass > m_max or query_coupling < c_min or query_coupling > c_max:
        # 外推（使用最近值）
        query_mass = np.clip(query_mass, m_min, m_max)
        query_coupling = np.clip(query_coupling, c_min, c_max)

    # 找到包围的网格索引
    i = np.searchsorted(mass_grid, query_mass, side='right') - 1
    j = np.searchsorted(coupling_grid, query_coupling, side='right') - 1

    i = max(0, min(i, len(mass_grid) - 2))
    j = max(0, min(j, len(coupling_grid) - 2))

    m1, m2 = mass_grid[i], mass_grid[i + 1]
    c1, c2 = coupling_grid[j], coupling_grid[j + 1]

    v11 = cross_section_table[i, j]
    v12 = cross_section_table[i, j + 1]
    v21 = cross_section_table[i + 1, j]
    v22 = cross_section_table[i + 1, j + 1]

    # 双线性插值
    dm = m2 - m1
    dc = c2 - c1
    if abs(dm) < 1e-14 or abs(dc) < 1e-14:
        return v11

    t = (query_mass - m1) / dm
    s = (query_coupling - c1) / dc

    return (1.0 - t) * (1.0 - s) * v11 + (1.0 - t) * s * v12 \
         + t * (1.0 - s) * v21 + t * s * v22
