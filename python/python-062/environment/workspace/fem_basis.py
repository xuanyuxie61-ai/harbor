"""
fem_basis.py
================================================================================
三维有限元基函数模块 —— 基于种子项目 419_fem3d_sample

在行星边界层（PBL）的垂直方向离散化中，有限元方法（FEM）因其在复杂地形
和边界层底部的灵活性而被广泛采用。本模块提供四面体单元（T4）的线性
基函数及其梯度计算，用于构建 LES 的空间离散算子。

核心物理公式
--------------------------------------------------------------------------------
对于四面体单元 T，体积由下式给出：
    V = | det([x1 x2 x3 x4; y1 y2 y3 y4; z1 z2 z3 z4; 1 1 1 1]) | / 6

节点 i 的线性基函数 φ_i 满足 φ_i(x_j) = δ_{ij}，且在单元内为：
    φ_i(x,y,z) = (a_i + b_i x + c_i y + d_i z) / (6V)

其中系数 a_i, b_i, c_i, d_i 由节点坐标代数余子式确定。

LES 中的 Galerkin 弱形式：
    ∫_Ω  (∂u_i/∂t) φ_j dV  +  ∫_Ω  u_k ∂u_i/∂x_k φ_j dV
    = - (1/ρ) ∫_Ω  ∂p/∂x_i φ_j dV  +  ∫_Ω  ν_eff (∂²u_i/∂x_k²) φ_j dV  +  ∫_Ω  f_i φ_j dV

其中 ν_eff = ν_mol + ν_sgs 为分子粘性加亚格子涡粘性。
"""

import numpy as np


def tetrahedron_volume(nodes):
    """
    计算四面体体积。

    参数
    ----------
    nodes : np.ndarray, shape (4, 3)
        四个顶点的坐标

    返回
    -------
    volume : float
        四面体体积（严格为正）
    """
    if nodes.shape != (4, 3):
        raise ValueError("tetrahedron_volume: nodes 形状必须为 (4, 3)")

    M = np.vstack([nodes.T, np.ones(4)])
    volume = abs(np.linalg.det(M)) / 6.0

    if volume < 1e-15:
        raise ValueError("tetrahedron_volume: 四面体体积过小或退化")

    return volume


def basis_mn_tet4(nodes, points):
    """
    计算 T4 单元在所有采样点处的线性基函数值。

    参数
    ----------
    nodes : np.ndarray, shape (4, 3)
        四面体顶点
    points : np.ndarray, shape (n, 3)
        采样点

    返回
    -------
    phi : np.ndarray, shape (4, n)
        phi[i,j] = φ_i(points[j])
    """
    points = np.atleast_2d(points)
    if points.shape[1] != 3:
        points = points.T

    n = points.shape[0]
    phi = np.zeros((4, n), dtype=np.float64)

    # 计算有符号体积（用于重心坐标）
    M_full = np.vstack([nodes.T, np.ones(4)])
    vol_signed = np.linalg.det(M_full) / 6.0

    if abs(vol_signed) < 1e-15:
        raise ValueError("basis_mn_tet4: 四面体体积退化")

    # 计算每个节点的体积坐标（重心坐标）
    for i in range(4):
        for j in range(n):
            sub_nodes = np.copy(nodes)
            sub_nodes[i] = points[j]
            M = np.vstack([sub_nodes.T, np.ones(4)])
            det_i = np.linalg.det(M)
            phi[i, j] = (det_i / 6.0) / vol_signed

    # 数值裁剪与归一化
    phi = np.clip(phi, 0.0, 1.0)
    col_sum = phi.sum(axis=0)
    col_sum = np.where(np.abs(col_sum) < 1e-15, 1.0, col_sum)
    phi = phi / col_sum

    return phi


def basis_gradient_tet4(nodes):
    """
    计算 T4 单元基函数的梯度（在单元内为常数）。

    参数
    ----------
    nodes : np.ndarray, shape (4, 3)
        四面体顶点

    返回
    -------
    grad_phi : np.ndarray, shape (4, 3)
        grad_phi[i] = ∇φ_i = [∂φ_i/∂x, ∂φ_i/∂y, ∂φ_i/∂z]
    """
    volume = tetrahedron_volume(nodes)

    grad_phi = np.zeros((4, 3), dtype=np.float64)

    for i in range(4):
        # 使用循环索引计算对面三角形的法向量
        idx = [k for k in range(4) if k != i]
        p1, p2, p3 = nodes[idx[0]], nodes[idx[1]], nodes[idx[2]]

        # 面法向量（指向节点 i）
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)

        # 梯度方向
        grad_phi[i] = normal / (6.0 * volume)

        # 确保方向指向外侧（通过符号检查）
        center = nodes.mean(axis=0)
        to_center = center - nodes[i]
        if np.dot(grad_phi[i], to_center) > 0:
            grad_phi[i] = -grad_phi[i]

    return grad_phi


def fem_laplacian_matrix(nodes, element_nodes, nu_eff):
    """
    组装有限元 Laplacian 刚度矩阵（对应扩散项）。

    参数
    ----------
    nodes : np.ndarray, shape (n_node, 3)
        全局节点坐标
    element_nodes : np.ndarray, shape (n_elem, 4)
        每个四面体的节点索引
    nu_eff : float
        有效粘性系数（m²/s）

    返回
    -------
    L : np.ndarray, shape (n_node, n_node)
        稀疏形式的 Laplacian 矩阵（这里返回稠密矩阵用于小规模演示）
    """
    n_node = nodes.shape[0]
    n_elem = element_nodes.shape[0]
    L = np.zeros((n_node, n_node), dtype=np.float64)

    for e in range(n_elem):
        en = element_nodes[e]
        elem_nodes = nodes[en]

        vol = tetrahedron_volume(elem_nodes)
        grad = basis_gradient_tet4(elem_nodes)

        # 单元刚度矩阵: K_{ij} = ∫_T ∇φ_i · ∇φ_j dV = vol * ∇φ_i · ∇φ_j
        for i_loc in range(4):
            for j_loc in range(4):
                i_glob = en[i_loc]
                j_glob = en[j_loc]
                L[i_glob, j_glob] += nu_eff * vol * np.dot(grad[i_loc], grad[j_loc])

    return L
