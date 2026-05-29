# -*- coding: utf-8 -*-
"""
fem_basis.py
三维有限元基函数与坐标变换模块

核心算法来源：
- fem3d_pack: TET4 四面体线性基函数、物理/参考坐标变换、Jacobian、体积计算

物理背景：
在三维高超声速边界层稳定性分析中，扰动方程需在物理域离散。
本模块提供参考四面体到物理四面体的等参映射，用于：
1. 构造三维线性稳定性算子的有限元离散矩阵
2. 计算局部 Jacobian 与度量张量
3. 数值积分权重的坐标变换
"""

import numpy as np


def tet4_basis(t, p):
    """
    计算 TET4 线性基函数在物理坐标处的值。

    参考算法：fem3d_pack/basis_mn_tet4

    四面体顶点 T(:,0:3)，物理点 P(:,0:2)。
    基函数 φ_i(P) 由体积坐标定义：
        φ_i = V_i / V
    其中 V 为四面体体积，V_i 为替换第 i 个顶点为 P 后的子四面体体积。

    体积公式:
        V = det([x1 x2 x3 x4; y1 y2 y3 y4; z1 z2 z3 z4; 1 1 1 1]) / 6

    参数:
        t (np.ndarray): 顶点坐标, shape (3, 4)
        p (np.ndarray): 求值点, shape (3, n) 或 (3,)

    返回:
        np.ndarray: phi, shape (4, n) 或 (4,)
    """
    if p.ndim == 1:
        p = p.reshape(3, 1)
        squeeze = True
    else:
        squeeze = False
    n = p.shape[1]

    # 计算体积 V
    volume = (
        t[0, 0] * (t[1, 1] * (t[2, 2] - t[2, 3]) - t[1, 2] * (t[2, 1] - t[2, 3]) + t[1, 3] * (t[2, 1] - t[2, 2]))
        - t[0, 1] * (t[1, 0] * (t[2, 2] - t[2, 3]) - t[1, 2] * (t[2, 0] - t[2, 3]) + t[1, 3] * (t[2, 0] - t[2, 2]))
        + t[0, 2] * (t[1, 0] * (t[2, 1] - t[2, 3]) - t[1, 1] * (t[2, 0] - t[2, 3]) + t[1, 3] * (t[2, 0] - t[2, 1]))
        - t[0, 3] * (t[1, 0] * (t[2, 1] - t[2, 2]) - t[1, 1] * (t[2, 0] - t[2, 2]) + t[1, 2] * (t[2, 0] - t[2, 1]))
    )

    if abs(volume) < 1e-15:
        raise ValueError("fem_basis: 四面体体积为零，网格退化")

    phi = np.zeros((4, n))
    # φ_1: 替换顶点 1
    phi[0, :] = (
        p[0, :] * (t[1, 1] * (t[2, 2] - t[2, 3]) - t[1, 2] * (t[2, 1] - t[2, 3]) + t[1, 3] * (t[2, 1] - t[2, 2]))
        - t[0, 1] * (p[1, :] * (t[2, 2] - t[2, 3]) - t[1, 2] * (p[2, :] - t[2, 3]) + t[1, 3] * (p[2, :] - t[2, 2]))
        + t[0, 2] * (p[1, :] * (t[2, 1] - t[2, 3]) - t[1, 1] * (p[2, :] - t[2, 3]) + t[1, 3] * (p[2, :] - t[2, 1]))
        - t[0, 3] * (p[1, :] * (t[2, 1] - t[2, 2]) - t[1, 1] * (p[2, :] - t[2, 2]) + t[1, 2] * (p[2, :] - t[2, 1]))
    ) / volume

    # φ_2: 替换顶点 2
    phi[1, :] = (
        t[0, 0] * (p[1, :] * (t[2, 2] - t[2, 3]) - t[1, 2] * (p[2, :] - t[2, 3]) + t[1, 3] * (p[2, :] - t[2, 2]))
        - p[0, :] * (t[1, 0] * (t[2, 2] - t[2, 3]) - t[1, 2] * (t[2, 0] - t[2, 3]) + t[1, 3] * (t[2, 0] - t[2, 2]))
        + t[0, 2] * (t[1, 0] * (p[2, :] - t[2, 3]) - p[1, :] * (t[2, 0] - t[2, 3]) + t[1, 3] * (t[2, 0] - p[2, :]))
        - t[0, 3] * (t[1, 0] * (p[2, :] - t[2, 2]) - p[1, :] * (t[2, 0] - t[2, 2]) + t[1, 2] * (t[2, 0] - p[2, :]))
    ) / volume

    # φ_3: 替换顶点 3
    phi[2, :] = (
        t[0, 0] * (t[1, 1] * (p[2, :] - t[2, 3]) - p[1, :] * (t[2, 1] - t[2, 3]) + t[1, 3] * (t[2, 1] - p[2, :]))
        - t[0, 1] * (t[1, 0] * (p[2, :] - t[2, 3]) - p[1, :] * (t[2, 0] - t[2, 3]) + t[1, 3] * (t[2, 0] - p[2, :]))
        + p[0, :] * (t[1, 0] * (t[2, 1] - t[2, 3]) - t[1, 1] * (t[2, 0] - t[2, 3]) + t[1, 3] * (t[2, 0] - t[2, 1]))
        - t[0, 3] * (t[1, 0] * (t[2, 1] - p[2, :]) - t[1, 1] * (t[2, 0] - p[2, :]) + p[1, :] * (t[2, 0] - t[2, 1]))
    ) / volume

    # φ_4: 替换顶点 4
    phi[3, :] = (
        t[0, 0] * (t[1, 1] * (t[2, 2] - p[2, :]) - t[1, 2] * (t[2, 1] - p[2, :]) + p[1, :] * (t[2, 1] - t[2, 2]))
        - t[0, 1] * (t[1, 0] * (t[2, 2] - p[2, :]) - t[1, 2] * (t[2, 0] - p[2, :]) + p[1, :] * (t[2, 0] - t[2, 2]))
        + t[0, 2] * (t[1, 0] * (t[2, 1] - p[2, :]) - t[1, 1] * (t[2, 0] - p[2, :]) + p[1, :] * (t[2, 0] - t[2, 1]))
        - p[0, :] * (t[1, 0] * (t[2, 1] - t[2, 2]) - t[1, 1] * (t[2, 0] - t[2, 2]) + t[1, 2] * (t[2, 0] - t[2, 1]))
    ) / volume

    if squeeze:
        return phi[:, 0]
    return phi


def tetrahedron_volume(t):
    """
    计算四面体体积。

    V = |det([x2-x1, x3-x1, x4-x1; y2-y1, y3-y1, y4-y1; z2-z1, z3-z1, z4-z1])| / 6

    参数:
        t (np.ndarray): 顶点, shape (3, 4)

    返回:
        float: 体积（非负）
    """
    A = np.column_stack((t[:, 1] - t[:, 0], t[:, 2] - t[:, 0], t[:, 3] - t[:, 0]))
    vol = abs(np.linalg.det(A)) / 6.0
    return vol


def reference_to_physical_tet4(t, xi):
    """
    将参考四面体坐标 ξ 映射到物理坐标 x。

    参考四面体顶点:
        ξ1=(0,0,0), ξ2=(1,0,0), ξ3=(0,1,0), ξ4=(0,0,1)

    等参映射:
        x(ξ) = Σ_{i=1}^4 x_i φ_i(ξ)
    其中 φ_1=1-ξ-η-ζ, φ_2=ξ, φ_3=η, φ_4=ζ

    参数:
        t (np.ndarray): 物理顶点, shape (3, 4)
        xi (np.ndarray): 参考坐标, shape (3, n) 或 (3,)

    返回:
        np.ndarray: 物理坐标
    """
    if xi.ndim == 1:
        xi = xi.reshape(3, 1)
        squeeze = True
    else:
        squeeze = False

    n = xi.shape[1]
    phi = np.zeros((4, n))
    phi[0, :] = 1.0 - xi[0, :] - xi[1, :] - xi[2, :]
    phi[1, :] = xi[0, :]
    phi[2, :] = xi[1, :]
    phi[3, :] = xi[2, :]

    x_phys = t @ phi  # (3,4) @ (4,n) = (3,n)
    if squeeze:
        return x_phys[:, 0]
    return x_phys


def physical_to_reference_tet4(t, x):
    """
    将物理坐标 x 逆映射到参考四面体坐标 ξ。

    通过求解线性方程组: x = t[:,0] + J * ξ，其中 J = [t[:,1]-t[:,0], t[:,2]-t[:,0], t[:,3]-t[:,0]]

    参数:
        t (np.ndarray): 物理顶点, shape (3, 4)
        x (np.ndarray): 物理坐标, shape (3, n) 或 (3,)

    返回:
        np.ndarray: 参考坐标 ξ
    """
    if x.ndim == 1:
        x = x.reshape(3, 1)
        squeeze = True
    else:
        squeeze = False

    J = np.column_stack((t[:, 1] - t[:, 0], t[:, 2] - t[:, 0], t[:, 3] - t[:, 0]))
    detJ = np.linalg.det(J)
    if abs(detJ) < 1e-14:
        raise ValueError("physical_to_reference_tet4: Jacobian 行列式接近零")

    rhs = x - t[:, 0][:, None]
    xi = np.linalg.solve(J, rhs)

    if squeeze:
        return xi[:, 0]
    return xi


def reference_tet4_sample(n):
    """
    在参考四面体内随机采样 n 个点（均匀分布）。

    采样方法：在单位立方体中均匀采样后拒绝不在四面体中的点；
    或直接利用体积坐标 (u,v,w) 的变换:
        ξ = u
        η = v * (1 - u)
        ζ = w * (1 - u - v)
    其中 u,v,w ~ Uniform(0,1)，并通过排序保证约束。

    参数:
        n (int): 采样点数

    返回:
        np.ndarray: shape (3, n)
    """
    samples = np.random.rand(3, n)
    s1 = np.sort(samples, axis=0)
    xi = np.zeros((3, n))
    xi[0, :] = s1[0, :]
    xi[1, :] = s1[1, :] - s1[0, :]
    xi[2, :] = s1[2, :] - s1[1, :]
    return xi


def build_fem_mass_matrix(nodes, tetrahedra, rho=None):
    """
    构造有限元一致质量矩阵。

    M_{ij} = ∫_Ω ρ φ_i φ_j dΩ ≈ Σ_e ρ_e V_e/20 * (1+δ_{ij})

    参数:
        nodes (np.ndarray): 节点坐标, shape (n_node, 3)
        tetrahedra (np.ndarray): 单元连接, shape (n_elem, 4)
        rho (np.ndarray or float): 密度场，默认 1.0

    返回:
        np.ndarray: 质量矩阵, shape (n_node, n_node)
    """
    n_node = nodes.shape[0]
    n_elem = tetrahedra.shape[0]
    M = np.zeros((n_node, n_node))

    if rho is None:
        rho = 1.0
    if np.isscalar(rho):
        rho_vals = np.full(n_elem, rho)
    else:
        rho_vals = np.asarray(rho)

    for e in range(n_elem):
        idx = tetrahedra[e]
        t = nodes[idx].T  # (3, 4)
        vol = tetrahedron_volume(t)
        if vol < 1e-15:
            continue
        fac = rho_vals[e] * vol / 20.0
        for i in range(4):
            for j in range(4):
                ii = idx[i]
                jj = idx[j]
                add = fac * (2.0 if i == j else 1.0)
                M[ii, jj] += add
    return M


def build_fem_stiffness_matrix(nodes, tetrahedra, mu_field=None):
    """
    构造有限元刚度矩阵（扩散算子）。

    K_{ij} = ∫_Ω μ ∇φ_i · ∇φ_j dΩ

    对于线性四面体，梯度 ∇φ_i 为常数，故:
        K_{ij} = μ_e V_e * (∇φ_i · ∇φ_j)

    参数:
        nodes (np.ndarray): 节点坐标
        tetrahedra (np.ndarray): 单元连接
        mu_field (np.ndarray or float): 扩散系数场

    返回:
        np.ndarray: 刚度矩阵
    """
    n_node = nodes.shape[0]
    n_elem = tetrahedra.shape[0]
    K = np.zeros((n_node, n_node))

    if mu_field is None:
        mu_field = 1.0
    if np.isscalar(mu_field):
        mu_vals = np.full(n_elem, mu_field)
    else:
        mu_vals = np.asarray(mu_field)

    for e in range(n_elem):
        idx = tetrahedra[e]
        t = nodes[idx].T
        vol = tetrahedron_volume(t)
        if vol < 1e-15:
            continue

        # 计算梯度: ∇φ = inv(J^T) * ∇_ref φ
        J = np.column_stack((t[:, 1] - t[:, 0], t[:, 2] - t[:, 0], t[:, 3] - t[:, 0]))
        try:
            invJT = np.linalg.inv(J.T)
        except np.linalg.LinAlgError:
            continue

        # 参考梯度
        grad_ref = np.array([[-1.0, -1.0, -1.0],
                             [ 1.0,  0.0,  0.0],
                             [ 0.0,  1.0,  0.0],
                             [ 0.0,  0.0,  1.0]])  # shape (4, 3)
        grad_phys = grad_ref @ invJT  # (4, 3)

        fac = mu_vals[e] * vol
        for i in range(4):
            for j in range(4):
                ii = idx[i]
                jj = idx[j]
                K[ii, jj] += fac * np.dot(grad_phys[i], grad_phys[j])
    return K
