"""
mesh_boundary.py
================================================================================
网格边界处理模块 —— 基于种子项目 1332_triangulation_boundary_edges
与 790_navier_stokes_mesh3d

在行星边界层模拟中，边界条件的精确处理至关重要：
- 下边界（地表）：无滑移条件 u=0，给定热通量或温度
- 上边界（自由大气）：应力自由，或给定辐射边界条件
- 侧边界：周期性或开放边界

本模块提供二维/三维三角化网格的边界边/面识别，以及边界节点标记。

核心物理公式
--------------------------------------------------------------------------------
边界层近地层（surface layer）Monin-Obukhov 相似理论：
    u(z) = (u* / κ) [ ln(z/z_0) - ψ_m(ζ) ]

其中 ζ = z/L 为稳定度参数，L = -u*³ θ_v / (κ g w'θ'_v) 为 Obukhov 长度，
κ ≈ 0.4 为 von Kármán 常数，u* 为摩擦速度。

壁面剪切应力：
    τ_w = ρ u*²

对于 LES 中的壁面模型（wall model），下边界速度不是直接设为零，
而是通过对数律或强制边界条件施加。
"""

import numpy as np


def triangulation_boundary_edges(triangles):
    """
    识别二维三角网格中的边界边。

    参数
    ----------
    triangles : np.ndarray, shape (n_tri, 3)
        每个三角形由三个节点索引组成（0-based）

    返回
    -------
    boundary_edges : np.ndarray, shape (n_edge, 2)
        边界边列表，每条边由两个节点索引组成
    """
    triangles = np.asarray(triangles)
    if triangles.shape[1] != 3:
        raise ValueError("triangulation_boundary_edges: 三角形必须为 3 节点")

    # 统计每条边的出现次数
    edge_dict = {}
    for tri in triangles:
        edges = [
            tuple(sorted((tri[0], tri[1]))),
            tuple(sorted((tri[1], tri[2]))),
            tuple(sorted((tri[2], tri[0]))),
        ]
        for e in edges:
            edge_dict[e] = edge_dict.get(e, 0) + 1

    # 只出现一次的边为边界边
    boundary_edges = [list(e) for e, count in edge_dict.items() if count == 1]
    return np.array(boundary_edges, dtype=int)


def extract_boundary_nodes_3d(element_nodes, all_nodes, lower_z=0.0, upper_z=1.0, tol=1e-6):
    """
    从三维四面体网格中提取边界节点，并分类为下边界、上边界和侧边界。

    参数
    ----------
    element_nodes : np.ndarray, shape (n_elem, 4)
        四面体节点索引
    all_nodes : np.ndarray, shape (n_node, 3)
        节点坐标 [x, y, z]
    lower_z : float
        下边界 z 坐标（地表）
    upper_z : float
        上边界 z 坐标（PBL 顶）
    tol : float
        容差

    返回
    -------
    bdry_info : dict
        {'lower': array, 'upper': array, 'side': array, 'all': array}
    """
    n_node = all_nodes.shape[0]

    # 标记位于下边界和上边界上的节点
    lower_nodes = np.where(np.abs(all_nodes[:, 2] - lower_z) < tol)[0]
    upper_nodes = np.where(np.abs(all_nodes[:, 2] - upper_z) < tol)[0]

    # 侧边界：通过边界面法向量判断
    # 简化处理：计算所有边界面的外法向量，z 分量接近 0 的为侧面
    side_nodes = set()

    # 统计每个面的出现次数
    face_dict = {}
    for elem in element_nodes:
        faces = [
            tuple(sorted((elem[0], elem[1], elem[2]))),
            tuple(sorted((elem[0], elem[1], elem[3]))),
            tuple(sorted((elem[0], elem[2], elem[3]))),
            tuple(sorted((elem[1], elem[2], elem[3]))),
        ]
        for f in faces:
            face_dict[f] = face_dict.get(f, 0) + 1

    # 只出现一次的面为边界面
    boundary_faces = [f for f, count in face_dict.items() if count == 1]

    for f in boundary_faces:
        p1, p2, p3 = all_nodes[f[0]], all_nodes[f[1]], all_nodes[f[2]]
        normal = np.cross(p2 - p1, p3 - p1)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-15:
            continue
        normal = normal / normal_norm

        # 如果法向量 z 分量很小，认为是侧边界
        if abs(normal[2]) < 0.5:
            side_nodes.update(f)

    side_nodes = np.array(sorted(side_nodes), dtype=int)

    return {
        'lower': lower_nodes,
        'upper': upper_nodes,
        'side': side_nodes,
        'all': np.unique(np.concatenate([lower_nodes, upper_nodes, side_nodes]))
    }


def apply_surface_layer_bc(u, v, w, theta, nodes, lower_nodes, u_star=0.3, z0=0.1, kappa=0.4):
    """
    应用近地层 Monin-Obukhov 边界条件。

    参数
    ----------
    u, v, w : np.ndarray, shape (n_node,)
        速度分量
    theta : np.ndarray, shape (n_node,)
        位温（K）
    nodes : np.ndarray, shape (n_node, 3)
        节点坐标
    lower_nodes : np.ndarray
        下边界节点索引
    u_star : float
        摩擦速度（m/s）
    z0 : float
        粗糙度长度（m）
    kappa : float
        von Kármán 常数

    返回
    -------
    u, v, w : np.ndarray
        更新后的速度场
    """
    u_new = np.copy(u)
    v_new = np.copy(v)
    w_new = np.copy(w)

    for idx in lower_nodes:
        z = nodes[idx, 2]
        if z < z0:
            z = z0

        # 对数律：u(z) = u* / κ * ln(z/z0)
        u_mag = (u_star / kappa) * np.log(z / z0)

        # 保持风向不变，仅调整大小
        u_old = u[idx]
        v_old = v[idx]
        uv_mag = np.sqrt(u_old**2 + v_old**2)

        if uv_mag > 1e-12:
            scale = u_mag / uv_mag
            u_new[idx] = u_old * scale
            v_new[idx] = v_old * scale
        else:
            # 默认沿 x 方向
            u_new[idx] = u_mag
            v_new[idx] = 0.0

        w_new[idx] = 0.0  # 无穿透

    return u_new, v_new, w_new
