"""
mesh_vertex_to_element.py — 网格顶点-单元关联与 2D FEM 组装

科学背景
========
在 2D 有限元方法中, 需要建立顶点 (vertex) 到单元 (element) 的
关联结构 (vertex-to-element, VTOE). 这对于:
1. 组装全局刚度矩阵
2. 施加节点载荷
3. 后处理梯度计算
至关重要.

算法来源 (种子项目 756_mesh_vtoe, 414_fem2d_scalar_display)
==========================================================
种子 756: 从 ETOV (element-to-vertex) 构造 VTOE
  - VTOE_POINTER[v] 到 VTOE_POINTER[v+1]-1 存储含顶点 v 的所有单元
种子 414: 2D FEM 标量场的三角剖分与可视化

在本项目中的角色
================
1. 构造 2D 随机域上的三角剖分
2. 建立顶点-单元关联 (用于 FEM 组装)
3. 计算 2D 随机场上的 FEM 插值

核心公式
========
1. 线性三角形刚度矩阵:
   K_e = (area_e)^{-1} · B_e^T · D · B_e
2. 全局组装:  K[I,J] += K_e[local_I, local_J]
3. 面积坐标:  L_i = (a_i + b_i·x + c_i·y) / (2·A_e)
"""

import numpy as np


def build_vtoe(etov, v_num):
    """从 ETOV 构造 VTOE (种子 756).

    参数
    ----
    etov : ndarray, shape (e_order, e_num)
        element-to-vertex 映射
    v_num : int
        顶点总数

    返回
    ----
    vtoe_pointer : ndarray, shape (v_num+1,)
    vtoe : ndarray
        顶点 v 关联的单元为 vtoe[vtoe_pointer[v]:vtoe_pointer[v+1]]
    """
    e_order, e_num = etov.shape

    # 计数: 每个顶点被多少个单元包含
    v_count = np.zeros(v_num, dtype=int)
    for e in range(e_num):
        for local_v in range(e_order):
            global_v = etov[local_v, e]
            v_count[global_v] += 1

    # 构造指针
    vtoe_pointer = np.zeros(v_num + 1, dtype=int)
    for v in range(v_num):
        vtoe_pointer[v + 1] = vtoe_pointer[v] + v_count[v]

    # 填充 VTOE
    vtoe = np.zeros(vtoe_pointer[v_num], dtype=int)
    v_cursor = vtoe_pointer[:-1].copy()

    for e in range(e_num):
        for local_v in range(e_order):
            global_v = etov[local_v, e]
            vtoe[v_cursor[global_v]] = e
            v_cursor[global_v] += 1

    return vtoe_pointer, vtoe


def create_2d_triangulation(x_nodes, y_nodes):
    """在规则矩形网格上创建简单三角剖分.

    将每个矩形单元分为两个三角形.

    参数
    ----
    x_nodes : ndarray, shape (nx,)
    y_nodes : ndarray, shape (ny,)

    返回
    ----
    nodes_xy : ndarray, shape (v_num, 2)
    etov : ndarray, shape (3, e_num)
    """
    nx = len(x_nodes)
    ny = len(y_nodes)
    v_num = nx * ny

    # 节点坐标
    xx, yy = np.meshgrid(x_nodes, y_nodes, indexing='ij')
    nodes_xy = np.column_stack([xx.ravel(), yy.ravel()])

    # 三角剖分: 每个矩形分为 2 个三角形
    e_num = 2 * (nx - 1) * (ny - 1)
    etov = np.zeros((3, e_num), dtype=int)

    e_idx = 0
    for j in range(ny - 1):
        for i in range(nx - 1):
            v00 = i * ny + j
            v10 = (i + 1) * ny + j
            v01 = i * ny + (j + 1)
            v11 = (i + 1) * ny + (j + 1)

            # 三角形 1: v00, v10, v01
            etov[:, e_idx] = [v00, v10, v01]
            e_idx += 1
            # 三角形 2: v10, v11, v01
            etov[:, e_idx] = [v10, v11, v01]
            e_idx += 1

    return nodes_xy, etov


def compute_triangle_geometry(nodes_xy, etov):
    """计算三角形的面积和几何信息.

    参数
    ----
    nodes_xy : ndarray, shape (v_num, 2)
    etov : ndarray, shape (3, e_num)

    返回
    ----
    areas : ndarray, shape (e_num,)
    centroids : ndarray, shape (e_num, 2)
    """
    e_num = etov.shape[1]
    areas = np.zeros(e_num)
    centroids = np.zeros((e_num, 2))

    for e in range(e_num):
        v = etov[:, e]
        xy = nodes_xy[v]  # shape (3, 2)
        x0, y0 = xy[0]
        x1, y1 = xy[1]
        x2, y2 = xy[2]
        # 有向面积
        area = 0.5 * abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
        areas[e] = area
        centroids[e] = np.mean(xy, axis=0)

    return areas, centroids


def assemble_fem_laplacian_2d(nodes_xy, etov, kappa_field_at_nodes):
    """组装 2D Laplace 算子的 FEM 刚度矩阵.

    -∇·(κ∇u) = f 的弱形式:
        K·U = F
    其中 K[I,J] = Σ_e ∫_e κ·∇φ_I·∇φ_J dx

    对于线性三角形:
        ∇φ_i = (1/(2A)) · [y_{i+1}-y_{i+2}, x_{i+2}-x_{i+1}]

    参数
    ----
    nodes_xy : ndarray, shape (v_num, 2)
    etov : ndarray, shape (3, e_num)
    kappa_field_at_nodes : ndarray, shape (v_num,)
        各节点处的扩散系数

    返回
    ----
    K : ndarray, shape (v_num, v_num)
        全局刚度矩阵
    """
    v_num = nodes_xy.shape[0]
    e_num = etov.shape[1]
    K = np.zeros((v_num, v_num))

    for e in range(e_num):
        v = etov[:, e]
        xy = nodes_xy[v]

        # 面积
        x0, y0 = xy[0]
        x1, y1 = xy[1]
        x2, y2 = xy[2]
        area = 0.5 * abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
        if area < 1.0e-30:
            continue

        # 梯度系数 b_i, c_i
        b = np.array([y1 - y2, y2 - y0, y0 - y1])
        c = np.array([x2 - x1, x0 - x2, x1 - x0])

        # 单元内 κ (取平均)
        kappa_e = np.mean(kappa_field_at_nodes[v])

        # 单元刚度矩阵: K_e[i,j] = κ·(b_i·b_j + c_i·c_j) / (4·A)
        for i in range(3):
            for j in range(3):
                K_e_ij = kappa_e * (b[i] * b[j] + c[i] * c[j]) / (4.0 * area)
                K[v[i], v[j]] += K_e_ij

    return K
