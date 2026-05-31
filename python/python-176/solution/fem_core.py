"""
fem_core.py
================================================================================
二维有限元方法（FEM）核心离散模块

本模块融合以下种子项目的核心算法：
  - 418_fem3d_project   : 3D FEM L2 投影、质量矩阵组装、TET4 基函数思想
  - 1319_triangle_symq_to_ref : 参考三角形上的对称求积规则转换

科学背景
--------
在最优控制伴随方程方法中，PDE 约束的有限元离散是核心步骤。
对于二维椭圆域上的反应-扩散方程：
    ∂y/∂t − ν Δy + c y³ = f    in Ω × (0,T)
    ν ∂y/∂n = q                on ∂Ω × (0,T)
其弱形式为：对任意测试函数 v ∈ H¹(Ω)，
    ∫_Ω ∂y/∂t · v + ν ∇y·∇v + c y³ v = ∫_Ω f v + ν ∫_{∂Ω} q v

使用 P1（线性）有限元离散，状态 y 近似为
    y_h(x,t) = Σ_{i=1}^{N} y_i(t) φ_i(x)
其中 φ_i 是节点 i 上的帽子函数（hat function）。

半离散系统为：
    M dy/dt + ν A y + c M(y³) = F + B q
其中
  - M_{ij} = ∫_Ω φ_i φ_j          (质量矩阵)
  - A_{ij} = ∫_Ω ∇φ_i · ∇φ_j      (刚度矩阵)
  - F_i    = ∫_Ω f φ_i            (载荷向量）
  - B_{ij} = ν ∫_{∂Ω} φ_i φ_j     (边界控制矩阵)

三角形 P1 元的局部矩阵公式
---------------------------
对于顶点为 (x1,y1), (x2,y2), (x3,y3) 的三角形单元 T：
    b1 = y2 − y3,  b2 = y3 − y1,  b3 = y1 − y2
    c1 = x3 − x2,  c2 = x1 − x3,  c3 = x2 − x1
    |T| = ½ |b1·c2 − b2·c1|

局部质量矩阵：
    M_loc = |T|/12 · [[2, 1, 1], [1, 2, 1], [1, 1, 2]]

局部刚度矩阵：
    A_loc = 1/(4|T|) · (b_i b_j + c_i c_j)_{i,j=1}^3
"""

import numpy as np


def assemble_fem_matrices(nodes, elements, boundary_nodes, nu=1.0):
    """
    组装 FEM 质量矩阵 M、刚度矩阵 A 和边界控制矩阵 B。

    参数
    ----
    nodes          : (N_nodes, 2) 节点坐标
    elements       : (N_elements, 3) 三角形单元
    boundary_nodes : 边界节点索引列表
    nu             : 扩散系数 ν

    返回
    ----
    M : 质量矩阵（稠密）
    A : 刚度矩阵（稠密）
    B : 边界控制矩阵（稠密，仅边界节点非零）
    """
    n_nodes = nodes.shape[0]
    M = np.zeros((n_nodes, n_nodes), dtype=float)
    A = np.zeros((n_nodes, n_nodes), dtype=float)
    B = np.zeros((n_nodes, n_nodes), dtype=float)

    # 1) 遍历单元，组装 M 和 A
    for e in elements:
        i, j, k = e
        p1, p2, p3 = nodes[i], nodes[j], nodes[k]

        b1 = p2[1] - p3[1]
        b2 = p3[1] - p1[1]
        b3 = p1[1] - p2[1]
        c1 = p3[0] - p2[0]
        c2 = p1[0] - p3[0]
        c3 = p2[0] - p1[0]

        area = 0.5 * abs(b1 * c2 - b2 * c1)
        if area < 1.0e-15:
            continue

        # 局部质量矩阵
        m_loc = (area / 12.0) * np.array([[2.0, 1.0, 1.0],
                                          [1.0, 2.0, 1.0],
                                          [1.0, 1.0, 2.0]])
        # 局部刚度矩阵
        a_loc = np.zeros((3, 3), dtype=float)
        inv4area = 1.0 / (4.0 * area)
        bb = [b1, b2, b3]
        cc = [c1, c2, c3]
        for ii in range(3):
            for jj in range(3):
                a_loc[ii, jj] = inv4area * (bb[ii] * bb[jj] + cc[ii] * cc[jj])

        # 组装到全局
        idx = [i, j, k]
        for ii in range(3):
            for jj in range(3):
                M[idx[ii], idx[jj]] += m_loc[ii, jj]
                A[idx[ii], idx[jj]] += a_loc[ii, jj]

    # 2) 组装边界控制矩阵 B（仅边界边贡献）
    # 边界边只属于一个三角形，通过 edge_count 找到
    edge_count = {}
    for e in elements:
        edges = [(e[0], e[1]), (e[1], e[2]), (e[2], e[0])]
        for v1, v2 in edges:
            key = tuple(sorted((int(v1), int(v2))))
            edge_count[key] = edge_count.get(key, 0) + 1

    boundary_edges = [e for e, c in edge_count.items() if c == 1]

    for v1, v2 in boundary_edges:
        p1 = nodes[v1]
        p2 = nodes[v2]
        length = np.linalg.norm(p2 - p1)
        if length < 1.0e-15:
            continue
        # 边界质量矩阵（1D P1 元）
        b_loc = (length / 6.0) * np.array([[2.0, 1.0],
                                           [1.0, 2.0]])
        B[v1, v1] += nu * b_loc[0, 0]
        B[v1, v2] += nu * b_loc[0, 1]
        B[v2, v1] += nu * b_loc[1, 0]
        B[v2, v2] += nu * b_loc[1, 1]

    return M, A, B, boundary_edges


def assemble_rhs_source(nodes, elements, f_fn, t=0.0):
    """
    组装源项载荷向量 F_i = ∫_Ω f(x,t) φ_i dx。
    使用单元重心近似：f 在单元上取重心处的值，乘以局部质量矩阵。
    """
    n_nodes = nodes.shape[0]
    F = np.zeros(n_nodes, dtype=float)

    for e in elements:
        i, j, k = e
        p1, p2, p3 = nodes[i], nodes[j], nodes[k]
        xc = (p1[0] + p2[0] + p3[0]) / 3.0
        yc = (p1[1] + p2[1] + p3[1]) / 3.0
        f_val = f_fn(xc, yc, t)

        area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
        if area < 1.0e-15:
            continue

        # 局部载荷：f_val * [area/3, area/3, area/3]
        # 等价于 M_loc * [f_val, f_val, f_val]
        F[i] += area * f_val / 3.0
        F[j] += area * f_val / 3.0
        F[k] += area * f_val / 3.0

    return F


def evaluate_fem_at_point(nodes, elements, y_coeffs, xq, yq):
    """
    使用 P1 有限元插值，在查询点 (xq, yq) 处计算解的值。
    通过寻找包含该点的三角形，然后计算重心坐标实现插值。

    边界处理：若点不在任何三角形内，返回最近节点的值。
    """
    # 简单方法：计算到所有节点的距离，找到最近的三个节点
    # 更好的方法：遍历所有三角形，检查点是否在三角形内
    best_bary = None
    best_elem = -1
    min_neg = -1.0e-6

    for idx, e in enumerate(elements):
        p1, p2, p3 = nodes[e]
        # 计算重心坐标
        denom = (p2[1] - p3[1]) * (p1[0] - p3[0]) + (p3[0] - p2[0]) * (p1[1] - p3[1])
        if abs(denom) < 1.0e-15:
            continue
        w1 = ((p2[1] - p3[1]) * (xq - p3[0]) + (p3[0] - p2[0]) * (yq - p3[1])) / denom
        w2 = ((p3[1] - p1[1]) * (xq - p3[0]) + (p1[0] - p3[0]) * (yq - p3[1])) / denom
        w3 = 1.0 - w1 - w2
        if w1 >= min_neg and w2 >= min_neg and w3 >= min_neg:
            # 找到包含该点的三角形
            w1 = max(w1, 0.0)
            w2 = max(w2, 0.0)
            w3 = max(w3, 0.0)
            s = w1 + w2 + w3
            w1 /= s
            w2 /= s
            w3 /= s
            return w1 * y_coeffs[e[0]] + w2 * y_coeffs[e[1]] + w3 * y_coeffs[e[2]]

    # 回退：最近节点
    dists = (nodes[:, 0] - xq) ** 2 + (nodes[:, 1] - yq) ** 2
    return y_coeffs[np.argmin(dists)]


def l2_projection(nodes, elements, g_fn, t=0.0):
    """
    L2 投影：求解 M u = b，其中 b_i = ∫_Ω g(x,t) φ_i dx。
    这是将任意函数投影到有限元空间的经典操作，
    融合 418_fem3d_project 的 L2-Galerkin 投影思想。
    """
    M, _, _, _ = assemble_fem_matrices(nodes, elements, [])
    n_nodes = nodes.shape[0]
    b = np.zeros(n_nodes, dtype=float)

    for e in elements:
        i, j, k = e
        p1, p2, p3 = nodes[i], nodes[j], nodes[k]
        area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
        if area < 1.0e-15:
            continue
        # 使用 3 点求积（重心）
        xc = (p1[0] + p2[0] + p3[0]) / 3.0
        yc = (p1[1] + p2[1] + p3[1]) / 3.0
        g_val = g_fn(xc, yc, t)

        b[i] += area * g_val / 3.0
        b[j] += area * g_val / 3.0
        b[k] += area * g_val / 3.0

    u = np.linalg.solve(M, b)
    return u


def fem_norm_l2(nodes, elements, y_coeffs):
    """
    计算 FEM 解的 L2 范数：‖y_h‖_{L²} = √(y^T M y)
    """
    M, _, _, _ = assemble_fem_matrices(nodes, elements, [])
    val = np.dot(y_coeffs, M @ y_coeffs)
    return np.sqrt(max(val, 0.0))


def fem_norm_h1(nodes, elements, y_coeffs, nu=1.0):
    """
    计算 FEM 解的 H1 范数：‖y_h‖_{H¹} = √(y^T M y + ν y^T A y)
    """
    M, A, _, _ = assemble_fem_matrices(nodes, elements, [], nu)
    val = np.dot(y_coeffs, M @ y_coeffs) + nu * np.dot(y_coeffs, A @ y_coeffs)
    return np.sqrt(max(val, 0.0))
