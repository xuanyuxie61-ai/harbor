"""
reactor_mesh.py
===============
基于 753_mesh_boundary 与 446_fractal_coastline 改造的网格生成与边界处理模块。

气泡柱反应器通常为圆柱形，但工业装置中内部构件（换热管束、分布板）
导致复杂边界。本模块提供：
1. 结构化圆柱网格生成
2. 网格边界段提取（mesh_boundary）
3. 边界几何扰动（coastline_perturb）用于粗糙壁面或分布板孔道的分形建模
4. 网格正交性检查

核心公式
--------
1. 圆柱坐标结构化网格：
       (r_i, z_j) = (i · Δr, j · Δz),  i=0..Nr, j=0..Nz
       x = r cosθ, y = r sinθ

2. 网格边界提取算法：
       对多边形单元列表 element_node[e][v]，
       将每个单元分解为边 segment = [(v1,v2), (v2,v3), ..., (v_n,v1)]。
       对全局边列表排序后，仅出现一次的边（无反向匹配）即为边界边。

3. 边界分形扰动（Coastline）：
       q_{2i} = p_i                         （原始节点）
       q_{2i+1} = 0.5(p_i + p_{i+1}) + w_i · perturb_i
       perturb_i = 0.5(p_i + p_{i+1}) + w(p_i + p_{i+1})
                   - w(p_{i-1} + p_{i+2})
       w ~ N(μ, μ²)   (μ 控制扰动幅度)

4. Jacobian 行列式（坐标变换）：
       J = |∂(x,y)/∂(ξ,η)| = |x_ξ y_η - x_η y_ξ|
       用于网格质量评价：|J| > 0 保证映射非奇异。
"""

import numpy as np


# ---------------------------------------------------------------------------
# Mesh boundary extraction (from 753_mesh_boundary)
# ---------------------------------------------------------------------------

def mesh_boundary_segments(element_node):
    """
    从单元节点连接表提取网格边界线段序列。

    Parameters
    ----------
    element_node : ndarray, shape (n_elements, n_vertices)
        单元顶点编号（逆时针）。

    Returns
    -------
    boundary_segments : ndarray, shape (n_boundary, 2)
        边界线段，按逆时针顺序排列。
    """
    element_node = np.asarray(element_node, dtype=int)
    n_elements, n_vertices = element_node.shape

    n_segments = n_elements * n_vertices
    segments = np.zeros((n_segments, 2), dtype=int)

    s = 0
    for e in range(n_elements):
        j = n_vertices - 1
        for jp1 in range(n_vertices):
            segments[s, 0] = element_node[e, j]
            segments[s, 1] = element_node[e, jp1]
            j = jp1
            s += 1

    # 对每段排序（小节点在前）用于匹配
    segments_sorted = np.sort(segments, axis=1)
    # 使用字典统计
    seg_dict = {}
    for i in range(n_segments):
        key = (segments_sorted[i, 0], segments_sorted[i, 1])
        # 记录原始方向
        direction = 1 if segments[i, 0] < segments[i, 1] else -1
        if key not in seg_dict:
            seg_dict[key] = [direction]
        else:
            seg_dict[key].append(direction)

    boundary = []
    for key, dirs in seg_dict.items():
        # 若仅出现一次，则为边界
        if len(dirs) == 1:
            if dirs[0] == 1:
                boundary.append([key[0], key[1]])
            else:
                boundary.append([key[1], key[0]])

    if not boundary:
        return np.empty((0, 2), dtype=int)

    boundary = np.array(boundary, dtype=int)

    # 重排成连续环
    n_b = boundary.shape[0]
    for b1 in range(n_b - 1):
        for b2 in range(b1 + 1, n_b):
            if boundary[b2, 0] == boundary[b1, 1]:
                boundary[[b1 + 1, b2]] = boundary[[b2, b1 + 1]]
                break

    return boundary


# ---------------------------------------------------------------------------
# Coastline perturbation (from 446_fractal_coastline)
# ---------------------------------------------------------------------------

def boundary_perturb(p, mu=0.1, seed=42):
    """
    对闭合多边形边界施加分形扰动，模拟粗糙壁面或孔道几何。

    Parameters
    ----------
    p : ndarray, shape (n, 2)
        原始多边形节点坐标（闭合，逆时针）。
    mu : float
        扰动控制参数，推荐 0 ≤ μ ≤ 0.25。
    seed : int
        随机种子。

    Returns
    -------
    q : ndarray, shape (2n, 2)
        扰动后的边界（节点数翻倍）。
    """
    rng = np.random.default_rng(seed)
    p = np.asarray(p, dtype=float)
    n, d = p.shape

    sig = mu ** 2
    w = mu + sig * rng.standard_normal(n)
    w = np.clip(w, -0.5, 0.5)

    # circshift 模拟
    p_prev = np.roll(p, 1, axis=0)
    p_next = np.roll(p, -1, axis=0)
    p_next2 = np.roll(p, -2, axis=0)

    w_col = w.reshape(-1, 1)
    perturb = (0.5 * (p + p_prev)
               + w_col * (p + p_prev)
               - w_col * (p_next + p_next2))

    perturb = np.roll(perturb, -1, axis=0)

    q = np.zeros((2 * n, d))
    q[0:2 * n:2] = p
    q[1:2 * n:2] = perturb

    return q


# ---------------------------------------------------------------------------
# Structured cylindrical mesh generation
# ---------------------------------------------------------------------------

def generate_cylindrical_mesh(R, H, Nr, Nz):
    """
    生成二维轴对称结构化网格（r-z 平面）。

    Parameters
    ----------
    R : float
        反应器半径 [m]。
    H : float
        反应器高度 [m]。
    Nr, Nz : int
        径向与轴向网格数。

    Returns
    -------
    nodes : ndarray, shape ((Nr+1)*(Nz+1), 2)
        节点坐标 (r, z)。
    elements : ndarray, shape (Nr*Nz, 4)
        四边形单元节点编号（逆时针）。
    """
    dr = R / Nr
    dz = H / Nz

    n_nodes = (Nr + 1) * (Nz + 1)
    nodes = np.zeros((n_nodes, 2))

    for j in range(Nz + 1):
        for i in range(Nr + 1):
            idx = j * (Nr + 1) + i
            nodes[idx, 0] = i * dr
            nodes[idx, 1] = j * dz

    n_elements = Nr * Nz
    elements = np.zeros((n_elements, 4), dtype=int)

    for j in range(Nz):
        for i in range(Nr):
            e = j * Nr + i
            n1 = j * (Nr + 1) + i
            n2 = n1 + 1
            n4 = (j + 1) * (Nr + 1) + i
            n3 = n4 + 1
            elements[e] = [n1, n2, n3, n4]

    return nodes, elements


def compute_jacobian_2d(nodes, element):
    """
    计算二维四边形单元的 Jacobian 行列式（在单元中心近似）。

    Parameters
    ----------
    nodes : ndarray, shape (n_nodes, 2)
    element : ndarray, shape (4,)
        单元节点编号。

    Returns
    -------
    J : float
        中心点 Jacobian 近似值。
    """
    x = nodes[element, 0]
    y = nodes[element, 1]

    # 使用叉积近似计算
    # 对角线向量
    dx1 = x[2] - x[0]
    dy1 = y[2] - y[0]
    dx2 = x[3] - x[1]
    dy2 = y[3] - y[1]

    J = 0.5 * abs(dx1 * dy2 - dx2 * dy1)
    return J


def mesh_quality_report(nodes, elements):
    """
    网格质量报告：最小 Jacobian、平均 Jacobian、边界段数。
    """
    jac_values = []
    for e in elements:
        jac = compute_jacobian_2d(nodes, e)
        jac_values.append(jac)

    jac_values = np.array(jac_values)
    boundary = mesh_boundary_segments(elements)

    return {
        'jacobian_min': float(np.min(jac_values)),
        'jacobian_mean': float(np.mean(jac_values)),
        'jacobian_negative_count': int(np.sum(jac_values <= 0)),
        'n_elements': elements.shape[0],
        'n_nodes': nodes.shape[0],
        'n_boundary_segments': boundary.shape[0],
    }
