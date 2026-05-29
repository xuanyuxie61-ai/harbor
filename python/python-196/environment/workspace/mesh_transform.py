"""
mesh_transform.py
网格几何变换与自适应细化模块

包含：
- 2D/3D仿射变换：旋转、平移、缩放、线性变换（源自 hand_dilation, hand_rotation, hand_translation, hand_linear）
- 多边形表面节点拓扑处理（源自 polygonal_surface_display）
- 网格质量评估与自适应细化标记

科学背景：
在异构HPC任务调度中，有限元网格需要根据热-电耦合场的梯度进行
自适应细化。网格变换算子用于将粗网格映射到细网格，同时保持
单元质量。
"""

import numpy as np


def rotation_matrix_2d(angle_rad):
    """
    2D旋转矩阵（源自 hand_rotation）。

    R(theta) = [[cos(theta), -sin(theta)],
                [sin(theta),  cos(theta)]]
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[c, -s], [s, c]], dtype=float)


def dilation_matrix_2d(sx, sy):
    """
    2D缩放矩阵（源自 hand_dilation）。

    D = [[sx,  0],
         [ 0, sy]]
    """
    return np.array([[sx, 0.0], [0.0, sy]], dtype=float)


def translation_vector_2d(tx, ty, n_points):
    """
    生成平移向量（源自 hand_translation）。
    """
    return np.tile(np.array([[tx], [ty]], dtype=float), (1, n_points))


def affine_transform_2d(points, A=None, b=None):
    """
    2D仿射变换: y = A @ x + b（源自 hand_linear）。

    参数:
        points: ndarray, shape (2, N)
        A: ndarray, shape (2, 2), 线性部分
        b: ndarray, shape (2, N) 或 (2,), 平移部分

    返回:
        transformed: ndarray, shape (2, N)
    """
    points = np.array(points, dtype=float)
    if points.shape[0] != 2:
        raise ValueError("points must have shape (2, N)")
    if A is None:
        A = np.eye(2)
    if b is None:
        b = np.zeros((2, points.shape[1]))
    elif b.ndim == 1:
        b = np.tile(b[:, np.newaxis], (1, points.shape[1]))
    return A @ points + b


def transform_mesh(nodes, elements, A=None, b=None):
    """
    对有限元网格进行仿射变换。
    保持拓扑(elements)不变，仅变换节点坐标。

    参数:
        nodes: ndarray, shape (2, node_num)
        elements: ndarray, shape (3, elem_num), 三角形单元节点索引
        A, b: 变换参数

    返回:
        new_nodes, elements
    """
    new_nodes = affine_transform_2d(nodes, A, b)
    return new_nodes, elements


def polygon_surface_quality(nodes, elements):
    """
    计算三角形网格的质量指标（源自 polygonal_surface_display 的网格处理思想）。

    对每个三角形单元计算面积与边长比：
        Q = 4 * sqrt(3) * A / (a^2 + b^2 + c^2)
    Q 的取值范围 (0, 1]，1 为等边三角形。

    参数:
        nodes: ndarray, shape (2, N)
        elements: ndarray, shape (3, M)

    返回:
        quality: ndarray, shape (M,), 各单元质量
        min_quality: float
        mean_quality: float
    """
    nodes = np.array(nodes, dtype=float)
    elements = np.array(elements, dtype=int)
    nelem = elements.shape[1]
    quality = np.zeros(nelem)
    # TODO Hole_3: 完成三角形网格质量指标计算。
    # 关键知识点：
    #   1) 三角形单元面积计算（鞋带公式）
    #   2) 质量指标 Q = 4*sqrt(3)*A / (a^2+b^2+c^2)，取值范围 (0,1]
    #   3) 注意 elements 的索引基制 (0-based 或 1-based) 必须与 fem_thermal_solver.py 中的 Hole_2 保持一致
    raise NotImplementedError("Hole_3: polygon_surface_quality 质量计算循环待实现")
    return quality, float(np.min(quality)), float(np.mean(quality))


def adaptive_refinement_markers(nodes, elements, gradient_field, threshold_ratio=0.8):
    """
    根据梯度场标记需要细化的单元。

    计算每个单元的梯度模:
        |grad T|_e = sqrt( (dTdx)^2 + (dTdy)^2 )
    标记前 threshold_ratio 比例的单元进行细化。

    参数:
        nodes: ndarray, shape (2, N)
        elements: ndarray, shape (3, M)
        gradient_field: ndarray, shape (2, N), 每个节点上的梯度 (dTdx, dTdy)
        threshold_ratio: float, 需要细化的单元比例

    返回:
        marker: ndarray, bool, shape (M,)
    """
    nodes = np.array(nodes, dtype=float)
    elements = np.array(elements, dtype=int)
    gradient_field = np.array(gradient_field, dtype=float)
    nelem = elements.shape[1]
    grad_norm = np.zeros(nelem)
    for e in range(nelem):
        i1, i2, i3 = elements[:, e] - 1
        g1 = np.linalg.norm(gradient_field[:, i1])
        g2 = np.linalg.norm(gradient_field[:, i2])
        g3 = np.linalg.norm(gradient_field[:, i3])
        grad_norm[e] = (g1 + g2 + g3) / 3.0
    sorted_idx = np.argsort(grad_norm)[::-1]
    n_mark = max(1, int(np.ceil(threshold_ratio * nelem)))
    marker = np.zeros(nelem, dtype=bool)
    marker[sorted_idx[:n_mark]] = True
    return marker


def refine_marked_elements(nodes, elements, marker):
    """
    对标记的三角形单元进行1-to-4细化（将每条边中点连接）。

    参数:
        nodes: ndarray, shape (2, N)
        elements: ndarray, shape (3, M)
        marker: ndarray, bool, shape (M,)

    返回:
        new_nodes, new_elements
    """
    nodes = np.array(nodes, dtype=float)
    elements = np.array(elements, dtype=int)
    node_num = nodes.shape[1]
    elem_num = elements.shape[1]

    # 边中点映射: (min(i,j), max(i,j)) -> 新节点索引
    edge_mid = {}
    new_nodes_list = [nodes[:, i] for i in range(node_num)]

    def get_mid(i, j):
        i, j = int(i), int(j)
        if i > j:
            i, j = j, i
        key = (i, j)
        if key not in edge_mid:
            mid_pt = (nodes[:, i] + nodes[:, j]) / 2.0
            idx = node_num + len(edge_mid)
            edge_mid[key] = idx
            new_nodes_list.append(mid_pt)
        return edge_mid[key]

    new_elements_list = []
    for e in range(elem_num):
        i1, i2, i3 = elements[:, e] - 1  # 0-based
        if marker[e]:
            m12 = get_mid(i1, i2)
            m23 = get_mid(i2, i3)
            m31 = get_mid(i3, i1)
            new_elements_list.append([i1 + 1, m12 + 1, m31 + 1])
            new_elements_list.append([m12 + 1, i2 + 1, m23 + 1])
            new_elements_list.append([m31 + 1, m23 + 1, i3 + 1])
            new_elements_list.append([m12 + 1, m23 + 1, m31 + 1])
        else:
            new_elements_list.append([i1 + 1, i2 + 1, i3 + 1])

    new_nodes = np.column_stack(new_nodes_list)
    new_elements = np.array(new_elements_list, dtype=int).T
    return new_nodes, new_elements
