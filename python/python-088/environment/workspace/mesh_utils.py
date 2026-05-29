"""
mesh_utils.py
有限元网格生成与边界处理模块

融入种子项目:
  - 107_boundary_word_equilateral: 等边三角形网格、边界词处理、三角剖分

功能:
  - 规则三角形网格生成（六边形/等边三角形排布）
  - 边界节点标记与处理
  - 三角形单元面积和重心坐标计算
  - 节点重编号优化（减少带宽）
"""

import numpy as np
from typing import List, Tuple, Optional


def generate_triangular_mesh(
    nx: int, ny: int, domain: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    quadratic: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成规则三角形网格，覆盖矩形域 [xmin, xmax] x [ymin, ymax]。

    每个矩形单元被剖分为两个三角形。
    当 quadratic=True 时，生成 T6 二次三角形单元（6节点）。

    T6 节点排布顺序:
        节点 0-2: 三角形顶点（逆时针）
        节点 3-5: 边中点（对应边 0-1, 1-2, 2-0）

    参数:
        nx: x 方向顶点节点数
        ny: y 方向顶点节点数
        domain: (xmin, xmax, ymin, ymax)
        quadratic: 是否生成二次单元

    返回:
        (nodes, elements, boundary_nodes)
        nodes: 形状 (n_nodes, 2)，节点坐标
        elements: 形状 (n_elements, 3) 或 (n_elements, 6)
        boundary_nodes: 边界节点编号数组
    """
    xmin, xmax, ymin, ymax = domain
    n_vertex = nx * ny
    vertices = np.zeros((n_vertex, 2))

    dx = (xmax - xmin) / (nx - 1) if nx > 1 else 1.0
    dy = (ymax - ymin) / (ny - 1) if ny > 1 else 1.0

    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            vertices[idx, 0] = xmin + i * dx
            vertices[idx, 1] = ymin + j * dy

    # 先生成3节点线性单元
    n_elements = 2 * (nx - 1) * (ny - 1)
    lin_elements = np.zeros((n_elements, 3), dtype=int)

    e = 0
    for j in range(ny - 1):
        for i in range(nx - 1):
            n1 = j * nx + i
            n2 = j * nx + (i + 1)
            n3 = (j + 1) * nx + (i + 1)
            n4 = (j + 1) * nx + i

            lin_elements[e, :] = [n1, n2, n3]
            e += 1
            lin_elements[e, :] = [n1, n3, n4]
            e += 1

    if not quadratic:
        # 标记边界节点
        boundary_nodes = []
        for j in range(ny):
            for i in range(nx):
                if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                    boundary_nodes.append(j * nx + i)
        return vertices, lin_elements, np.array(boundary_nodes, dtype=int)

    # ===== 提升到 T6 二次单元 =====
    # 为每条唯一的边创建中点节点
    edge_to_midnode = {}
    midnodes = []

    def get_edge_key(na, nb):
        return (min(na, nb), max(na, nb))

    for elem in lin_elements:
        # 三条边
        edges = [(elem[0], elem[1]), (elem[1], elem[2]), (elem[2], elem[0])]
        for na, nb in edges:
            ekey = get_edge_key(na, nb)
            if ekey not in edge_to_midnode:
                mid_coord = 0.5 * (vertices[na] + vertices[nb])
                mid_idx = n_vertex + len(midnodes)
                edge_to_midnode[ekey] = mid_idx
                midnodes.append(mid_coord)

    # 合并节点数组
    nodes = np.vstack([vertices, np.array(midnodes)])

    # 构建6节点单元
    elements = np.zeros((n_elements, 6), dtype=int)
    for e_idx, elem in enumerate(lin_elements):
        # 顶点
        elements[e_idx, 0] = elem[0]
        elements[e_idx, 1] = elem[1]
        elements[e_idx, 2] = elem[2]
        # 中点
        elements[e_idx, 3] = edge_to_midnode[get_edge_key(elem[0], elem[1])]
        elements[e_idx, 4] = edge_to_midnode[get_edge_key(elem[1], elem[2])]
        elements[e_idx, 5] = edge_to_midnode[get_edge_key(elem[2], elem[0])]

    # 标记边界节点（顶点 + 边中点）
    boundary_nodes_set = set()
    for j in range(ny):
        for i in range(nx):
            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                boundary_nodes_set.add(j * nx + i)

    # 添加边界边的中点
    for e_idx, elem in enumerate(lin_elements):
        edges = [(elem[0], elem[1]), (elem[1], elem[2]), (elem[2], elem[0])]
        for na, nb in edges:
            # 检查边是否在边界上
            xa, ya = vertices[na]
            xb, yb = vertices[nb]
            on_boundary = False
            if abs(xa - xmin) < 1e-10 and abs(xb - xmin) < 1e-10:
                on_boundary = True
            elif abs(xa - xmax) < 1e-10 and abs(xb - xmax) < 1e-10:
                on_boundary = True
            elif abs(ya - ymin) < 1e-10 and abs(yb - ymin) < 1e-10:
                on_boundary = True
            elif abs(ya - ymax) < 1e-10 and abs(yb - ymax) < 1e-10:
                on_boundary = True
            if on_boundary:
                boundary_nodes_set.add(edge_to_midnode[get_edge_key(na, nb)])

    boundary_nodes = np.array(sorted(boundary_nodes_set), dtype=int)

    return nodes, elements, boundary_nodes


def generate_equilateral_triangular_mesh(
    nx: int, ny: int, side_length: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成等边三角形网格（蜂窝状排布）。

    基于 107_boundary_word_equilateral 的等边三角形思想。
    节点在六边形/三角晶格上排布，相邻节点间距为 side_length。

    对于等边三角形晶格，基矢为:
        a_1 = L (1, 0)
        a_2 = L (1/2, \\sqrt{3}/2)

    参数:
        nx: x 方向单元数
        ny: y 方向单元数
        side_length: 边长 L

    返回:
        (nodes, elements, boundary_nodes)
    """
    L = side_length
    sqrt3 = np.sqrt(3.0)

    # 节点排布：交错网格
    n_nodes = nx * ny
    nodes = np.zeros((n_nodes, 2))

    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            x = i * L
            y = j * L * sqrt3 / 2.0
            if j % 2 == 1:
                x += L / 2.0  # 奇数行偏移
            nodes[idx, 0] = x
            nodes[idx, 1] = y

    # 等边三角形单元
    elements = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n00 = j * nx + i
            n10 = j * nx + (i + 1)
            n01 = (j + 1) * nx + i
            n11 = (j + 1) * nx + (i + 1)

            if j % 2 == 0:
                # 偶数行: 上三角和下三角
                elements.append([n00, n10, n01])
                elements.append([n10, n11, n01])
            else:
                elements.append([n00, n11, n01])
                elements.append([n00, n10, n11])

    elements = np.array(elements, dtype=int)

    # 边界节点
    boundary_nodes = []
    for j in range(ny):
        for i in range(nx):
            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                boundary_nodes.append(j * nx + i)
    boundary_nodes = np.array(boundary_nodes, dtype=int)

    return nodes, elements, boundary_nodes


def triangle_area(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
) -> float:
    """
    计算三角形面积（二维）。

    对于顶点 (x1,y1), (x2,y2), (x3,y3)，面积为:
        A = \\frac{1}{2} |x_1(y_2 - y_3) + x_2(y_3 - y_1) + x_3(y_1 - y_2)|
          = \\frac{1}{2} |\\det(\vec{p_2}-\vec{p_1}, \vec{p_3}-\vec{p_1})|

    参数:
        p1, p2, p3: 顶点坐标，形状 (2,)

    返回:
        面积（非负）
    """
    area = 0.5 * abs(
        p1[0] * (p2[1] - p3[1])
        + p2[0] * (p3[1] - p1[1])
        + p3[0] * (p1[1] - p2[1])
    )
    return area


def barycentric_coordinates(
    p: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
) -> np.ndarray:
    """
    计算点 p 相对于三角形 (p1, p2, p3) 的重心坐标。

    重心坐标 (\\lambda_1, \\lambda_2, \\lambda_3) 满足:
        p = \\lambda_1 p_1 + \\lambda_2 p_2 + \\lambda_3 p_3
        \\lambda_1 + \\lambda_2 + \\lambda_3 = 1

    且:
        \\lambda_1 = A_{p p_2 p_3} / A_{p_1 p_2 p_3}
        \\lambda_2 = A_{p_1 p p_3} / A_{p_1 p_2 p_3}
        \\lambda_3 = A_{p_1 p_2 p} / A_{p_1 p_2 p_3}

    参数:
        p: 目标点
        p1, p2, p3: 三角形顶点

    返回:
        重心坐标数组 (3,)
    """
    A_total = triangle_area(p1, p2, p3)
    if A_total < 1e-15:
        return np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    lambda1 = triangle_area(p, p2, p3) / A_total
    lambda2 = triangle_area(p1, p, p3) / A_total
    lambda3 = triangle_area(p1, p2, p) / A_total

    # 归一化保证和为1
    s = lambda1 + lambda2 + lambda3
    if s > 0:
        return np.array([lambda1, lambda2, lambda3]) / s
    return np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])


def is_point_in_triangle(
    p: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, tol: float = 1e-10
) -> bool:
    """
    判断点 p 是否在三角形 (p1, p2, p3) 内（含边界）。

    使用重心坐标判定：\\lambda_i \\ge -tol 对所有 i。

    参数:
        p: 目标点
        p1, p2, p3: 三角形顶点
        tol: 数值容差

    返回:
        是否在三角形内
    """
    lam = barycentric_coordinates(p, p1, p2, p3)
    return np.all(lam >= -tol)


def compute_element_centroids(
    nodes: np.ndarray, elements: np.ndarray
) -> np.ndarray:
    """
    计算所有三角形单元的重心坐标。

    重心 (centroid) 为:
        c = \\frac{1}{3} (p_1 + p_2 + p_3)

    参数:
        nodes: 节点坐标数组
        elements: 单元节点编号数组

    返回:
        重心坐标数组 (n_elements, 2)
    """
    centroids = np.zeros((len(elements), 2))
    for e, elem in enumerate(elements):
        centroids[e] = (nodes[elem[0]] + nodes[elem[1]] + nodes[elem[2]]) / 3.0
    return centroids


def reverse_cuthill_mckee(
    adjacency: np.ndarray
) -> np.ndarray:
    """
    Reverse Cuthill-McKee (RCM) 算法：重编号节点以减少稀疏矩阵带宽。

    对于有限元刚度矩阵，带宽 B 定义为:
        B = \\max_{i,j: A_{ij} \\ne 0} |i - j|

    RCM 通过广度优先搜索从周长节点开始，将高度连接的节点
    排在编号末尾，从而显著降低带宽。

    参数:
        adjacency: 邻接矩阵 (n_nodes, n_nodes) 或稀疏表示

    返回:
        新的节点编号排列
    """
    n = adjacency.shape[0]
    visited = np.zeros(n, dtype=bool)
    ordering = []

    # 找度数最小的节点作为起始点
    degrees = np.sum(adjacency > 0, axis=1)
    start = int(np.argmin(degrees))

    queue = [start]
    visited[start] = True

    while queue:
        current = queue.pop(0)
        ordering.append(current)

        # 获取未访问的邻居，按度数排序
        neighbors = np.where(adjacency[current] > 0)[0]
        unvisited = [v for v in neighbors if not visited[v]]
        unvisited.sort(key=lambda v: degrees[v])

        for v in unvisited:
            visited[v] = True
            queue.append(v)

    # Reverse Cuthill-McKee: 反转编号
    ordering = ordering[::-1]
    return np.array(ordering, dtype=int)


def build_adjacency_from_elements(
    n_nodes: int, elements: np.ndarray
) -> np.ndarray:
    """
    从三角形单元构建节点邻接矩阵。

    参数:
        n_nodes: 节点总数
        elements: 单元数组

    返回:
        邻接矩阵 (n_nodes, n_nodes)
    """
    adj = np.zeros((n_nodes, n_nodes), dtype=int)
    for elem in elements:
        for i in range(3):
            for j in range(i + 1, 3):
                n1, n2 = elem[i], elem[j]
                adj[n1, n2] = 1
                adj[n2, n1] = 1
    return adj


def mesh_quality_metrics(
    nodes: np.ndarray, elements: np.ndarray
) -> dict:
    """
    计算网格质量指标。

    指标包括:
      - 最小/最大/平均面积
      - 最小角（衡量单元畸形程度）

    对于三角形，内角可通过余弦定理计算:
        \\cos \\theta_k = \\frac{a_j^2 + a_l^2 - a_k^2}{2 a_j a_l}

    参数:
        nodes, elements: 网格数据

    返回:
        字典包含质量指标
    """
    areas = []
    min_angles = []

    for elem in elements:
        p1, p2, p3 = nodes[elem[0]], nodes[elem[1]], nodes[elem[2]]
        a = np.linalg.norm(p2 - p3)
        b = np.linalg.norm(p1 - p3)
        c = np.linalg.norm(p1 - p2)

        area = triangle_area(p1, p2, p3)
        areas.append(area)

        # 计算角度（余弦定理）
        if a > 1e-14 and b > 1e-14 and c > 1e-14:
            cos_alpha = (b**2 + c**2 - a**2) / (2 * b * c)
            cos_beta = (a**2 + c**2 - b**2) / (2 * a * c)
            cos_gamma = (a**2 + b**2 - c**2) / (2 * a * b)
            angles = [
                np.arccos(np.clip(cos_alpha, -1, 1)),
                np.arccos(np.clip(cos_beta, -1, 1)),
                np.arccos(np.clip(cos_gamma, -1, 1)),
            ]
            min_angles.append(min(angles))

    areas = np.array(areas)
    min_angles = np.array(min_angles)

    return {
        "min_area": float(np.min(areas)) if len(areas) > 0 else 0.0,
        "max_area": float(np.max(areas)) if len(areas) > 0 else 0.0,
        "mean_area": float(np.mean(areas)) if len(areas) > 0 else 0.0,
        "min_angle_deg": float(np.degrees(np.min(min_angles))) if len(min_angles) > 0 else 0.0,
        "num_elements": len(elements),
    }
