"""
spatial_mesh.py
空间异质性建模与有限元采样模块

基于以下种子项目融合:
- 413_fem2d_sample: 2D有限元函数采样
- 1238_tet_mesh_refine: 四面体网格细化
"""

import numpy as np
from typing import Tuple, List, Optional


def generate_2d_triangular_mesh(nx: int, ny: int,
                                x_min: float = 0.0, x_max: float = 1.0,
                                y_min: float = 0.0, y_max: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成规则三角形网格。

    节点坐标:
        x_i = x_min + i * hx,  hx = (x_max - x_min) / nx
        y_j = y_min + j * hy,  hy = (y_max - y_min) / ny

    每个矩形单元划分为2个三角形:
        T1: (i,j) -> (i+1,j) -> (i,j+1)
        T2: (i+1,j) -> (i+1,j+1) -> (i,j+1)

    参数:
        nx, ny: x,y方向单元数
        x_min, x_max, y_min, y_max: 区域范围

    返回:
        nodes: (n_nodes, 2) 节点坐标数组
        elements: (n_elements, 3) 三角形单元连通性
    """
    hx = (x_max - x_min) / nx
    hy = (y_max - y_min) / ny

    n_nodes_x = nx + 1
    n_nodes_y = ny + 1
    n_nodes = n_nodes_x * n_nodes_y

    nodes = np.zeros((n_nodes, 2), dtype=np.float64)
    for j in range(n_nodes_y):
        for i in range(n_nodes_x):
            idx = j * n_nodes_x + i
            nodes[idx, 0] = x_min + i * hx
            nodes[idx, 1] = y_min + j * hy

    n_elements = 2 * nx * ny
    elements = np.zeros((n_elements, 3), dtype=np.int32)

    e = 0
    for j in range(ny):
        for i in range(nx):
            n1 = j * n_nodes_x + i
            n2 = j * n_nodes_x + (i + 1)
            n3 = (j + 1) * n_nodes_x + i
            n4 = (j + 1) * n_nodes_x + (i + 1)

            elements[e, :] = [n1, n2, n3]
            elements[e + 1, :] = [n2, n4, n3]
            e += 2

    return nodes, elements


def triangle_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    计算三角形面积 (鞋带公式)。

    公式:
        A = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))


def barycentric_coordinates(p: np.ndarray,
                            p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Tuple[float, float, float]:
    """
    计算点 p 关于三角形 (p1,p2,p3) 的重心坐标。

    公式:
        lambda_1 = A(p,p2,p3) / A(p1,p2,p3)
        lambda_2 = A(p1,p,p3) / A(p1,p2,p3)
        lambda_3 = A(p1,p2,p) / A(p1,p2,p3)

    其中 A 表示三角形有向面积。
    """
    A = triangle_area(p1, p2, p3)
    if A < 1e-14:
        return -1.0, -1.0, -1.0

    lambda1 = triangle_area(p, p2, p3) / A
    lambda2 = triangle_area(p1, p, p3) / A
    lambda3 = 1.0 - lambda1 - lambda2

    return lambda1, lambda2, lambda3


def locate_point_in_mesh(nodes: np.ndarray, elements: np.ndarray,
                         point: np.ndarray) -> int:
    """
    在三角网格中定位包含给定点 p 的单元。

    使用暴力搜索 + 重心坐标测试。
    若点在边界上，取 lambda >= -1e-10 作为容差。
    """
    n_elements = elements.shape[0]
    best_elem = -1
    best_min_lambda = -1.0

    for e in range(n_elements):
        n1, n2, n3 = elements[e, :]
        p1 = nodes[n1, :]
        p2 = nodes[n2, :]
        p3 = nodes[n3, :]

        l1, l2, l3 = barycentric_coordinates(point, p1, p2, p3)
        min_lambda = min(l1, l2, l3)

        if min_lambda >= -1e-10:
            return e

        # 记录最接近的单元
        if min_lambda > best_min_lambda:
            best_min_lambda = min_lambda
            best_elem = e

    return best_elem


def refine_mesh_midpoint(nodes: np.ndarray, elements: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    中点细分: 每个三角形细分为4个小三角形。

    算法:
        1. 找到所有唯一边
        2. 在边中点插入新节点
        3. 每个原三角形被替换为4个子三角形

    返回:
        new_nodes, new_elements
    """
    n_nodes = nodes.shape[0]
    n_elements = elements.shape[0]

    # 收集所有边
    edges = []
    edge_map = {}

    for e in range(n_elements):
        tri = elements[e, :]
        edge_pairs = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
        for a, b in edge_pairs:
            key = tuple(sorted((int(a), int(b))))
            if key not in edge_map:
                edge_map[key] = len(edges)
                edges.append(key)

    n_edges = len(edges)
    new_n_nodes = n_nodes + n_edges
    new_nodes = np.zeros((new_n_nodes, 2), dtype=np.float64)
    new_nodes[:n_nodes, :] = nodes

    # 中点节点
    mid_node_idx = {}
    for idx, (a, b) in enumerate(edges):
        new_nodes[n_nodes + idx, :] = 0.5 * (nodes[a, :] + nodes[b, :])
        mid_node_idx[(a, b)] = n_nodes + idx

    # 构建新单元
    new_elements = np.zeros((4 * n_elements, 3), dtype=np.int32)

    for e in range(n_elements):
        tri = elements[e, :]
        v0, v1, v2 = int(tri[0]), int(tri[1]), int(tri[2])

        m01 = mid_node_idx[tuple(sorted((v0, v1)))]
        m12 = mid_node_idx[tuple(sorted((v1, v2)))]
        m20 = mid_node_idx[tuple(sorted((v2, v0)))]

        new_elements[4 * e + 0, :] = [v0, m01, m20]
        new_elements[4 * e + 1, :] = [m01, v1, m12]
        new_elements[4 * e + 2, :] = [m20, m12, v2]
        new_elements[4 * e + 3, :] = [m01, m12, m20]

    return new_nodes, new_elements


def fem_basis_t3(xi: float, eta: float) -> np.ndarray:
    """
    参考单元上的线性T3基函数。

    参考三角形顶点: (0,0), (1,0), (0,1)

    基函数:
        N1(xi, eta) = 1 - xi - eta
        N2(xi, eta) = xi
        N3(xi, eta) = eta
    """
    N = np.array([1.0 - xi - eta, xi, eta], dtype=np.float64)
    return N


def fem_sample_on_mesh(nodes: np.ndarray, elements: np.ndarray,
                       node_values: np.ndarray,
                       sample_points: np.ndarray) -> np.ndarray:
    """
    在三角网格上采样有限元函数值。

    对于每个采样点 p:
        1. 找到包含 p 的三角形单元 e
        2. 计算重心坐标 lambda
        3. 插值: f(p) = sum_{i=1}^3 lambda_i * f(v_i)

    参数:
        nodes: (n_nodes, 2)
        elements: (n_elements, 3)
        node_values: (n_nodes,) 节点上的函数值
        sample_points: (n_samples, 2)

    返回:
        sample_values: (n_samples,)
    """
    n_samples = sample_points.shape[0]
    sample_values = np.zeros(n_samples, dtype=np.float64)

    for s in range(n_samples):
        p = sample_points[s, :]
        e = locate_point_in_mesh(nodes, elements, p)

        if e < 0:
            # 点在网格外，使用最近节点值
            dists = np.sum((nodes - p)**2, axis=1)
            nearest = int(np.argmin(dists))
            sample_values[s] = node_values[nearest]
            continue

        n1, n2, n3 = elements[e, :]
        p1 = nodes[n1, :]
        p2 = nodes[n2, :]
        p3 = nodes[n3, :]

        l1, l2, l3 = barycentric_coordinates(p, p1, p2, p3)
        sample_values[s] = l1 * node_values[n1] + l2 * node_values[n2] + l3 * node_values[n3]

    return sample_values


def spatial_diffusion_operator(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """
    构造空间扩散算子的有限元刚度矩阵 (Galerkin方法)。

    对于泊松方程 -nabla^2 u = f，刚度矩阵元素:
        K_{ij} = sum_e int_{T_e} nabla(phi_i) . nabla(phi_j) dA

    使用线性T3单元的解析公式:
        K_e = A_e * G^T * G

    其中 G 是梯度矩阵 (3x2):
        G = [dN1/dx, dN1/dy;
             dN2/dx, dN2/dy;
             dN3/dx, dN3/dy]
    """
    n_nodes = nodes.shape[0]
    K = np.zeros((n_nodes, n_nodes), dtype=np.float64)

    for e in range(elements.shape[0]):
        n1, n2, n3 = elements[e, :]
        p1 = nodes[n1, :]
        p2 = nodes[n2, :]
        p3 = nodes[n3, :]

        A = triangle_area(p1, p2, p3)
        if A < 1e-14:
            continue

        # 计算梯度
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        # dN1/dx = (y2 - y3) / (2A), dN1/dy = (x3 - x2) / (2A)
        # dN2/dx = (y3 - y1) / (2A), dN2/dy = (x1 - x3) / (2A)
        # dN3/dx = (y1 - y2) / (2A), dN3/dy = (x2 - x1) / (2A)

        dNdx = np.array([y2 - y3, y3 - y1, y1 - y2], dtype=np.float64) / (2.0 * A)
        dNdy = np.array([x3 - x2, x1 - x3, x2 - x1], dtype=np.float64) / (2.0 * A)

        # 单元刚度矩阵
        Ke = A * (np.outer(dNdx, dNdx) + np.outer(dNdy, dNdy))

        # 组装到全局
        idx = [n1, n2, n3]
        for i_loc in range(3):
            for j_loc in range(3):
                K[idx[i_loc], idx[j_loc]] += Ke[i_loc, j_loc]

    return K
