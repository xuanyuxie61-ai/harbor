r"""
mesh_refinement.py
==================
三角网格细化升级模块（T3 到 T4 转换）

科学背景：
---------
在有限元方法中，三节点三角形（T3，线性元）的精度为 O(h^2)。
通过引入形心节点将其升级为四节点三角形（T4，bubble 元），
可以在不显著增加自由度的情况下提高局部逼近精度。

T3 到 T4 转换（来自项目 1353_triangulation_t3_to_t4）：
------------------------------------------------------
对于每个三节点三角形，在其形心处添加一个新节点：
    x_{centroid} = \frac{x_1 + x_2 + x_3}{3}
    y_{centroid} = \frac{y_1 + y_2 + y_3}{3}

四节点单元的形函数为：
    N_1 = 1 - \xi - \eta
    N_2 = \xi
    N_3 = \eta
    N_4 = 27 \xi \eta (1 - \xi - \eta)   （bubble 函数）

其中 bubble 函数在三条边上为零，在形心处取最大值 1。
这种单元可以更好地捕捉图像中的局部奇异性（如边缘）。

应用：
-----
在压缩感知重建中，T4 单元用于：
1. 在边缘区域提供更高阶的逼近
2. 通过 bubble 函数增强局部稀疏表示能力
3. 改善梯度计算精度
"""

import numpy as np
from typing import Tuple


def triangulation_t3_to_t4(nodes: np.ndarray, triangles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 T3 三角剖分转换为 T4 三角剖分。

    参数:
        nodes: T3 节点坐标，形状为 (n_nodes, 2)
        triangles: T3 三角形索引，形状为 (n_triangles, 3)
    返回:
        (nodes_t4, triangles_t4): T4 节点坐标 (n_nodes + n_triangles, 2)
                                  T4 三角形索引 (n_triangles, 4)
    """
    nodes = np.asarray(nodes, dtype=float)
    triangles = np.asarray(triangles, dtype=int)

    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles 必须是形状为 (n_tri, 3) 的数组")

    n_nodes_t3 = len(nodes)
    n_tri = len(triangles)

    # 初始化 T4 节点数组（原始节点 + 形心节点）
    nodes_t4 = np.zeros((n_nodes_t3 + n_tri, 2), dtype=float)
    nodes_t4[:n_nodes_t3] = nodes

    # T4 三角形索引数组
    triangles_t4 = np.zeros((n_tri, 4), dtype=int)
    triangles_t4[:, :3] = triangles

    # 为每个三角形添加形心节点
    node_count = n_nodes_t3
    for t in range(n_tri):
        idx = triangles[t]
        # 边界检查
        if np.any(idx < 0) or np.any(idx >= n_nodes_t3):
            raise ValueError(f"三角形 {t} 包含越界节点索引")

        # 形心坐标
        centroid = np.mean(nodes[idx], axis=0)
        nodes_t4[node_count] = centroid
        triangles_t4[t, 3] = node_count
        node_count += 1

    return nodes_t4, triangles_t4


def t4_shape_functions(xi: float, eta: float) -> np.ndarray:
    r"""
    计算 T4 单元在参考坐标 (\xi, \eta) 处的形函数值。

    参考三角形：
        (0,0) ---- (1,0)
          \        /
           \      /
            (0,1)

    形函数：
        N_1 = 1 - \xi - \eta
        N_2 = \xi
        N_3 = \eta
        N_4 = 27 \xi \eta (1 - \xi - \eta)

    参数:
        xi, eta: 参考坐标（必须满足 xi >= 0, eta >= 0, xi + eta <= 1）
    返回:
        形函数值数组，形状为 (4,)
    """
    if xi < -1e-10 or eta < -1e-10 or xi + eta > 1.0 + 1e-10:
        raise ValueError(f"参考坐标 ({xi}, {eta}) 超出参考三角形")

    N = np.zeros(4, dtype=float)
    N[0] = 1.0 - xi - eta
    N[1] = xi
    N[2] = eta
    N[3] = 27.0 * xi * eta * (1.0 - xi - eta)
    return N


def t4_shape_derivatives(xi: float, eta: float) -> np.ndarray:
    r"""
    计算 T4 单元形函数对参考坐标的偏导数。

    返回:
        dN_dxi, dN_deta，每个形状为 (4,)
    """
    if xi < -1e-10 or eta < -1e-10 or xi + eta > 1.0 + 1e-10:
        raise ValueError(f"参考坐标 ({xi}, {eta}) 超出参考三角形")

    dN_dxi = np.zeros(4, dtype=float)
    dN_deta = np.zeros(4, dtype=float)

    dN_dxi[0] = -1.0
    dN_dxi[1] = 1.0
    dN_dxi[2] = 0.0
    dN_dxi[3] = 27.0 * eta * (1.0 - 2.0 * xi - eta)

    dN_deta[0] = -1.0
    dN_deta[1] = 0.0
    dN_deta[2] = 1.0
    dN_deta[3] = 27.0 * xi * (1.0 - xi - 2.0 * eta)

    return dN_dxi, dN_deta


def interpolate_on_t4_mesh(nodes_t4: np.ndarray, triangles_t4: np.ndarray,
                           node_values: np.ndarray, query_points: np.ndarray) -> np.ndarray:
    """
    在 T4 网格上对给定的节点值进行插值。

    参数:
        nodes_t4: T4 节点坐标
        triangles_t4: T4 三角形索引
        node_values: 每个节点上的标量值
        query_points: 查询点坐标，形状为 (n_query, 2)
    返回:
        插值结果，形状为 (n_query,)
    """
    node_values = np.asarray(node_values, dtype=float)
    query_points = np.asarray(query_points, dtype=float)

    results = np.zeros(len(query_points), dtype=float)

    for q in range(len(query_points)):
        pt = query_points[q]
        found = False

        for t in range(len(triangles_t4)):
            tri = triangles_t4[t]
            p = nodes_t4[tri[:3]]  # 使用 T3 顶点判断所在三角形

            # 计算重心坐标
            A = triangle_area(p[0], p[1], p[2])
            if abs(A) < 1e-14:
                continue

            w0 = triangle_area(pt, p[1], p[2]) / A
            w1 = triangle_area(pt, p[2], p[0]) / A
            w2 = 1.0 - w0 - w1

            if w0 >= -1e-10 and w1 >= -1e-10 and w2 >= -1e-10:
                # 点在三角形内或边界上
                # 映射到参考坐标
                xi = w1
                eta = w2
                N = t4_shape_functions(xi, eta)
                vals = node_values[tri]
                results[q] = np.dot(N, vals)
                found = True
                break

        if not found:
            # 若未找到包含点，取最近节点值
            dists = np.linalg.norm(nodes_t4 - pt, axis=1)
            results[q] = node_values[np.argmin(dists)]

    return results


def triangle_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """计算三角形有向面积（辅助函数）。"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return 0.5 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
