"""
三角形网格细化模块

融合自:
- 1350_triangulation_refine: 三角形网格的边二分法细化

细化策略:
    对每个标记为细化的三角形，将其每条边的中点作为新节点插入，
    将原三角形细分为 4 个子三角形。

数学原理:
    设原三角形顶点为 v1, v2, v3，则中点为:
        m12 = (v1 + v2) / 2
        m23 = (v2 + v3) / 2
        m31 = (v3 + v1) / 2
    
    细分为 4 个子三角形:
        T1 = [v1, m12, m31]
        T2 = [m12, v2, m23]
        T3 = [m31, m23, v3]
        T4 = [m12, m23, m31]

嵌套细化空间:
    设 V_h 为粗网格上的分片线性有限元空间，
    V_h/2 为细网格上的空间，则 V_h ⊂ V_h/2。
    对于任意 u_h ∈ V_h，其在细网格上的限制 u_h|_{T_i} 仍然是线性的。
"""

import numpy as np


def refine_triangulation(nodes, triangles, refine_flags=None):
    """
    对三角形网格进行一致细化或自适应细化。
    
    Parameters
    ----------
    nodes : ndarray, shape (n_nodes, 2)
        节点坐标
    triangles : ndarray, shape (n_tri, 3)
        三角形顶点索引
    refine_flags : ndarray, shape (n_tri,), optional
        标记哪些三角形需要细化 (True/1 表示细化)
        若 None，则对所有三角形进行细化
    
    Returns
    -------
    new_nodes : ndarray, shape (n_new_nodes, 2)
        细化后的节点坐标
    new_triangles : ndarray, shape (n_new_tri, 3)
        细化后的三角形
    parent_map : ndarray, shape (n_new_tri,)
        每个新三角形对应的父三角形索引
    node_level : ndarray, shape (n_new_nodes,)
        每个节点的细化层级
    """
    n_nodes = nodes.shape[0]
    n_tri = triangles.shape[0]

    if refine_flags is None:
        refine_flags = np.ones(n_tri, dtype=bool)
    else:
        refine_flags = np.array(refine_flags, dtype=bool)

    # 边到全局中点节点的映射
    edge_midpoint = {}
    midpoint_index = {}

    def get_midpoint(i, j):
        """获取边 (i,j) 的中点节点索引，若不存在则创建。"""
        key = tuple(sorted([i, j]))
        if key not in midpoint_index:
            mid = (nodes[i] + nodes[j]) / 2.0
            idx = n_nodes + len(edge_midpoint)
            edge_midpoint[key] = mid
            midpoint_index[key] = idx
        return midpoint_index[key]

    new_triangles_list = []
    parent_map_list = []

    for t in range(n_tri):
        v1, v2, v3 = triangles[t]

        if refine_flags[t]:
            m12 = get_midpoint(v1, v2)
            m23 = get_midpoint(v2, v3)
            m31 = get_midpoint(v3, v1)

            m12_idx = midpoint_index[tuple(sorted([v1, v2]))]
            m23_idx = midpoint_index[tuple(sorted([v2, v3]))]
            m31_idx = midpoint_index[tuple(sorted([v3, v1]))]

            new_triangles_list.append([v1, m12_idx, m31_idx])
            new_triangles_list.append([m12_idx, v2, m23_idx])
            new_triangles_list.append([m31_idx, m23_idx, v3])
            new_triangles_list.append([m12_idx, m23_idx, m31_idx])

            parent_map_list.extend([t, t, t, t])
        else:
            new_triangles_list.append([v1, v2, v3])
            parent_map_list.append(t)

    # 组装新的节点数组
    n_new_midpoints = len(edge_midpoint)
    new_nodes = np.zeros((n_nodes + n_new_midpoints, 2))
    new_nodes[:n_nodes] = nodes

    node_level = np.zeros(n_nodes + n_new_midpoints, dtype=int)
    # 原始节点层级设为 0，中点节点层级设为 1
    node_level[:n_nodes] = 0
    node_level[n_nodes:] = 1

    for key, idx in midpoint_index.items():
        new_nodes[idx] = edge_midpoint[key]

    new_triangles = np.array(new_triangles_list, dtype=int)
    parent_map = np.array(parent_map_list, dtype=int)

    return new_nodes, new_triangles, parent_map, node_level


def refine_marked_elements(nodes, triangles, element_errors, threshold_ratio=0.5):
    """
    基于误差指示器自适应标记并细化单元。
    
    标记策略 (Dörfler marking):
        1. 计算所有单元误差的最大值
        2. 标记误差大于 threshold_ratio * max_error 的单元
    
    Parameters
    ----------
    nodes : ndarray, shape (n_nodes, 2)
        节点坐标
    triangles : ndarray, shape (n_tri, 3)
        三角形索引
    element_errors : ndarray, shape (n_tri,)
        每个三角形的误差估计值
    threshold_ratio : float
        相对于最大误差的阈值比例
    
    Returns
    -------
    new_nodes : ndarray
    new_triangles : ndarray
    parent_map : ndarray
    node_level : ndarray
    refined_count : int
        实际被细化的单元数
    """
    max_error = np.max(element_errors)
    if max_error < 1e-15:
        # 所有误差极小，无需细化
        return nodes.copy(), triangles.copy(), np.arange(len(triangles)), np.zeros(len(nodes), dtype=int), 0

    threshold = threshold_ratio * max_error
    refine_flags = element_errors >= threshold

    refined_count = np.sum(refine_flags)

    new_nodes, new_triangles, parent_map, node_level = refine_triangulation(
        nodes, triangles, refine_flags
    )

    return new_nodes, new_triangles, parent_map, node_level, int(refined_count)


def coarsen_mesh(nodes, triangles, coarsen_flags):
    """
    粗化网格：移除被标记的三角形及其独享的节点。
    
    注意：此实现为简化版，实际 AMR 粗化需要保证网格合法性和
    层次结构一致性。
    
    Parameters
    ----------
    nodes : ndarray
    triangles : ndarray
    coarsen_flags : ndarray, shape (n_tri,)
        标记需要移除的三角形
    
    Returns
    -------
    new_nodes : ndarray
    new_triangles : ndarray
    node_remap : ndarray
        旧节点到新节点的映射
    """
    coarsen_flags = np.array(coarsen_flags, dtype=bool)
    keep_triangles = ~coarsen_flags

    if np.all(keep_triangles):
        return nodes.copy(), triangles.copy(), np.arange(len(nodes))

    # 统计每个节点被多少个保留的三角形使用
    node_usage = np.zeros(len(nodes), dtype=int)
    for tri in triangles[keep_triangles]:
        node_usage[tri] += 1

    keep_nodes = node_usage > 0
    node_remap = np.cumsum(keep_nodes) - 1
    node_remap[~keep_nodes] = -1

    new_nodes = nodes[keep_nodes]
    new_triangles = node_remap[triangles[keep_triangles]]

    # 检查三角形的合法性
    valid = np.all(new_triangles >= 0, axis=1)
    new_triangles = new_triangles[valid]

    return new_nodes, new_triangles, node_remap


def compute_mesh_quality(nodes, triangles):
    """
    计算三角形网格的质量度量。
    
    质量指标:
        q = 4 * sqrt(3) * A / (a^2 + b^2 + c^2)
    其中 A 是面积，a,b,c 是边长。q ∈ [0,1]，q=1 为等边三角形。
    
    Parameters
    ----------
    nodes : ndarray, shape (n_nodes, 2)
    triangles : ndarray, shape (n_tri, 3)
    
    Returns
    -------
    quality : ndarray, shape (n_tri,)
        每个三角形的质量指标
    """
    n_tri = len(triangles)
    quality = np.zeros(n_tri)

    for t in range(n_tri):
        p1 = nodes[triangles[t, 0]]
        p2 = nodes[triangles[t, 1]]
        p3 = nodes[triangles[t, 2]]

        a = np.linalg.norm(p2 - p3)
        b = np.linalg.norm(p1 - p3)
        c = np.linalg.norm(p1 - p2)

        area = 0.5 * abs(
            (p2[0] - p1[0]) * (p3[1] - p1[1]) -
            (p3[0] - p1[0]) * (p2[1] - p1[1])
        )

        denom = a ** 2 + b ** 2 + c ** 2
        if denom > 1e-14:
            quality[t] = 4.0 * np.sqrt(3.0) * area / denom
        else:
            quality[t] = 0.0

    return quality
