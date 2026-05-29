"""
mesh_topology.py
================================================================================
网格拓扑与图结构模块 —— 基于种子项目 489_grf_display 与 790_navier_stokes_mesh3d

在 LES 中，网格的拓扑结构（节点-单元-面的邻接关系）决定了离散算子的
稀疏模式与并行通信图。本模块将网格视为无向图，提供邻接矩阵、节点度
与连通分量分析。

核心物理公式
--------------------------------------------------------------------------------
网格质量度量：
    - 纵横比 AR = h_max / h_min，其中 h 为单元特征尺度
    - 正交质量 Q = min( cos(θ_ij) )，θ_ij 为面法向量与节点连线的夹角

对于谱-有限元混合方法，水平方向采用球谐/谱展开，垂直方向采用 FEM，
网格拓扑主要影响垂直算子的组装与边界条件施加。
"""

import numpy as np


def build_mesh_graph(element_nodes, n_nodes):
    """
    从单元节点列表构建网格邻接图。

    参数
    ----------
    element_nodes : np.ndarray, shape (n_elem, elem_order)
        单元节点索引（0-based）
    n_nodes : int
        总节点数

    返回
    -------
    adjacency : dict
        {node_id: set(neighbor_ids)}
    degrees : np.ndarray
        每个节点的度
    """
    adjacency = {i: set() for i in range(n_nodes)}

    for elem in element_nodes:
        for i in range(len(elem)):
            for j in range(i + 1, len(elem)):
                ni, nj = elem[i], elem[j]
                if 0 <= ni < n_nodes and 0 <= nj < n_nodes:
                    adjacency[ni].add(nj)
                    adjacency[nj].add(ni)

    degrees = np.array([len(adjacency[i]) for i in range(n_nodes)], dtype=int)
    return adjacency, degrees


def element_neighbor_tets(element_nodes):
    """
    计算四面体网格的单元邻接关系（通过共享面）。

    参数
    ----------
    element_nodes : np.ndarray, shape (n_elem, 4)
        四面体节点索引

    返回
    -------
    neighbors : np.ndarray, shape (n_elem, 4)
        neighbors[e][f] = 与单元 e 的第 f 个面相邻的单元索引，-1 表示边界
    """
    n_elem = element_nodes.shape[0]
    neighbors = np.full((n_elem, 4), -1, dtype=int)

    # 面 → 单元映射
    face_to_elem = {}
    for e in range(n_elem):
        elem = element_nodes[e]
        faces = [
            tuple(sorted((elem[0], elem[1], elem[2]))),
            tuple(sorted((elem[0], elem[1], elem[3]))),
            tuple(sorted((elem[0], elem[2], elem[3]))),
            tuple(sorted((elem[1], elem[2], elem[3]))),
        ]
        for f in faces:
            if f not in face_to_elem:
                face_to_elem[f] = []
            face_to_elem[f].append(e)

    # 填充邻接关系
    for e in range(n_elem):
        elem = element_nodes[e]
        faces = [
            tuple(sorted((elem[0], elem[1], elem[2]))),
            tuple(sorted((elem[0], elem[1], elem[3]))),
            tuple(sorted((elem[0], elem[2], elem[3]))),
            tuple(sorted((elem[1], elem[2], elem[3]))),
        ]
        for f_idx, f in enumerate(faces):
            elems = face_to_elem[f]
            if len(elems) == 2:
                neighbors[e, f_idx] = elems[0] if elems[1] == e else elems[1]

    return neighbors


def mesh_quality_metrics(nodes, element_nodes):
    """
    计算网格质量指标。

    参数
    ----------
    nodes : np.ndarray, shape (n_node, 3)
    element_nodes : np.ndarray, shape (n_elem, 4)

    返回
    -------
    metrics : dict
        {'min_volume', 'max_volume', 'mean_volume', 'volume_ratio'}
    """
    from fem_basis import tetrahedron_volume

    volumes = []
    for e in range(element_nodes.shape[0]):
        en = element_nodes[e]
        try:
            vol = tetrahedron_volume(nodes[en])
            volumes.append(vol)
        except ValueError:
            volumes.append(0.0)

    volumes = np.array(volumes)
    valid = volumes > 1e-15

    if not np.any(valid):
        raise ValueError("mesh_quality_metrics: 所有单元体积退化")

    metrics = {
        'min_volume': np.min(volumes[valid]),
        'max_volume': np.max(volumes[valid]),
        'mean_volume': np.mean(volumes[valid]),
        'volume_ratio': np.max(volumes[valid]) / np.min(volumes[valid]),
        'n_degenerate': np.sum(~valid)
    }
    return metrics
