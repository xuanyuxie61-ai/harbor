#!/usr/bin/env python3
"""
mesh_generator.py
三维四面体网格生成与细化模块（源自 tet_mesh_refine 项目）

对 PEMFC 三维几何区域（流道-脊-膜-催化层-GDL）进行四面体剖分，
并实施 Liu & Joe 的 8-子四面体细化算法，以支持多物理场有限元求解。

细化规则：
    - 每条边插入中点，形成 8 个子四面体
    - 共享边去重，保证网格连续性
"""

import numpy as np


def generate_pemfc_mesh():
    """
    生成简化的 PEMFC 三维四面体初始网格。
    几何域：
        x ∈ [0, 1]   (流道方向)
        y ∈ [0, 1]   (脊/膜面内)
        z ∈ [0, 0.3] (膜厚 + GDL)

    分层：
        z ∈ [0.0, 0.1] : 阴极 GDL
        z ∈ [0.1, 0.12]: 阴极 CL
        z ∈ [0.12, 0.18]: PEM (Nafion)
        z ∈ [0.18, 0.20]: 阳极 CL
        z ∈ [0.20, 0.30]: 阳极 GDL
    """
    # 简化为单位立方体的四面体剖分
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.3],
        [1.0, 0.0, 0.3],
        [1.0, 1.0, 0.3],
        [0.0, 1.0, 0.3],
    ], dtype=float)

    # 将立方体剖分为 6 个四面体（标准分解）
    elements = np.array([
        [0, 1, 3, 4],
        [1, 3, 4, 5],
        [1, 2, 3, 5],
        [2, 3, 5, 6],
        [3, 4, 5, 7],
        [3, 5, 6, 7],
        [4, 5, 7, 6],
        [0, 1, 2, 4],
    ], dtype=int)

    return nodes, elements


def tetrahedron_volume(nodes, tet):
    """
    计算单个四面体的有向体积。
    对应原项目 tetrahedron_volume.m 与 r8mat_det_4d.m。
    V = |det([x1-x0, x2-x0, x3-x0])| / 6
    """
    p0 = nodes[tet[0]]
    p1 = nodes[tet[1]]
    p2 = nodes[tet[2]]
    p3 = nodes[tet[3]]

    M = np.array([
        [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]],
        [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]],
        [p3[0] - p0[0], p3[1] - p0[1], p3[2] - p0[2]],
    ], dtype=float)

    vol = abs(np.linalg.det(M)) / 6.0
    return vol


def refine_mesh(nodes, elements):
    """
    对 4 节点线性四面体网格执行一次 8-子四面体细化。
    对应原项目 tet_mesh_order4_refine_size 与 tet_mesh_order4_refine_compute。

    算法：
        1. 列出所有元素的 6 条边
        2. 对边去重（排序 + 唯一化）
        3. 在每条唯一边的中点插入新节点
        4. 按 Liu & Joe (1996) 规则将每个原四面体分解为 8 个子四面体
    """
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    n_nodes = nodes.shape[0]
    n_tets = elements.shape[0]

    # Step 1: 收集所有边
    edges = []
    for t in range(n_tets):
        tet = elements[t]
        edge_list = [
            (tet[0], tet[1]), (tet[0], tet[2]), (tet[0], tet[3]),
            (tet[1], tet[2]), (tet[1], tet[3]), (tet[2], tet[3]),
        ]
        for e in edge_list:
            edges.append(tuple(sorted(e)))

    # Step 2: 去重
    edges_unique = sorted(list(set(edges)))
    n_edges = len(edges_unique)

    # Step 3: 创建新节点（边中点）
    new_nodes = np.zeros((n_nodes + n_edges, 3), dtype=float)
    new_nodes[:n_nodes] = nodes

    edge_to_node = {}
    for k, (i, j) in enumerate(edges_unique):
        new_nodes[n_nodes + k] = 0.5 * (nodes[i] + nodes[j])
        edge_to_node[(i, j)] = n_nodes + k

    # Step 4: 构建新单元（8-subtetrahedron 分解）
    # 每个原四面体产生 8 个子四面体
    new_elements = np.zeros((n_tets * 8, 4), dtype=int)

    for t in range(n_tets):
        tet = elements[t]
        v0, v1, v2, v3 = tet

        # 6 条边的中点节点编号
        n01 = edge_to_node[tuple(sorted((v0, v1)))]
        n02 = edge_to_node[tuple(sorted((v0, v2)))]
        n03 = edge_to_node[tuple(sorted((v0, v3)))]
        n12 = edge_to_node[tuple(sorted((v1, v2)))]
        n13 = edge_to_node[tuple(sorted((v1, v3)))]
        n23 = edge_to_node[tuple(sorted((v2, v3)))]

        # 8 个子四面体（Liu & Joe 规则）
        sub_tets = [
            [v0, n01, n02, n03],
            [n01, v1, n12, n13],
            [n02, n12, v2, n23],
            [n03, n13, n23, v3],
            [n01, n02, n03, n13],
            [n01, n02, n12, n13],
            [n02, n03, n13, n23],
            [n02, n12, n13, n23],
        ]

        for s in range(8):
            new_elements[8 * t + s] = sub_tets[s]

    # 去除退化单元（体积为零）
    valid = []
    for e in range(new_elements.shape[0]):
        vol = tetrahedron_volume(new_nodes, new_elements[e])
        if vol > 1e-14:
            valid.append(e)
    new_elements = new_elements[valid]

    return new_nodes, new_elements


def compute_mesh_quality(nodes, elements):
    """
    计算四面体网格质量指标：最小与平均体积。
    """
    vols = np.array([tetrahedron_volume(nodes, elements[e])
                     for e in range(elements.shape[0])])
    return {
        'n_nodes': nodes.shape[0],
        'n_elements': elements.shape[0],
        'min_volume': float(np.min(vols)),
        'max_volume': float(np.max(vols)),
        'mean_volume': float(np.mean(vols)),
    }


if __name__ == '__main__':
    nodes, elements = generate_pemfc_mesh()
    print("Initial:", compute_mesh_quality(nodes, elements))
    nodes_r, elements_r = refine_mesh(nodes, elements)
    print("Refined:", compute_mesh_quality(nodes_r, elements_r))
