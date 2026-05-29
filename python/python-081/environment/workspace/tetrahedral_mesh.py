"""
四面体网格生成与自适应细化模块
=====================================
基于种子项目:
  - 1247_tetrahedron_grid: 四面体内部均匀网格生成
  - 1238_tet_mesh_refine: 四面体网格八倍细分

科学背景:
  在大变形非线性有限元分析中，三维实体域的离散化采用4节点线性
  四面体单元(P1)。对于超弹性材料的大变形问题，网格质量直接决定
  数值精度与收敛性。本模块实现：
  1. 标准立方体域的四面体网格剖分
  2. 基于边细分的8-子四面体自适应加密(8-subtetrahedron subdivision)

关键公式:
  - 四面体体积: V = |det([x2-x1, x3-x1, x4-x1])| / 6
  - 细化后节点数: N_new = N_old + N_edges_unique
  - 细化后单元数: E_new = 8 * E_old
"""

import numpy as np
from typing import Tuple, List, Optional


def generate_cube_tetrahedral_mesh(nx: int = 4, ny: int = 4, nz: int = 4,
                                    xlim: Tuple[float, float] = (0.0, 1.0),
                                    ylim: Tuple[float, float] = (0.0, 1.0),
                                    zlim: Tuple[float, float] = (0.0, 1.0)) -> Tuple[np.ndarray, np.ndarray]:
    """
    在单位立方体 [xlim]×[ylim]×[zlim] 内生成结构化四面体网格。
    先将立方体划分为六面体，再将每个六面体切分为5或6个四面体。
    此处采用6-四面体剖分以保证对角线一致性。

    参数:
        nx, ny, nz: 各方向六面体层数
        xlim, ylim, zlim: 几何边界

    返回:
        nodes: (N_nodes, 3) 节点坐标数组
        elements: (N_elements, 4) 单元-节点连接表(0-based)
    """
    if nx < 1 or ny < 1 or nz < 1:
        raise ValueError("网格层数必须至少为1")

    dx = (xlim[1] - xlim[0]) / nx
    dy = (ylim[1] - ylim[0]) / ny
    dz = (zlim[1] - zlim[0]) / nz

    # 生成节点
    n_nodes = (nx + 1) * (ny + 1) * (nz + 1)
    nodes = np.zeros((n_nodes, 3), dtype=np.float64)
    idx = 0
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                nodes[idx, 0] = xlim[0] + i * dx
                nodes[idx, 1] = ylim[0] + j * dy
                nodes[idx, 2] = zlim[0] + k * dz
                idx += 1

    def node_id(i: int, j: int, k: int) -> int:
        return k * (nx + 1) * (ny + 1) + j * (nx + 1) + i

    # 每个六面体切分为6个四面体
    # 方法: 将六面体沿面对角线分成2个三棱柱,
    # 每个三棱柱再分成3个四面体
    elements = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                n0 = node_id(i, j, k)
                n1 = node_id(i + 1, j, k)
                n2 = node_id(i + 1, j + 1, k)
                n3 = node_id(i, j + 1, k)
                n4 = node_id(i, j, k + 1)
                n5 = node_id(i + 1, j, k + 1)
                n6 = node_id(i + 1, j + 1, k + 1)
                n7 = node_id(i, j + 1, k + 1)

                # 三棱柱1: 底面 [n0,n1,n2] -> 顶面 [n4,n5,n6]
                tets = [
                    [n0, n1, n2, n4],
                    [n1, n2, n4, n5],
                    [n2, n4, n5, n6],
                    # 三棱柱2: 底面 [n0,n2,n3] -> 顶面 [n4,n6,n7]
                    [n0, n2, n3, n4],
                    [n2, n3, n4, n6],
                    [n3, n4, n6, n7],
                ]
                elements.extend(tets)

    elements = np.array(elements, dtype=np.int32)
    return nodes, elements


def tetrahedron_volume(nodes: np.ndarray, element: np.ndarray) -> float:
    """
    计算单个四面体单元的有向体积。
    V = det([x2-x1, x3-x1, x4-x1]) / 6
    """
    x1, x2, x3, x4 = nodes[element[0]], nodes[element[1]], nodes[element[2]], nodes[element[3]]
    mat = np.array([x2 - x1, x3 - x1, x4 - x1], dtype=np.float64)
    vol = np.linalg.det(mat) / 6.0
    return vol


def check_mesh_quality(nodes: np.ndarray, elements: np.ndarray) -> dict:
    """
    检查网格质量，返回最小/平均体积、负体积单元数等。
    """
    vols = []
    negative_count = 0
    for e in elements:
        v = tetrahedron_volume(nodes, e)
        vols.append(v)
        if v <= 0:
            negative_count += 1
    vols = np.array(vols)
    return {
        "min_volume": float(np.min(vols)),
        "max_volume": float(np.max(vols)),
        "mean_volume": float(np.mean(vols)),
        "negative_count": negative_count,
        "total_elements": len(elements),
    }


def refine_tetrahedral_mesh(nodes: np.ndarray, elements: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    对4节点线性四面体网格执行8-子四面体细分(8-subtetrahedron subdivision)。
    每条边中点插入新节点，每个原四面体按固定拓扑模板拆分为8个子四面体。

    参数:
        nodes: (N, 3) 原节点坐标
        elements: (E, 4) 原单元连接表

    返回:
        new_nodes: 细化后节点坐标
        new_elements: 细化后单元连接表
    """
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError("nodes 必须是 (N, 3) 数组")
    if elements.ndim != 2 or elements.shape[1] != 4:
        raise ValueError("elements 必须是 (E, 4) 数组")

    n_nodes_old = nodes.shape[0]

    # 提取所有边并去重
    edges = []
    for e in elements:
        edges.append(tuple(sorted((e[0], e[1]))))
        edges.append(tuple(sorted((e[0], e[2]))))
        edges.append(tuple(sorted((e[0], e[3]))))
        edges.append(tuple(sorted((e[1], e[2]))))
        edges.append(tuple(sorted((e[1], e[3]))))
        edges.append(tuple(sorted((e[2], e[3]))))

    unique_edges = []
    edge_map = {}
    for ed in edges:
        if ed not in edge_map:
            edge_map[ed] = len(unique_edges)
            unique_edges.append(ed)

    n_edges = len(unique_edges)
    n_nodes_new = n_nodes_old + n_edges
    n_elements_new = 8 * elements.shape[0]

    # 新节点数组
    new_nodes = np.zeros((n_nodes_new, 3), dtype=np.float64)
    new_nodes[:n_nodes_old, :] = nodes

    # 边中点
    for idx, (i, j) in enumerate(unique_edges):
        new_nodes[n_nodes_old + idx, :] = 0.5 * (nodes[i] + nodes[j])

    # 8-subtet 细分模板 (参考 Red-Green 细化中的标准8分)
    # 原节点: 0,1,2,3
    # 边中点: 01,02,03,12,13,23 (对应 edge_map)
    new_elements = np.zeros((n_elements_new, 4), dtype=np.int32)
    e_count = 0
    for e in elements:
        n0, n1, n2, n3 = e
        m01 = n_nodes_old + edge_map[tuple(sorted((n0, n1)))]
        m02 = n_nodes_old + edge_map[tuple(sorted((n0, n2)))]
        m03 = n_nodes_old + edge_map[tuple(sorted((n0, n3)))]
        m12 = n_nodes_old + edge_map[tuple(sorted((n1, n2)))]
        m13 = n_nodes_old + edge_map[tuple(sorted((n1, n3)))]
        m23 = n_nodes_old + edge_map[tuple(sorted((n2, n3)))]

        # 8个子四面体
        subtets = [
            [n0, m01, m02, m03],
            [n1, m01, m12, m13],
            [n2, m02, m12, m23],
            [n3, m03, m13, m23],
            [m01, m02, m03, m13],
            [m01, m02, m12, m13],
            [m02, m03, m13, m23],
            [m02, m12, m13, m23],
        ]
        for st in subtets:
            new_elements[e_count, :] = st
            e_count += 1

    return new_nodes, new_elements


def get_surface_triangles(elements: np.ndarray) -> np.ndarray:
    """
    从四面体网格提取表面三角形。表面三角形仅属于一个四面体。
    """
    face_count = {}
    for e in elements:
        faces = [
            tuple(sorted((e[0], e[1], e[2]))),
            tuple(sorted((e[0], e[1], e[3]))),
            tuple(sorted((e[0], e[2], e[3]))),
            tuple(sorted((e[1], e[2], e[3]))),
        ]
        for f in faces:
            face_count[f] = face_count.get(f, 0) + 1

    surface_faces = [f for f, c in face_count.items() if c == 1]
    return np.array(surface_faces, dtype=np.int32)
