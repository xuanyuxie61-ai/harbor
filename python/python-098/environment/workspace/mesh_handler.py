# -*- coding: utf-8 -*-
"""
mesh_handler.py
基于 ice_to_medit（NETCDF/MEDIT 网格转换）与 gmsh_to_fem（GMSH 到 FEM 转换），
实现超表面仿真域的网格 I/O、节点/单元管理与质量评估。

核心科学问题：
  电磁仿真（FDTD/FEM/BEM）需要高质量的非结构化网格来离散化 Maxwell 方程。
  对于超表面结构，通常采用四面体或三角形网格剖分介电/金属区域。
  网格质量直接影响数值色散误差与矩阵条件数。

关键公式：
  1. 四面体体积（行列式公式）:
       V = |det( [v1-v0, v2-v0, v3-v0] )| / 6
  2. 三角形面积:
       A = | (v1-v0) × (v2-v0) | / 2
  3. 单元质量指标（aspect ratio）:
       Q = 6 √6 · V / (Σ l_i²)^{3/2}
     其中 l_i 为六条边长，Q ∈ [0,1]，1 为正四面体。
  4. 节点坐标范围检测:
       dim = 3 if z_max ≠ z_min else 2 if y_max ≠ y_min else 1
"""

import numpy as np


def read_simple_mesh(filename):
    """
    读取简化的文本格式网格文件（参考 gmsh_to_fem 的节点/单元读取）。
    格式：
      NODES
      n_nodes
      x y z [label]
      ...
      ELEMENTS
      n_elements
      elem_type node1 node2 ... [label]
    """
    nodes = []
    node_labels = []
    elements = []
    element_labels = []
    elem_types = []
    mode = None

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.upper() == 'NODES':
                mode = 'nodes_count'
                continue
            elif line.upper() == 'ELEMENTS':
                mode = 'elements_count'
                continue

            if mode == 'nodes_count':
                n_nodes = int(line)
                mode = 'nodes_data'
                continue
            elif mode == 'elements_count':
                n_elements = int(line)
                mode = 'elements_data'
                continue

            if mode == 'nodes_data':
                parts = line.split()
                if len(parts) < 2:
                    continue
                coords = list(map(float, parts[:3]))
                # 若只有 2D 坐标，补零
                while len(coords) < 3:
                    coords.append(0.0)
                nodes.append(coords[:3])
                if len(parts) > 3:
                    node_labels.append(int(parts[3]))
                else:
                    node_labels.append(0)
            elif mode == 'elements_data':
                parts = line.split()
                if len(parts) < 2:
                    continue
                # 第一个整数为单元类型（3=triangle, 4=tetrahedron）
                elem_type = int(parts[0])
                conn = list(map(int, parts[1:1 + elem_type]))
                elements.append(conn)
                elem_types.append(elem_type)
                if len(parts) > 1 + elem_type:
                    element_labels.append(int(parts[1 + elem_type]))
                else:
                    element_labels.append(0)

    nodes = np.array(nodes, dtype=float)
    node_labels = np.array(node_labels, dtype=int)
    elements = np.array(elements, dtype=int)
    elem_types = np.array(elem_types, dtype=int)
    element_labels = np.array(element_labels, dtype=int)
    return nodes, node_labels, elements, elem_types, element_labels


def write_simple_mesh(filename, nodes, elements, elem_types=None,
                      node_labels=None, element_labels=None):
    """
    写出简化网格文件（参考 ice_to_medit 的 mesh_write 思想）。
    """
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    n_nodes = nodes.shape[0]
    n_elements = elements.shape[0]

    if node_labels is None:
        node_labels = np.zeros(n_nodes, dtype=int)
    if element_labels is None:
        element_labels = np.zeros(n_elements, dtype=int)
    if elem_types is None:
        elem_types = np.array([elements.shape[1]] * n_elements, dtype=int)

    with open(filename, 'w') as f:
        f.write("# Simple mesh format (synthesized from ice_to_medit + gmsh_to_fem)\n")
        f.write("NODES\n")
        f.write(f"{n_nodes}\n")
        for i in range(n_nodes):
            f.write(f"{nodes[i, 0]:.8e} {nodes[i, 1]:.8e} {nodes[i, 2]:.8e} {node_labels[i]}\n")
        f.write("ELEMENTS\n")
        f.write(f"{n_elements}\n")
        for i in range(n_elements):
            conn = elements[i]
            f.write(f"{elem_types[i]}")
            for c in conn:
                f.write(f" {c}")
            f.write(f" {element_labels[i]}\n")


def detect_dimension(nodes):
    """
    检测节点数据的维度（参考 gmsh_size_read 中的维度推断）。
    若 z 坐标全同则 dim=2，若 y 也全同则 dim=1。
    """
    nodes = np.asarray(nodes, dtype=float)
    if nodes.shape[0] == 0:
        return 0
    x_min, x_max = np.min(nodes[:, 0]), np.max(nodes[:, 0])
    y_min, y_max = np.min(nodes[:, 1]), np.max(nodes[:, 1])
    z_min, z_max = np.min(nodes[:, 2]), np.max(nodes[:, 2])

    tol = 1e-12
    dim = 3
    if abs(z_max - z_min) < tol:
        dim = 2
        if abs(y_max - y_min) < tol:
            dim = 1
    return dim


def compute_tetrahedron_volume(nodes, tet):
    """
    计算单个四面体体积（参考 ice_to_medit 的 tetrahedron 几何）。
    V = |det(v1-v0, v2-v0, v3-v0)| / 6
    """
    v0 = nodes[tet[0]]
    v1 = nodes[tet[1]]
    v2 = nodes[tet[2]]
    v3 = nodes[tet[3]]
    mat = np.array([v1 - v0, v2 - v0, v3 - v0])
    vol = abs(np.linalg.det(mat)) / 6.0
    return vol


def compute_triangle_area(nodes, tri):
    """
    计算单个三角形面积。
    """
    v0 = nodes[tri[0]]
    v1 = nodes[tri[1]]
    v2 = nodes[tri[2]]
    cross = np.cross(v1 - v0, v2 - v0)
    area = 0.5 * np.linalg.norm(cross)
    return area


def compute_mesh_quality(nodes, elements, elem_types):
    """
    计算网格单元质量指标：
      - 四面体：Q = 6√6 V / (Σ l_i²)^{3/2}
      - 三角形：Q = 4√3 A / (a² + b² + c²)
    返回各单元的质量值与平均质量。
    """
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    elem_types = np.asarray(elem_types, dtype=int)
    n_elem = elements.shape[0]
    qualities = np.zeros(n_elem)

    for i in range(n_elem):
        etype = elem_types[i]
        elem = elements[i]
        if etype == 3 or len(elem) == 3:
            # 三角形
            v0, v1, v2 = nodes[elem[0]], nodes[elem[1]], nodes[elem[2]]
            a = np.linalg.norm(v1 - v0)
            b = np.linalg.norm(v2 - v1)
            c = np.linalg.norm(v0 - v2)
            area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            denom = a ** 2 + b ** 2 + c ** 2
            if denom > 1e-15:
                qualities[i] = 4.0 * np.sqrt(3.0) * area / denom
            else:
                qualities[i] = 0.0
        elif etype == 4 or len(elem) == 4:
            # 四面体
            vol = compute_tetrahedron_volume(nodes, elem)
            edge_sum_sq = 0.0
            edges = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
            for (p,q) in edges:
                l = np.linalg.norm(nodes[elem[p]] - nodes[elem[q]])
                edge_sum_sq += l ** 2
            if edge_sum_sq > 1e-15:
                qualities[i] = 6.0 * np.sqrt(6.0) * vol / (edge_sum_sq ** 1.5)
            else:
                qualities[i] = 0.0
        else:
            qualities[i] = 0.0

    avg_quality = float(np.mean(qualities)) if n_elem > 0 else 0.0
    min_quality = float(np.min(qualities)) if n_elem > 0 else 0.0
    return qualities, avg_quality, min_quality


def generate_unit_cube_mesh(nx=5, ny=5, nz=5):
    """
    生成单位立方体的简单结构化四面体网格，用于测试。
    返回 nodes, elements, elem_types。
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.linspace(0, 1, nz)
    nodes = []
    idx_map = {}
    index = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                nodes.append([x[i], y[j], z[k]])
                idx_map[(i, j, k)] = index
                index += 1
    nodes = np.array(nodes, dtype=float)

    elements = []
    elem_types = []
    for k in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):
                # 将一个立方体拆分为 5 或 6 个四面体，这里简化为 6 个
                p0 = idx_map[(i, j, k)]
                p1 = idx_map[(i + 1, j, k)]
                p2 = idx_map[(i + 1, j + 1, k)]
                p3 = idx_map[(i, j + 1, k)]
                p4 = idx_map[(i, j, k + 1)]
                p5 = idx_map[(i + 1, j, k + 1)]
                p6 = idx_map[(i + 1, j + 1, k + 1)]
                p7 = idx_map[(i, j + 1, k + 1)]
                tets = [
                    [p0, p1, p3, p4],
                    [p1, p2, p3, p6],
                    [p4, p5, p1, p6],
                    [p4, p6, p3, p7],
                    [p1, p3, p4, p6],
                ]
                for t in tets:
                    elements.append(t)
                    elem_types.append(4)
    elements = np.array(elements, dtype=int)
    elem_types = np.array(elem_types, dtype=int)
    return nodes, elements, elem_types
