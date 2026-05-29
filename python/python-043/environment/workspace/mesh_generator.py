"""
mesh_generator.py — 球壳网格生成器

融合以下种子项目：
- 254_cvt_circle_uniform : CVT (Centroidal Voronoi Tessellation) 在圆盘上的生成思想
- 1310_triangle_io       : Triangle 网格文件读写格式
- 1349_triangulation_rcm : 三角网格的 RCM 重排序

功能：
1. 在球壳（内半径 r_icb，外半径 r_cmb）内生成三维 CVT 网格节点
2. 构建 Delaunay 三角化（二维球面投影 + 三维四面体）
3. 应用 Reverse Cuthill-McKee (RCM) 算法对节点重排序以降低稀疏矩阵带宽
4. 导出/导入 Triangle 格式网格文件
"""

import numpy as np
from scipy.spatial import Delaunay


def cvt_sphere_uniform(n_points, n_samples=10000, n_iter=20):
    """
    在单位球面上生成近似 CVT (Centroidal Voronoi Tessellation) 节点。
    算法思想源自 254_cvt_circle_uniform：
      1. 随机采样大量点
      2. 对每个生成元，找最近的采样点集
      3. 用质心替换生成元
      4. 迭代直至收敛
    """
    # 在球面上均匀初始化生成元
    generators = _random_sphere_points(n_points)

    for it in range(n_iter):
        # 在球面上大量采样
        samples = _random_sphere_points(n_samples)

        # 对每个采样点，找最近的生成元
        closest = _nearest_neighbor(samples, generators)

        # 计算每个 Voronoi 单元的质心
        new_generators = np.zeros_like(generators)
        counts = np.zeros(n_points)
        for i in range(n_points):
            mask = (closest == i)
            if np.any(mask):
                pts = samples[mask]
                centroid = np.mean(pts, axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 1e-15:
                    centroid /= norm
                new_generators[i] = centroid
                counts[i] = np.sum(mask)
            else:
                new_generators[i] = generators[i]
                counts[i] = 1

        generators = new_generators

    return generators


def _random_sphere_points(n):
    """在单位球面上均匀生成 n 个随机点。"""
    p = np.random.normal(size=(n, 3))
    norms = np.linalg.norm(p, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return p / norms


def _nearest_neighbor(points, generators):
    """对每个 point，返回最近 generator 的索引。"""
    dists = np.linalg.norm(points[:, None, :] - generators[None, :, :], axis=2)
    return np.argmin(dists, axis=1)


def spherical_shell_mesh(n_radial, n_theta, n_phi,
                          r_icb=0.35, r_cmb=1.0,
                          use_cvt=False, cvt_samples=5000, cvt_iter=15):
    """
    生成球壳结构化/半结构化网格节点。

    参数：
      n_radial : 径向层数
      n_theta  : 极角方向节点数
      n_phi    : 方位角方向节点数
      r_icb    : 内核边界半径（归一化）
      r_cmb    : 核幔边界半径（归一化）
      use_cvt  : 是否在球面上使用 CVT 优化

    返回：
      nodes    : (N, 3) 节点坐标数组
      elements : (M, 4) 四面体单元索引（如可用）
      r_levels : 径向分层半径
      theta_levels : 极角分层
      phi_levels   : 方位角分层
    """
    # 径向分层（对数网格，增强边界层分辨率）
    s = np.linspace(0.0, 1.0, n_radial)
    # 使用映射 r = r_icb + (r_cmb - r_icb) * s^2 增强边界层
    r_levels = r_icb + (r_cmb - r_icb) * (s ** 2)

    theta_levels = np.linspace(0.0, np.pi, n_theta)
    phi_levels = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

    nodes_list = []
    for r in r_levels:
        if use_cvt and r > r_icb + 1e-6:
            # 在中层使用 CVT 优化球面网格
            n_surf = max(50, n_theta * n_phi // 4)
            surf_pts = cvt_sphere_uniform(n_surf, cvt_samples, cvt_iter)
            surf_pts *= r
            nodes_list.append(surf_pts)
        else:
            # 结构化网格
            layer_pts = []
            for theta in theta_levels:
                for phi in phi_levels:
                    x = r * np.sin(theta) * np.cos(phi)
                    y = r * np.sin(theta) * np.sin(phi)
                    z = r * np.cos(theta)
                    layer_pts.append([x, y, z])
            nodes_list.append(np.array(layer_pts))

    nodes = np.vstack(nodes_list)

    # 构建 Delaunay 四面体（仅在节点不太多的情况下）
    if len(nodes) <= 5000:
        elements = Delaunay(nodes).simplices
    else:
        elements = np.array([])

    return nodes, elements, r_levels, theta_levels, phi_levels


def build_adjacency_matrix(nodes, elements):
    """
    从四面体单元构建节点邻接矩阵（用于 RCM 重排序）。
    源自 1349_triangulation_rcm 的邻接图构建思想。
    """
    n_nodes = len(nodes)
    adjacency = [set() for _ in range(n_nodes)]

    if elements.size == 0:
        # 无单元信息时，用距离阈值构建邻接
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                dist = np.linalg.norm(nodes[i] - nodes[j])
                if dist < 0.3:
                    adjacency[i].add(j)
                    adjacency[j].add(i)
        return adjacency

    for elem in elements:
        for i in range(len(elem)):
            for j in range(i + 1, len(elem)):
                a, b = elem[i], elem[j]
                adjacency[a].add(b)
                adjacency[b].add(a)

    return adjacency


def reverse_cuthill_mckee(adjacency):
    """
    Reverse Cuthill-McKee (RCM) 算法实现。
    源自 1349_triangulation_rcm。

    目标：对节点重新编号，使邻接矩阵的带宽最小化，
          从而提升稀疏矩阵求解器的缓存效率。
    """
    n = len(adjacency)
    visited = [False] * n
    permutation = []

    # 从最小度节点开始（优先处理边界节点）
    degrees = [len(adjacency[i]) for i in range(n)]

    while len(permutation) < n:
        # 找未访问的最小度节点
        unvisited_degrees = [(degrees[i], i) for i in range(n) if not visited[i]]
        if not unvisited_degrees:
            break
        _, start = min(unvisited_degrees)

        queue = [start]
        visited[start] = True

        while queue:
            # 按度数升序排列当前层的邻居
            current = queue.pop(0)
            permutation.append(current)

            neighbors = sorted([v for v in adjacency[current] if not visited[v]],
                               key=lambda v: degrees[v])
            for v in neighbors:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)

    # Reverse
    permutation = permutation[::-1]
    return permutation


def apply_rcm_permutation(nodes, elements, permutation):
    """
    将 RCM 重排序应用于节点和单元。
    返回新节点顺序及逆映射。
    """
    n = len(nodes)
    inv_perm = [0] * n
    for i, p in enumerate(permutation):
        inv_perm[p] = i

    new_nodes = nodes[permutation]
    new_elements = np.copy(elements)
    for i in range(elements.shape[0]):
        for j in range(elements.shape[1]):
            new_elements[i, j] = inv_perm[elements[i, j]]

    return new_nodes, new_elements, permutation, inv_perm


def write_triangle_node_file(filename, nodes, attributes=None, markers=None):
    """
    写入 Triangle 格式的 .node 文件（源自 1310_triangle_io）。
    格式：
      <顶点数> <维数> <属性数> <边界标记数>
      <编号> <x> <y> <z> [属性...] [边界标记]
    """
    n_nodes = len(nodes)
    dim = 3
    n_att = 0 if attributes is None else attributes.shape[1]
    n_marker = 0 if markers is None else 1

    with open(filename, 'w') as f:
        f.write(f"{n_nodes} {dim} {n_att} {n_marker}\n")
        for i, node in enumerate(nodes):
            line = f"{i + 1} {node[0]:.15e} {node[1]:.15e} {node[2]:.15e}"
            if attributes is not None:
                for att in attributes[i]:
                    line += f" {att:.15e}"
            if markers is not None:
                line += f" {int(markers[i])}"
            f.write(line + "\n")


def write_triangle_element_file(filename, elements, attributes=None):
    """
    写入 Triangle 格式的 .ele 文件（源自 1310_triangle_io）。
    格式：
      <单元数> <每单元节点数> <属性数>
      <编号> <n1> <n2> <n3> <n4> [属性...]
    """
    n_elem = len(elements)
    n_nodes_per_elem = elements.shape[1]
    n_att = 0 if attributes is None else attributes.shape[1]

    with open(filename, 'w') as f:
        f.write(f"{n_elem} {n_nodes_per_elem} {n_att}\n")
        for i, elem in enumerate(elements):
            line = f"{i + 1}"
            for idx in elem:
                line += f" {int(idx) + 1}"
            if attributes is not None:
                for att in attributes[i]:
                    line += f" {att:.15e}"
            f.write(line + "\n")


def read_triangle_node_file(filename):
    """
    读取 Triangle 格式的 .node 文件（源自 1310_triangle_io）。
    返回 nodes, attributes, markers。
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # 跳过空行和注释
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith('#'):
            clean_lines.append(stripped)

    header = clean_lines[0].split()
    n_nodes = int(header[0])
    dim = int(header[1])
    n_att = int(header[2])
    n_marker = int(header[3])

    nodes = np.zeros((n_nodes, dim))
    attributes = np.zeros((n_nodes, n_att)) if n_att > 0 else None
    markers = np.zeros(n_nodes, dtype=int) if n_marker > 0 else None

    for i, line in enumerate(clean_lines[1:1 + n_nodes]):
        parts = line.split()
        for d in range(dim):
            nodes[i, d] = float(parts[1 + d])
        offset = 1 + dim
        if n_att > 0:
            for a in range(n_att):
                attributes[i, a] = float(parts[offset + a])
            offset += n_att
        if n_marker > 0:
            markers[i] = int(parts[offset])

    return nodes, attributes, markers


def read_triangle_element_file(filename):
    """
    读取 Triangle 格式的 .ele 文件（源自 1310_triangle_io）。
    返回 elements, attributes。
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith('#'):
            clean_lines.append(stripped)

    header = clean_lines[0].split()
    n_elem = int(header[0])
    n_nodes_per_elem = int(header[1])
    n_att = int(header[2])

    elements = np.zeros((n_elem, n_nodes_per_elem), dtype=int)
    attributes = np.zeros((n_elem, n_att)) if n_att > 0 else None

    for i, line in enumerate(clean_lines[1:1 + n_elem]):
        parts = line.split()
        for j in range(n_nodes_per_elem):
            elements[i, j] = int(parts[1 + j]) - 1  # 转为 0 基索引
        offset = 1 + n_nodes_per_elem
        if n_att > 0:
            for a in range(n_att):
                attributes[i, a] = float(parts[offset + a])

    return elements, attributes


def estimate_mesh_quality(nodes, elements):
    """
    估计网格质量：计算最小二面角、最大纵横比。
    """
    if elements.size == 0:
        return {"min_dihedral": 0.0, "max_aspect_ratio": 1.0}

    quality_metrics = []
    for elem in elements:
        pts = nodes[elem]
        edges = []
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append(np.linalg.norm(pts[i] - pts[j]))
        edges = np.array(edges)
        min_edge = np.min(edges)
        max_edge = np.max(edges)
        aspect = max_edge / (min_edge + 1e-30)
        quality_metrics.append(aspect)

    return {
        "max_aspect_ratio": float(np.max(quality_metrics)),
        "mean_aspect_ratio": float(np.mean(quality_metrics)),
    }
