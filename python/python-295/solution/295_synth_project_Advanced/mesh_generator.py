#!/usr/bin/env python3
"""
mesh_generator.py
=================
二维轴对称 (r-z) 网格生成模块，融合种子项目:
  - [756] mesh_vtoe: 顶点-单元邻接关系
  - [1310] triangle_io: TRIANGLE 格式 I/O
  - [1331] triangulation_boundary: 边界边检测
  - [1413] welzl/icosahedron: 正二十面体球面网格 + 最小包围球

物理背景:
  ICF 内爆模拟需要在 (r,z) 半平面生成结构化/非结构化网格。
  对于球形靶丸，需要极坐标下的曲面贴合网格。
  边界识别用于施加 Dirichlet/Neumann 边界条件。
  正二十面体细分网格用于球谐函数展开时的球面离散化。
"""

import numpy as np
from scipy.spatial import ConvexHull


class MeshVtoe:
    """
    顶点-单元邻接关系 (融合 [756] mesh_vtoe)。
    构建 vertex-to-element 索引, 用于梯度重构和通量计算。
    """

    def __init__(self, element_order, element_num, vertex_num, etov):
        """
        参数:
            element_order: 每个单元的节点数 (三角形=3, 四边形=4)
            element_num: 单元总数
            vertex_num: 节点总数
            etov: 单元-节点映射表, shape=(element_num, element_order)
                  etov[e, k] 为单元 e 的第 k 个节点的全局编号 (0-indexed)
        """
        self.element_order = element_order
        self.element_num = element_num
        self.vertex_num = vertex_num
        self.etov = np.asarray(etov, dtype=np.int64)

        # 构建顶点-单元邻接
        self.vtoe_pointer = np.zeros(vertex_num + 1, dtype=np.int64)
        vtoe_list = [[] for _ in range(vertex_num)]

        for e in range(element_num):
            for k in range(element_order):
                v = self.etov[e, k]
                vtoe_list[v].append(e)

        # 排序并扁平化
        self.vtoe = []
        for v in range(vertex_num):
            self.vtoe_pointer[v] = len(self.vtoe)
            self.vtoe.extend(sorted(vtoe_list[v]))
        self.vtoe_pointer[vertex_num] = len(self.vtoe)
        self.vtoe = np.array(self.vtoe, dtype=np.int64)

    def get_elements_around_vertex(self, v):
        """获取包含节点 v 的所有单元"""
        start = self.vtoe_pointer[v]
        end = self.vtoe_pointer[v + 1]
        return self.vtoe[start:end]

    def vertex_degree(self, v):
        """获取节点 v 的度 (相邻单元数)"""
        return self.vtoe_pointer[v + 1] - self.vtoe_pointer[v]


class BoundaryDetector:
    """
    边界边检测器 (融合 [1331] triangulation_boundary)。
    通过统计每条边的出现次数, 识别边界边。
    """

    def __init__(self, triangle_node):
        """
        参数:
            triangle_node: 三角形节点索引, shape=(n_tri, 3)
        """
        self.triangle_node = np.asarray(triangle_node, dtype=np.int64)
        self.n_tri = self.triangle_node.shape[0]
        self.boundary_edges = self._detect_boundary()

    def _detect_boundary(self):
        """
        边界检测算法:
          1. 枚举所有边 (每个三角形 3 条)
          2. 对每条边, 将其节点排序后作为 key
          3. 仅出现一次的边为边界边
        """
        edge_count = {}
        edge_orient = {}

        for t in range(self.n_tri):
            for k in range(3):
                v1 = self.triangle_node[t, k]
                v2 = self.triangle_node[t, (k + 1) % 3]
                key = (min(v1, v2), max(v1, v2))
                edge_count[key] = edge_count.get(key, 0) + 1
                if key not in edge_orient:
                    edge_orient[key] = (v1, v2)

        boundary = []
        for key, count in edge_count.items():
            if count == 1:
                v1, v2 = edge_orient[key]
                boundary.append((v1, v2))

        return np.array(boundary, dtype=np.int64) if boundary else np.zeros((0, 2), dtype=np.int64)

    def get_boundary_nodes(self):
        """获取所有边界节点"""
        if self.boundary_edges.size == 0:
            return np.array([], dtype=np.int64)
        return np.unique(self.boundary_edges.ravel())

    def get_boundary_count(self):
        """获取边界边数量"""
        return self.boundary_edges.shape[0]


class TriangleIO:
    """
    TRIANGLE 格式 I/O (融合 [1310] triangle_io)。
    简化版, 仅保留节点和单元读写。
    """

    @staticmethod
    def write_node(filename, node_xy, node_marker=None):
        """
        写入 TRIANGLE .node 文件。
        格式:
          node_num  dim  n_attr  n_marker
          idx  x  y  [attrs]  [marker]
        """
        n_nodes = node_xy.shape[0]
        dim = node_xy.shape[1]
        n_attr = 0
        n_marker = 1 if node_marker is not None else 0

        with open(filename, 'w') as f:
            f.write(f"{n_nodes}  {dim}  {n_attr}  {n_marker}\n")
            for i in range(n_nodes):
                line = f"{i}  {node_xy[i, 0]:.15e}  {node_xy[i, 1]:.15e}"
                if node_marker is not None:
                    line += f"  {node_marker[i]}"
                f.write(line + "\n")

    @staticmethod
    def read_node(filename):
        """读取 TRIANGLE .node 文件"""
        with open(filename, 'r') as f:
            header = f.readline().split()
            n_nodes, dim = int(header[0]), int(header[1])
            n_attr, n_marker = int(header[2]), int(header[3])

            node_xy = np.zeros((n_nodes, dim))
            node_marker = np.zeros(n_nodes, dtype=np.int64) if n_marker > 0 else None

            for i in range(n_nodes):
                parts = f.readline().split()
                node_xy[i, 0] = float(parts[1])
                node_xy[i, 1] = float(parts[2])
                if n_marker > 0:
                    node_marker[i] = int(parts[-1])

        return node_xy, node_marker

    @staticmethod
    def write_element(filename, triangle_node):
        """写入 TRIANGLE .ele 文件"""
        n_tri, n_local = triangle_node.shape
        with open(filename, 'w') as f:
            f.write(f"{n_tri}  {n_local}  0\n")
            for i in range(n_tri):
                line = f"{i}"
                for k in range(n_local):
                    line += f"  {triangle_node[i, k]}"
                f.write(line + "\n")


class IcosahedronSphereMesh:
    """
    正二十面体球面网格生成与细分 (融合 [1413] welzl/icosahedron)。
    用于球谐函数展开的球面离散化。
    """

    @staticmethod
    def generate(tessellation_level=2):
        """
        生成正二十面体并细分到指定层级。
        细分后所有顶点投影到单位球面。

        参数:
            tessellation_level: 细分层级 (0=正二十面体, 每+1面数×4)
        返回:
            vertices: (N_v, 3) 单位球面上的顶点坐标
            faces: (N_f, 3) 三角形面片
        """
        phi = (1.0 + np.sqrt(5.0)) / 2.0  # 黄金比例

        # 正二十面体的 12 个顶点
        verts = np.array([
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
        ], dtype=np.float64)

        # 归一化到单位球面
        norms = np.linalg.norm(verts, axis=1, keepdims=True)
        verts = verts / norms

        # 20 个三角面
        faces = np.array([
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ], dtype=np.int64)

        # 细分
        for _ in range(tessellation_level):
            verts, faces = IcosahedronSphereMesh._subdivide(verts, faces)

        return verts, faces

    @staticmethod
    def _subdivide(vertices, faces):
        """一级细分: 每个三角形 → 4 个三角形"""
        edge_midpoints = {}
        new_verts = list(vertices)

        def get_midpoint(i, j):
            key = (min(i, j), max(i, j))
            if key in edge_midpoints:
                return edge_midpoints[key]
            mid = (vertices[i] + vertices[j]) / 2.0
            mid = mid / np.linalg.norm(mid)  # 投影到球面
            idx = len(new_verts)
            new_verts.append(mid)
            edge_midpoints[key] = idx
            return idx

        new_faces = []
        for tri in faces:
            a, b, c = tri
            ab = get_midpoint(a, b)
            bc = get_midpoint(b, c)
            ca = get_midpoint(c, a)
            new_faces.append([a, ab, ca])
            new_faces.append([b, bc, ab])
            new_faces.append([c, ca, bc])
            new_faces.append([ab, bc, ca])

        return np.array(new_verts), np.array(new_faces, dtype=np.int64)


class WelzlMinBoundingSphere:
    """
    Welzl 最小包围球算法 (融合 [1413] welzl)。
    用于确定 ICF 靶丸内爆区域的几何包络。
    """

    @staticmethod
    def compute(points, max_iter=10000):
        """
        Welzl 算法求最小包围球 (随机增量法)。

        参数:
            points: (N, D) 点集
            max_iter: 最大迭代次数
        返回:
            center: (D,) 球心
            radius: 球半径
        """
        pts = np.asarray(points, dtype=np.float64).copy()
        n, d = pts.shape
        if n == 0:
            return np.zeros(d), 0.0

        # 随机排列
        rng = np.random.default_rng(42)
        perm = rng.permutation(n)
        pts = pts[perm]

        # 初始化: 用第一个点
        center = pts[0].copy()
        radius = 0.0

        for i in range(1, n):
            if np.linalg.norm(pts[i] - center) > radius + 1.0e-14:
                # 点 i 在当前球外, 它必在边界球上
                center, radius = WelzlMinBoundingSphere._welzl_with_boundary(
                    pts[:i + 1], [pts[i]], d
                )

        return center, radius

    @staticmethod
    def _welzl_with_boundary(points, boundary, d):
        """Welzl 算法递归辅助"""
        if len(boundary) == d + 1:
            return WelzlMinBoundingSphere._sphere_from_boundary(boundary)

        center = boundary[0].copy() if boundary else points[0].copy()
        radius = 0.0
        if boundary:
            center, radius = WelzlMinBoundingSphere._sphere_from_boundary(boundary)

        for i in range(len(points)):
            if np.linalg.norm(points[i] - center) > radius + 1.0e-14:
                new_boundary = list(boundary) + [points[i]]
                center, radius = WelzlMinBoundingSphere._welzl_with_boundary(
                    points[:i], new_boundary, d
                )
        return center, radius

    @staticmethod
    def _sphere_from_boundary(boundary):
        """由边界点集计算包围球"""
        pts = np.array(boundary)
        n = pts.shape[0]
        d = pts.shape[1]

        if n == 1:
            return pts[0].copy(), 0.0
        elif n == 2:
            center = (pts[0] + pts[1]) / 2.0
            radius = np.linalg.norm(pts[1] - pts[0]) / 2.0
            return center, radius
        elif n == 3:
            # 三点确定一个圆 (在它们所在的平面上)
            # 使用 circumcenter 公式
            a, b, c = pts[0], pts[1], pts[2]
            ab = b - a
            ac = c - a
            ab2 = np.dot(ab, ab)
            ac2 = np.dot(ac, ac)
            ab_ac = np.dot(ab, ac)
            denom = 2.0 * (ab2 * ac2 - ab_ac ** 2)
            if abs(denom) < 1e-30:
                # 退化: 三点共线, 用两点包围
                d_ab = np.linalg.norm(b - a)
                d_ac = np.linalg.norm(c - a)
                d_bc = np.linalg.norm(c - b)
                max_d = max(d_ab, d_ac, d_bc)
                if max_d == d_ab:
                    return (a + b) / 2.0, d_ab / 2.0
                elif max_d == d_ac:
                    return (a + c) / 2.0, d_ac / 2.0
                else:
                    return (b + c) / 2.0, d_bc / 2.0
            s1 = (ab2 * ac2 - ac2 * ab_ac) / denom  # 简化
            s2 = (ac2 * ab2 - ab2 * ab_ac) / denom
            # 实际上 circumcenter = a + s1*ab + s2*ac (需要更准确的推导)
            # 使用线性方程组
            A = 2.0 * (pts[1:] - pts[0])
            b_rhs = np.sum(pts[1:] ** 2, axis=1) - np.sum(pts[0] ** 2)
            # 使用 lstsq 避免奇异
            center, _, _, _ = np.linalg.lstsq(A, b_rhs, rcond=None)
            radius = np.linalg.norm(pts[0] - center)
            return center, radius
        else:
            # 超定系统: 用最小二乘法
            # |x - c|² = R² 对所有边界点
            # 等价于线性方程组
            A = 2.0 * (pts[1:] - pts[0])
            b = np.sum(pts[1:] ** 2, axis=1) - np.sum(pts[0] ** 2)
            # 使用 lstsq 避免奇异
            center, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            radius = np.linalg.norm(pts[0] - center)
            return center, radius


class AxisymmetricMeshGenerator:
    """
    二维轴对称 (r-z) 结构化网格生成器。
    支持矩形区域和球形靶丸区域的网格。
    """

    @staticmethod
    def rectangular(nr, nz, r_min, r_max, z_min, z_max):
        """
        生成矩形区域的结构化四边形网格 (r-z 平面)。

        参数:
            nr, nz: r, z 方向的节点数
            r_min, r_max: 径向范围 [cm]
            z_min, z_max: 轴向范围 [cm]
        返回:
            node_xy: (N_nodes, 2) 节点坐标
            quad_nodes: (N_elements, 4) 四边形单元节点索引
        """
        r = np.linspace(r_min, r_max, nr)
        z = np.linspace(z_min, z_max, nz)
        rr, zz = np.meshgrid(r, z, indexing='ij')

        node_xy = np.column_stack([rr.ravel(), zz.ravel()])
        quad_nodes = []
        for i in range(nr - 1):
            for j in range(nz - 1):
                n0 = i * nz + j
                n1 = n0 + 1
                n2 = (i + 1) * nz + j + 1
                n3 = (i + 1) * nz + j
                quad_nodes.append([n0, n1, n2, n3])

        return node_xy, np.array(quad_nodes, dtype=np.int64)

    @staticmethod
    def spherical_capsule(nr, ntheta, r_inner, r_outer, theta_min=0.0, theta_max=np.pi):
        """
        生成球形靶丸的极坐标网格 (r-θ 平面)。
        用于 ICF 内爆模拟的靶丸区域。

        参数:
            nr: 径向节点数
            ntheta: 角向节点数
            r_inner: 内半径 [cm]
            r_outer: 外半径 [cm]
            theta_min, theta_max: 极角范围 [rad]
        返回:
            node_xy: (N, 2) 节点 (r, z) 坐标
            quad_nodes: (N_elem, 4) 四边形单元
            boundary_mask: (N,) 边界标记 (1=边界, 0=内部)
        """
        r = np.linspace(r_inner, r_outer, nr)
        theta = np.linspace(theta_min, theta_max, ntheta)
        rr, tt = np.meshgrid(r, theta, indexing='ij')

        # 转换为 (r, z) 坐标 (轴对称)
        node_r = rr.ravel()
        node_z = node_r * np.cos(tt.ravel())
        # 注意: 在轴对称中, r 是到对称轴的距离
        # 这里用球坐标: r_sph → R_cyl = r_sph sin θ, Z = r_sph cos θ
        R_cyl = rr.ravel() * np.sin(tt.ravel())
        Z = rr.ravel() * np.cos(tt.ravel())

        node_xy = np.column_stack([R_cyl, Z])

        quad_nodes = []
        for i in range(nr - 1):
            for j in range(ntheta - 1):
                n0 = i * ntheta + j
                n1 = n0 + 1
                n2 = (i + 1) * ntheta + j + 1
                n3 = (i + 1) * ntheta + j
                quad_nodes.append([n0, n1, n2, n3])

        # 边界标记
        boundary_mask = np.zeros(node_xy.shape[0], dtype=np.int64)
        for i in range(node_xy.shape[0]):
            ii, jj = divmod(i, ntheta)
            if ii == 0 or ii == nr - 1 or jj == 0 or jj == ntheta - 1:
                boundary_mask[i] = 1

        return node_xy, np.array(quad_nodes, dtype=np.int64), boundary_mask


def generate_icf_mesh(mesh_type='rectangular', **kwargs):
    """
    统一的 ICF 网格生成接口。

    参数:
        mesh_type: 'rectangular' 或 'spherical'
        **kwargs: 传递给具体生成器的参数
    返回:
        dict: {node_xy, elements, boundary_mask, vtoe}
    """
    if mesh_type == 'rectangular':
        nr = kwargs.get('nr', 21)
        nz = kwargs.get('nz', 21)
        r_min = kwargs.get('r_min', 0.0)
        r_max = kwargs.get('r_max', 0.2)
        z_min = kwargs.get('z_min', -0.2)
        z_max = kwargs.get('z_max', 0.2)
        node_xy, elements = AxisymmetricMeshGenerator.rectangular(nr, nz, r_min, r_max, z_min, z_max)
        boundary_mask = np.zeros(node_xy.shape[0], dtype=np.int64)
        for i in range(node_xy.shape[0]):
            ii, jj = divmod(i, nz)
            if ii == 0 or ii == nr - 1 or jj == 0 or jj == nz - 1:
                boundary_mask[i] = 1
    elif mesh_type == 'spherical':
        nr = kwargs.get('nr', 21)
        ntheta = kwargs.get('ntheta', 21)
        r_inner = kwargs.get('r_inner', 0.0)
        r_outer = kwargs.get('r_outer', 0.1)
        node_xy, elements, boundary_mask = AxisymmetricMeshGenerator.spherical_capsule(
            nr, ntheta, r_inner, r_outer
        )
    else:
        raise ValueError(f"未知网格类型: {mesh_type}")

    # 将四边形拆分为两个三角形
    tri_nodes = []
    for q in elements:
        tri_nodes.append([q[0], q[1], q[2]])
        tri_nodes.append([q[0], q[2], q[3]])
    tri_nodes = np.array(tri_nodes, dtype=np.int64)

    # 构建顶点-单元邻接
    vtoe = MeshVtoe(3, tri_nodes.shape[0], node_xy.shape[0], tri_nodes)

    # 检测边界
    bd = BoundaryDetector(tri_nodes)

    return {
        'node_xy': node_xy,
        'tri_elements': tri_nodes,
        'quad_elements': elements,
        'boundary_mask': boundary_mask,
        'vtoe': vtoe,
        'boundary_edges': bd.boundary_edges,
        'boundary_nodes': bd.get_boundary_nodes()
    }
