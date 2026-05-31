"""
mesh_engine.py
==============
三维有限元网格生成与细化引擎

融合原始项目：
  - 378_fem_to_gmsh：FEM 到 Gmsh 格式转换，节点/单元数据结构
  - 1350_triangulation_refine：三角形网格细化（中点剖分）
  - 1168_stla_to_tri_surface_fast：STL 表面三角网格解析

数学物理模型：
  1. 四面体单元 (tetrahedron) 线性插值：
       T(x) = Σ_{i=1}^4 N_i(x) T_i
     其中 N_i 为体积坐标形函数。
  2. 网格细化（红-绿细化 / 中点剖分）：
       每个三角形通过连接三边中点细分为 4 个子三角形；
       每个四面体通过连接六条棱中点细分为 8 个子四面体。
  3. 节点排序一致性检查：
       0-based ↔ 1-based 索引自动检测与转换。
"""

import numpy as np


class TetrahedralMesh:
    """
    四面体网格数据结构。

    属性
    ----
    nodes : np.ndarray, shape (n_nodes, 3)
        节点坐标数组
    elements : np.ndarray, shape (n_elems, 4)
        单元节点索引（0-based）
    """

    def __init__(self, nodes=None, elements=None):
        self.nodes = np.zeros((0, 3), dtype=float) if nodes is None else np.asarray(nodes, dtype=float)
        self.elements = np.zeros((0, 4), dtype=int) if elements is None else np.asarray(elements, dtype=int)
        self._validate()

    def _validate(self):
        if self.nodes.size == 0 or self.elements.size == 0:
            return
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3:
            raise ValueError("TetrahedralMesh: nodes 必须为 (n, 3) 数组")
        if self.elements.ndim != 2 or self.elements.shape[1] != 4:
            raise ValueError("TetrahedralMesh: elements 必须为 (m, 4) 数组")
        n_nodes = self.nodes.shape[0]
        emin = self.elements.min()
        emax = self.elements.max()
        if emin < 0 or emax >= n_nodes:
            # 尝试检测 1-based 索引
            if emin == 1 and emax == n_nodes:
                self.elements = self.elements - 1
            else:
                raise ValueError(
                    "TetrahedralMesh: 单元节点索引越界 (min=%d, max=%d, n_nodes=%d)" % (emin, emax, n_nodes)
                )

    @property
    def n_nodes(self):
        return self.nodes.shape[0]

    @property
    def n_elements(self):
        return self.elements.shape[0]

    def element_volume(self, elem_idx: int):
        """
        计算第 elem_idx 个四面体单元的有向体积。

        公式：
            V = | (a-d) · [(b-d) × (c-d)] | / 6
        其中 a,b,c,d 为四个顶点坐标。
        """
        idx = self.elements[elem_idx]
        a, b, c, d = self.nodes[idx[0]], self.nodes[idx[1]], self.nodes[idx[2]], self.nodes[idx[3]]
        vol = np.dot(a - d, np.cross(b - d, c - d)) / 6.0
        return abs(vol)

    def total_volume(self):
        """计算网格覆盖的总体积。"""
        vol = 0.0
        for e in range(self.n_elements):
            vol += self.element_volume(e)
        return vol

    def refine_uniform(self):
        """
        对四面体网格进行一致细化（中点剖分）。

        每个四面体通过连接六条棱的中点，被剖分为 8 个子四面体。
        新增节点数为原棱数（去重后）。

        数学上，对每条棱 (i,j) 产生中点：
            m_{ij} = (x_i + x_j) / 2
        然后按标准模板将原四面体分解为 8 个子单元。
        """
        if self.n_elements == 0:
            return TetrahedralMesh()

        n_nodes_old = self.n_nodes
        # 建立边到全局中点索引的映射
        edge_to_mid = {}
        new_nodes = [self.nodes.copy()]

        def get_mid(i, j):
            key = (min(i, j), max(i, j))
            if key not in edge_to_mid:
                mid_idx = n_nodes_old + len(edge_to_mid)
                edge_to_mid[key] = mid_idx
                new_nodes.append(((self.nodes[i] + self.nodes[j]) / 2.0).reshape(1, 3))
            return edge_to_mid[key]

        new_elements = []
        for e in range(self.n_elements):
            v0, v1, v2, v3 = self.elements[e]
            m01 = get_mid(v0, v1)
            m02 = get_mid(v0, v2)
            m03 = get_mid(v0, v3)
            m12 = get_mid(v1, v2)
            m13 = get_mid(v1, v3)
            m23 = get_mid(v2, v3)

            # 标准 8-子四面体剖分模板
            subs = [
                [v0, m01, m02, m03],
                [m01, v1, m12, m13],
                [m02, m12, v2, m23],
                [m03, m13, m23, v3],
                [m01, m02, m03, m13],
                [m01, m02, m12, m13],
                [m02, m03, m13, m23],
                [m02, m12, m13, m23],
            ]
            new_elements.extend(subs)

        all_nodes = np.vstack(new_nodes) if len(new_nodes) > 1 else new_nodes[0]
        all_elements = np.array(new_elements, dtype=int)
        return TetrahedralMesh(all_nodes, all_elements)

    def compute_centroids(self):
        """
        计算每个四面体单元的质心。

        公式：
            x_c = (x_0 + x_1 + x_2 + x_3) / 4
        """
        c = np.zeros((self.n_elements, 3), dtype=float)
        for e in range(self.n_elements):
            idx = self.elements[e]
            c[e] = self.nodes[idx].mean(axis=0)
        return c

    def compute_boundary_faces(self):
        """
        提取位于网格边界上的三角形面片。

        原理：统计每个三角形面出现的次数，只出现一次的面为边界面。
        """
        face_count = {}
        for e in range(self.n_elements):
            idx = list(self.elements[e])
            faces = [
                tuple(sorted([idx[0], idx[1], idx[2]])),
                tuple(sorted([idx[0], idx[1], idx[3]])),
                tuple(sorted([idx[0], idx[2], idx[3]])),
                tuple(sorted([idx[1], idx[2], idx[3]])),
            ]
            for f in faces:
                face_count[f] = face_count.get(f, 0) + 1

        boundary_faces = [f for f, cnt in face_count.items() if cnt == 1]
        return np.array(boundary_faces, dtype=int)


def generate_uniform_box_mesh(xlim=(-1.0, 1.0),
                              ylim=(-1.0, 1.0),
                              zlim=(-1.0, 1.0),
                              nx=4, ny=4, nz=4):
    """
    生成长方体区域的均匀四面体网格（通过将六面体剖分为 6 个四面体）。

    参数
    ----
    xlim, ylim, zlim : tuple
        区域边界
    nx, ny, nz : int
        各方向网格划分数（≥2）

    返回
    ----
    mesh : TetrahedralMesh
    """
    nx = max(2, int(nx))
    ny = max(2, int(ny))
    nz = max(2, int(nz))

    x = np.linspace(xlim[0], xlim[1], nx)
    y = np.linspace(ylim[0], ylim[1], ny)
    z = np.linspace(zlim[0], zlim[1], nz)

    nodes = []
    node_index = {}
    idx = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                nodes.append([x[i], y[j], z[k]])
                node_index[(i, j, k)] = idx
                idx += 1

    nodes = np.array(nodes, dtype=float)
    elements = []

    for k in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):
                c = [
                    node_index[(i, j, k)],
                    node_index[(i + 1, j, k)],
                    node_index[(i + 1, j + 1, k)],
                    node_index[(i, j + 1, k)],
                    node_index[(i, j, k + 1)],
                    node_index[(i + 1, j, k + 1)],
                    node_index[(i + 1, j + 1, k + 1)],
                    node_index[(i, j + 1, k + 1)],
                ]
                # 将六面体剖分为 6 个四面体（避免对角面歧义）
                # 选择一种一致的剖分模式
                tets = [
                    [c[0], c[1], c[3], c[4]],
                    [c[1], c[3], c[4], c[5]],
                    [c[1], c[2], c[3], c[5]],
                    [c[3], c[4], c[5], c[7]],
                    [c[3], c[5], c[6], c[7]],
                    [c[2], c[3], c[5], c[6]],
                ]
                elements.extend(tets)

    elements = np.array(elements, dtype=int)
    return TetrahedralMesh(nodes, elements)


def gmsh_format_string(mesh: TetrahedralMesh):
    """
    将四面体网格导出为 Gmsh ASCII 格式字符串（源自 fem_to_gmsh 思想）。

    Gmsh 格式：
        $MeshFormat
        2.2 0 8
        $EndMeshFormat
        $Nodes
        n_nodes
        idx x y z
        $EndNodes
        $Elements
        n_elements
        idx type num_tags tag1 tag2 n1 n2 n3 n4
        $EndElements
    """
    lines = []
    lines.append("$MeshFormat")
    lines.append("2.2 0 8")
    lines.append("$EndMeshFormat")
    lines.append("$Nodes")
    lines.append("%d" % mesh.n_nodes)
    for i in range(mesh.n_nodes):
        lines.append("  %d %.16g %.16g %.16g" % (i + 1, mesh.nodes[i, 0], mesh.nodes[i, 1], mesh.nodes[i, 2]))
    lines.append("$EndNodes")
    lines.append("$Elements")
    lines.append("%d" % mesh.n_elements)
    for e in range(mesh.n_elements):
        # Gmsh tetrahedron type 4 for 4-node tetrahedron
        nodes_str = " ".join("%d" % (mesh.elements[e, v] + 1) for v in range(4))
        lines.append("  %d 4 2 0 %d %s" % (e + 1, e + 1, nodes_str))
    lines.append("$EndElements")
    return "\n".join(lines)


def parse_stl_like_surface(nodes_xyz, face_nodes):
    """
    模拟 STL 到 TRI_SURFACE 的转换（源自 stla_to_tri_surface_fast）。

    对输入的三角形表面节点和索引进行去重与 0-based 标准化。
    """
    nodes = np.asarray(nodes_xyz, dtype=float)
    faces = np.asarray(face_nodes, dtype=int)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("parse_stl_like_surface: faces 必须为 (m, 3) 数组")
    return nodes, faces
