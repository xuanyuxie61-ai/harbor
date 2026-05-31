# -*- coding: utf-8 -*-
"""
spherical_mesh.py
球面三角剖分与像素化（HEALPix-like 层级网格）

核心物理：
    CMB 天图在球面上离散化。本模块实现基于二十面体的层级三角剖分：
    - 初始二十面体 20 个面，每个面为球面三角形
    - 每次细分将每个三角形 4 等分（中点归一化到球面）
    - 共形时间 η 类比为径向坐标，形成 3D 球壳网格

    节点邻接关系用于有限体积法求解玻尔兹曼方程或
    球谐变换中的快速求和。

融合种子项目 1336_triangulation_display 与 1345_triangulation_plot
（网格 I/O、连通性、邻居关系）。
"""

import numpy as np
from typing import List, Tuple, Dict
from utils import ensure_positive, clip_to_unit


# ---------------------------------------------------------------------------
# 单位球面几何工具
# ---------------------------------------------------------------------------
def spherical_to_cartesian(theta: float, phi: float) -> np.ndarray:
    """球坐标 (θ,φ) → 笛卡尔 (x,y,z)，单位球面。"""
    st = np.sin(theta)
    return np.array([st * np.cos(phi), st * np.sin(phi), np.cos(theta)])


def cartesian_to_spherical(v: np.ndarray) -> Tuple[float, float]:
    """笛卡尔 → 球坐标，返回 (θ, φ)。"""
    x, y, z = v
    r = np.linalg.norm(v)
    if r < 1e-15:
        return 0.0, 0.0
    theta = np.arccos(clip_to_unit(z / r))
    phi = np.arctan2(y, x)
    if phi < 0:
        phi += 2.0 * np.pi
    return theta, phi


def normalize_to_sphere(v: np.ndarray) -> np.ndarray:
    """将向量归一化到单位球面。"""
    r = np.linalg.norm(v)
    if r < 1e-15:
        return np.array([0.0, 0.0, 1.0])
    return v / r


# ---------------------------------------------------------------------------
# 二十面体初始网格
# ---------------------------------------------------------------------------
def create_icosahedron() -> Tuple[np.ndarray, np.ndarray]:
    """
    生成单位球内接二十面体的顶点和面。
    顶点数 = 12，面数 = 20。
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0  # 黄金比例
    verts = np.array([
        [-1.0,  phi, 0.0], [1.0,  phi, 0.0], [-1.0, -phi, 0.0], [1.0, -phi, 0.0],
        [0.0, -1.0,  phi], [0.0, 1.0,  phi], [0.0, -1.0, -phi], [0.0, 1.0, -phi],
        [ phi, 0.0, -1.0], [ phi, 0.0, 1.0], [-phi, 0.0, -1.0], [-phi, 0.0, 1.0],
    ], dtype=float)
    # 归一化
    verts = np.array([normalize_to_sphere(v) for v in verts])
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=int)
    return verts, faces


# ---------------------------------------------------------------------------
# 层级细分
# ---------------------------------------------------------------------------
def subdivide_mesh(verts: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    对每个三角形进行 4 等分细分：
        取三边中点，连接形成 4 个小三角形。
    使用边字典保证中点顶点的唯一性。
    """
    new_faces = []
    edge_midpoint: Dict[Tuple[int, int], int] = {}
    vert_list = verts.tolist()

    def get_midpoint(i: int, j: int) -> int:
        key = (min(i, j), max(i, j))
        if key in edge_midpoint:
            return edge_midpoint[key]
        mid = normalize_to_sphere(0.5 * (verts[i] + verts[j]))
        idx = len(vert_list)
        vert_list.append(mid)
        edge_midpoint[key] = idx
        return idx

    for tri in faces:
        a, b, c = tri
        ab = get_midpoint(a, b)
        bc = get_midpoint(b, c)
        ca = get_midpoint(c, a)
        new_faces.append([a, ab, ca])
        new_faces.append([b, bc, ab])
        new_faces.append([c, ca, bc])
        new_faces.append([ab, bc, ca])

    return np.array(vert_list), np.array(new_faces, dtype=int)


class SphericalMesh:
    """
    球面层级三角网格。
    支持任意细分次数 nsides，提供节点、单元、邻居关系。
    """

    def __init__(self, nsides: int = 2):
        """
        Parameters
        ----------
        nsides : int
            细分次数（≥0）。nsides=0 为原始二十面体。
        """
        self.nsides = ensure_positive(nsides, "nsides")
        self.vertices, self.faces = create_icosahedron()
        for _ in range(nsides):
            self.vertices, self.faces = subdivide_mesh(self.vertices, self.faces)
        self.n_vertices = len(self.vertices)
        self.n_faces = len(self.faces)
        self._compute_neighbors()

    def _compute_neighbors(self):
        """
        计算每个面的三个邻居面索引（共享边）。
        邻居索引按边 (v0-v1, v1-v2, v2-v0) 排序。
        -1 表示边界（球面封闭网格无边界，故不会出现）。
        """
        edge_to_faces: Dict[Tuple[int, int], List[int]] = {}
        for fi, tri in enumerate(self.faces):
            for ei in range(3):
                v1 = tri[ei]
                v2 = tri[(ei + 1) % 3]
                key = (min(v1, v2), max(v1, v2))
                edge_to_faces.setdefault(key, []).append(fi)

        self.neighbors = np.full((self.n_faces, 3), -1, dtype=int)
        for fi, tri in enumerate(self.faces):
            for ei in range(3):
                v1 = tri[ei]
                v2 = tri[(ei + 1) % 3]
                key = (min(v1, v2), max(v1, v2))
                faces_sharing = edge_to_faces[key]
                for fj in faces_sharing:
                    if fj != fi:
                        self.neighbors[fi, ei] = fj
                        break

    def face_area(self, face_idx: int) -> float:
        """
        计算球面三角形面积（单位球面）。
        使用 L'Huilier 定理：
            tan(E/4) = sqrt(tan(s/2) tan((s-a)/2) tan((s-b)/2) tan((s-c)/2))
        其中 E 为球面角盈，a,b,c 为大圆弧长（即边对应的中心角）。
        """
        tri = self.faces[face_idx]
        v0, v1, v2 = self.vertices[tri[0]], self.vertices[tri[1]], self.vertices[tri[2]]
        # 边长（中心角）
        a = np.arccos(clip_to_unit(np.dot(v1, v2)))
        b = np.arccos(clip_to_unit(np.dot(v2, v0)))
        c = np.arccos(clip_to_unit(np.dot(v0, v1)))
        s = 0.5 * (a + b + c)
        # 数值保护
        tan_s2 = np.tan(max(s / 2.0, 1e-12))
        tan_sa = np.tan(max((s - a) / 2.0, 1e-12))
        tan_sb = np.tan(max((s - b) / 2.0, 1e-12))
        tan_sc = np.tan(max((s - c) / 2.0, 1e-12))
        tan_E4 = np.sqrt(tan_s2 * tan_sa * tan_sb * tan_sc)
        E = 4.0 * np.arctan(tan_E4)
        return E

    def total_area(self) -> float:
        """所有面面积之和，理论上应等于 4π。"""
        return sum(self.face_area(i) for i in range(self.n_faces))

    def node_angles(self, node_idx: int) -> Tuple[float, float]:
        """返回节点对应的 (θ, φ)。"""
        return cartesian_to_spherical(self.vertices[node_idx])

    def write_mesh(self, prefix: str):
        """将网格输出为文本文件（节点 + 单元 + 邻居）。"""
        np.savetxt(f"{prefix}_nodes.txt", self.vertices, fmt="%.12e")
        np.savetxt(f"{prefix}_elements.txt", self.faces + 1, fmt="%d")  # 1-based
        np.savetxt(f"{prefix}_neighbors.txt", self.neighbors + 1, fmt="%d")
