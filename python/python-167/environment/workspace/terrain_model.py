"""
terrain_model.py
地形建模与足部接触分析模块。
融入种子项目：
  - 1310_triangle_io（三角形网格数据读写）
  - 1168_stla_to_tri_surface_fast（STL 文件转三角表面）
  - 956_quadrilateral_surface_display（四边形表面双线性插值，去除可视化）

科学背景：多足机器人在非结构化地形上行走时，足部-地面接触力学
是步态稳定性的核心。地形用三角网格 T = {V, F} 表示，其中 V 为顶点集，
F 为三角形面片集。每个三角形面片上的高度场通过重心坐标插值得到。
"""

import numpy as np
from typing import Tuple, List, Optional


class TriangulatedTerrain:
    """
    三角网格地形模型。
    支持从节点/元素文件读写（源自 triangle_io）以及 STL 快速解析（源自 stla_to_tri_surface_fast）。
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        """
        vertices: (N, 3) 顶点坐标数组
        faces:    (M, 3) 三角形顶点索引数组（0-based）
        """
        self.vertices = np.asarray(vertices, dtype=float)
        self.faces = np.asarray(faces, dtype=int)
        self._compute_face_normals()
        self._compute_aabb()

    def _compute_face_normals(self):
        """
        计算每个三角形面片的单位法向量。

        数学公式：
        对三角形 (v0, v1, v2)，边向量
            e1 = v1 - v0,  e2 = v2 - v0
        法向量 n = (e1 × e2) / ||e1 × e2||
        叉积 e1 × e2 的模长等于三角形面积的 2 倍。
        """
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        e1 = v1 - v0
        e2 = v2 - v0
        cross = np.cross(e1, e2)
        norms = np.linalg.norm(cross, axis=1)
        norms = np.where(norms < 1e-14, 1.0, norms)
        self.face_normals = cross / norms[:, np.newaxis]
        self.face_areas = 0.5 * norms

    def _compute_aabb(self):
        """计算轴对齐包围盒（AABB），用于快速碰撞剔除。"""
        self.aabb_min = np.min(self.vertices, axis=0)
        self.aabb_max = np.max(self.vertices, axis=0)

    @classmethod
    def from_node_element(cls, node_filename: str, element_filename: str):
        """
        源自 triangle_io：从节点文件与元素文件读取三角网格。
        文件格式兼容 Triangle 网格生成器输出。
        """
        # 读取节点文件：第一行为节点数，后续每行 node_id x y z [attributes]
        with open(node_filename, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        node_count = int(lines[0].split()[0])
        vertices = np.zeros((node_count, 3))
        for i in range(1, node_count + 1):
            parts = lines[i].split()
            vertices[i - 1] = [float(parts[1]), float(parts[2]), float(parts[3])]

        # 读取元素文件：第一行为元素数，后续每行 elem_id n1 n2 n3 [attributes]
        with open(element_filename, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        elem_count = int(lines[0].split()[0])
        faces = np.zeros((elem_count, 3), dtype=int)
        for i in range(1, elem_count + 1):
            parts = lines[i].split()
            faces[i - 1] = [int(parts[1]) - 1, int(parts[2]) - 1, int(parts[3]) - 1]
        return cls(vertices, faces)

    @classmethod
    def from_stl_ascii(cls, stl_filename: str):
        """
        源自 stla_to_tri_surface_fast：快速读取 ASCII STL 文件。
        对大型文件采用逐行扫描策略，统计 solid/facet/vertex 关键字。
        """
        with open(stl_filename, 'r') as f:
            lines = f.readlines()

        vertices = []
        faces = []
        vertex_buffer = []
        for line in lines:
            lower = line.lower().strip()
            if lower.startswith('vertex'):
                parts = lower.split()
                if len(parts) >= 4:
                    vertex_buffer.append([float(parts[1]), float(parts[2]), float(parts[3])])
                if len(vertex_buffer) == 3:
                    base = len(vertices)
                    faces.append([base, base + 1, base + 2])
                    vertices.extend(vertex_buffer)
                    vertex_buffer = []

        if len(vertices) == 0:
            # 退化情况：返回一个默认平面
            vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
            faces = np.array([[0, 1, 2], [1, 3, 2]])
            return cls(vertices, faces)
        return cls(np.array(vertices), np.array(faces, dtype=int))

    def barycentric_interpolation(self, face_idx: int, p: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        在指定三角形面片内用重心坐标计算高度 z 与法向量。

        数学公式：
        对三角形顶点 v0, v1, v2，点 p 的重心坐标 (λ0, λ1, λ2) 满足
            p = λ0·v0 + λ1·v1 + λ2·v2,   λ0 + λ1 + λ2 = 1
        解得
            λ1 = ((p - v0) · (v1 - v0)) / ||v1 - v0||^2
            λ2 = ((p - v0) · (v2 - v0)) / ||v2 - v0||^2
            λ0 = 1 - λ1 - λ2
        高度 z 通过线性插值得到。
        """
        v0, v1, v2 = self.vertices[self.faces[face_idx]]
        e1 = v1 - v0
        e2 = v2 - v0
        denom = np.dot(e1, e1) * np.dot(e2, e2) - np.dot(e1, e2) ** 2
        if abs(denom) < 1e-14:
            return v0[2], self.face_normals[face_idx]
        w = p - v0
        lambda1 = (np.dot(w, e1) * np.dot(e2, e2) - np.dot(w, e2) * np.dot(e1, e2)) / denom
        lambda2 = (np.dot(w, e2) * np.dot(e1, e1) - np.dot(w, e1) * np.dot(e1, e2)) / denom
        lambda0 = 1.0 - lambda1 - lambda2
        z = lambda0 * v0[2] + lambda1 * v1[2] + lambda2 * v2[2]
        # 法向量用重心坐标加权（面片法向量在此假设为常数）
        normal = self.face_normals[face_idx]
        return z, normal

    def query_height(self, x: float, y: float) -> Tuple[float, np.ndarray, int]:
        """
        查询给定 (x, y) 处的地形高度 z、法向量与所在面片索引。
        采用 AABB 预筛选 + 重心坐标包含测试。
        """
        p = np.array([x, y, 0.0])
        # 快速 AABB 剔除
        if x < self.aabb_min[0] or x > self.aabb_max[0] or y < self.aabb_min[1] or y > self.aabb_max[1]:
            return 0.0, np.array([0.0, 0.0, 1.0]), -1

        best_z = -np.inf
        best_normal = np.array([0.0, 0.0, 1.0])
        best_face = -1
        for i in range(self.faces.shape[0]):
            v0, v1, v2 = self.vertices[self.faces[i]]
            # 2D 重心坐标测试（投影到 xy 平面）
            denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
            if abs(denom) < 1e-14:
                continue
            lambda0 = ((v1[1] - v2[1]) * (x - v2[0]) + (v2[0] - v1[0]) * (y - v2[1])) / denom
            lambda1 = ((v2[1] - v0[1]) * (x - v2[0]) + (v0[0] - v2[0]) * (y - v2[1])) / denom
            lambda2 = 1.0 - lambda0 - lambda1
            if lambda0 >= -1e-9 and lambda1 >= -1e-9 and lambda2 >= -1e-9:
                z = lambda0 * v0[2] + lambda1 * v1[2] + lambda2 * v2[2]
                if z > best_z:
                    best_z = z
                    best_normal = self.face_normals[i]
                    best_face = i
        if best_face == -1:
            return 0.0, np.array([0.0, 0.0, 1.0]), -1
        return best_z, best_normal, best_face


class QuadrilateralTerrainPatch:
    """
    源自 quadrilateral_surface_display 的四边形表面插值。
    去除可视化，保留双线性插值数学核心。
    """

    def __init__(self, nodes: np.ndarray):
        """
        nodes: (4, 3) 四边形顶点，按逆时针顺序
        """
        self.nodes = np.asarray(nodes, dtype=float)
        if self.nodes.shape != (4, 3):
            raise ValueError("QuadrilateralTerrainPatch requires exactly 4 nodes with 3 coordinates each")

    def bilinear_interpolate(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        参数空间 [0,1]^2 到物理空间的双线性映射。

        数学公式：
        p(ξ, η) = (1-ξ)(1-η)·v0 + ξ(1-η)·v1 + ξη·v2 + (1-ξ)η·v3

        Jacobian：
        J = [ ∂p/∂ξ , ∂p/∂η ]
        ∂p/∂ξ = -(1-η)·v0 + (1-η)·v1 + η·v2 - η·v3
        ∂p/∂η = -(1-ξ)·v0 - ξ·v1 + ξ·v2 + (1-ξ)·v3
        """
        if not (0.0 <= xi <= 1.0 and 0.0 <= eta <= 1.0):
            xi = np.clip(xi, 0.0, 1.0)
            eta = np.clip(eta, 0.0, 1.0)
        v0, v1, v2, v3 = self.nodes
        p = ((1 - xi) * (1 - eta) * v0
             + xi * (1 - eta) * v1
             + xi * eta * v2
             + (1 - xi) * eta * v3)
        dpdxi = -(1 - eta) * v0 + (1 - eta) * v1 + eta * v2 - eta * v3
        dpdeta = -(1 - xi) * v0 - xi * v1 + xi * v2 + (1 - xi) * v3
        J = np.column_stack((dpdxi, dpdeta))
        return p, J

    def normal_at(self, xi: float, eta: float) -> np.ndarray:
        """
        计算四边形参数点处的单位法向量 n = (∂p/∂ξ × ∂p/∂η) / ||...||。
        """
        _, J = self.bilinear_interpolate(xi, eta)
        cross = np.cross(J[:, 0], J[:, 1])
        norm = np.linalg.norm(cross)
        if norm < 1e-14:
            return np.array([0.0, 0.0, 1.0])
        return cross / norm


def generate_sample_terrain() -> TriangulatedTerrain:
    """
    生成一个示例非平坦地形：带有凹凸的平面，用于零参数运行演示。
    """
    x = np.linspace(-2.0, 2.0, 21)
    y = np.linspace(-2.0, 2.0, 21)
    X, Y = np.meshgrid(x, y)
    Z = 0.3 * np.sin(2.0 * X) * np.cos(1.5 * Y) + 0.1 * np.cos(3.0 * (X + Y))
    vertices = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    faces = []
    nx, ny = X.shape
    for i in range(nx - 1):
        for j in range(ny - 1):
            v0 = i * ny + j
            v1 = v0 + 1
            v2 = (i + 1) * ny + j
            v3 = v2 + 1
            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])
    return TriangulatedTerrain(vertices, np.array(faces, dtype=int))
