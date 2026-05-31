# -*- coding: utf-8 -*-
"""
geometry_parser.py
三维裂隙几何数据解析模块

基于种子项目 821_obj_display 的 Wavefront OBJ 文件解析算法，
用于读取复杂三维裂隙网络的几何信息（顶点、面片、法向量）。

在裂隙介质渗流研究中，真实裂隙几何常通过三维激光扫描或 CT 成像获得，
并以 OBJ 等标准格式存储。本模块提供鲁棒的解析功能，支持：
    - 顶点坐标提取
    - 三角/多边形面片索引提取
    - 法向量计算与验证
    - 裂隙表面积与体积估算

核心公式：
    三角形面积（Heron 公式）:
        A = sqrt(s(s-a)(s-b)(s-c)), s = (a+b+c)/2
    
    多边形法向量（Newell 方法）:
        n_x = Σ(y_i - y_{i+1})(z_i + z_{i+1})
        n_y = Σ(z_i - z_{i+1})(x_i + x_{i+1})
        n_z = Σ(x_i - x_{i+1})(y_i + y_{i+1})
"""

import numpy as np
from typing import List, Tuple, Optional


class OBJGeometryParser:
    """
    OBJ 格式三维裂隙几何解析器

    支持解析标准 Wavefront OBJ 格式的三维裂隙表面数据，
    并计算裂隙几何参数（面积、粗糙度、方向分布等）。
    """

    def __init__(self):
        self.vertices = np.zeros((0, 3))
        self.faces = []
        self.normals = np.zeros((0, 3))
        self.face_normals = []
        self.face_areas = []

    def parse_string(self, obj_text: str) -> dict:
        """
        从字符串解析 OBJ 数据

        Parameters
        ----------
        obj_text : str
            OBJ 格式文本内容

        Returns
        -------
        dict
            解析结果字典
        """
        vertices = []
        normals = []
        faces = []

        lines = obj_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('$'):
                continue

            parts = line.split()
            if len(parts) == 0:
                continue

            keyword = parts[0].upper()

            if keyword == 'V' and len(parts) >= 4:
                # 顶点: v x y z
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    vertices.append([x, y, z])
                except (ValueError, IndexError):
                    continue

            elif keyword == 'VN' and len(parts) >= 4:
                # 法向量: vn nx ny nz
                try:
                    nx, ny, nz = float(parts[1]), float(parts[2]), float(parts[3])
                    normals.append([nx, ny, nz])
                except (ValueError, IndexError):
                    continue

            elif keyword == 'F':
                # 面片: f v1 v2 v3 ...
                face_indices = []
                for part in parts[1:]:
                    # 处理 v/vt/vn 格式，只取顶点索引
                    idx_str = part.split('/')[0]
                    try:
                        idx = int(idx_str)
                        # OBJ 使用 1-based 索引
                        face_indices.append(idx - 1 if idx > 0 else idx)
                    except ValueError:
                        continue
                if len(face_indices) >= 3:
                    faces.append(face_indices)

        self.vertices = np.array(vertices)
        self.normals = np.array(normals)
        self.faces = faces

        # 计算面片法向量和面积
        self._compute_face_properties()

        return {
            "n_vertices": len(vertices),
            "n_faces": len(faces),
            "n_normals": len(normals),
            "vertices": self.vertices,
            "faces": self.faces,
            "face_areas": np.array(self.face_areas),
            "face_normals": np.array(self.face_normals) if self.face_normals else np.zeros((0, 3))
        }

    def _compute_face_properties(self):
        """计算每个面片的法向量和面积"""
        self.face_areas = []
        self.face_normals = []

        if len(self.vertices) == 0:
            return

        for face in self.faces:
            n_vert = len(face)
            if n_vert < 3:
                self.face_areas.append(0.0)
                self.face_normals.append([0.0, 0.0, 1.0])
                continue

            # 获取面片顶点
            verts = []
            for idx in face:
                if 0 <= idx < len(self.vertices):
                    verts.append(self.vertices[idx])
            if len(verts) < 3:
                self.face_areas.append(0.0)
                self.face_normals.append([0.0, 0.0, 1.0])
                continue

            # 使用 Newell 方法计算法向量
            normal = self._newell_normal(verts)
            area = self._polygon_area(verts)

            self.face_normals.append(normal)
            self.face_areas.append(area)

    @staticmethod
    def _newell_normal(vertices: List[List[float]]) -> List[float]:
        """
        Newell 方法计算多边形法向量

        公式：
            n_x = Σ(y_i - y_{i+1})(z_i + z_{i+1})
            n_y = Σ(z_i - z_{i+1})(x_i + x_{i+1})
            n_z = Σ(x_i - x_{i+1})(y_i + y_{i+1})
        """
        n = [0.0, 0.0, 0.0]
        m = len(vertices)
        for i in range(m):
            j = (i + 1) % m
            n[0] += (vertices[i][1] - vertices[j][1]) * (vertices[i][2] + vertices[j][2])
            n[1] += (vertices[i][2] - vertices[j][2]) * (vertices[i][0] + vertices[j][0])
            n[2] += (vertices[i][0] - vertices[j][0]) * (vertices[i][1] + vertices[j][1])

        norm = np.sqrt(n[0]**2 + n[1]**2 + n[2]**2)
        if norm > 1e-12:
            n = [n[0]/norm, n[1]/norm, n[2]/norm]
        else:
            n = [0.0, 0.0, 1.0]
        return n

    @staticmethod
    def _polygon_area(vertices: List[List[float]]) -> float:
        """
        计算多边形面积（3D 空间中）

        将多边形三角剖分后求和。
        """
        if len(vertices) < 3:
            return 0.0

        total_area = 0.0
        v0 = np.array(vertices[0])
        for i in range(1, len(vertices) - 1):
            v1 = np.array(vertices[i])
            v2 = np.array(vertices[i + 1])
            # 叉积求三角形面积
            cross = np.cross(v1 - v0, v2 - v0)
            total_area += 0.5 * np.linalg.norm(cross)
        return total_area

    def total_surface_area(self) -> float:
        """计算裂隙总表面积"""
        return float(np.sum(self.face_areas))

    def mean_aperture_estimate(self, volume: float) -> float:
        """
        从已知体积估算平均裂隙开度

        公式：
            b_avg = V / A_total

        Parameters
        ----------
        volume : float
            裂隙体积 [m³]

        Returns
        -------
        float
            估算平均开度 [m]
        """
        area = self.total_surface_area()
        if area < 1e-12:
            return 0.0
        return volume / area

    def orientation_distribution(self, n_bins: int = 18) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算裂隙面片方向分布（走向-倾角直方图）

        Parameters
        ----------
        n_bins : int
            角度分箱数

        Returns
        -------
        tuple
            (bin_centers, histogram)
        """
        if len(self.face_normals) == 0:
            return np.zeros(n_bins), np.zeros(n_bins)

        normals = np.array(self.face_normals)
        areas = np.array(self.face_areas)

        # 计算倾角（与水平面夹角）
        # 法向量 z 分量: cos(dip)
        dips = np.arccos(np.clip(np.abs(normals[:, 2]), -1.0, 1.0))
        dips_deg = np.degrees(dips)

        bins = np.linspace(0, 90, n_bins + 1)
        hist, _ = np.histogram(dips_deg, bins=bins, weights=areas)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        return bin_centers, hist

    def roughness_coefficient(self) -> float:
        """
        计算裂隙表面粗糙度系数

        使用 JRC (Joint Roughness Coefficient) 相关统计量：
            σ_z = std(z_i)

        Returns
        -------
        float
            表面高程标准差 [m]
        """
        if len(self.vertices) == 0:
            return 0.0
        z_coords = self.vertices[:, 2]
        return float(np.std(z_coords))

    def generate_sample_fracture_obj(self, size: float = 1.0, amplitude: float = 0.01,
                                     n_segments: int = 20) -> str:
        """
        生成一个示例正弦波形裂隙表面的 OBJ 文本

        用于测试和验证解析功能。

        Parameters
        ----------
        size : float
            裂隙平面尺寸
        amplitude : float
            表面起伏振幅
        n_segments : int
            网格分段数

        Returns
        -------
        str
            OBJ 格式文本
        """
        lines = ["# Sample fractured surface OBJ"]
        dx = size / n_segments
        dy = size / n_segments

        # 生成顶点
        for i in range(n_segments + 1):
            for j in range(n_segments + 1):
                x = j * dx
                y = i * dy
                z = amplitude * np.sin(2 * np.pi * x / size) * np.cos(2 * np.pi * y / size)
                lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")

        # 生成面片（三角化）
        for i in range(n_segments):
            for j in range(n_segments):
                v0 = i * (n_segments + 1) + j + 1
                v1 = v0 + 1
                v2 = (i + 1) * (n_segments + 1) + j + 1
                v3 = v2 + 1
                lines.append(f"f {v0} {v1} {v3}")
                lines.append(f"f {v0} {v3} {v2}")

        return '\n'.join(lines)
