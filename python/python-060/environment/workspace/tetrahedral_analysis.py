"""
三维体积分析与等值面相交模块

本模块实现平流层三维体积（四面体网格）与等值面的相交计算，
用于臭氧层厚度、体积通量等物理量的精确计算。

科学背景:
平流层在大气化学传输模型中通常被离散化为三维多面体网格。
计算臭氧等值面（如 1e12 molec/cm³）与网格单元的相交体积
是评估臭氧层总体积、表面积和通量的关键步骤。

科学公式:
1. 平面与四面体相交:
   平面方程: n · (x - x0) = 0
   四面体顶点符号: d_i = n · (v_i - x0)
   若所有 d_i 同号，则无相交
   否则交点为边与平面的交点

2. 四面体体积:
   V = |det(v1-v0, v2-v0, v3-v0)| / 6

3. 等值面面积估计:
   通过相交多边形面积求和

4. 臭氧层体积:
   V_O3 = ∫_{[O3] > threshold} dV

融入原项目: 1255_tetrahedron_slice_movie (平面与四面体相交)
"""

import numpy as np
from typing import Tuple, Optional, List


class TetrahedralAnalyzer:
    """
    四面体网格分析器
    """

    def __init__(self):
        pass

    def tetrahedron_volume(self, v0: np.ndarray, v1: np.ndarray,
                           v2: np.ndarray, v3: np.ndarray) -> float:
        """
        计算四面体体积
        V = |det(v1-v0, v2-v0, v3-v0)| / 6
        """
        edges = np.array([v1 - v0, v2 - v0, v3 - v0])
        vol = abs(np.linalg.det(edges)) / 6.0
        return vol

    def sign_opposite_strict(self, a: float, b: float) -> bool:
        """
        检查两个数是否严格异号
        """
        return a * b < 0.0

    def plane_tetrahedron_intersect(self, plane_point: np.ndarray,
                                     plane_normal: np.ndarray,
                                     vertices: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        计算平面与四面体的相交多边形

        Parameters
        ----------
        plane_point : ndarray (3,)
            平面上一点
        plane_normal : ndarray (3,)
            平面法向量
        vertices : ndarray (3, 4)
            四面体四个顶点

        Returns
        -------
        n_int : int
            交点数量 (0-4)
        points : ndarray (3, 4)
            交点坐标
        """
        plane_normal = plane_normal / (np.linalg.norm(plane_normal) + 1e-30)

        # 计算各顶点到平面的有符号距离
        d = np.dot(plane_normal, vertices - plane_point[:, np.newaxis])

        # 检查是否全部同号
        if np.all(d < 0.0) or np.all(d > 0.0):
            return 0, np.zeros((3, 4))

        n_int = 0
        p_int = np.zeros((3, 4))

        # 遍历所有边
        edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

        for j1, j2 in edges:
            if d[j1] == 0.0:
                # 顶点在平面上
                if n_int < 4:
                    p_int[:, n_int] = vertices[:, j1]
                    n_int += 1
            elif self.sign_opposite_strict(d[j1], d[j2]):
                # 边与平面相交
                if n_int < 4:
                    t = d[j1] / (d[j1] - d[j2])
                    t = np.clip(t, 0.0, 1.0)
                    p_int[:, n_int] = vertices[:, j1] + t * (vertices[:, j2] - vertices[:, j1])
                    n_int += 1

        # 去重
        unique_points = []
        for i in range(n_int):
            is_dup = False
            for up in unique_points:
                if np.linalg.norm(p_int[:, i] - up) < 1e-10:
                    is_dup = True
                    break
            if not is_dup:
                unique_points.append(p_int[:, i])

        n_int = len(unique_points)
        p_int = np.zeros((3, 4))
        for i, up in enumerate(unique_points):
            p_int[:, i] = up

        return n_int, p_int

    def polygon_area_3d(self, points: np.ndarray, n_points: int) -> float:
        """
        计算3D空间中多边形面积 (通过投影到主平面)
        """
        if n_points < 3:
            return 0.0

        # 计算法向量 ( Newell 方法 )
        normal = np.zeros(3)
        for i in range(n_points):
            j = (i + 1) % n_points
            normal[0] += (points[1, i] - points[1, j]) * (points[2, i] + points[2, j])
            normal[1] += (points[2, i] - points[2, j]) * (points[0, i] + points[0, j])
            normal[2] += (points[0, i] - points[0, j]) * (points[1, i] + points[1, j])

        normal_len = np.linalg.norm(normal)
        if normal_len < 1e-30:
            return 0.0

        # 投影到法向量分量最大的平面
        max_comp = np.argmax(np.abs(normal))

        if max_comp == 0:
            area = 0.5 * abs(
                np.sum(points[1, :n_points] * np.roll(points[2, :n_points], -1) -
                       np.roll(points[1, :n_points], -1) * points[2, :n_points])
            )
        elif max_comp == 1:
            area = 0.5 * abs(
                np.sum(points[2, :n_points] * np.roll(points[0, :n_points], -1) -
                       np.roll(points[2, :n_points], -1) * points[0, :n_points])
            )
        else:
            area = 0.5 * abs(
                np.sum(points[0, :n_points] * np.roll(points[1, :n_points], -1) -
                       np.roll(points[0, :n_points], -1) * points[1, :n_points])
            )

        return area

    def extract_isosurface_cells(self, tetrahedra: np.ndarray,
                                  values: np.ndarray,
                                  threshold: float) -> List[dict]:
        """
        提取与等值面相交的四面体单元

        Parameters
        ----------
        tetrahedra : ndarray (n_tet, 4)
            四面体顶点索引
        values : ndarray (n_vertices,)
            顶点上的标量值 (如臭氧浓度)
        threshold : float
            等值面值

        Returns
        -------
        intersected_cells : list of dict
            相交单元信息
        """
        intersected = []

        for tet in tetrahedra:
            v_vals = values[tet]

            # 检查是否跨越等值面
            if np.all(v_vals < threshold) or np.all(v_vals > threshold):
                continue

            # 标记高于/低于等值面的顶点
            above = v_vals > threshold
            below = v_vals < threshold
            on_surface = np.abs(v_vals - threshold) < 1e-10 * threshold

            n_int = 0
            p_int = []

            edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
            for j1, j2 in edges:
                if on_surface[j1]:
                    p_int.append(tet[j1])
                    n_int += 1
                elif above[j1] != above[j2]:
                    # 线性插值找到交点
                    t = (threshold - v_vals[j1]) / (v_vals[j2] - v_vals[j1] + 1e-30)
                    t = np.clip(t, 0.0, 1.0)
                    p_int.append((tet[j1], tet[j2], t))
                    n_int += 1

            intersected.append({
                'tet_indices': tet,
                'vertex_values': v_vals,
                'intersection_type': n_int,
                'crossing_vertices': p_int
            })

        return intersected

    def compute_isosurface_area_and_volume(self, vertices: np.ndarray,
                                            tetrahedra: np.ndarray,
                                            values: np.ndarray,
                                            threshold: float) -> dict:
        """
        计算等值面的总面积和包围体积

        Parameters
        ----------
        vertices : ndarray (n_vertices, 3)
            顶点坐标
        tetrahedra : ndarray (n_tet, 4)
            四面体索引
        values : ndarray (n_vertices,)
            顶点标量值
        threshold : float
            等值面值

        Returns
        -------
        metrics : dict
            面积和体积指标
        """
        total_area = 0.0
        total_volume = 0.0
        n_crossing = 0

        for tet in tetrahedra:
            v = vertices[tet]  # (4, 3)
            v_vals = values[tet]

            # 四面体总体积
            tet_vol = self.tetrahedron_volume(v[0], v[1], v[2], v[3])

            if np.all(v_vals >= threshold):
                total_volume += tet_vol
            elif not (np.all(v_vals < threshold) or np.all(v_vals > threshold)):
                n_crossing += 1
                # 计算相交面积
                # 构造等值面平面 (简化: 用重心处线性近似)
                centroid = np.mean(v, axis=0)
                grad_approx = np.zeros(3)
                for i in range(4):
                    grad_approx += (v_vals[i] - threshold) * (v[i] - centroid)
                grad_norm = np.linalg.norm(grad_approx)
                if grad_norm > 1e-30:
                    plane_normal = grad_approx / grad_norm
                else:
                    plane_normal = np.array([0.0, 0.0, 1.0])

                n_int, p_int = self.plane_tetrahedron_intersect(
                    centroid, plane_normal, v.T)

                if n_int >= 3:
                    area = self.polygon_area_3d(p_int, n_int)
                    total_area += area

                # 体积贡献 (高于阈值的部分)
                frac_above = np.sum(v_vals >= threshold) / 4.0
                total_volume += frac_above * tet_vol

        return {
            'isosurface_area': total_area,
            'enclosed_volume': total_volume,
            'n_crossing_cells': n_crossing,
            'threshold': threshold
        }


class StratosphericVolumeAnalysis:
    """
    平流层体积分析器
    用于臭氧层结构的三维分析
    """

    def __init__(self, z_min: float = 10000.0, z_max: float = 50000.0,
                 horizontal_extent: float = 2.0e6):
        """
        Parameters
        ----------
        z_min, z_max : float
            垂直范围 (m)
        horizontal_extent : float
            水平范围 (m)
        """
        self.z_min = z_min
        self.z_max = z_max
        self.horizontal_extent = horizontal_extent
        self.analyzer = TetrahedralAnalyzer()

    def build_tetrahedral_mesh(self, n_z: int = 20,
                                n_xy: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建简化的三维四面体网格
        """
        # 构建规则六面体网格再分解为四面体
        x = np.linspace(-self.horizontal_extent / 2,
                        self.horizontal_extent / 2, n_xy)
        y = np.linspace(-self.horizontal_extent / 2,
                        self.horizontal_extent / 2, n_xy)
        z = np.linspace(self.z_min, self.z_max, n_z)

        # 生成顶点
        vertices = []
        for zi in z:
            for yi in y:
                for xi in x:
                    vertices.append([xi, yi, zi])
        vertices = np.array(vertices)

        # 每个六面体分解为 5 或 6 个四面体 (简化: 使用标准分解)
        tetrahedra = []
        nx, ny, nz = n_xy, n_xy, n_z
        for k in range(nz - 1):
            for j in range(ny - 1):
                for i in range(nx - 1):
                    # 六面体 8 个顶点索引
                    v000 = i + j * nx + k * nx * ny
                    v100 = v000 + 1
                    v010 = v000 + nx
                    v110 = v010 + 1
                    v001 = v000 + nx * ny
                    v101 = v001 + 1
                    v011 = v001 + nx
                    v111 = v011 + 1

                    # 分解为 5 个四面体
                    tetrahedra.append([v000, v100, v010, v001])
                    tetrahedra.append([v101, v100, v001, v110])
                    tetrahedra.append([v101, v001, v110, v111])
                    tetrahedra.append([v010, v110, v001, v011])
                    tetrahedra.append([v110, v001, v011, v111])

        return vertices, np.array(tetrahedra)

    def ozone_concentration_field(self, vertices: np.ndarray) -> np.ndarray:
        """
        生成臭氧浓度场 (molec/cm³)
        简化模型: 在 25km 附近峰值
        """
        z = vertices[:, 2]
        z_km = z / 1000.0

        # 基础剖面
        o3 = 5e12 * np.exp(-((z_km - 25.0) / 8.0) ** 2)

        # 水平变化 (简化)
        r_horiz = np.sqrt(vertices[:, 0] ** 2 + vertices[:, 1] ** 2)
        o3 *= (1.0 - 0.2 * r_horiz / (self.horizontal_extent / 2.0 + 1e-30))

        return np.clip(o3, 1e8, 1e15)

    def analyze_ozone_layer(self, threshold: float = 1e12) -> dict:
        """
        分析臭氧层三维结构
        """
        vertices, tetrahedra = self.build_tetrahedral_mesh(n_z=15, n_xy=8)
        o3_values = self.ozone_concentration_field(vertices)

        metrics = self.analyzer.compute_isosurface_area_and_volume(
            vertices, tetrahedra, o3_values, threshold)

        # 额外统计
        above_mask = o3_values > threshold
        n_above = np.sum(above_mask)

        # 臭氧层平均高度
        if n_above > 0:
            mean_altitude = np.mean(vertices[above_mask, 2]) / 1000.0
            std_altitude = np.std(vertices[above_mask, 2]) / 1000.0
        else:
            mean_altitude = 0.0
            std_altitude = 0.0

        metrics['mean_altitude_km'] = mean_altitude
        metrics['std_altitude_km'] = std_altitude
        metrics['n_vertices'] = len(vertices)
        metrics['n_tetrahedra'] = len(tetrahedra)
        metrics['fraction_above_threshold'] = n_above / len(vertices)

        return metrics
