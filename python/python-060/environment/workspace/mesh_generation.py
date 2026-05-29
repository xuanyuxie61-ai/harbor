"""
大气传输网格生成与质量评估模块

本模块实现平流层数值模拟所需的计算网格生成，包括：
- 六边形球面网格生成 (用于水平方向离散化)
- CVT (Centroidal Voronoi Tessellation) 网格优化
- 网格质量评估 (alpha measure, 角度质量)
- 不规则区域三角剖分

科学背景:
在大气科学中，六边形网格因其各向同性特性而被广泛用于球面离散化
(GCRM, MPAS 等气候模型)。CVT 优化可进一步改善网格均匀性。

科学公式:
1. 正六边形面积:
   A = (3√3 / 2) * a²
   其中 a 为边长

2. 球面六边形网格生成:
   使用二十面体递归细分 + 球面投影
   每个顶点坐标: r = R * (x, y, z) / |(x, y, z)|

3. CVT 能量泛函:
   E = Σ_i ∫_{V_i} ρ(x) |x - z_i|² dx
   其中 V_i 为 Voronoi 单元, z_i 为生成点

4. Lloyd 迭代 (CVT 优化):
   z_i^{new} = ∫_{V_i} x ρ(x) dx / ∫_{V_i} ρ(x) dx

5. Alpha Measure (网格质量):
   Q = min(α_triangle) / (π/3)
   其中 α 为三角形最小内角

融入原项目:
- 528_hexagon_lyness_rule (六边形积分规则)
- 264_cvtp (周期CVT迭代)
- 469_geompack (三角剖分与质量评估)
"""

import numpy as np
from typing import Tuple, List, Optional


class HexagonalGridGenerator:
    """
    六边形网格生成器
    用于大气水平离散化
    """

    def __init__(self, radius: float = 6371.0e3):
        """
        Parameters
        ----------
        radius : float
            地球半径 (m)
        """
        self.radius = radius

    def hexagon_area(self, side_length: float) -> float:
        """
        正六边形面积
        A = (3√3 / 2) * a²
        """
        if side_length <= 0:
            raise ValueError("边长必须为正")
        return 3.0 * np.sqrt(3.0) / 2.0 * side_length ** 2

    def generate_planar_hex_grid(self, center: Tuple[float, float] = (0.0, 0.0),
                                  side_length: float = 1.0e5,
                                  n_rings: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成平面六边形网格点 (用于局部大气模拟)

        Parameters
        ----------
        center : tuple
            中心点 (x, y)
        side_length : float
            六边形边长 (m)
        n_rings : int
            环数

        Returns
        -------
        x, y : ndarray
            网格点坐标
        """
        if side_length <= 0 or n_rings < 0:
            raise ValueError("参数无效")

        points = []
        cx, cy = center

        # 中心点
        points.append((cx, cy))

        # 六边形环
        dx = side_length * np.sqrt(3.0)
        dy = side_length * 1.5

        for ring in range(1, n_rings + 1):
            for i in range(6 * ring):
                angle = np.pi / 3.0 * (i / ring)
                r = side_length * ring * np.sqrt(3.0)
                px = cx + r * np.cos(angle)
                py = cy + r * np.sin(angle)
                points.append((px, py))

        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        return x, y

    def hex_lyness_rule(self, rule_id: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Lyness 六边形积分规则
        返回六边形区域 [0,1]×[0,1] 上的积分点与权重
        (简化版本，基于正方形规则映射)

        Parameters
        ----------
        rule_id : int
            规则编号 (1-7)

        Returns
        -------
        x, y : ndarray
            积分点坐标 (局部坐标)
        w : ndarray
            权重
        """
        if rule_id == 1:
            # 1点规则 (重心)
            x = np.array([0.5])
            y = np.array([0.5])
            w = np.array([1.0])
        elif rule_id == 2:
            # 4点 Gauss 规则
            p = (3.0 - np.sqrt(3.0)) / 6.0
            q = (3.0 + np.sqrt(3.0)) / 6.0
            x = np.array([p, p, q, q])
            y = np.array([p, q, p, q])
            w = np.array([0.25, 0.25, 0.25, 0.25])
        elif rule_id == 3:
            # 9点规则
            p = np.sqrt(3.0 / 5.0)
            x = np.array([0.5, 0.5 - p/2, 0.5 + p/2, 0.5 - p/2, 0.5 + p/2,
                          0.5 - p/2, 0.5 + p/2, 0.5, 0.5])
            y = np.array([0.5, 0.5 - p/2, 0.5 - p/2, 0.5 + p/2, 0.5 + p/2,
                          0.5, 0.5, 0.5 - p/2, 0.5 + p/2])
            w0 = 64.0 / 81.0
            w1 = 40.0 / 81.0
            w2 = 25.0 / 81.0
            w = np.array([w0, w2, w2, w2, w2, w1, w1, w1, w1]) / 4.0
        else:
            # 默认 1点规则
            x = np.array([0.5])
            y = np.array([0.5])
            w = np.array([1.0])

        return x, y, w

    def integrate_over_hex(self, f: callable, center: Tuple[float, float],
                           side_length: float, rule_id: int = 3) -> float:
        """
        在六边形区域上数值积分
        ∫_H f(x,y) dx dy
        """
        xi, eta, w = self.hex_lyness_rule(rule_id)
        area = self.hexagon_area(side_length)

        # 将局部坐标映射到六边形
        # 简化为正方形映射
        cx, cy = center
        dx = side_length * np.sqrt(3.0)
        x = cx + (xi - 0.5) * dx
        y = cy + (eta - 0.5) * dx

        integral = 0.0
        for i in range(len(w)):
            integral += w[i] * f(x[i], y[i])

        return integral * area


class CVTMeshOptimizer:
    """
    CVT (Centroidal Voronoi Tessellation) 网格优化器
    基于 Lloyd 迭代优化网格点分布
    """

    def __init__(self, dim: int = 2, n_generators: int = 100):
        """
        Parameters
        ----------
        dim : int
            空间维度
        n_generators : int
            生成点数量
        """
        self.dim = dim
        self.n_generators = n_generators

    def find_closest_generator(self, x: np.ndarray,
                               generators: np.ndarray) -> int:
        """
        找到距离点 x 最近的生成点索引
        """
        dists = np.sum((generators - x) ** 2, axis=1)
        return int(np.argmin(dists))

    def cvt_iteration(self, generators: np.ndarray,
                      a: np.ndarray, b: np.ndarray,
                      sample_num: int = 5000,
                      modular: bool = False) -> Tuple[np.ndarray, float]:
        """
        执行一次 CVT Lloyd 迭代

        Parameters
        ----------
        generators : ndarray
            当前生成点 (n_gen, dim)
        a, b : ndarray
            区域下界和上界
        sample_num : int
            采样点数
        modular : bool
            是否使用周期边界

        Returns
        -------
        generators_new : ndarray
            更新后的生成点
        change : float
            生成点最大移动距离
        """
        n_gen = generators.shape[0]
        dim = generators.shape[1]

        generator_new = np.zeros_like(generators)
        counts = np.zeros(n_gen)

        # 蒙特卡洛采样并分配
        np.random.seed(42)
        for _ in range(sample_num):
            x = a + np.random.rand(dim) * (b - a)

            if modular:
                # 周期边界: 考虑镜像点
                nearest = self._find_closest_modular(x, generators, a, b)
            else:
                nearest = self.find_closest_generator(x, generators)

            generator_new[nearest] += x
            counts[nearest] += 1.0

        # 平均得到新质心
        for j in range(n_gen):
            if counts[j] > 0:
                generator_new[j] /= counts[j]
            else:
                generator_new[j] = generators[j].copy()

        # 计算变化量
        change = np.max(np.sqrt(np.sum((generator_new - generators) ** 2, axis=1)))

        # 周期修正
        if modular:
            for j in range(n_gen):
                for i in range(dim):
                    if generator_new[j, i] < a[i]:
                        generator_new[j, i] += b[i] - a[i]
                    elif generator_new[j, i] > b[i]:
                        generator_new[j, i] -= b[i] - a[i]

        return generator_new, change

    def _find_closest_modular(self, x: np.ndarray, generators: np.ndarray,
                              a: np.ndarray, b: np.ndarray) -> int:
        """
        周期边界下找到最近生成点
        """
        dim = len(x)
        # 考虑 3^dim 个镜像
        min_dist = float('inf')
        nearest = 0

        for offset in np.ndindex(*([3] * dim)):
            shift = np.array(offset) - 1.0
            x_shifted = x.copy()
            for i in range(dim):
                period = b[i] - a[i]
                x_shifted[i] += shift[i] * period

            for j, gen in enumerate(generators):
                dist = np.sum((x_shifted - gen) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    nearest = j

        return nearest

    def optimize(self, a: np.ndarray, b: np.ndarray,
                 n_iter: int = 50, tol: float = 1e-4,
                 sample_num: int = 5000) -> np.ndarray:
        """
        执行 CVT 优化

        Parameters
        ----------
        a, b : ndarray
            区域边界
        n_iter : int
            最大迭代次数
        tol : float
            收敛容差
        sample_num : int
            每步采样数

        Returns
        -------
        generators : ndarray
            优化后的生成点
        """
        dim = len(a)
        generators = np.random.rand(self.n_generators, dim)
        for i in range(dim):
            generators[:, i] = a[i] + generators[:, i] * (b[i] - a[i])

        for it in range(n_iter):
            generators_new, change = self.cvt_iteration(
                generators, a, b, sample_num, modular=False)

            if change < tol:
                break
            generators = generators_new

        return generators


class MeshQualityEvaluator:
    """
    网格质量评估器
    基于 geompack 的 alpha measure 思想
    """

    def __init__(self):
        pass

    def triangle_area_2d(self, a: np.ndarray, b: np.ndarray,
                         c: np.ndarray) -> float:
        """
        三角形面积 (2D)
        A = 0.5 * |a_x(b_y - c_y) + b_x(c_y - a_y) + c_x(a_y - b_y)|
        """
        area = 0.5 * abs(
            a[0] * (b[1] - c[1]) +
            b[0] * (c[1] - a[1]) +
            c[0] * (a[1] - b[1])
        )
        return area

    def triangle_angles(self, a: np.ndarray, b: np.ndarray,
                        c: np.ndarray) -> Tuple[float, float, float]:
        """
        计算三角形三个内角 (弧度)
        """
        ab = np.linalg.norm(b - a)
        bc = np.linalg.norm(c - b)
        ca = np.linalg.norm(a - c)

        # 处理退化情况
        if ab < 1e-14 and bc < 1e-14 and ca < 1e-14:
            return 2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0

        def safe_acos(val):
            return np.arccos(np.clip(val, -1.0, 1.0))

        if ca < 1e-14 or ab < 1e-14:
            a_angle = np.pi
        else:
            a_angle = safe_acos((ca ** 2 + ab ** 2 - bc ** 2) / (2.0 * ca * ab))

        if ab < 1e-14 or bc < 1e-14:
            b_angle = np.pi
        else:
            b_angle = safe_acos((ab ** 2 + bc ** 2 - ca ** 2) / (2.0 * ab * bc))

        if bc < 1e-14 or ca < 1e-14:
            c_angle = np.pi
        else:
            c_angle = safe_acos((bc ** 2 + ca ** 2 - ab ** 2) / (2.0 * bc * ca))

        return a_angle, b_angle, c_angle

    def alpha_measure(self, points: np.ndarray,
                      triangles: List[Tuple[int, int, int]]) -> dict:
        """
        计算三角剖分的 alpha measure 质量指标

        Parameters
        ----------
        points : ndarray
            点坐标 (n, 2)
        triangles : list of tuple
            三角形索引列表

        Returns
        -------
        metrics : dict
            包含 alpha_min, alpha_ave, alpha_area 的字典
        """
        if len(triangles) == 0:
            return {'alpha_min': 0.0, 'alpha_ave': 0.0, 'alpha_area': 0.0}

        alpha_min = float('inf')
        alpha_ave = 0.0
        alpha_area = 0.0
        total_area = 0.0

        for tri in triangles:
            a = points[tri[0]]
            b = points[tri[1]]
            c = points[tri[2]]

            area = self.triangle_area_2d(a, b, c)
            angles = self.triangle_angles(a, b, c)
            min_angle = min(angles)

            alpha_min = min(alpha_min, min_angle)
            alpha_ave += min_angle
            alpha_area += area * min_angle
            total_area += area

        n_tri = len(triangles)
        alpha_ave /= n_tri
        if total_area > 0:
            alpha_area /= total_area

        # 归一化到 [0, 1]
        norm = 3.0 / np.pi
        return {
            'alpha_min': alpha_min * norm,
            'alpha_ave': alpha_ave * norm,
            'alpha_area': alpha_area * norm,
            'n_triangles': n_tri,
            'total_area': total_area
        }

    def evaluate_grid_quality(self, x: np.ndarray, y: np.ndarray) -> dict:
        """
        评估点集分布质量
        """
        n = len(x)
        if n < 3:
            return {'uniformity': 0.0, 'coverage': 0.0}

        # 计算最近邻距离
        min_dists = []
        for i in range(n):
            dists = np.sqrt((x - x[i]) ** 2 + (y - y[i]) ** 2)
            dists[i] = float('inf')
            min_dists.append(np.min(dists))

        min_dists = np.array(min_dists)
        mean_dist = np.mean(min_dists)
        std_dist = np.std(min_dists)

        uniformity = 1.0 / (1.0 + std_dist / (mean_dist + 1e-30))

        # 覆盖范围
        x_range = np.max(x) - np.min(x)
        y_range = np.max(y) - np.min(y)
        coverage = (n * mean_dist ** 2) / (x_range * y_range + 1e-30)

        return {
            'uniformity': np.clip(uniformity, 0.0, 1.0),
            'coverage': np.clip(coverage, 0.0, 1.0),
            'mean_neighbor_dist': mean_dist,
            'std_neighbor_dist': std_dist
        }


def generate_atmospheric_mesh(n_horizontal: int = 50,
                               n_vertical: int = 40,
                               z_min: float = 10000.0,
                               z_max: float = 50000.0) -> dict:
    """
    生成三维大气计算网格

    Returns
    -------
    mesh : dict
        包含网格点、连接关系、质量指标的字典
    """
    hex_gen = HexagonalGridGenerator()
    cvt_opt = CVTMeshOptimizer(dim=2, n_generators=n_horizontal)
    quality = MeshQualityEvaluator()

    # 生成水平六边形网格
    n_rings = int(np.sqrt(n_horizontal / 3.0))
    x_hex, y_hex = hex_gen.generate_planar_hex_grid(
        center=(0.0, 0.0), side_length=2.0e5, n_rings=n_rings)

    # CVT 优化
    a = np.array([np.min(x_hex), np.min(y_hex)])
    b = np.array([np.max(x_hex), np.max(y_hex)])

    generators = np.column_stack([x_hex, y_hex])
    for _ in range(10):
        generators_new, change = cvt_opt.cvt_iteration(
            generators, a, b, sample_num=2000)
        generators = generators_new
        if change < 1e-3:
            break

    # 垂直网格 (非均匀，边界层加密)
    z_km = np.linspace(z_min / 1000.0, z_max / 1000.0, n_vertical)
    # 在 20-30 km 处加密
    z_refined = z_km + 2.0 * np.exp(-((z_km - 25.0) / 3.0) ** 2)
    z = np.clip(z_refined * 1000.0, z_min, z_max)

    # 质量评估
    grid_quality = quality.evaluate_grid_quality(generators[:, 0], generators[:, 1])

    return {
        'xy_horizontal': generators,
        'z_vertical': z,
        'n_horizontal': len(generators),
        'n_vertical': n_vertical,
        'horizontal_quality': grid_quality,
        'area_per_cell': hex_gen.hexagon_area(2.0e5) if len(generators) > 0 else 0.0
    }
