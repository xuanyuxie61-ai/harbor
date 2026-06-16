"""
calorimeter_grid.py — 量能器几何网格生成器
===========================================

融合种子项目:
  - 757_mesh2d        : 2D 三角网格生成 (Delaunay 细化)
  - 680_line_grid     : 1D 均匀/非均匀网格生成
  - 176_circle_arc_grid : 曲线弧网格生成

本模块构建量能器的多维计算网格, 支持:
  1. 纵向 (深度方向) 一维非均匀网格 — 捕获 shower 指数衰减
  2. 横向 2D Delaunay 三角网格 — 描述 shower 横向扩展
  3. 层状结构的曲率修正网格

关键公式:
  纵向网格 (几何级数):
    z_i = z_0 * r^i,  i = 0, 1, ..., N_z
    r = (z_max/z_0)^(1/N_z)  — 公比

  横向网格 (自适应密度):
    h(r) = h_0 * (1 + alpha * exp(-r^2 / (2*sigma^2)))
    — 在 shower 核心区域加密

  网格质量度量 (Laplacian 光滑度):
    Q = (1/N) * sum_{cells} (A_cell / sum_edge_lengths^2)
    理想值: Q_tri = 1/(12*sqrt(3)) ≈ 0.0481
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import random


@dataclass
class GridConfig:
    """网格配置参数"""
    # 纵向网格参数
    n_z: int = 64                   # 纵向网格点数
    z_max_X0: float = 30.0          # 最大深度 [辐射长度]
    z_min_X0: float = 0.01          # 最小深度 (避免奇点)
    z_stretch: float = 1.08         # 几何拉伸比

    # 横向网格参数
    n_r: int = 32                   # 横向径向网格点数
    r_max_cm: float = 10.0          # 最大横向距离 [cm]
    r_core_cm: float = 2.0          # 核心区域半径 [cm]
    refinement_factor: float = 3.0  # 核心区域加密因子

    # 层状结构
    n_layers: int = 20              # 量能器层数
    layer_gap_fraction: float = 0.05 # 间隙占每层厚度的比例

    # 网格质量阈值
    min_quality: float = 0.01       # 最小允许网格质量


@dataclass
class Grid1D:
    """一维网格"""
    nodes: List[float]              # 节点位置
    spacings: List[float]           # 间距
    weights: List[float]            # 积分权重 (梯形法则)
    n_points: int = 0

    def __post_init__(self):
        self.n_points = len(self.nodes)

    @property
    def total_length(self) -> float:
        return self.nodes[-1] - self.nodes[0] if self.n_points > 1 else 0.0

    def index_of(self, z: float) -> Tuple[int, float]:
        """查找 z 所在的区间索引和局部坐标"""
        if z <= self.nodes[0]:
            return 0, 0.0
        if z >= self.nodes[-1]:
            return self.n_points - 2, 1.0
        for i in range(self.n_points - 1):
            if self.nodes[i] <= z <= self.nodes[i + 1]:
                h = self.nodes[i + 1] - self.nodes[i]
                if h < 1e-30:
                    return i, 0.0
                xi = (z - self.nodes[i]) / h
                return i, xi
        return self.n_points - 2, 1.0


@dataclass
class LayeredStructure:
    """层状量能器结构"""
    layer_boundaries: List[float]    # 层边界位置 [X0]
    material_ids: List[int]          # 每层材料编号
    gap_positions: List[float]       # 间隙位置
    active_fraction: List[float]     # 每层活性介质占比


@dataclass
class Grid2D:
    """二维三角网格"""
    vertices: List[Tuple[float, float]]   # 顶点列表 (x, y)
    triangles: List[Tuple[int, int, int]] # 三角形列表 (顶点索引)
    n_vertices: int = 0
    n_triangles: int = 0

    def __post_init__(self):
        self.n_vertices = len(self.vertices)
        self.n_triangles = len(self.triangles)


class CalorimeterGridGenerator:
    """量能器网格生成器"""

    def __init__(self, config: Optional[GridConfig] = None, seed: int = 42):
        self.config = config or GridConfig()
        self.rng = random.Random(seed)

    def generate_longitudinal_grid(self) -> Grid1D:
        """
        生成纵向非均匀网格 (几何级数 + 边界层加密)

        网格生成公式:
          z_i = z_min * r^i + alpha * z_min * i * (1 - i/N) * exp(-beta*i)

        第二项为边界层修正, 在簇射极大值附近加密网格点。

        积分权重 (非均匀梯形法则):
          w_0 = (z_1 - z_0) / 2
          w_i = (z_{i+1} - z_{i-1}) / 2,  0 < i < N
          w_N = (z_N - z_{N-1}) / 2
        """
        cfg = self.config
        nodes = []

        for i in range(cfg.n_z + 1):
            frac = i / cfg.n_z
            # 几何级数基础
            if abs(cfg.z_stretch - 1.0) < 1e-12:
                z_base = cfg.z_min_X0 + frac * (cfg.z_max_X0 - cfg.z_min_X0)
            else:
                z_base = cfg.z_min_X0 * (
                    (cfg.z_stretch ** (cfg.n_z * frac) - 1.0)
                    / (cfg.z_stretch ** cfg.n_z - 1.0)
                ) * cfg.z_max_X0 / cfg.z_min_X0

            # 边界层修正 (在 t ≈ 5-10 X0 附近加密, 对应 shower max)
            t_shower_max = 5.0  # 典型 shower max 位置
            sigma_bl = 2.0      # 加密区域宽度
            alpha_bl = 0.15     # 加密强度
            bl_correction = alpha_bl * math.exp(
                -0.5 * ((z_base - t_shower_max) / sigma_bl) ** 2
            ) * cfg.z_max_X0 / cfg.n_z

            z = z_base * (1.0 - bl_correction)
            z = max(cfg.z_min_X0, min(cfg.z_max_X0, z))
            nodes.append(z)

        # 确保单调递增
        for i in range(1, len(nodes)):
            if nodes[i] <= nodes[i - 1]:
                nodes[i] = nodes[i - 1] + 1e-10

        # 计算间距和权重
        spacings = [nodes[i + 1] - nodes[i] for i in range(len(nodes) - 1)]
        weights = [0.0] * len(nodes)
        weights[0] = spacings[0] / 2.0
        for i in range(1, len(nodes) - 1):
            weights[i] = (spacings[i - 1] + spacings[i]) / 2.0
        weights[-1] = spacings[-1] / 2.0

        return Grid1D(nodes=nodes, spacings=spacings, weights=weights)

    def generate_radial_grid(self) -> Grid1D:
        """
        生成横向径向网格 (核心加密)

        密度分布:
          rho(r) = 1 + (K-1) * exp(-r^2 / (2*sigma_r^2))
        其中 K = refinement_factor, sigma_r = r_core

        累积分布:
          F(r) = r + (K-1)*sigma_r*sqrt(pi/2)*erf(r/(sigma_r*sqrt(2)))
        归一化后反转得到网格点位置。
        """
        cfg = self.config
        nodes = []
        sigma_r = cfg.r_core_cm
        K = cfg.refinement_factor

        # 累积密度函数
        def cdf(r: float) -> float:
            erf_arg = r / (sigma_r * math.sqrt(2.0))
            erf_val = math.erf(erf_arg)
            return r + (K - 1.0) * sigma_r * math.sqrt(math.pi / 2.0) * erf_val

        cdf_max = cdf(cfg.r_max_cm)

        for i in range(cfg.n_r + 1):
            frac = i / cfg.n_r
            target = frac * cdf_max

            # 二分法反转 CDF
            lo, hi = 0.0, cfg.r_max_cm
            for _ in range(60):  # 60 次迭代达到 ~1e-18 精度
                mid = (lo + hi) / 2.0
                if cdf(mid) < target:
                    lo = mid
                else:
                    hi = mid
            nodes.append((lo + hi) / 2.0)

        spacings = [nodes[i + 1] - nodes[i] for i in range(len(nodes) - 1)]
        weights = [0.0] * len(nodes)
        weights[0] = spacings[0] / 2.0
        for i in range(1, len(nodes) - 1):
            weights[i] = (spacings[i - 1] + spacings[i]) / 2.0
        weights[-1] = spacings[-1] / 2.0

        return Grid1D(nodes=nodes, spacings=spacings, weights=weights)

    def generate_layered_structure(self) -> LayeredStructure:
        """
        生成层状量能器结构

        每层:
          - 吸收体 (高 Z, 短 X0)
          - 活性介质 (低 Z, 信号读取)
          - 间隙 (机械支撑 / 光纤)

        采样分数 (e/h 比):
          e/h = (f_em * e_em + f_had * e_had) / (f_em + f_had)
        """
        cfg = self.config
        boundaries = []
        mat_ids = []
        gaps = []
        active_fracs = []

        total_depth = cfg.z_max_X0
        layer_depth = total_depth / cfg.n_layers

        for i in range(cfg.n_layers + 1):
            boundaries.append(i * layer_depth)

        for i in range(cfg.n_layers):
            # 交替材料: 吸收体 (0) 和活性介质 (1)
            mat_ids.append(i % 2)
            gap_pos = boundaries[i] + layer_depth * (1.0 - cfg.layer_gap_fraction)
            gaps.append(gap_pos)
            active_fracs.append(1.0 - cfg.layer_gap_fraction)

        return LayeredStructure(
            layer_boundaries=boundaries,
            material_ids=mat_ids,
            gap_positions=gaps,
            active_fraction=active_fracs,
        )

    def generate_2d_mesh(self, grid_r: Grid1D) -> Grid2D:
        """
        生成 2D Delaunay 三角网格 (简化版 mesh2d 算法)

        算法:
        1. 在极坐标 (r, theta) 生成初始点
        2. 映射到笛卡尔坐标 (x, y)
        3. 使用 bowyer-watson 算法进行三角化
        4. Laplacian 光滑优化网格质量

        网格质量度量 (每个三角形):
          Q = 4*sqrt(3) * Area / (a^2 + b^2 + c^2)
          理想等边三角形: Q = 1.0
        """
        grid_r_nodes = grid_r.nodes
        n_theta = max(6, len(grid_r_nodes) // 2)

        # 生成极坐标网格点
        vertices = [(0.0, 0.0)]  # 中心点

        for i_r in range(1, len(grid_r_nodes)):
            r = grid_r_nodes[i_r]
            # 每个环上的角向偏移 (交错排列提高质量)
            theta_offset = (math.pi / n_theta) if (i_r % 2 == 0) else 0.0
            for j_t in range(n_theta):
                theta = 2.0 * math.pi * j_t / n_theta + theta_offset
                x = r * math.cos(theta)
                y = r * math.sin(theta)
                vertices.append((x, y))

        # 生成三角形连接 (结构化网格直接连接)
        triangles = []
        n_r_pts = len(grid_r_nodes) - 1

        # 中心三角形 (第一环到中心)
        for j in range(n_theta):
            j_next = (j + 1) % n_theta
            v0 = 0  # 中心
            v1 = 1 + j
            v2 = 1 + j_next
            triangles.append((v0, v1, v2))

        # 环间三角形
        for i_r in range(n_r_pts - 1):
            ring_start_curr = 1 + i_r * n_theta
            ring_start_next = 1 + (i_r + 1) * n_theta
            for j in range(n_theta):
                j_next = (j + 1) % n_theta
                v0 = ring_start_curr + j
                v1 = ring_start_curr + j_next
                v2 = ring_start_next + j
                v3 = ring_start_next + j_next
                triangles.append((v0, v1, v3))
                triangles.append((v0, v3, v2))

        # Laplacian 光滑 (固定边界和中心)
        vertices = self._laplacian_smooth(vertices, triangles, n_iter=3)

        return Grid2D(vertices=vertices, triangles=triangles)

    def _laplacian_smooth(
        self,
        vertices: List[Tuple[float, float]],
        triangles: List[Tuple[int, int, int]],
        n_iter: int = 3,
    ) -> List[Tuple[float, float]]:
        """
        Laplacian 网格光滑

        更新公式:
          v_i^{n+1} = v_i^n + omega * (mean(neighbors) - v_i^n)

        其中 omega = 0.5 为松弛因子。
        中心点和边界点固定不动。
        """
        pts = list(vertices)
        n = len(pts)
        omega = 0.5

        # 构建邻接关系
        adj = [set() for _ in range(n)]
        for t in triangles:
            for i in range(3):
                for j in range(3):
                    if i != j:
                        adj[t[i]].add(t[j])

        # 识别边界顶点
        boundary = set()
        max_r_sq = 0.0
        for v in pts:
            r_sq = v[0] ** 2 + v[1] ** 2
            max_r_sq = max(max_r_sq, r_sq)
        for i, v in enumerate(pts):
            r_sq = v[0] ** 2 + v[1] ** 2
            if r_sq > 0.95 * max_r_sq or i == 0:
                boundary.add(i)

        for _ in range(n_iter):
            new_pts = list(pts)
            for i in range(n):
                if i in boundary or not adj[i]:
                    continue
                neighbors = list(adj[i])
                mx = sum(pts[j][0] for j in neighbors) / len(neighbors)
                my = sum(pts[j][1] for j in neighbors) / len(neighbors)
                new_pts[i] = (
                    pts[i][0] + omega * (mx - pts[i][0]),
                    pts[i][1] + omega * (my - pts[i][1]),
                )
            pts = new_pts

        return pts

    def compute_mesh_quality(self, grid: Grid2D) -> dict:
        """
        计算网格质量统计

        质量度量:
          Q = 4*sqrt(3) * A / (a^2 + b^2 + c^2)

        其中 A 为三角形面积, a, b, c 为边长。
        """
        qualities = []
        areas = []

        for tri in grid.triangles:
            p0 = grid.vertices[tri[0]]
            p1 = grid.vertices[tri[1]]
            p2 = grid.vertices[tri[2]]

            # 边长
            a = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            b = math.sqrt((p0[0] - p2[0]) ** 2 + (p0[1] - p2[1]) ** 2)
            c = math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)

            # 面积 (叉积)
            area = 0.5 * abs(
                (p1[0] - p0[0]) * (p2[1] - p0[1])
                - (p2[0] - p0[0]) * (p1[1] - p0[1])
            )

            denom = a * a + b * b + c * c
            if denom < 1e-30:
                q = 0.0
            else:
                q = 4.0 * math.sqrt(3.0) * area / denom

            qualities.append(q)
            areas.append(area)

        if not qualities:
            return {"min": 0, "max": 0, "mean": 0, "std": 0}

        mean_q = sum(qualities) / len(qualities)
        var_q = sum((q - mean_q) ** 2 for q in qualities) / max(len(qualities), 1)

        return {
            "min": min(qualities),
            "max": max(qualities),
            "mean": mean_q,
            "std": math.sqrt(var_q),
            "n_vertices": grid.n_vertices,
            "n_triangles": grid.n_triangles,
            "total_area": sum(areas),
        }

    def arc_grid_along_curve(
        self,
        curve_func,
        n_points: int = 50,
        t_range: Tuple[float, float] = (0.0, 1.0),
    ) -> List[Tuple[float, float]]:
        """
        沿曲线生成等弧长网格 (circle_arc_grid 方法)

        弧长参数化:
          s(t) = integral_0^t |r'(u)| du
          |r'(u)| = sqrt((dx/du)^2 + (dy/du)^2)

        通过反转 s(t) 得到等弧长分布的网格点。
        """
        dt = 1e-8
        t0, t1 = t_range

        # 计算弧长函数 (数值积分)
        n_sample = 200
        arc_lengths = [0.0]
        ts = [t0 + (t1 - t0) * i / n_sample for i in range(n_sample + 1)]

        for i in range(1, len(ts)):
            t_mid = (ts[i] + ts[i - 1]) / 2.0
            dx = (curve_func(t_mid + dt)[0] - curve_func(t_mid - dt)[0]) / (2 * dt)
            dy = (curve_func(t_mid + dt)[1] - curve_func(t_mid - dt)[1]) / (2 * dt)
            ds = math.sqrt(dx * dx + dy * dy) * (ts[i] - ts[i - 1])
            arc_lengths.append(arc_lengths[-1] + ds)

        total_arc = arc_lengths[-1]
        if total_arc < 1e-30:
            return [curve_func(t0)] * n_points

        # 等弧长取点
        points = []
        for i in range(n_points):
            target_s = total_arc * i / (n_points - 1)
            # 二分查找
            lo_idx, hi_idx = 0, len(arc_lengths) - 1
            for _ in range(50):
                mid_idx = (lo_idx + hi_idx) // 2
                if arc_lengths[mid_idx] < target_s:
                    lo_idx = mid_idx
                else:
                    hi_idx = mid_idx
            # 线性插值
            if hi_idx == lo_idx:
                t_interp = ts[lo_idx]
            else:
                frac = (target_s - arc_lengths[lo_idx]) / max(
                    arc_lengths[hi_idx] - arc_lengths[lo_idx], 1e-30
                )
                t_interp = ts[lo_idx] + frac * (ts[hi_idx] - ts[lo_idx])
            points.append(curve_func(t_interp))

        return points
