# -*- coding: utf-8 -*-
"""
shell_geometry.py
圆柱壳几何描述与测地距离计算

融合种子项目:
  - 236_cube_surface_distance: 表面距离统计
  - 107_boundary_word_equilateral: 边界节点识别与排序

科学背景:
  薄壁圆柱壳的中曲面参数化为
    r(θ, x) = [ R cos θ, R sin θ, x ]
  第一基本形式系数:
    E = r_θ · r_θ = R²
    F = r_θ · r_x = 0
    G = r_x · r_x = 1
  第二基本形式系数:
    L = n · r_θθ = R
    M = n · r_θx = 0
    N = n · r_xx = 0
  高斯曲率 K = (LN - M²)/(EG - F²) = 0
  平均曲率 H = (EN + GL - 2FM)/(2(EG-F²)) = 1/(2R)
"""

import numpy as np
from numpy.linalg import norm


class CylindricalShellGeometry:
    """
    圆柱壳中曲面几何描述类
    """

    def __init__(self, radius: float, length: float, thickness: float):
        """
        Parameters
        ----------
        radius : float
            圆柱壳中曲面半径 R (m)
        length : float
            壳体轴向长度 L (m)
        thickness : float
            壳壁厚度 t (m)
        """
        if radius <= 0.0:
            raise ValueError("半径必须为正")
        if length <= 0.0:
            raise ValueError("长度必须为正")
        if thickness <= 0.0 or thickness >= radius:
            raise ValueError("厚度必须为正且小于半径")
        self.R = float(radius)
        self.L = float(length)
        self.t = float(thickness)
        # 高斯曲率 K = 0 (可展曲面)
        self.K = 0.0
        # 平均曲率 H = 1/(2R)
        self.H = 1.0 / (2.0 * self.R)

    def parametric_surface(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        参数化中曲面 r(θ, x) = [R cos θ, R sin θ, x]

        Returns
        -------
        coords : (..., 3) ndarray
        """
        theta = np.asarray(theta)
        x = np.asarray(x)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        return np.stack([
            self.R * cos_t,
            self.R * sin_t,
            x
        ], axis=-1)

    def surface_normal(self, theta: np.ndarray) -> np.ndarray:
        """
        单位外法向量 n(θ) = [cos θ, sin θ, 0]
        """
        theta = np.asarray(theta)
        return np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=-1)

    def first_fundamental_form(self) -> tuple:
        """
        第一基本形式 (E, F, G)
        ds² = E dθ² + 2F dθ dx + G dx²
        对于圆柱壳: E = R², F = 0, G = 1
        """
        return self.R ** 2, 0.0, 1.0

    def second_fundamental_form(self, theta: np.ndarray) -> tuple:
        """
        第二基本形式 (L, M, N)
        对于圆柱壳: L = R, M = 0, N = 0
        """
        theta = np.asarray(theta)
        return self.R * np.ones_like(theta), np.zeros_like(theta), np.zeros_like(theta)

    def principal_curvatures(self) -> tuple:
        """
        主曲率 κ₁ = 1/R (环向), κ₂ = 0 (轴向)
        """
        return 1.0 / self.R, 0.0

    def geodesic_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """
        圆柱面上两点的测地距离

        圆柱面是可展曲面，可展开为平面矩形:
          u = R·θ,  v = x
        展开后两点间直线距离即测地距离:
          d = sqrt( (R·Δθ)² + (Δx)² )
        其中 Δθ 取最短弧: min(|Δθ|, 2π - |Δθ|)

        Parameters
        ----------
        p1, p2 : array_like, shape (3,)
            三维笛卡尔坐标 [x, y, z]

        Returns
        -------
        d : float
            测地距离
        """
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        if p1.shape != (3,) or p2.shape != (3,):
            raise ValueError("输入点必须是三维坐标")
        # 反解参数
        theta1 = np.arctan2(p1[1], p1[0])
        theta2 = np.arctan2(p2[1], p2[0])
        x1, x2 = p1[2], p2[2]
        dtheta = np.abs(theta2 - theta1)
        dtheta = np.minimum(dtheta, 2.0 * np.pi - dtheta)
        dx = x2 - x1
        d = np.sqrt((self.R * dtheta) ** 2 + dx ** 2)
        return float(d)

    def boundary_sort(self, nodes: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        """
        边界节点排序 (基于 107_boundary_word_equilateral 的边界排序思想)

        将边界节点按圆周角 θ 排序，用于施加周期性边界条件。

        Parameters
        ----------
        nodes : (N, 3) ndarray
            节点坐标
        tol : float
            边界判定容差

        Returns
        -------
        sorted_indices : (N,) ndarray
            排序后的节点索引
        """
        nodes = np.asarray(nodes, dtype=float)
        if nodes.ndim != 2 or nodes.shape[1] != 3:
            raise ValueError("nodes 必须是 (N,3) 数组")
        # 判定边界节点: x=0 或 x=L 的节点
        x = nodes[:, 2]
        is_boundary = (np.abs(x) < tol) | (np.abs(x - self.L) < tol)
        indices = np.where(is_boundary)[0]
        if len(indices) == 0:
            return np.array([], dtype=int)
        # 按极角排序
        theta = np.arctan2(nodes[indices, 1], nodes[indices, 0])
        # 处理 atan2 的周期性: 保证 [-π, π] 内连续
        theta = np.mod(theta + 2.0 * np.pi, 2.0 * np.pi)
        order = np.argsort(theta)
        return indices[order]

    def surface_area(self) -> float:
        """
        中曲面面积 A = 2πRL
        """
        return 2.0 * np.pi * self.R * self.L

    def aspect_ratio(self) -> float:
        """
        长径比 L/R
        """
        return self.L / self.R

    def batdorf_parameter(self, E: float, nu: float) -> float:
        """
        Batdorf 参数 Z = (L² / R t) * sqrt(1 - ν²)
        用于壳体屈曲分类:
          Z < 2.85: 短壳 (类似平板)
          2.85 ≤ Z ≤ 1000: 中等长度壳
          Z > 1000: 长壳
        """
        return (self.L ** 2 / (self.R * self.t)) * np.sqrt(1.0 - nu ** 2)
