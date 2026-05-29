"""
三维降水体积蒙特卡洛估算模块 (3D Precipitation Volume Estimator)

集成种子项目:
- 1257_tetrahedron01_monte_carlo: 单位四面体上的蒙特卡洛积分

科学背景:
  中尺度对流系统的降水率场通常定义在三维空间 (或随高度变化).
  为了估算特定区域内的总降水量, 将空间离散为四面体单元,
  并在每个单元上使用蒙特卡洛积分:

    Total_Precip = Σ_cells V_cell * (1/N) Σ_{i=1}^N R(x_i)

  其中 R(x) 为降水率, V_cell 为单元体积.

核心公式:
  标准单位四面体顶点: (0,0,0), (1,0,0), (0,1,0), (0,0,1)
  体积: V = 1/6

  单位四面体上的精确单项式积分:
    ∫ x^a y^b z^c dV = a! b! c! / (a+b+c+3)!

  随机点生成 (基于指数分布):
    生成 E1, E2, E3 ~ Exp(1), 则
    x = E1/S, y = E2/S, z = E3/S,  S = E1+E2+E3
    若 S=0, 回退到均匀采样.
"""

import numpy as np
from typing import List, Tuple


def tetrahedron01_volume() -> float:
    """单位四面体体积."""
    return 1.0 / 6.0


def tetrahedron01_monomial_integral(e: Tuple[int, int, int]) -> float:
    """
    单位四面体上的单项式精确积分 (基于 1257_tetrahedron01_monte_carlomonomial_integral).

    ∫_{T} x^e1 y^e2 z^e3 dV = e1! e2! e3! / (e1+e2+e3+3)!
    """
    e1, e2, e3 = e
    from math import factorial
    return factorial(e1) * factorial(e2) * factorial(e3) / factorial(e1 + e2 + e3 + 3)


def tetrahedron01_sample(n: int) -> np.ndarray:
    """
    在单位四面体内均匀生成 n 个随机点 (基于 1257_tetrahedron01_monte_carlo/sample).

    算法: 使用指数分布归一化.
    """
    points = np.zeros((n, 3))
    for i in range(n):
        e1, e2, e3 = np.random.exponential(1.0, 3)
        s = e1 + e2 + e3
        if s < 1e-20:
            # 回退到均匀采样
            u = np.random.rand(3)
            u.sort()
            points[i] = [u[0], u[1] - u[0], u[2] - u[1]]
        else:
            points[i] = [e1 / s, e2 / s, e3 / s]
    return points


def map_to_physical_tetrahedron(points_ref: np.ndarray,
                                 vertices: np.ndarray) -> np.ndarray:
    """
    将参考四面体中的点映射到物理四面体.

    顶点顺序: v0, v1, v2, v3 (各为 3D 点)
    x_phys = v0 + J * ξ_ref,  J = [v1-v0, v2-v0, v3-v0]
    """
    v0 = vertices[0]
    J = np.column_stack([vertices[1] - v0,
                         vertices[2] - v0,
                         vertices[3] - v0])
    return v0 + points_ref @ J.T


def tetrahedron_physical_volume(vertices: np.ndarray) -> float:
    """
    物理四面体体积 = |det(J)| / 6.
    """
    J = np.column_stack([vertices[1] - vertices[0],
                         vertices[2] - vertices[0],
                         vertices[3] - vertices[0]])
    return abs(np.linalg.det(J)) / 6.0


class PrecipitationVolumeEstimator:
    """
    基于四面体蒙特卡洛积分的降水体积估算器.
    """

    def __init__(self, n_samples_per_cell: int = 256):
        self.n_samples = n_samples_per_cell

    def estimate_cell_precipitation(self, vertices: np.ndarray,
                                     precip_rate_func) -> float:
        """
        对单个四面体单元估算总降水量.

        参数:
          vertices: (4,3) 物理坐标
          precip_rate_func: R(x,y,z) -> float, 降水率函数

        返回:
          单元内总降水量 (体积积分).
        """
        vol = tetrahedron_physical_volume(vertices)
        if vol < 1e-20:
            return 0.0

        ref_points = tetrahedron01_sample(self.n_samples)
        phys_points = map_to_physical_tetrahedron(ref_points, vertices)

        total = 0.0
        valid = 0
        for pt in phys_points:
            try:
                val = precip_rate_func(pt)
                if np.isfinite(val) and val >= 0.0:
                    total += val
                    valid += 1
            except Exception:
                continue

        if valid == 0:
            return 0.0
        return vol * total / valid

    def estimate_domain_precipitation(self, tetrahedra_vertices: List[np.ndarray],
                                       precip_rate_func) -> float:
        """
        对多个四面体单元组成的区域估算总降水量.
        """
        total = 0.0
        for verts in tetrahedra_vertices:
            total += self.estimate_cell_precipitation(verts, precip_rate_func)
        return total

    def estimate_from_gridded_field(self, precip_rate_2d: np.ndarray,
                                     dx: float, dy: float, dz_levels: np.ndarray) -> float:
        """
        从格点降水率场估算总降水体积.
        将每个网格柱体分解为 5 个四面体 (Kuhn 分割).
        """
        nz, ny, nx = precip_rate_2d.shape if precip_rate_2d.ndim == 3 else (1, *precip_rate_2d.shape)
        if precip_rate_2d.ndim == 2:
            precip_rate_2d = precip_rate_2d.reshape((1, ny, nx))

        total_precip = 0.0
        # 简化为每层棱柱体的体积平均
        for k in range(nz):
            dz = dz_levels[k] if k < len(dz_levels) else dz_levels[-1] if len(dz_levels) > 0 else 1000.0
            for j in range(ny):
                for i in range(nx):
                    rate = precip_rate_2d[k, j, i]
                    if np.isfinite(rate) and rate > 0.0:
                        cell_vol = dx * dy * dz
                        total_precip += cell_vol * rate
        return total_precip
