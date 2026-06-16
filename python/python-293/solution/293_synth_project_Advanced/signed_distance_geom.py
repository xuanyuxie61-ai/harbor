# -*- coding: utf-8 -*-
"""
signed_distance_geom.py
========================
磁层边界几何与符号距离函数

融合种子项目:
  305_dist_plot — DistMesh 符号距离函数库 (Persson & Strang 2004)
  329_ellipse_distance — 椭圆上的距离统计与采样

物理背景
--------
空间等离子体中的波-粒子相互作用发生在特定的几何结构中:

1. 磁层顶 (Magnetopause):
   太阳风与地球磁层的界面，由压力平衡定义。
   可用符号距离函数描述: φ(x,y) < 0 在磁层内部。

2. 等离子体片 (Plasma Sheet):
   磁尾中的高密度等离子体区域。

3. 辐射带 (Radiation Belts):
   被地磁场捕获的高能粒子区域，边界由绝热不变量定义。

符号距离函数
------------
φ(p) 满足 |∇φ| = 1 在边界附近。

常用几何:
  - 圆:   φ = |p - c| - r
  - 椭圆: φ ≈ √((x/a)² + (y/b)²) - 1
  - 矩形: φ = max(|x-xc|-w, |y-yc|-h)
  - 多边形: min distance to edges (with sign)

椭圆上的采样
-----------
在椭圆 (x/a)² + (y/b)² = 1 上均匀采样:
  x = a·cos(t), y = b·sin(t), t ∈ [0, 2π)
弧长元素: ds = √(a²sin²t + b²cos²t) dt
"""

import numpy as np
import math
from typing import Tuple, Optional


# ===== 基本符号距离函数 ====================================================

def dist_circle(p: np.ndarray, xc: float, yc: float,
                 r: float) -> np.ndarray:
    """
    点到圆的符号距离:

    .. math:: d(p) = \\sqrt{(p_x - x_c)^2 + (p_y - y_c)^2} - r

    d < 0: 圆内, d = 0: 圆上, d > 0: 圆外
    """
    p = np.atleast_2d(p)
    return np.sqrt((p[:, 0] - xc) ** 2 + (p[:, 1] - yc) ** 2) - r


def dist_rectangle(p: np.ndarray, xc: float, yc: float,
                    w: float, h: float) -> np.ndarray:
    """
    点到矩形的符号距离:

    .. math:: d(p) = \\max(|p_x - x_c| - w, |p_y - y_c| - h)

    (近似符号距离，非精确 SDF)
    """
    p = np.atleast_2d(p)
    dx = np.abs(p[:, 0] - xc) - w
    dy = np.abs(p[:, 1] - yc) - h
    return np.maximum(dx, dy)


def dist_ellipse(p: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    点到椭圆的近似符号距离:

    椭圆: (x/a)² + (y/b)² = 1

    近似距离:
      d ≈ (√((x/a)² + (y/b)²) - 1) / |∇g|
    其中 g = (x/a)² + (y/b)² - 1
    """
    p = np.atleast_2d(p)
    xa = p[:, 0] / max(a, 1e-300)
    yb = p[:, 1] / max(b, 1e-300)
    g = xa ** 2 + yb ** 2 - 1.0
    grad_g = 2.0 * np.sqrt((xa / max(a, 1e-300)) ** 2
                            + (yb / max(b, 1e-300)) ** 2)
    grad_g = np.maximum(grad_g, 1e-300)
    return g / grad_g


def dist_segment(p: np.ndarray, a: np.ndarray,
                  b: np.ndarray) -> np.ndarray:
    """
    点到线段 AB 的距离.

    使用投影参数 t:
      t = (p-a)·(b-a) / |b-a|²
    若 0 ≤ t ≤ 1: 距离 = |p - (a + t(b-a))|
    否则: 距离 = min(|p-a|, |p-b|)
    """
    p = np.atleast_2d(p)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ab = b - a
    ab2 = np.dot(ab, ab)

    if ab2 < 1e-300:
        return np.sqrt(np.sum((p - a) ** 2, axis=1))

    t = np.dot(p - a, ab) / ab2
    t = np.clip(t, 0.0, 1.0)
    proj = a[np.newaxis, :] + t[:, np.newaxis] * ab[np.newaxis, :]
    return np.sqrt(np.sum((p - proj) ** 2, axis=1))


# ===== CSG 布尔运算 ========================================================

def sdf_union(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    """
    符号距离函数的并集:

    φ_{A∪B} = min(φ_A, φ_B)

    用于组合多个等离子体区域 (如磁层 + 等离子体片).
    """
    return np.minimum(d1, d2)


def sdf_intersection(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    """
    符号距离函数的交集:

    φ_{A∩B} = max(φ_A, φ_B)
    """
    return np.maximum(d1, d2)


def sdf_difference(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    """
    符号距离函数的差集:

    φ_{A\\B} = max(φ_A, -φ_B)
    """
    return np.maximum(d1, -d2)


def sdf_complement(d: np.ndarray) -> np.ndarray:
    """符号距离函数的补集: φ_{A^c} = -φ_A."""
    return -d


# ===== 椭圆采样与距离统计 ==================================================

class EllipseSampler:
    """
    椭圆上的均匀采样与距离统计

    用于分析等离子体分布函数在各向异性温度下的等值面性质。

    椭圆参数方程:
      x(t) = a cos(t)
      y(t) = b sin(t)

    弧长:
      s(t) = ∫₀ᵗ √(a²sin²τ + b²cos²τ) dτ
    """

    def __init__(self, a: float, b: float):
        """
        Parameters
        ----------
        a, b : float
            椭圆半长轴和半短轴.
        """
        if a <= 0 or b <= 0:
            raise ValueError("EllipseSampler: semi-axes must be positive")
        self.a = a
        self.b = b
        self.eccentricity = math.sqrt(abs(1.0 - min(a, b) ** 2
                                           / max(a, b) ** 2))
        self.total_arc = self._arc_length(2.0 * math.pi)

    def _arc_length(self, t: float, n_quad: int = 100) -> float:
        """计算从 0 到 t 的弧长 (数值积分)."""
        ts = np.linspace(0, t, n_quad + 1)
        ds = np.sqrt((self.a * np.sin(ts)) ** 2
                      + (self.b * np.cos(ts)) ** 2)
        return float(np.trapz(ds, ts))

    def sample_uniform(self, n: int) -> np.ndarray:
        """
        在椭圆周长上均匀采样 n 个点

        Returns
        -------
        np.ndarray, shape (n, 2)
            采样点坐标.
        """
        # 通过弧长参数化实现均匀采样
        s_targets = np.linspace(0, self.total_arc, n, endpoint=False)
        points = np.zeros((n, 2))

        for i, s in enumerate(s_targets):
            # 二分法求 t(s)
            t = self._invert_arc_length(s)
            points[i, 0] = self.a * math.cos(t)
            points[i, 1] = self.b * math.sin(t)

        return points

    def _invert_arc_length(self, s_target: float,
                            tol: float = 1e-10) -> float:
        """通过二分法反解弧长参数 t(s)."""
        t_lo, t_hi = 0.0, 2.0 * math.pi
        for _ in range(60):
            t_mid = 0.5 * (t_lo + t_hi)
            s = self._arc_length(t_mid)
            if s < s_target:
                t_lo = t_mid
            else:
                t_hi = t_mid
            if t_hi - t_lo < tol:
                break
        return 0.5 * (t_lo + t_hi)

    def distance_stats(self, n_samples: int = 1000
                        ) -> Tuple[float, float]:
        """
        估计椭圆上两点间距离的统计量

        随机采样 n 对点，计算距离的均值和方差。

        Returns
        -------
        mu : float
            平均距离.
        var : float
            距离方差.
        """
        p = self.sample_uniform(n_samples)
        q = self.sample_uniform(n_samples)

        dists = np.sqrt(np.sum((p - q) ** 2, axis=1))
        mu = float(np.mean(dists))
        var = float(np.var(dists)) if n_samples > 1 else 0.0

        return mu, var

    def curvature(self, t: float) -> float:
        """
        椭圆在参数 t 处的曲率:

        .. math::
            \\kappa(t) = \\frac{ab}{(a^2 \\sin^2 t + b^2 \\cos^2 t)^{3/2}}
        """
        num = self.a * self.b
        denom = ((self.a * math.sin(t)) ** 2
                 + (self.b * math.cos(t)) ** 2) ** 1.5
        return num / max(denom, 1e-300)


# ===== 磁层拓扑模型 ========================================================

class MagnetosphereTopology:
    """
    简化磁层拓扑模型

    使用符号距离函数的 CSG 组合描述磁层结构:
      - 磁层腔: 圆形区域 (地球为中心)
      - 磁尾: 拉伸的椭圆区域
      - 磁层顶: 日侧压缩，夜侧拉伸
    """

    def __init__(self, r_mp: float = 10.0, r_tail: float = 30.0,
                 epsilon: float = 0.3):
        """
        Parameters
        ----------
        r_mp : float
            日侧磁层顶距离 (地球半径).
        r_tail : float
            夜侧磁尾延伸距离.
        epsilon : float
            磁层顶压缩率.
        """
        self.r_mp = r_mp
        self.r_tail = r_tail
        self.eps = epsilon

    def magnetopause_sdf(self, p: np.ndarray) -> np.ndarray:
        """
        磁层顶的符号距离函数

        日侧: 压缩半圆 (半径 r_mp)
        夜侧: 拉伸半圆 (半径 r_tail)

        使用椭圆近似:
          (x/a)² + (y/b)² = 1
        其中 a = (r_mp + r_tail)/2, 中心偏移 = (r_tail - r_mp)/2
        """
        p = np.atleast_2d(p)
        a = 0.5 * (self.r_mp + self.r_tail)
        x_shift = p[:, 0] - 0.5 * (self.r_tail - self.r_mp)
        b = self.r_mp * (1.0 - 0.5 * self.eps)

        return dist_ellipse(
            np.column_stack([x_shift, p[:, 1]]), a, b)

    def plasma_sheet_sdf(self, p: np.ndarray,
                          thickness: float = 2.0) -> np.ndarray:
        """
        等离子体片的符号距离函数

        磁尾中的薄片状高密度区域:
          |y| < h(x) 且 x < 0 (夜侧)
        """
        p = np.atleast_2d(p)
        # 厚度随 x 变化: h(x) = thickness * (1 + |x|/r_tail)
        h = thickness * (1.0 + np.abs(p[:, 0]) / max(self.r_tail, 1.0))
        d_sheet = np.abs(p[:, 1]) - h
        # 仅限夜侧 (x < 0)
        d_night = p[:, 0]
        return sdf_intersection(d_sheet, d_night)

    def is_inside_magnetosphere(self, p: np.ndarray) -> np.ndarray:
        """判断点是否在磁层内部."""
        return self.magnetopause_sdf(p) < 0

    def l_shell(self, p: np.ndarray) -> np.ndarray:
        """
        计算 L 壳参数 (偶极场近似):

        L = r / cos²(λ)
        其中 r = √(x² + y²), λ = arctan(y/x) 为磁纬
        """
        p = np.atleast_2d(p)
        r = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)
        r = np.maximum(r, 1e-300)
        cos_lat = np.abs(p[:, 0]) / r  # 简化: x 为径向
        cos_lat = np.maximum(cos_lat, 1e-10)
        return r / (cos_lat ** 2)
