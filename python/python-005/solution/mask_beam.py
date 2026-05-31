# -*- coding: utf-8 -*-
"""
mask_beam.py
巡天掩膜几何与波束卷积

核心物理：
    1. 巡天掩膜的几何矩（面积、质心、惯性矩、椭圆率）：
        ν_{pq} = ∬_Ω x^p y^q dx dy
        通过 Green 定理分解为边界线积分求和。
    2. 圆形波束的矩积分：
        ∬_{disk} x^{e1} y^{e2} dx dy
        利用极坐标与 Gamma 函数解析求解。
    3. 波束窗函数 B_l：
        B_l = 2π ∫_0^{θ_FWHM} dθ sinθ P_l(cosθ) B(θ)
        对高斯波束 B(θ) = exp(-θ^2 / 2σ^2) / (2πσ^2)。

融合种子项目 109_boundary_word_right（边界多边形遍历、点在多边形内）、
886_polygon_integrals（多边形几何矩）、294_disk_integrals（圆盘矩积分）。
"""

import numpy as np
from typing import List, Tuple
from utils import gamma_lanczos, binomial, robust_divide


# ---------------------------------------------------------------------------
# 多边形几何矩（Green 定理 / Steger 方法）
# ---------------------------------------------------------------------------
def polygon_moment(nv: int, x: np.ndarray, y: np.ndarray,
                   p: int, q: int) -> float:
    """
    计算多边形 ν_{pq} = ∬ x^p y^q dx dy。
    使用 Steger 方法将面积积分转化为边界线积分：
        ν_{pq} = Σ_{边(i,j)} (x_j y_i - x_i y_j) s_{pq} / [(p+q+2)(p+q+1) C(p+q,p)]
    其中 s_{pq} = Σ_{a=0}^p Σ_{b=0}^q C(p,a) C(q,b) x_i^{p-a} x_j^a y_i^{q-b} y_j^b。
    """
    if nv < 3:
        return 0.0
    nu = 0.0
    for i in range(nv):
        j = (i + 1) % nv
        xi, yi = x[i], y[i]
        xj, yj = x[j], y[j]
        s = 0.0
        for a in range(p + 1):
            for b in range(q + 1):
                s += (binomial(p, a) * binomial(q, b) *
                      (xi ** (p - a)) * (xj ** a) *
                      (yi ** (q - b)) * (yj ** b))
        cross = xj * yi - xi * yj
        denom = (p + q + 2) * (p + q + 1) * binomial(p + q, p)
        if denom == 0:
            continue
        nu += cross * s / denom
    return nu


def polygon_area(x: np.ndarray, y: np.ndarray) -> float:
    """多边形有向面积（Green 定理）。"""
    nv = len(x)
    if nv < 3:
        return 0.0
    area = 0.0
    for i in range(nv):
        j = (i + 1) % nv
        area += x[i] * y[j] - x[j] * y[i]
    return 0.5 * area


def polygon_centroid(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """多边形质心。"""
    A = polygon_area(x, y)
    if abs(A) < 1e-15:
        return 0.0, 0.0
    cx = polygon_moment(len(x), x, y, 1, 0) / A
    cy = polygon_moment(len(x), x, y, 0, 1) / A
    return cx, cy


def polygon_central_moment(x: np.ndarray, y: np.ndarray,
                           p: int, q: int) -> float:
    """中心矩 μ_{pq} = ∬ (x-cx)^p (y-cy)^q dx dy。"""
    cx, cy = polygon_centroid(x, y)
    x_shifted = x - cx
    y_shifted = y - cy
    return polygon_moment(len(x), x_shifted, y_shifted, p, q)


# ---------------------------------------------------------------------------
# 点在多边形内（射线交叉法，来自 boundary_word_right）
# ---------------------------------------------------------------------------
def point_in_polygon(px: float, py: float,
                     x: np.ndarray, y: np.ndarray) -> bool:
    """
    射线交叉法判断点 (px,py) 是否在多边形内。
    处理点在边上的边界情况。
    """
    nv = len(x)
    inside = False
    j = nv - 1
    for i in range(nv):
        xi, yi = x[i], y[i]
        xj, yj = x[j], y[j]
        # 检查是否恰好在边上
        if abs((py - yi) * (xj - xi) - (px - xi) * (yj - yi)) < 1e-12:
            if min(xi, xj) <= px <= max(xi, xj) and min(yi, yj) <= py <= max(yi, yj):
                return True
        # 射线交叉
        if ((yi > py) != (yj > py)):
            xinters = (xj - xi) * (py - yi) / (yj - yi + 1e-15) + xi
            if px < xinters:
                inside = not inside
        j = i
    return inside


# ---------------------------------------------------------------------------
# 圆盘矩积分（来自 disk_integrals）
# ---------------------------------------------------------------------------
def disk_monomial_integral(r: float, e1: int, e2: int) -> float:
    """
    计算 ∬_{x^2+y^2≤r^2} x^{e1} y^{e2} dx dy 的解析值。
    若 e1 或 e2 为奇数，结果为 0。
    否则：
        I = 2 Γ((e1+1)/2) Γ((e2+1)/2) / [Γ((e1+e2)/2 + 1) (e1+e2+2)] * r^{e1+e2+2}
    """
    if e1 < 0 or e2 < 0:
        return 0.0
    if e1 % 2 == 1 or e2 % 2 == 1:
        return 0.0
    if r <= 0.0:
        return 0.0
    g1 = gamma_lanczos((e1 + 1.0) / 2.0)
    g2 = gamma_lanczos((e2 + 1.0) / 2.0)
    g3 = gamma_lanczos((e1 + e2) / 2.0 + 1.0)
    exponent = e1 + e2 + 2
    return 2.0 * g1 * g2 / (g3 * exponent) * (r ** exponent)


def disk_area(r: float) -> float:
    """圆盘面积 π r^2。"""
    return np.pi * r * r


def disk_uniform_sample(r: float, n: int) -> np.ndarray:
    """
    在圆盘内均匀随机采样 n 个点。
    方法：生成二维正态分布 → 归一化 → 径向缩放 sqrt(U(0,1))。
    """
    samples = np.random.randn(n, 2)
    norms = np.linalg.norm(samples, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    samples = samples / norms
    radii = r * np.sqrt(np.random.rand(n, 1))
    return samples * radii


# ---------------------------------------------------------------------------
# 高斯波束窗函数
# ---------------------------------------------------------------------------
def gaussian_beam_window(l: int, fwhm_arcmin: float) -> float:
    """
    高斯波束的球谐窗函数：
        B_l = exp[-l(l+1) σ^2 / 2]
    其中 σ = FWHM / sqrt(8 ln 2)（弧度）。
    """
    sigma = np.radians(fwhm_arcmin / 60.0) / np.sqrt(8.0 * np.log(2.0))
    return np.exp(-l * (l + 1.0) * sigma ** 2 / 2.0)


def beam_convolved_Cl(Cl: np.ndarray, lmax: int, fwhm_arcmin: float) -> np.ndarray:
    """
    对原始 C_l 进行波束卷积：
        C_l^{obs} = B_l^2 C_l^{sky}
    """
    out = np.zeros(len(Cl))
    for idx, l in enumerate(range(2, lmax + 3)):
        if idx >= len(Cl):
            break
        Bl = gaussian_beam_window(l, fwhm_arcmin)
        out[idx] = Bl * Bl * Cl[idx]
    return out


# ---------------------------------------------------------------------------
# 掩膜-波束联合几何分析
# ---------------------------------------------------------------------------
class SurveyMask:
    """
    球面巡天掩膜：用二维投影多边形表示（例如 Mollweide 投影后的边界）。
    计算面积、椭圆率、边界矩等几何描述子。
    """

    def __init__(self, boundary_x: np.ndarray, boundary_y: np.ndarray):
        """
        Parameters
        ----------
        boundary_x, boundary_y : np.ndarray
            掩膜边界顶点（逆时针或顺时针排列）。
        """
        self.x = boundary_x
        self.y = boundary_y
        self.nv = len(boundary_x)
        self._area = polygon_area(self.x, self.y)
        self._cx, self._cy = polygon_centroid(self.x, self.y)

    def area(self) -> float:
        return abs(self._area)

    def centroid(self) -> Tuple[float, float]:
        return self._cx, self._cy

    def ellipticity(self) -> float:
        """
        椭圆率 e = 1 - b/a，其中 a,b 为等效椭圆的半长/短轴。
        由二阶中心矩计算：
            μ20 = ∬ (x-cx)^2 dxdy
            μ02 = ∬ (y-cy)^2 dxdy
            μ11 = ∬ (x-cx)(y-cy) dxdy
            λ_{±} = (μ20+μ02)/2 ± sqrt[((μ20-μ02)/2)^2 + μ11^2]
        """
        mu20 = polygon_central_moment(self.x, self.y, 2, 0)
        mu02 = polygon_central_moment(self.x, self.y, 0, 2)
        mu11 = polygon_central_moment(self.x, self.y, 1, 1)
        trace = mu20 + mu02
        det = mu20 * mu02 - mu11 * mu11
        disc = np.sqrt(max(((mu20 - mu02) / 2.0) ** 2 + mu11 ** 2, 0.0))
        lambda_plus = trace / 2.0 + disc
        lambda_minus = trace / 2.0 - disc
        if lambda_plus <= 1e-15:
            return 0.0
        a = np.sqrt(lambda_plus)
        b = np.sqrt(max(lambda_minus, 0.0))
        return 1.0 - b / a

    def contains(self, px: float, py: float) -> bool:
        return point_in_polygon(px, py, self.x, self.y)

    def fsky(self) -> float:
        """球面覆盖比例 f_sky = A_mask / (4π)。这里用投影面积近似。"""
        return self.area() / (4.0 * np.pi)
