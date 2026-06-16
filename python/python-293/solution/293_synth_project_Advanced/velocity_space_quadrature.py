# -*- coding: utf-8 -*-
"""
velocity_space_quadrature.py
==============================
速度空间求积与高阶矩计算

融合种子项目:
  886_polygon_integrals — 多边形精确矩积分 (Steger 1996)

物理背景
--------
Vlasov 方程的速度矩通过速度空间积分定义:
  M_n(x,t) = ∫ v^n f(x,v,t) dv

高阶矩 (>2) 决定热流、粘滞等输运系数。
精确的数值积分对守恒律的满足至关重要。

积分方法
--------
1. 梯形法则: O(Δv²) 精度
2. Simpson 法则: O(Δv⁴) 精度
3. Gauss-Hermite 求积: 对 Maxwellian 权重精确

多边形区域上的精确矩:
  ν_{pq} = ∫∫_P x^p y^q dx dy
通过 Steger 算法转化为边界线积分，精确到机器精度。
"""

import numpy as np
import math
from typing import Tuple, Optional


class VelocityQuadrature:
    """
    速度空间求积规则

    支持多种积分方法:
      - trapezoidal: 梯形法则
      - simpson: Simpson 法则
      - gauss_hermite: Gauss-Hermite 求积
    """

    def __init__(self, v_grid: np.ndarray, method: str = 'simpson'):
        self.v = np.asarray(v_grid, dtype=np.float64)
        self.nv = len(self.v)
        self.dv = self.v[1] - self.v[0] if self.nv > 1 else 1.0
        self.method = method

        # 预计算权重
        self.weights = self._compute_weights()

    def _compute_weights(self) -> np.ndarray:
        """预计算积分权重."""
        if self.method == 'trapezoidal':
            w = np.ones(self.nv) * self.dv
            w[0] *= 0.5
            w[-1] *= 0.5
            return w
        elif self.method == 'simpson':
            if self.nv < 3:
                return np.ones(self.nv) * self.dv
            w = np.ones(self.nv) * self.dv
            # Simpson's 1/3 rule
            w[0] = self.dv / 3.0
            w[-1] = self.dv / 3.0
            for i in range(1, self.nv - 1, 2):
                w[i] = 4.0 * self.dv / 3.0
            for i in range(2, self.nv - 2, 2):
                w[i] = 2.0 * self.dv / 3.0
            if self.nv % 2 == 0:
                # 偶数点: 最后一段用梯形
                w[-2] = self.dv / 2.0
                w[-1] = self.dv / 2.0
            return w
        else:
            return np.ones(self.nv) * self.dv

    def integrate(self, f: np.ndarray) -> float:
        """
        ∫ f(v) dv

        Parameters
        ----------
        f : np.ndarray, shape (nv,)
            被积函数值.

        Returns
        -------
        float
            积分值.
        """
        return float(np.sum(f * self.weights))

    def moment(self, f: np.ndarray, order: int) -> float:
        """
        速度矩: M_n = ∫ v^n f(v) dv

        Parameters
        ----------
        f : np.ndarray
            分布函数.
        order : int
            矩的阶数.

        Returns
        -------
        float
            第 order 阶矩.
        """
        integrand = f * (self.v ** order)
        return self.integrate(integrand)

    def central_moment(self, f: np.ndarray, order: int) -> float:
        """
        中心矩: μ_n = ∫ (v - <v>)^n f(v) dv

        <v> = M_1 / M_0 为平均速度.
        """
        M0 = self.moment(f, 0)
        if abs(M0) < 1.0e-300:
            return 0.0
        v_mean = self.moment(f, 1) / M0
        v_shift = self.v - v_mean
        integrand = f * (v_shift ** order)
        return self.integrate(integrand)

    def thermal_speed(self, f: np.ndarray) -> float:
        """
        热速度: v_th = √(<v²> - <v>²)

        等价于 v_th = √(2T/m).
        """
        M0 = self.moment(f, 0)
        if abs(M0) < 1.0e-300:
            return 0.0
        v_mean = self.moment(f, 1) / M0
        v2_mean = self.moment(f, 2) / M0
        return math.sqrt(max(v2_mean - v_mean ** 2, 0.0))


class GaussHermiteQuadrature:
    """
    Gauss-Hermite 求积

    适用于带 Maxwellian 权重的积分:
      ∫ g(v) exp(-v²/v_th²) dv ≈ Σ w_i g(v_i)

    n 点 Gauss-Hermite 精确到 2n-1 阶多项式.
    """

    def __init__(self, n_points: int, v_th: float = 1.0):
        self.n = n_points
        self.v_th = v_th
        self.nodes, self.raw_weights = np.polynomial.hermite.hermgauss(n_points)
        # 缩放: v = v_th * x / √2  (物理速度)
        self.v = self.nodes * v_th / math.sqrt(2.0)
        self.weights = self.raw_weights * v_th / math.sqrt(2.0)

    def integrate(self, g: np.ndarray) -> float:
        """∫ g(v) exp(-v²/v_th²) dv ≈ Σ w_i g(v_i)."""
        return float(np.sum(g * self.weights))


class PolygonMomentCalculator:
    """
    多边形区域上的精确矩积分

    基于 Steger (1996) 算法:
      ν_{pq} = ∫∫_P x^p y^q dx dy

    将面积分通过 Green 定理转化为边界线积分:
      ν_{pq} = (1/(p+q+2)) Σ_edges ∮ (x^p y^q)(x dy - y dx) / 2

    用于计算速度空间中截断区域的各阶矩。
    """

    def __init__(self, vertices_x: np.ndarray,
                 vertices_y: np.ndarray):
        """
        Parameters
        ----------
        vertices_x, vertices_y : np.ndarray
            多边形顶点坐标 (按逆时针排列, 首尾可重合).
        """
        self.vx = np.asarray(vertices_x, dtype=np.float64)
        self.vy = np.asarray(vertices_y, dtype=np.float64)
        # 确保首尾重合
        if not (abs(self.vx[0] - self.vx[-1]) < 1e-12 and
                abs(self.vy[0] - self.vy[-1]) < 1e-12):
            self.vx = np.append(self.vx, self.vx[0])
            self.vy = np.append(self.vy, self.vy[0])
        self.n = len(self.vx) - 1  # 有效顶点数

    def area(self) -> float:
        """
        多边形面积 (Shoelace 公式):
          A = (1/2) |Σ (x_i y_{i+1} - x_{i+1} y_i)|
        """
        return abs(self.moment(0, 0))

    def centroid(self) -> Tuple[float, float]:
        """
        质心: (x̄, ȳ) = (ν₁₀/A, ν₀₁/A)
        """
        A = self.area()
        if abs(A) < 1e-300:
            return (0.0, 0.0)
        return (self.moment(1, 0) / A, self.moment(0, 1) / A)

    def moment(self, p: int, q: int) -> float:
        """
        非归一化矩 ν(p,q) = ∫∫_P x^p y^q dx dy

        使用 Steger 算法精确计算.
        """
        from grid_integer_lib import i4_choose

        nu_pq = 0.0
        xj = self.vx[self.n - 1]
        yj = self.vy[self.n - 1]

        for i in range(self.n):
            xi = self.vx[i]
            yi = self.vy[i]

            s_pq = 0.0
            for k in range(p + 1):
                for l in range(q + 1):
                    c1 = i4_choose(k + l, l)
                    c2 = i4_choose(p + q - k - l, q - l)
                    s_pq += (c1 * c2
                             * (xi ** k) * (xj ** (p - k))
                             * (yi ** l) * (yj ** (q - l)))

            cross = xj * yi - xi * yj
            nu_pq += cross * s_pq

            xj = xi
            yj = yi

        denom = ((p + q + 2) * (p + q + 1)
                 * i4_choose(p + q, p))
        if abs(denom) < 1e-300:
            return 0.0
        return nu_pq / denom

    def inertia_tensor(self) -> np.ndarray:
        """
        惯性张量 (2D):
          I_xx = ∫∫ y² dx dy = ν₀₂
          I_yy = ∫∫ x² dx dy = ν₂₀
          I_xy = -∫∫ xy dx dy = -ν₁₁
        """
        cx, cy = self.centroid()
        A = self.area()

        # 中心矩
        I_xx = self.moment(0, 2) - A * cy ** 2
        I_yy = self.moment(2, 0) - A * cx ** 2
        I_xy = -(self.moment(1, 1) - A * cx * cy)

        return np.array([[I_xx, I_xy], [I_xy, I_yy]])

    def boundary_integrals(self, p: int, q: int) -> np.ndarray:
        """
        各边对矩的贡献 (用于分析哪些边界最重要)

        Returns
        -------
        np.ndarray, shape (n,)
            各边贡献值.
        """
        from grid_integer_lib import i4_choose

        contributions = np.zeros(self.n, dtype=np.float64)
        for i in range(self.n):
            xi = self.vx[i]
            yi = self.vy[i]
            xj = self.vx[(i + 1) % self.n]
            yj = self.vy[(i + 1) % self.n]

            s_pq = 0.0
            for k in range(p + 1):
                for l in range(q + 1):
                    c1 = i4_choose(k + l, l)
                    c2 = i4_choose(p + q - k - l, q - l)
                    s_pq += (c1 * c2
                             * (xi ** k) * (xj ** (p - k))
                             * (yi ** l) * (yj ** (q - l)))

            contributions[i] = (xj * yi - xi * yj) * s_pq

        denom = ((p + q + 2) * (p + q + 1)
                 * i4_choose(p + q, p))
        if abs(denom) > 1e-300:
            contributions /= denom

        return contributions
