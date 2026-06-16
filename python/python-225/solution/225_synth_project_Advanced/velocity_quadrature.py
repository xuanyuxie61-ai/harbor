# -*- coding: utf-8 -*-
"""
velocity_quadrature.py
======================

速度空间积分求积模块

本模块为暗物质反冲率的速度空间积分提供高精度数值求积方案。
对应种子项目:
  - 302_disk01_rule: 单位圆盘上 Legendre-Gauss 求积
  - 559_hypercube_integrals: 超立方体单项式积分

科学公式
--------
反冲率积分:
  R = (ρ_χ N_T σ_n A²) / (2 m_χ μ_n²)
      * ∫∫∫ f(v) / v * Θ(v - v_min) d³v

在球坐标下:
  R ∝ ∫_0^∞ dv v ∫_{-1}^{+1} d(cos θ) ∫_0^{2π} dφ
          * f(v, θ, φ) * Θ(v cos θ - v_min)

简化为 1D 积分:
  R ∝ 2π ∫_{v_min}^{∞} dv v² f(v) / v  (各向同性时)

求积方案:
  1. Gauss-Legendre (径向)
  2. Gauss-Chebyshev (角向)
  3. 复合 Simpson
  4. 自适应 Gauss-Kronrod (高精度需求)
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Callable, Optional


# ---------------------------------------------------------------------------
# 第一部分: Gauss-Legendre 求积 (源自 302_disk01_rule)
# ---------------------------------------------------------------------------

def gauss_legendre_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 Gauss-Legendre 求积节点和权重在 [-1, 1] 上。

    使用 Golub-Welsch 算法 (三对角矩阵特征值):
      J = tridiag(b_i, a_i, b_i)
      a_i = 0
      b_i = i / sqrt(4 i² - 1)
    特征值 = 节点, 特征向量第一分量平方 * 2 = 权重
    """
    if n < 1:
        raise ValueError(f"节点数 n 必须 >= 1, 得到 n={n}")
    if n == 1:
        return np.array([0.0]), np.array([2.0])

    i = np.arange(1, n, dtype=np.float64)
    b = i / np.sqrt(4.0 * i**2 - 1.0)
    # 三对角矩阵
    J = np.diag(b, -1) + np.diag(b, 1)
    eigvals, eigvecs = np.linalg.eigh(J)
    nodes = eigvals
    weights = 2.0 * eigvecs[0, :]**2
    return nodes, weights


def shift_legendre_to_interval(
    nodes: np.ndarray,
    weights: np.ndarray,
    a: float,
    b: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """将 [-1, 1] 上的 Gauss-Legendre 规则映射到 [a, b]。"""
    mid = 0.5 * (b + a)
    half = 0.5 * (b - a)
    new_nodes = mid + half * nodes
    new_weights = half * weights
    return new_nodes, new_weights


def disk_quadrature(
    n_r: int,
    n_theta: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    单位圆盘上的求积规则 (源自 302_disk01_rule 核心算法)。

    Q(f) = π Σ_j Σ_i W_i * F(R_i cos Θ_j, R_i sin Θ_j)

    径向: 使用 Gauss-Legendre on [0, 1] 映射后取 √r
    角向: 均匀分点

    用于速度空间的速度-角度积分
    """
    # 径向: [-1, 1] 上的 GL 规则
    xr, wr = gauss_legendre_nodes_weights(n_r)
    # 映射到 [0, 1]
    xr = (xr + 1.0) / 2.0
    wr = wr / 2.0
    # 取平方根使权重适应 r dr 积分
    r_nodes = np.sqrt(xr)
    r_weights = wr / n_theta

    # 角向: 均匀分布
    theta_nodes = np.array([
        2.0 * np.pi * j / n_theta for j in range(n_theta)
    ])

    return r_nodes, r_weights, theta_nodes, np.ones(n_theta) / n_theta


# ---------------------------------------------------------------------------
# 第二部分: 超立方体积分 (源自 559_hypercube_integrals)
# ---------------------------------------------------------------------------

def hypercube_monomial_integral(e: np.ndarray) -> float:
    """
    超立方体 [0,1]^M 上的单项式积分 (源自 559_hypercube_integrals):

    ∫_[0,1]^M x_1^{e_1} x_2^{e_2} ... x_M^{e_M} dx
      = Π_{i=1}^{M} 1 / (e_i + 1)

    用于: 相空间体积的解析计算, 用于归一化检验
    """
    if np.any(e < 0):
        raise ValueError("所有指数必须非负")
    return float(np.prod(1.0 / (e + 1)))


def hypercube_quadrature_product(
    n_1d: int,
    M: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    M 维超立方体 [0,1]^M 上的张量积 Gauss-Legendre 求积。

    返回 (nodes, weights):
      nodes: shape (n_1d^M, M)
      weights: shape (n_1d^M,)
    """
    nodes_1d, weights_1d = gauss_legendre_nodes_weights(n_1d)
    nodes_1d, weights_1d = shift_legendre_to_interval(nodes_1d, weights_1d, 0.0, 1.0)

    # 张量积
    grids = [nodes_1d] * M
    weight_grids = [weights_1d] * M

    nodes_M = np.array(np.meshgrid(*grids, indexing='ij')).reshape(M, -1).T
    weights_M = np.array(np.meshgrid(*weight_grids, indexing='ij')).reshape(-1)
    for i in range(1, M):
        weights_M = None  # 重新计算
    # 正确实现
    from itertools import product as iter_product
    indices = list(iter_product(range(n_1d), repeat=M))
    nodes_M = np.array([[nodes_1d[i[d]] for d in range(M)] for i in indices])
    weights_M = np.array([
        np.prod([weights_1d[i[d]] for d in range(M)]) for i in indices
    ])
    return nodes_M, weights_M


# ---------------------------------------------------------------------------
# 第三部分: DM 速度空间专用积分
# ---------------------------------------------------------------------------

class VelocityIntegral:
    """
    暗物质速度空间积分器。

    提供多种积分策略计算:
      I(v_min) = ∫_{v>v_min} (f(v) / |v|) d³v

    以及加权版本:
      I_n(v_min) = ∫_{v>v_min} |v|^n f(v) d³v
    """

    def __init__(
        self,
        f_velocity: Callable,
        v_esc: float = 544.0,
        v_0: float = 220.0,
    ):
        self.f_velocity = f_velocity
        self.v_esc = v_esc
        self.v_0 = v_0

    def integrate_eta_1d(
        self,
        v_min: float,
        v_earth_mag: float,
        n_quad: int = 100,
    ) -> float:
        """
        1D Gauss-Legendre 积分计算 η(v_min) (各向同性近似):

          η(v_min) ≈ 2π ∫_{v_min}^{v_esc+v_E} (v - v_min + ...) f(v) dv

        精确公式 (Lewin & Smith):
          η(v_min) = (1/(2β N_esc)) * [erf(x_-) - erf(x_+)] 等
        此处用于数值验证。
        """
        if v_min >= self.v_esc + v_earth_mag:
            return 0.0
        v_lower = max(v_min, 0.0)
        v_upper = self.v_esc + v_earth_mag + 10 * self.v_0
        v_upper = min(v_upper, 2000.0)  # 截断
        if v_upper <= v_lower:
            return 0.0

        nodes, weights = gauss_legendre_nodes_weights(n_quad)
        nodes, weights = shift_legendre_to_interval(nodes, weights, v_lower, v_upper)

        integral = 0.0
        for i in range(len(nodes)):
            v = nodes[i]
            if v <= 0:
                continue
            # 各向同性 Maxwellian
            f_val = np.exp(-(v / self.v_0)**2) / (
                (np.pi**1.5) * self.v_0**3
            )
            integrand = f_val / v * 4.0 * np.pi * v**2
            integral += weights[i] * integrand

        return integral

    def integrate_rate_shell(
        self,
        v_min: float,
        v_earth: np.ndarray,
        n_r: int = 50,
        n_theta: int = 30,
    ) -> float:
        """
        使用 disk_quadrature (球壳) 积分计算率积分。

        在球坐标 (v, θ, φ) 下:
          I = ∫_{v>v_min} (f(v+v_E)/v) v² sin θ dθ dφ dv
        """
        r_nodes, r_weights, theta_nodes, _ = disk_quadrature(n_r, n_theta)

        # 映射 r ∈ [0, 1] → v ∈ [0, v_max]
        v_max = self.v_esc + np.linalg.norm(v_earth) + 100.0
        v_nodes = r_nodes * v_max
        v_weights = r_weights * v_max  # Jacobian

        integral = 0.0
        v_earth_mag = np.linalg.norm(v_earth)
        for i, v in enumerate(v_nodes):
            if v <= 0:
                continue
            for j, theta in enumerate(theta_nodes):
                # 假设各向同性: f 仅依赖 |v + v_E|
                # 简化: v_E 沿 z 轴
                v_gal = np.sqrt(v**2 + v_earth_mag**2 + 2 * v * v_earth_mag * np.cos(theta))
                if v_gal >= self.v_esc:
                    continue
                f_val = np.exp(-(v_gal / self.v_0)**2) / (
                    (np.pi**1.5) * self.v_0**3
                )
                # v > v_min 条件: 仅当 v cos θ > v_min 的投影
                v_proj = v * np.cos(theta)
                if v < v_min:
                    # 只有部分角度满足
                    if v_proj < v_min:
                        continue
                integrand = f_val / v * v**2 * np.sin(theta)
                integral += v_weights[i] * (2 * np.pi / n_theta) * integrand

        return integral

    def integrate_hypercube_phase(
        self,
        integrand: Callable,
        M: int = 3,
        n_1d: int = 10,
    ) -> float:
        """
        超立方体求积用于相空间体积元积分 (源自 559_hypercube_integrals)。

        ∫_[0,1]^M f(u) du ≈ Σ_i w_i f(x_i)
        """
        nodes_M, weights_M = hypercube_quadrature_product(n_1d, M)
        result = 0.0
        for i in range(len(nodes_M)):
            result += weights_M[i] * integrand(nodes_M[i])
        return result


# ---------------------------------------------------------------------------
# 第四部分: 复合 Simpson 求积
# ---------------------------------------------------------------------------

def simpson_1d(
    f_values: np.ndarray,
    h: float,
) -> float:
    """
    复合 Simpson 1/3 法则。

    ∫_a^b f(x) dx ≈ (h/3) * [f_0 + 4f_1 + 2f_2 + 4f_3 + ... + 4f_{n-1} + f_n]

    要求 n 为偶数, 否则使用 Simpson 3/8 在末端。
    """
    n = len(f_values) - 1
    if n < 2:
        raise ValueError(f"至少需要 3 个点, 得到 {n + 1}")
    if n % 2 == 0:
        # 复合 Simpson 1/3
        S = f_values[0] + f_values[-1]
        S += 4 * np.sum(f_values[1:-1:2])
        S += 2 * np.sum(f_values[2:-2:2])
        return S * h / 3.0
    else:
        # 前 n-3 个用 Simpson 1/3, 最后 3 个用 3/8
        if n >= 4:
            S13 = f_values[0] + f_values[-4]
            S13 += 4 * np.sum(f_values[1:-4:2])
            S13 += 2 * np.sum(f_values[2:-4:2])
            integral = S13 * h / 3.0
            # Simpson 3/8 在最后三个子区间
            integral += (3 * h / 8) * (
                f_values[-4] + 3*f_values[-3] + 3*f_values[-2] + f_values[-1]
            )
            return integral
        else:
            # n = 3: 直接用 Simpson 3/8
            return (3 * h / 8) * (
                f_values[0] + 3*f_values[1] + 3*f_values[2] + f_values[3]
            )
