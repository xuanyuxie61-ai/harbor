"""
垂直积分高斯求积模块 (Vertical Quadrature Module)

集成种子项目:
- 665_legendre_rule: Gauss-Legendre 求积规则生成

用于大气科学中垂直方向上的质量加权积分, 如:
  - 气柱水汽含量 (Precipitable Water)
  - 对流有效位能的精细计算
  - 质量通量积分

核心公式:
  将气压坐标 p ∈ [p_top, p_sfc] 映射到标准区间 [-1,1]:
    p = (b-a)/2 * ξ + (b+a)/2,  ξ ∈ [-1,1]
  则 ∫_{p_top}^{p_sfc} f(p) dp = (b-a)/2 * Σ_i w_i * f(p(ξ_i))

  质量坐标下的积分:
    ∫_0^z ρ(z') dz' = p_sfc - p(z)  (静力平衡)
"""

import numpy as np
from typing import Tuple


def legendre_gauss_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 Golub-Welsch 算法生成 n 点 Gauss-Legendre 求积节点与权重
    (基于 665_legendre_rule 的核心算法).

    节点 x_i 和权重 w_i 满足:
      ∫_{-1}^{1} P(x) dx = Σ_i w_i * P(x_i)
    对任意次数 ≤ 2n-1 的多项式 P(x) 精确成立.
    """
    if n < 1:
        raise ValueError("Order must be >= 1")
    if n == 1:
        return np.array([0.0]), np.array([2.0])

    # Jacobi 矩阵构造 (Legendre 多项式的三项递推)
    # β_0 = 2, β_j = j^2 / (4j^2 - 1) for j >= 1
    # α_j = 0 (Legendre 多项式)
    j = np.arange(1.0, n)
    beta = np.sqrt(j**2 / (4.0 * j**2 - 1.0))

    # 对称三对角矩阵
    J = np.diag(beta, k=1) + np.diag(beta, k=-1)

    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eigh(J)

    # 节点 = 特征值, 权重 = 2 * (第一个分量)^2
    x = eigenvalues
    w = 2.0 * eigenvectors[0, :]**2

    return x, w


def gauss_legendre_quadrature(f, a: float, b: float, n: int = 64) -> float:
    """
    在区间 [a, b] 上使用 n 点 Gauss-Legendre 求积计算 ∫_a^b f(x) dx.
    """
    if not np.isfinite(a) or not np.isfinite(b) or a >= b:
        return 0.0
    x, w = legendre_gauss_nodes_weights(n)
    # 坐标变换: x ∈ [-1,1] -> t ∈ [a,b]
    t = 0.5 * (b - a) * x + 0.5 * (b + a)
    jac = 0.5 * (b - a)
    ft = np.array([f(ti) for ti in t])
    # 处理 inf/nan
    ft = np.where(np.isfinite(ft), ft, 0.0)
    return float(jac * np.sum(w * ft))


def precipitable_water(pressure_levels: np.ndarray, qv: np.ndarray,
                       p_sfc: float, T_sfc: float) -> float:
    """
    计算整层可降水量 (Precipitable Water, kg/m²).

    公式:
      PW = (1/g) ∫_{p_top}^{p_sfc} qv(p) dp

    使用 Gauss-Legendre 求积在气压坐标上进行精确积分.
    """
    g = 9.80665
    if len(pressure_levels) < 2:
        return 0.0
    p_top = max(100.0, pressure_levels[-1])
    p_bot = min(p_sfc, pressure_levels[0])

    # 构建 qv(p) 插值函数 (保护性线性插值)
    def qv_of_p(p: float) -> float:
        if p <= pressure_levels[-1]:
            return float(qv[-1])
        if p >= pressure_levels[0]:
            return float(qv[0])
        # 线性插值 (气压递减)
        idx = np.searchsorted(pressure_levels[::-1], p)
        idx = len(pressure_levels) - 1 - idx
        idx = max(0, min(idx, len(pressure_levels) - 2))
        dp = pressure_levels[idx] - pressure_levels[idx+1]
        if abs(dp) < 1e-6:
            return float(qv[idx])
        w = (p - pressure_levels[idx+1]) / dp
        return float(qv[idx+1] + w * (qv[idx] - qv[idx+1]))

    # 32点高斯求积已足够精确
    pw = gauss_legendre_quadrature(qv_of_p, p_top, p_bot, n=32) / g
    return max(0.0, pw)


def mass_weighted_integral(pressure_levels: np.ndarray, field: np.ndarray,
                           p_sfc: float) -> float:
    """
    气压坐标下的质量加权积分:
      I = (1/g) ∫ field(p) dp
    """
    g = 9.80665
    if len(pressure_levels) < 2:
        return 0.0
    p_top = max(100.0, pressure_levels[-1])
    p_bot = min(p_sfc, pressure_levels[0])

    def field_of_p(p: float) -> float:
        if p <= pressure_levels[-1]:
            return float(field[-1])
        if p >= pressure_levels[0]:
            return float(field[0])
        idx = np.searchsorted(pressure_levels[::-1], p)
        idx = len(pressure_levels) - 1 - idx
        idx = max(0, min(idx, len(pressure_levels) - 2))
        dp = pressure_levels[idx] - pressure_levels[idx+1]
        if abs(dp) < 1e-6:
            return float(field[idx])
        w = (p - pressure_levels[idx+1]) / dp
        return float(field[idx+1] + w * (field[idx] - field[idx+1]))

    return gauss_legendre_quadrature(field_of_p, p_top, p_bot, n=24) / g


def convective_inhibition_integral(pressure_levels: np.ndarray,
                                   buoyancy: np.ndarray,
                                   p_sfc: float) -> float:
    """
    使用高精度 Gauss-Legendre 求积计算 CIN.
    仅对负浮力区积分.
    """
    g = 9.80665
    Rd = 287.05

    if len(pressure_levels) < 2:
        return 0.0

    p_top = max(100.0, pressure_levels[-1])
    p_bot = min(p_sfc, pressure_levels[0])

    def neg_buoyancy(p: float) -> float:
        if p <= pressure_levels[-1]:
            return min(0.0, float(buoyancy[-1]))
        if p >= pressure_levels[0]:
            return min(0.0, float(buoyancy[0]))
        idx = np.searchsorted(pressure_levels[::-1], p)
        idx = len(pressure_levels) - 1 - idx
        idx = max(0, min(idx, len(pressure_levels) - 2))
        dp = pressure_levels[idx] - pressure_levels[idx+1]
        if abs(dp) < 1e-6:
            return min(0.0, float(buoyancy[idx]))
        w = (p - pressure_levels[idx+1]) / dp
        b = float(buoyancy[idx+1] + w * (buoyancy[idx] - buoyancy[idx+1]))
        return min(0.0, b)

    return gauss_legendre_quadrature(neg_buoyancy, p_top, p_bot, n=48)
