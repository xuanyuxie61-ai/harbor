# -*- coding: utf-8 -*-
"""
opacity_engine.py
======================================================================
不透明度核计算 —— Minkowski 卷积 + 金字塔积分

物理背景:
    大气不透明度 (opacity) 由多种吸收源贡献:
        kappa(lambda, z) = sum_s kappa_s(lambda, z)          (1)
    其中每种源 s 包括:
        - 分子吸收 (H2O, CO2, CH4, CO, TiO, VO, ...)
        - 原子吸收 (Na, K, Fe, ...)
        - 连续吸收 (H-, Rayleigh, CIA)
        - 云/霾散射

    每条吸收线的线型函数 (line profile) 为 Voigt 轮廓:
        phi(nu) = integral G(nu - nu') * L(nu') dnu'
    其中 G 为 Doppler (Gauss) 轮廓, L 为 Lorentz 轮廓。
    这等价于两个凸集 (Gauss 核 + Lorentz 核) 的 Minkowski 卷积
    (来自 887_polygon_minkowski)。

    多层大气的不透明度积分采用金字塔求积法则
    (来自 931_pyramid_felippa_rule), 对光学深度维度进行
    高精度积分。

数学公式:
    Voigt 函数 H(a, u):
        H(a, u) = (a/pi) integral_{-inf}^{inf}
                      exp(-y^2) / ((u - y)^2 + a^2) dy       (2)
    其中 a = Gamma_L / (4 pi Delta_nu_D) 为阻尼参数,
          u = (nu - nu_0) / Delta_nu_D 为归一化频率偏移。

    线吸收截面:
        sigma(nu) = (pi e^2 / m_e c) * f * phi(nu)            (3)
    其中 f 为振子强度。

依赖: numpy, scipy
======================================================================
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from scipy.special import wofz


# ============================================================
# Minkowski 卷积 (移植自 887_polygon_minkowski)
# ============================================================
def polygon_vertices_to_minkowski(v: np.ndarray) -> np.ndarray:
    """
    从多边形顶点转换为 Minkowski 表示 (来自 887_polygon_minkowski)。

    输入: v(n, 2) -- 多边形顶点
    输出: u(n, 2) -- Minkowski 表示 (边法向量)

    物理应用: 吸收线型的凸包络表示。
    每条谱线的轮廓可视为频域上的凸多边形,
    多线叠加等价于 Minkowski 和。
    """
    nv = len(v)
    nu = nv
    u = np.zeros((nu, 2), dtype=np.float64)

    for iu in range(nu):
        iv0 = iu
        iv1 = (iu + 1) % nv
        u[iu, 0] = v[iv1, 1] - v[iv0, 1]
        u[iu, 1] = -(v[iv1, 0] - v[iv0, 0])
    return u


def polygon_minkowski_to_vertices(u: np.ndarray) -> np.ndarray:
    """
    从 Minkowski 表示重建顶点 (来自 887_polygon_minkowski)。
    """
    nu = len(u)
    nv = nu + 1
    v = np.zeros((nv, 2), dtype=np.float64)
    for iv in range(1, nv):
        v[iv, 0] = v[iv - 1, 0] - u[iv - 1, 1]
        v[iv, 1] = v[iv - 1, 1] + u[iv - 1, 0]
    return v


def minkowski_convolution_profile(
    freq_grid: np.ndarray,
    profile_a: np.ndarray,
    profile_b: np.ndarray,
) -> np.ndarray:
    """
    两个线型函数的 Minkowski 卷积 (Voigt 轮廓的离散版本)。

    物理意义:
        Gauss (Doppler) 轮廓: G(nu) ~ exp(-((nu-nu0)/Delta_nu_D)^2)
        Lorentz (pressure) 轮廓: L(nu) ~ Gamma / ((nu-nu0)^2 + Gamma^2)
        Voigt 轮廓 V = G (*) L (卷积)
    """
    result = np.convolve(profile_a, profile_b, mode="same")
    df = freq_grid[1] - freq_grid[0] if len(freq_grid) > 1 else 1.0
    result = result * df
    norm = np.sum(result) * df
    if norm > 1.0e-30:
        result = result / norm
    return result


# ============================================================
# Voigt 函数与线型
# ============================================================
def voigt_profile(a: float, u: np.ndarray) -> np.ndarray:
    """
    Voigt 函数 H(a, u) 的数值计算 (Faddeeva 函数实现)。

    H(a, u) = Re[w(z)], z = u + i a
    w(z) = exp(-z^2) * erfc(-i z)
    """
    z = u + 1j * a
    return np.real(wofz(z))


def doppler_width(nu_0: float, temperature: float, mass_kg: float) -> float:
    """
    Doppler 半宽:
        Delta_nu_D = (nu_0 / c) * sqrt(2 k_B T / m)
    """
    return (nu_0 / 2.99792458e8) * np.sqrt(
        2.0 * 1.380649e-23 * temperature / mass_kg
    )


def lorentz_width(pressure_pa: float, temperature: float, gamma_0: float = 0.1) -> float:
    """
    Lorentz 半宽 (压力展宽):
        Gamma_L = gamma_0 * (P / P_0) * (T_0 / T)^0.7
    """
    p_0 = 1.01325e5
    t_0 = 296.0
    return gamma_0 * (pressure_pa / p_0) * (t_0 / max(temperature, 1.0)) ** 0.7


def compute_line_opacity(
    freq_grid: np.ndarray,
    nu_0: float,
    temperature: float,
    pressure_pa: float,
    line_strength: float,
    mass_kg: float,
    gamma_0: float = 0.1,
) -> np.ndarray:
    """
    计算单条吸收线的不透明度谱。
    sigma(nu) = S * V(nu - nu_0, a, Delta_nu_D) / Delta_nu_D
    """
    dnu_D = doppler_width(nu_0, temperature, mass_kg)
    gamma_L = lorentz_width(pressure_pa, temperature, gamma_0)

    dnu_D = max(dnu_D, 1.0e-10)
    a = gamma_L / dnu_D
    u = (freq_grid - nu_0) / dnu_D

    V = voigt_profile(a, u)
    sigma = line_strength * V / dnu_D
    sigma = np.maximum(sigma, 0.0)
    return sigma


# ============================================================
# 金字塔积分法则 (移植自 931_pyramid_felippa_rule)
# ============================================================
def pyramid_unit_o05() -> Tuple[np.ndarray, np.ndarray]:
    """
    5 点 5 阶金字塔积分法则 (来自 931_pyramid_felippa_rule)。

    积分区域:
        -(1-z) <= x <= 1-z
        -(1-z) <= y <= 1-z
                 0 <= z <= 1

    体积 = 4/3 (来自 pyramid_unit_volume)
    """
    w = np.array([0.2109375, 0.2109375, 0.2109375, 0.2109375, 0.15625])
    xyz = np.array([
        [-0.48686449556014766, -0.48686449556014766, 0.16666666666666666],
        [0.48686449556014766, -0.48686449556014766, 0.16666666666666666],
        [0.48686449556014766, 0.48686449556014766, 0.16666666666666666],
        [-0.48686449556014766, 0.48686449556014766, 0.16666666666666666],
        [0.0, 0.0, 0.7],
    ])
    return w, xyz


def pyramid_unit_volume() -> float:
    """单位金字塔体积: 4/3"""
    return 4.0 / 3.0


def integrate_opacity_over_layer(
    opacity_func,
    z_bottom: float,
    z_top: float,
    wavelength_m: float,
) -> float:
    """
    使用金字塔积分法则计算单层内的积分不透明度。

    物理意义: 大气层 j 到 j+1 之间的柱密度积分:
        tau_j = integral_{z_j}^{z_{j+1}} kappa(lambda, z) rho(z) dz
    """
    w, xyz = pyramid_unit_o05()
    vol = pyramid_unit_volume()

    dz = z_top - z_bottom
    z_mid = 0.5 * (z_top + z_bottom)

    integral = 0.0
    for k in range(len(w)):
        z_local = z_mid + 0.5 * dz * xyz[k, 2]
        kappa = opacity_func(wavelength_m, z_local)
        integral += w[k] * kappa

    integral = integral * vol * (dz / 2.0)
    return abs(integral)


# ============================================================
# 完整大气不透明度计算
# ============================================================
class OpacityDatabase:
    """
    简化的不透明度数据库。
    包含主要分子吸收线的参数 (HITRAN 格式简化版)。
    """

    # 主要吸收线 [分子名, 中心频率 Hz, 振子强度, 质量 kg, 阻尼参数]
    LINES_H2O = [
        (1.0e13, 1.0e-20, 2.99e-26, 0.08),
        (3.0e13, 5.0e-21, 2.99e-26, 0.08),
        (6.0e13, 2.0e-21, 2.99e-26, 0.08),
    ]
    LINES_CO = [
        (2.0e13, 8.0e-21, 4.65e-26, 0.07),
        (6.4e13, 3.0e-21, 4.65e-26, 0.07),
    ]
    LINES_NA = [
        (5.08e14, 6.0e-16, 3.82e-26, 0.10),  # Na D lines
        (5.09e14, 3.0e-16, 3.82e-26, 0.10),
    ]
    LINES_K = [
        (3.89e14, 3.5e-16, 6.49e-26, 0.09),  # K resonance
        (7.69e14, 1.5e-16, 6.49e-26, 0.09),
    ]

    def __init__(self, molecules: Optional[List[str]] = None):
        if molecules is None:
            molecules = ["H2O", "CO", "Na", "K"]
        # Store lines as (molecule_name, line_params) tuples for direct lookup
        self.lines = []
        mol_line_map = {
            "H2O": self.LINES_H2O,
            "CO": self.LINES_CO,
            "Na": self.LINES_NA,
            "K": self.LINES_K,
        }
        for mol in molecules:
            if mol in mol_line_map:
                for line in mol_line_map[mol]:
                    self.lines.append((mol, line))

    def compute_opacity(
        self,
        freq_grid: np.ndarray,
        temperature: float,
        pressure_pa: float,
        abundances: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        计算给定温度、气压下的总不透明度。
        """
        if abundances is None:
            abundances = {"H2O": 1.0e-4, "CO": 1.0e-4, "Na": 1.0e-9, "K": 1.0e-9}

        kappa_total = np.zeros_like(freq_grid, dtype=np.float64)

        for mol_name, line in self.lines:
            nu_0, S_0, mass, gamma_0 = line
            abund = abundances.get(mol_name, 1.0e-6)

            S = S_0 * abund
            sigma = compute_line_opacity(
                freq_grid, nu_0, temperature, pressure_pa, S, mass, gamma_0
            )
            kappa_total += sigma

        # Rayleigh 散射 (H2 + He)
        rayleigh = 1.0e-29 * (freq_grid / 5.0e14) ** 4 * (pressure_pa / 1.0e5)
        kappa_total += rayleigh

        return kappa_total


def compute_optical_depth_profile(
    wavelength_m: float,
    z_grid: np.ndarray,
    p_grid: np.ndarray,
    t_grid: np.ndarray,
    rho_grid: np.ndarray,
    opacity_db: OpacityDatabase,
) -> np.ndarray:
    """
    计算给定波长的光学深度剖面 tau(z)。
        tau(z) = integral_z^inf kappa(lambda, T(z'), P(z')) rho(z') dz'
    """
    n = len(z_grid)
    tau = np.zeros(n, dtype=np.float64)

    for i in range(n - 2, -1, -1):
        freq_grid = np.array([2.99792458e8 / wavelength_m])
        kappa = opacity_db.compute_opacity(
            freq_grid, t_grid[i], p_grid[i]
        )[0]
        dz = z_grid[i + 1] - z_grid[i]
        tau[i] = tau[i + 1] + kappa * rho_grid[i] * dz

    return tau
