"""
potential_deformed.py
=====================
变形核势场计算与二维插值模块

本模块实现：
1. 球形 Woods-Saxon 势及其导数
2. 四极/八极形变 Woods-Saxon 势：
   V(r, θ, φ) = V₀ / [1 + exp((r - R(θ, φ)) / a)]
3. 自旋-轨道耦合势：
   V_{so}(r) = -V_{so}^0 (ħ / m_π c)² · (1/r) · dV_{WS}/dr · (l·s)
4. 二维插值用于势能在形变参数平面 (β₂, γ) 上的计算

数学公式：
-  Woods-Saxon 势：V_{WS}(r) = V₀ / {1 + exp[(r - R) / a]}
-  自旋-轨道势：V_{so}(r) = λ V_{so}^0 r_{so}² (1/r) d/dr [1 / (1 + exp[(r - R_{so})/a_{so}])]
-  离心势：V_{cent}(r) = ħ² l(l+1) / (2M r²)
"""

import numpy as np
from math import exp, sqrt, pi, sin, cos


# 物理常数（自然单位制近似）
HBARC = 197.3269804  # MeV·fm
M_NUCLEON = 939.0    # MeV/c²


def woods_saxon_potential(r, V0, R, a):
    """
    标准 Woods-Saxon 势。

    V(r) = V₀ / [1 + exp((r - R) / a)]

    参数
    ----
    r : float 或 ndarray
        径向距离 (fm)
    V0 : float
        势阱深度 (MeV)
    R : float
        势半径 (fm)
    a : float
        表面弥散参数 (fm)

    返回
    ----
    float 或 ndarray
        势值 (MeV)
    """
    r = np.asarray(r, dtype=float)
    # 避免溢出
    arg = (r - R) / a
    # 对于大正 arg，exp → ∞，势 → 0
    # 对于大负 arg，exp → 0，势 → V0
    V = np.zeros_like(r)
    mask_pos = arg > 700
    mask_neg = arg < -700
    mask_mid = ~(mask_pos | mask_neg)
    V[mask_pos] = 0.0
    V[mask_neg] = V0
    V[mask_mid] = V0 / (1.0 + np.exp(arg[mask_mid]))
    return V


def woods_saxon_derivative(r, V0, R, a):
    """
    Woods-Saxon 势对 r 的导数。

    dV/dr = - (V₀ / a) · exp((r-R)/a) / [1 + exp((r-R)/a)]²
    """
    r = np.asarray(r, dtype=float)
    arg = (r - R) / a
    dV = np.zeros_like(r)
    mask_pos = arg > 700
    mask_neg = arg < -700
    mask_mid = ~(mask_pos | mask_neg)
    dV[mask_pos] = 0.0
    dV[mask_neg] = 0.0
    earg = np.exp(arg[mask_mid])
    dV[mask_mid] = -(V0 / a) * earg / ((1.0 + earg) ** 2)
    return dV


def spin_orbit_potential(r, Vso0, Rso, aso, l, s, kappa=-0.5):
    """
    自旋-轨道耦合势。

    V_{so}(r) = - V_{so}^0 · (ħ / m_π c)² · (1/r) · dV_{WS}/dr · ⟨l·s⟩

    其中 ⟨l·s⟩ = [j(j+1) - l(l+1) - s(s+1)] / 2

    参数
    ----
    r : float
        径向坐标 (fm)
    Vso0 : float
        自旋-轨道势强度 (MeV)
    Rso, aso : float
        自旋-轨道势半径与弥散参数 (fm)
    l : int
        轨道角动量量子数
    s : float
        自旋量子数 (1/2)
    kappa : float
        相对论量子数，κ = -(j + 1/2) 对 j = l + 1/2，κ = j + 1/2 对 j = l - 1/2

    返回
    ----
    float
        自旋-轨道势值 (MeV)
    """
    if r < 1e-6:
        return 0.0
    dVdr = woods_saxon_derivative(r, 1.0, Rso, aso)
    # ⟨l·s⟩ = -ħ² κ (κ + 1) 的关系，这里简化为标量形式
    # 实际上 j(j+1) - l(l+1) - s(s+1) = 2⟨l·s⟩
    j_plus = l + s
    j_minus = l - s
    # 取平均耦合
    ls_coupling = 0.5 * (j_plus * (j_plus + 1) - l * (l + 1) - s * (s + 1))
    # 自然单位制下 (ħ/mc)² ~ 2.0 fm²（近似）
    hbar_over_mc_sq = 2.0
    return -Vso0 * hbar_over_mc_sq * (1.0 / r) * dVdr * ls_coupling


def deformed_woods_saxon(r, theta, phi, V0, R0, a, beta2, gamma,
                         beta3=0.0, beta4=0.0):
    """
    变形 Woods-Saxon 势。

    核表面形变：
    R(θ, φ) = R₀ [1 + β₂ cosγ Y₂₀ + β₂ sinγ (Y₂₂ + Y₂₋₂)/√2
                  + β₃ Y₃₀ + β₄ Y₄₀]

    参数
    ----
    r, theta, phi : float
        球坐标 (fm, rad, rad)
    V0, R0, a : float
        势参数
    beta2, gamma : float
        四极形变参数
    beta3, beta4 : float
        八极、十六极形变参数

    返回
    ----
    float
        势值 (MeV)
    """
    # 计算球谐函数 Y_{λ0} 的近似值（轴对称简化）
    # Y_20 = √(5/16π) (3cos²θ - 1)
    Y20 = sqrt(5.0 / (16.0 * pi)) * (3.0 * cos(theta) ** 2 - 1.0)
    # Y_22 实部
    Y22_real = sqrt(15.0 / (32.0 * pi)) * sin(theta) ** 2 * cos(2.0 * phi)
    # Y_30
    Y30 = sqrt(7.0 / (16.0 * pi)) * (5.0 * cos(theta) ** 3 - 3.0 * cos(theta))
    # Y_40
    Y40 = sqrt(9.0 / (256.0 * pi)) * (35.0 * cos(theta) ** 4
                                        - 30.0 * cos(theta) ** 2 + 3.0)

    R_def = R0 * (1.0 + beta2 * (cos(gamma) * Y20 + sin(gamma) * Y22_real)
                  + beta3 * Y30 + beta4 * Y40)

    return woods_saxon_potential(r, V0, R_def, a)


def bilinear_interpolate_2d(x, y, x_grid, y_grid, Z):
    """
    二维双线性插值。

    在势能量曲面 (β₂, γ) 平面上进行插值计算。
    基于 test_interp_2d 的核心思想：利用网格数据通过局部线性近似得到任意点值。

    参数
    ----
    x, y : float
        目标坐标
    x_grid, y_grid : 1D ndarray
        网格坐标
    Z : 2D ndarray
        网格上的函数值，shape 为 (len(y_grid), len(x_grid))

    返回
    ----
    float
        插值结果
    """
    # 找到包围 (x, y) 的矩形索引
    if x < x_grid[0] or x > x_grid[-1] or y < y_grid[0] or y > y_grid[-1]:
        raise ValueError("插值点超出网格范围")

    ix = np.searchsorted(x_grid, x) - 1
    iy = np.searchsorted(y_grid, y) - 1
    ix = max(0, min(ix, len(x_grid) - 2))
    iy = max(0, min(iy, len(y_grid) - 2))

    x0, x1 = x_grid[ix], x_grid[ix + 1]
    y0, y1 = y_grid[iy], y_grid[iy + 1]

    dx = (x - x0) / (x1 - x0) if x1 != x0 else 0.0
    dy = (y - y0) / (y1 - y0) if y1 != y0 else 0.0

    z00 = Z[iy, ix]
    z10 = Z[iy, ix + 1]
    z01 = Z[iy + 1, ix]
    z11 = Z[iy + 1, ix + 1]

    return (z00 * (1 - dx) * (1 - dy) +
            z10 * dx * (1 - dy) +
            z01 * (1 - dx) * dy +
            z11 * dx * dy)


def build_potential_energy_surface(n_beta, n_gamma, V0, R0, a, l_max=6):
    """
    构建 (β₂, γ) 平面上的单粒子势能量曲面。

    对每个 (β₂, γ) 网格点，计算前几个壳层单粒子能级的平均值，
    作为该形变下核势能的近似。

    返回
    ----
    beta_grid, gamma_grid : ndarray
        网格坐标
    energy_surface : ndarray, shape (n_gamma, n_beta)
        能量曲面
    """
    beta_grid = np.linspace(-0.3, 0.5, n_beta)
    gamma_grid = np.linspace(0.0, pi / 3.0, n_gamma)
    energy_surface = np.zeros((n_gamma, n_beta))

    for i, gamma in enumerate(gamma_grid):
        for j, beta2 in enumerate(beta_grid):
            # 近似单粒子能量：势阱深度 + 形变修正
            # 简谐振子近似：E_n = ħω (2n + l + 3/2)
            # 形变修正：ΔE = -β₂² / (4π) · ħω
            hbar_omega = 41.0 * (A_eff := 100) ** (-1.0 / 3.0)  # MeV
            deform_correction = -beta2 ** 2 / (4.0 * pi) * hbar_omega
            # 离心项平均
            avg_centrifugal = 0.0
            for l in range(l_max + 1):
                avg_centrifugal += HBARC ** 2 * l * (l + 1) / (2.0 * M_NUCLEON * R0 ** 2)
            avg_centrifugal /= (l_max + 1)

            energy_surface[i, j] = V0 + deform_correction + avg_centrifugal

    return beta_grid, gamma_grid, energy_surface


def total_single_particle_potential(r, theta, phi, l, j, params):
    """
    计算总单粒子势：中心势 + 自旋-轨道耦合 + 离心势。

    参数
    ----
    r, theta, phi : float
        球坐标
    l : int
        轨道角动量
    j : float
        总角动量
    params : dict
        包含 V0, R0, a, Vso0, Rso, aso, beta2, gamma 等

    返回
    ----
    float
        总势值 (MeV)
    """
    V0 = params.get('V0', -50.0)
    R0 = params.get('R0', 5.0)
    a = params.get('a', 0.65)
    Vso0 = params.get('Vso0', 12.0)
    Rso = params.get('Rso', R0)
    aso = params.get('aso', a)
    beta2 = params.get('beta2', 0.0)
    gamma = params.get('gamma', 0.0)
    s = 0.5

    V_central = deformed_woods_saxon(r, theta, phi, V0, R0, a, beta2, gamma)
    V_so = spin_orbit_potential(r, Vso0, Rso, aso, l, s)
    # TODO [Hole 1]: 填入离心势 V_cent(r) 的物理公式
    # 离心势来源于角动量 barrier：V_cent = ħ² l(l+1) / (2M r²)
    # 注意 r → 0 时的奇异性处理
    V_cent = 0.0  # 占位符，需要正确实现

    return V_central + V_so + V_cent
