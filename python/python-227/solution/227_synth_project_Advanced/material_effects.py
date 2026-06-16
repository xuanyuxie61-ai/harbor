"""
material_effects.py — 物质效应：能量损失与多次散射
===================================================

本模块实现带电粒子穿过探测器材料时的物理效应:

    1. 电离能量损失 (Bethe-Bloch 公式)
    2. 多次库仑散射 (Highland 公式)
    3. 辐射能量损失 (Bremsstrahlung, 对轻粒子)
    4. 能量损失涨落 (Landau-Vavilov 分布)

核心公式:

[Bethe-Bloch 电离能量损失]
    -dE/dx = K·z²·(Z/A)·(ρ/β²)·[½ln(2m_e·c²·β²γ²·T_max/I²) - β² - δ/2]

    其中:
        K = 4π·N_A·r_e²·m_e·c² = 0.307075 MeV·cm²/g
        z = 入射粒子电荷数
        Z, A = 材料原子序数和原子量
        ρ = 材料密度 [g/cm³]
        β = v/c
        γ = Lorentz 因子
        I = 平均激发能 [eV]
        T_max = 最大单次碰撞能量转移
        δ = 密度效应修正

[Highland 多次散射]
    θ₀ = (13.6 MeV / βcp) · z · √(x/X₀) · [1 + 0.038·ln(x/X₀)]

    散射角分布近似为 Gaussian:
        f(θ) = (1/√(2π)θ₀) · exp(-θ²/(2θ₀²))

[辐射能量损失 (Bremsstrahlung)]
    -dE/dx|_rad = E / X₀  (对电子/正电子)
    临界能量 E_c: 当辐射损失 = 电离损失时
        E_c ≈ 610 MeV / (Z + 1.24)
"""

import math
from typing import Tuple, Optional


# 常数
K_BETHE_BLOCH = 0.307075   # MeV·cm²/g
ME_ELECTRON = 0.510999     # MeV/c² (电子质量)


class MaterialEffects:
    """
    物质效应计算器

    处理带电粒子在探测器材料中的:
    - 平均能量损失 (Bethe-Bloch)
    - 多次散射角 (Highland)
    - 能量损失涨落 (Landau 最概然值)
    - 辐射修正

    Parameters
    ----------
    particle_charge : int
        粒子电荷数 z (±1, ±2, ...)
    particle_mass_mev : float
        粒子质量 [MeV/c²]
    """

    def __init__(self, particle_charge=1, particle_mass_mev=139.57):
        self.z_particle = abs(particle_charge)
        self.mass = particle_mass_mev

    # ============================================================
    # Bethe-Bloch 电离能量损失
    # ============================================================
    def bethe_bloch(self, momentum_mev, material_Z, material_A,
                    material_density, ionization_eV):
        """
        Bethe-Bloch 平均电离能量损失率

        -dE/dx = K·z²·(Z/A)·(ρ/β²)·[½ln(2m_e·c²·β²γ²·T_max/I²) - β² - δ/2]

        Parameters
        ----------
        momentum_mev : float
            粒子动量 [MeV/c]
        material_Z : int
            材料原子序数
        material_A : float
            材料原子量 [g/mol]
        material_density : float
            材料密度 [g/cm³]
        ionization_eV : float
            平均激发能 I [eV]

        Returns
        -------
        float : -dE/dx [MeV/cm]
        """
        # 运动学
        p = max(abs(momentum_mev), 1e-6)
        m = self.mass
        beta = p / math.sqrt(p * p + m * m)
        gamma = math.sqrt(p * p + m * m) / m
        beta2 = beta * beta

        if beta2 < 1e-10:
            return 0.0  # 静止粒子无电离损失

        # 最大能量转移
        # T_max = 2·m_e·c²·β²γ² / [1 + 2γm_e/M + (m_e/M)²]
        mass_ratio = ME_ELECTRON / m
        t_max = (2.0 * ME_ELECTRON * beta2 * gamma * gamma /
                 (1.0 + 2.0 * gamma * mass_ratio + mass_ratio * mass_ratio))

        # 对数项
        if t_max < 1e-15 or ionization_eV < 1e-15:
            return 0.0
        log_term = 0.5 * math.log(2.0 * ME_ELECTRON * beta2 * gamma * gamma *
                                   t_max / (ionization_eV * ionization_eV))

        # 密度效应修正 (简化 Sternheimer 参数化)
        delta = self._density_correction(beta, gamma, material_Z)

        # Bethe-Bloch 公式
        z2 = self.z_particle * self.z_particle
        prefactor = K_BETHE_BLOCH * z2 * (material_Z / material_A) * material_density / beta2
        bracket = log_term - beta2 - delta / 2.0

        # 确保非负
        dedx = prefactor * max(bracket, 0.0)

        return dedx

    def _density_correction(self, beta, gamma, Z):
        """
        密度效应修正 δ (简化 Sternheimer 参数化)

        δ(βγ) = 2·ln(ℏω_p / I) + ln(βγ) - 0.5  (高能 plateau)
        对于 βγ < x₀:  δ = 0 (低能无修正)
        对于 x₀ < βγ < x₁: 平滑过渡
        对于 βγ > x₁:  δ = plateau

        简化实现: 使用近似公式
        """
        beta_gamma = beta * gamma

        if beta_gamma < 0.1:
            return 0.0

        # 等离子体频率 (简化)
        # ℏω_p ≈ 28.8 eV · √(ρ·Z/A)  (ρ in g/cm³)
        # 这里使用典型值
        x = math.log10(beta_gamma)

        # 简化参数化 (适用于硅)
        x0 = 0.2   # 起始点
        x1 = 3.0   # plateau 起始点

        if x < x0:
            return 0.0
        elif x > x1:
            # Plateau 区域
            return 2.0 * math.log(beta_gamma) + 2.0 * math.log(0.0346)
        else:
            # 过渡区域 (线性插值)
            t = (x - x0) / (x1 - x0)
            plateau = 2.0 * math.log(10**x1) + 2.0 * math.log(0.0346)
            return plateau * t * t * (3.0 - 2.0 * t)

    # ============================================================
    # Highland 多次散射
    # ============================================================
    def highland_scattering(self, momentum_mev, thickness_x0):
        """
        Highland 多次散射投影角

        θ₀ = (13.6 MeV / βcp) · z · √(x/X₀) · [1 + 0.038·ln(x/X₀)]

        这是 PDG Review of Particle Physics 推荐的标准公式

        Parameters
        ----------
        momentum_mev : float
            粒子动量 [MeV/c]
        thickness_x0 : float
            材料厚度 (辐射长度分数 x/X₀)

        Returns
        -------
        float : 散射角 θ₀ [rad]
        """
        p = max(abs(momentum_mev), 1e-3)
        m = self.mass
        beta = p / math.sqrt(p * p + m * m)
        betap = beta * p  # βcp = β·p·c (c=1 时 = βp)

        if betap < 1e-10:
            return math.pi  # 完全散射

        t = max(thickness_x0, 1e-10)
        sqrt_t = math.sqrt(t)

        # Highland 公式
        theta0 = (13.6 / betap) * self.z_particle * sqrt_t

        # 对数修正项
        log_correction = 1.0 + 0.038 * math.log(t)
        theta0 *= max(log_correction, 0.5)  # 防止过度修正

        return abs(theta0)

    def scattering_sigma_space(self, theta0, thickness_cm):
        """
        多次散射导致的空间位移

        在薄层近似下，横向位移标准差:
            σ_space ≈ θ₀ · L / √3

        其中 L 为层厚度，√3 因子来自均匀分布的几何因子

        Parameters
        ----------
        theta0 : float
            散射角 [rad]
        thickness_cm : float
            层厚度 [cm]

        Returns
        -------
        float : 空间位移 σ [cm]
        """
        return theta0 * thickness_cm / math.sqrt(3.0)

    # ============================================================
    # Landau 能量损失涨落
    # ============================================================
    def landau_most_probable(self, momentum_mev, thickness_cm,
                             material_Z, material_A, material_density,
                             ionization_eV):
        """
        Landau-Vavilov 最概然能量损失

        Δ_p = ξ · [ln(2m_e·c²·β²γ²/I) + ln(ξ/I) + 0.2 - β²]

        其中 ξ = (K/2)·(Z/A)·ρ·t / β²

        Parameters
        ----------
        momentum_mev : float
            粒子动量 [MeV/c]
        thickness_cm : float
            材料厚度 [cm]
        material_Z : int
        material_A : float
        material_density : float
        ionization_eV : float
            [eV]

        Returns
        -------
        float : 最概然能量损失 [MeV]
        """
        p = max(abs(momentum_mev), 1e-3)
        m = self.mass
        beta2 = (p * p) / (p * p + m * m)
        gamma2 = (p * p + m * m) / (m * m)

        if beta2 < 1e-10:
            return 0.0

        # ξ 参数
        z2 = self.z_particle * self.z_particle
        xi = (K_BETHE_BLOCH / 2.0) * z2 * (material_Z / material_A) * \
             material_density * thickness_cm / beta2

        if xi < 1e-15:
            return 0.0

        # Landau 最概然值
        ionization_mev = ionization_eV * 1e-6  # eV → MeV
        if ionization_mev < 1e-15:
            return 0.0

        log_arg = 2.0 * ME_ELECTRON * beta2 * gamma2 / ionization_mev
        if log_arg < 1e-15:
            return 0.0

        delta_p = xi * (math.log(log_arg) + math.log(xi / ionization_mev)
                        + 0.2 - beta2)

        return max(delta_p, 0.0)

    # ============================================================
    # 综合能量损失计算
    # ============================================================
    def energy_loss_in_layer(self, momentum_mev, layer):
        """
        计算粒子穿过探测器层的总能量损失

        综合 Bethe-Bloch 平均损失和 Landau 最概然值

        Parameters
        ----------
        momentum_mev : float
            入射动量 [MeV/c]
        layer : DetectorLayer
            探测器层对象

        Returns
        -------
        dict : {
            'dedx_bethe': float,     # Bethe-Bloch 平均损失率 [MeV/cm]
            'delta_E_mean': float,   # 平均能量损失 [MeV]
            'delta_E_mp': float,     # Landau 最概然损失 [MeV]
            'theta0': float,         # Highland 散射角 [rad]
            'sigma_space': float,    # 散射空间位移 [mm]
            'p_out': float,          # 出射动量 [MeV/c]
        }
        """
        # 层厚度
        thickness_cm = layer.thickness_cm()

        # Bethe-Bloch 能量损失
        dedx = self.bethe_bloch(
            momentum_mev, layer.atomic_number, layer.atomic_mass,
            layer.density_g_cm3, layer.ionization_eV)

        delta_E_mean = dedx * thickness_cm

        # Landau 最概然损失
        delta_E_mp = self.landau_most_probable(
            momentum_mev, thickness_cm,
            layer.atomic_number, layer.atomic_mass,
            layer.density_g_cm3, layer.ionization_eV)

        # Highland 多次散射
        theta0 = self.highland_scattering(momentum_mev, layer.thickness_ratio)

        # 空间位移
        sigma_space = self.scattering_sigma_space(theta0, thickness_cm)
        sigma_space_mm = sigma_space * 10.0  # cm → mm

        # 出射动量 (使用最概然能量损失)
        delta_E = max(delta_E_mp, 0.0)
        e_in = math.sqrt(momentum_mev**2 + self.mass**2)
        e_out = max(e_in - delta_E, self.mass)  # 不能低于静止质量
        p_out = math.sqrt(max(e_out**2 - self.mass**2, 0.0))

        return {
            'dedx_bethe': dedx,
            'delta_E_mean': delta_E_mean,
            'delta_E_mp': delta_E_mp,
            'theta0': theta0,
            'sigma_space_mm': sigma_space_mm,
            'p_in': momentum_mev,
            'p_out': p_out,
        }

    def covariance_process_noise(self, momentum_mev, layer):
        """
        计算 Kalman 滤波的过程噪声协方差矩阵

        物质效应在状态向量上引入额外不确定性:

        Q = diag(σ²_θ, σ²_θ, σ²_s, σ²_p, σ²_p)

        其中:
            σ_θ = θ₀ (多次散射角)
            σ_s = θ₀·L/√3 (散射位移)
            σ_p = σ_E/p (能量损失相对涨落)

        Parameters
        ----------
        momentum_mev : float
            粒子动量 [MeV/c]
        layer : DetectorLayer

        Returns
        -------
        list of list : 5×5 过程噪声协方差矩阵
        """
        effects = self.energy_loss_in_layer(momentum_mev, layer)
        theta0 = effects['theta0']
        sigma_s = effects['sigma_space_mm']

        # 能量损失涨落 (简化: 使用 FWHM ≈ 4ξ 的 Landau 分布)
        thickness_cm = layer.thickness_cm()
        beta2 = (momentum_mev**2) / (momentum_mev**2 + self.mass**2)
        if beta2 < 1e-10:
            beta2 = 1e-10
        z2 = self.z_particle * self.z_particle
        xi = (K_BETHE_BLOCH / 2.0) * z2 * (layer.atomic_number / layer.atomic_mass) * \
             layer.density_g_cm3 * thickness_cm / beta2
        sigma_E = max(4.0 * xi, 1e-6)  # Landau FWHM 近似
        sigma_p = sigma_E / max(momentum_mev, 1e-3)

        # 构建对角过程噪声矩阵 (5×5)
        # 状态向量: (κ, λ, φ, d₀, z₀)
        # 散射主要影响角度 (λ, φ)，能量损失影响 κ
        q = [[0.0] * 5 for _ in range(5)]
        q[0][0] = (sigma_p * 0.01)**2     # κ (曲率) 噪声
        q[1][1] = theta0**2                # λ (偶极角) 散射
        q[2][2] = theta0**2                # φ (方位角) 散射
        q[3][3] = sigma_s**2               # d₀ (横向偏移) 散射位移
        q[4][4] = sigma_s**2               # z₀ (纵向偏移) 散射位移

        return q
