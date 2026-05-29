"""
atmospheric_model.py
系外行星大气物理模型模块。

融合原始项目：780_mortality（统计概率分布、CDF/PDF、期望计算）
             116_box_plot（数据分箱与离散化）

本模块建立系外行星大气的物理参数化模型，包括：
- 温度-压强（T-P）剖面
- 化学平衡丰度计算
- 行星半径与重力场
- 不确定性统计建模（融合 mortality 的概率分布思想）
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from scipy.special import erf


class AtmosphericProfile:
    """
    行星大气一维垂直剖面模型。

    物理坐标系:
        z: 海拔高度 (m)，z=0 为参考半径 R_p 处
        P: 压强 (Pa)
        T: 温度 (K)
        ρ: 质量密度 (kg/m³)
        g: 重力加速度 (m/s²)，随高度变化 g(z) = GM / (R_p + z)²

    流体静力学平衡:
        dP/dz = -ρ(z) g(z)

    理想气体状态方程:
        P = ρ k_B T / (μ m_u)
        其中 μ 是平均分子量，m_u 是原子质量单位
    """

    # 物理常数
    K_B = 1.380649e-23       # 玻尔兹曼常数 (J/K)
    AMU = 1.66053906660e-27  # 原子质量单位 (kg)
    G = 6.67430e-11          # 万有引力常数 (m³ kg⁻¹ s⁻²)
    M_SUN = 1.98847e30       # 太阳质量 (kg)
    R_JUPITER = 6.9911e7     # 木星半径 (m)
    R_EARTH = 6.371e6        # 地球半径 (m)

    def __init__(self, planet_mass_kg: float, planet_radius_m: float,
                 star_mass_kg: float, orbital_distance_m: float):
        """
        参数:
            planet_mass_kg: 行星质量 (kg)
            planet_radius_m: 行星半径 (m)
            star_mass_kg: 恒星质量 (kg)
            orbital_distance_m: 轨道距离 (m)
        """
        if planet_mass_kg <= 0 or planet_radius_m <= 0:
            raise ValueError("行星质量和半径必须为正")
        if star_mass_kg <= 0 or orbital_distance_m <= 0:
            raise ValueError("恒星质量和轨道距离必须为正")

        self.M_p = planet_mass_kg
        self.R_p = planet_radius_m
        self.M_star = star_mass_kg
        self.a = orbital_distance_m

        # 计算平衡温度（假设反照率为零，无温室效应）
        self.T_eq = self._equilibrium_temperature()

    def _equilibrium_temperature(self, albedo: float = 0.0, redistribution: float = 0.25) -> float:
        """
        计算行星平衡温度。

        公式:
            T_eq = T_star * √(R_star / 2a) * (1 - A_B)^{1/4} * f^{1/4}

        其中:
            T_star: 恒星有效温度
            R_star: 恒星半径（近似取太阳半径）
            a: 轨道半长轴
            A_B: 邦德反照率
            f: 能量再分配因子（0.25 表示全球平均，0.5 表示仅昼侧）
        """
        R_star = 6.957e8  # 太阳半径 (m)
        T_star = 5778.0   # 太阳有效温度 (K)
        T_eq = T_star * np.sqrt(R_star / (2.0 * self.a)) * ((1.0 - albedo) * redistribution)**0.25
        return T_eq

    def gravity(self, z: np.ndarray) -> np.ndarray:
        """
        计算高度 z 处的重力加速度。

        公式:
            g(z) = G M_p / (R_p + z)²
        """
        z = np.asarray(z, dtype=np.float64)
        r = self.R_p + z
        r = np.maximum(r, 1e-6)
        return self.G * self.M_p / r**2

    def guillot_temperature_profile(self, pressure: np.ndarray,
                                     T_int: float = 100.0,
                                     T_irr: float = None,
                                     gamma: float = 16.0 / 3.0,
                                     kappa_ir: float = 1e-2,
                                     kappa_v1: float = 6e-3,
                                     kappa_v2: float = 1e-4,
                                     f: float = 1.0 / 4.0) -> np.ndarray:
        """
        Guillot (2010) 温度-压强剖面模型。

        用于强烈辐射加热的行星大气（如热木星）。
        模型假设大气受内部热流和恒星辐照共同加热。

        公式:
            T⁴ = (3 T_int⁴ / 4) (2/3 + τ)
               + (3 T_irr⁴ / 4) f (2/3 + 1/(γ √3) + (γ/√3 - 1/(γ √3)) e^{-γ τ √3})

        其中光学厚度 τ 与压强的关系:
            τ = P κ_ir / g

        参数:
            pressure: 压强数组 (Pa)，从高层（低P）到低层（高P）
            T_int: 内部有效温度 (K)
            T_irr: 辐照有效温度 (K)，默认使用 T_eq
            gamma: 可见光与红外不透明度的比值参数
            kappa_ir: 红外不透明度 (m²/kg)
            kappa_v1, kappa_v2: 可见光不透明度参数
            f: 再分配因子

        返回:
            温度数组 (K)
        """
        pressure = np.asarray(pressure, dtype=np.float64)
        if np.any(pressure <= 0):
            raise ValueError("压强必须为正")

        if T_irr is None:
            T_irr = self.T_eq

        g_surf = self.gravity(0.0)
        tau = pressure * kappa_ir / g_surf

        term1 = 0.75 * T_int**4 * (2.0 / 3.0 + tau)

        term2_coeff = 2.0 / 3.0 + 1.0 / (gamma * np.sqrt(3.0))
        term2_exp = (gamma / np.sqrt(3.0) - 1.0 / (gamma * np.sqrt(3.0))) * np.exp(-gamma * tau * np.sqrt(3.0))
        term2 = 0.75 * T_irr**4 * f * (term2_coeff + term2_exp)

        T4 = term1 + term2
        T4 = np.maximum(T4, 1e-10)
        return T4**0.25

    def isothermal_profile(self, pressure: np.ndarray, T0: float) -> np.ndarray:
        """等温剖面。"""
        pressure = np.asarray(pressure, dtype=np.float64)
        if np.any(pressure <= 0):
            raise ValueError("压强必须为正")
        return np.full_like(pressure, T0, dtype=np.float64)

    def hydrostatic_pressure_grid(self, n_layers: int, P_top: float, P_bot: float) -> np.ndarray:
        """
        构造对数等间距的压强分层网格。

        融合 box_plot 的分箱离散化思想：将连续大气按压强对数均匀分箱。

        公式:
            log10(P_i) = log10(P_top) + i * Δ,  i = 0, ..., n_layers-1
            Δ = [log10(P_bot) - log10(P_top)] / (n_layers - 1)

        参数:
            n_layers: 层数
            P_top: 顶层压强 (Pa)
            P_bot: 底层压强 (Pa)

        返回:
            压强数组 (Pa)，从高层到低层递增
        """
        if n_layers < 2:
            raise ValueError("层数至少为 2")
        if P_top <= 0 or P_bot <= 0 or P_top >= P_bot:
            raise ValueError("压强范围不合法，需要 P_top < P_bot 且均为正")

        logP = np.linspace(np.log10(P_top), np.log10(P_bot), n_layers)
        return 10.0**logP

    def scale_height(self, T: float, mu: float, z: float = 0.0) -> float:
        """
        计算大气标高。

        公式:
            H = k_B T / (μ m_u g(z))
        """
        if T <= 0 or mu <= 0:
            raise ValueError("温度和平均分子量必须为正")
        g = self.gravity(z)
        if g <= 0:
            raise ValueError("重力加速度必须为正")
        return self.K_B * T / (mu * self.AMU * g)

    def altitude_from_pressure(self, pressure: np.ndarray, T: np.ndarray,
                               mu: np.ndarray, P_ref: float = None) -> np.ndarray:
        """
        从压强剖面计算海拔高度（流体静力学积分）。

        公式:
            z(P) = -∫_{P_ref}^{P} (k_B T(P') / μ m_u g(z) P') dP'

        近似处理：假设 g ≈ g(0)，T 取局部值。
        """
        pressure = np.asarray(pressure, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)
        if pressure.shape != T.shape:
            raise ValueError("压强和温度数组形状必须一致")
        if mu.shape != pressure.shape and mu.size != 1:
            raise ValueError("mu 必须是标量或与压强同形状数组")
        if mu.size == 1:
            mu = np.full_like(pressure, float(mu))

        # TODO Hole 2: 实现流体静力学积分计算海拔高度
        # 公式: dz = -(k_B * T / (μ * amu * g)) * dP/P = -H * dlnP
        # 从参考压强 P_ref 开始积分，归一化使 P_ref 处 z=0
        return np.zeros_like(pressure)


class ChemicalEquilibrium:
    """
    大气化学丰度模型。

    融合 mortality 的概率统计思想：将化学物种丰度的不确定性建模为
    对数正态分布，其参数由化学平衡计算确定。
    """

    def __init__(self, species_list: List[str]):
        self.species = species_list
        self.molar_masses = {
            'H2': 2.016, 'He': 4.003, 'H2O': 18.015, 'CH4': 16.043,
            'CO': 28.010, 'CO2': 44.010, 'NH3': 17.031, 'N2': 28.014,
            'O2': 31.999, 'Na': 22.990, 'K': 39.098, 'TiO': 63.866,
            'VO': 66.941, 'FeH': 56.853, 'HCN': 27.026, 'C2H2': 26.038,
            'PH3': 33.998, 'H2S': 34.082
        }

    def equilibrium_abundance(self, species: str, T: np.ndarray, P: np.ndarray,
                              metallicity: float = 1.0, C_O_ratio: float = 0.54) -> np.ndarray:
        """
        简化化学平衡丰度计算。

        基于吉布斯自由能最小化原理的简化参数化模型。
        对于太阳系丰度（金属丰度=1，C/O=0.54）：
            - H2: 主导成分，体积混合比 ~ 0.85
            - He: 次要成分，体积混合比 ~ 0.15
            - 其他痕量气体按温度和压强调节

        公式（简化参数化，基于化学平衡趋势）:
            VMR_i(P, T) = VMR_0 * (P / P_0)^{α_i} * exp(-E_i / k_B T)

        其中:
            VMR_0: 参考混合比
            α_i: 压强调节指数（与凝结/解离有关）
            E_i: 有效解离能

        参数:
            species: 化学物种名
            T: 温度数组 (K)
            P: 压强数组 (Pa)
            metallicity: 金属丰度倍数（相对于太阳）
            C_O_ratio: 碳氧比

        返回:
            体积混合比数组 (VMR)
        """
        T = np.asarray(T, dtype=np.float64)
        P = np.asarray(P, dtype=np.float64)
        if T.shape != P.shape:
            raise ValueError("温度和压强数组形状必须一致")
        if np.any(T <= 0) or np.any(P <= 0):
            raise ValueError("温度和压强必须为正")

        P_bar = P / 1e5  # 转换为 bar

        # 简化参数化化学模型（基于文献趋势拟合）
        if species == 'H2':
            vmr = 0.85 * np.ones_like(T)
        elif species == 'He':
            vmr = 0.15 * np.ones_like(T)
        elif species == 'H2O':
            # 水蒸气：高温时解离，高压时凝结
            vmr = 1e-3 * metallicity * (P_bar**0.1) * np.exp(-8000.0 / T)
            vmr = np.minimum(vmr, 1e-2 * metallicity)
        elif species == 'CH4':
            # 甲烷：低温富集
            vmr = 1e-4 * metallicity * (P_bar**0.05) * np.exp(-12000.0 / T)
            if C_O_ratio > 1.0:
                vmr *= (C_O_ratio / 0.54)**0.5
            vmr = np.minimum(vmr, 5e-3 * metallicity)
        elif species == 'CO':
            # 一氧化碳：高温富集
            vmr = 1e-3 * metallicity * (P_bar**0.15) * np.exp(-4000.0 / T)
            vmr = np.minimum(vmr, 1e-2 * metallicity)
        elif species == 'CO2':
            vmr = 1e-6 * metallicity * (P_bar**0.2) * np.exp(-6000.0 / T)
            vmr = np.minimum(vmr, 1e-4 * metallicity)
        elif species == 'NH3':
            vmr = 1e-5 * metallicity * (P_bar**0.1) * np.exp(-10000.0 / T)
            vmr = np.minimum(vmr, 1e-3 * metallicity)
        elif species == 'Na':
            vmr = 1e-7 * metallicity * np.exp(-5000.0 / T)
        elif species == 'K':
            vmr = 5e-8 * metallicity * np.exp(-5000.0 / T)
        elif species == 'TiO':
            vmr = 1e-9 * metallicity * np.exp(-12000.0 / T)
        elif species == 'VO':
            vmr = 5e-10 * metallicity * np.exp(-12000.0 / T)
        elif species == 'HCN':
            vmr = 1e-7 * metallicity * (C_O_ratio / 0.54) * np.exp(-9000.0 / T)
        else:
            vmr = 1e-12 * np.ones_like(T)

        vmr = np.maximum(vmr, 1e-30)
        return vmr

    def mean_molecular_weight(self, abundances: Dict[str, np.ndarray]) -> np.ndarray:
        """
        计算平均分子量。

        公式:
            μ = Σ_i (X_i μ_i) / Σ_i X_i

        其中 X_i 是体积混合比，μ_i 是摩尔质量。
        """
        if not abundances:
            raise ValueError("丰度字典为空")

        shape = None
        total_mass = None
        total_moles = None

        for sp, vmr in abundances.items():
            vmr = np.asarray(vmr, dtype=np.float64)
            if shape is None:
                shape = vmr.shape
                total_mass = np.zeros(shape, dtype=np.float64)
                total_moles = np.zeros(shape, dtype=np.float64)
            if vmr.shape != shape:
                raise ValueError(f"物种 {sp} 的丰度数组形状不一致")

            mu_i = self.molar_masses.get(sp, 20.0)
            total_mass += vmr * mu_i
            total_moles += vmr

        total_moles = np.maximum(total_moles, 1e-30)
        return total_mass / total_moles

    def sample_abundance_uncertainty(self, vmr_mean: np.ndarray,
                                      sigma_log: float = 0.5,
                                      n_samples: int = 1,
                                      seed: Optional[int] = None) -> np.ndarray:
        """
        对丰度不确定性进行对数正态采样。

        融合 mortality 的统计分布思想：化学丰度的不确定性常服从对数正态分布。

        若 log10(VMR) ~ N(log10(VMR_mean), σ²)，则:
            VMR_sample = 10^{log10(VMR_mean) + σ * Z}
            其中 Z ~ N(0, 1)

        参数:
            vmr_mean: 平均体积混合比
            sigma_log: log10 空间的标准差
            n_samples: 采样数
            seed: 随机种子

        返回:
            采样数组，形状 (n_samples, *vmr_mean.shape)
        """
        if seed is not None:
            np.random.seed(seed)

        vmr_mean = np.asarray(vmr_mean, dtype=np.float64)
        log_vmr = np.log10(np.maximum(vmr_mean, 1e-30))
        noise = np.random.normal(0.0, sigma_log, size=(n_samples,) + vmr_mean.shape)
        samples = 10.0**(log_vmr + noise)
        return np.maximum(samples, 1e-30)


class CloudModel:
    """
    行星大气云层模型。

    采用简化参数化方案：
        - 云层存在于特定压强范围内 [P_cloud_top, P_cloud_base]
        - 云粒子尺度分布遵循对数正态分布
        - 消光系数与凝结物质量相关
    """

    def __init__(self, P_cloud_top: float = 1e2, P_cloud_base: float = 1e4,
                 cloud_opacity: float = 1.0, particle_radius_m: float = 1e-6):
        self.P_top = P_cloud_top
        self.P_base = P_cloud_base
        self.cloud_opacity = cloud_opacity
        self.r_particle = particle_radius_m

    def cloud_optical_depth(self, pressure: np.ndarray) -> np.ndarray:
        """
        计算云层光学厚度贡献。

        参数化模型:
            τ_cloud(P) = τ_0 * exp( -[(log P - log P_c) / σ_P]² )

        其中 P_c 是云层中心压强，σ_P 控制云层厚度。
        """
        pressure = np.asarray(pressure, dtype=np.float64)
        logP = np.log10(pressure)
        logP_c = 0.5 * (np.log10(self.P_top) + np.log10(self.P_base))
        sigma_P = 0.5 * abs(np.log10(self.P_base) - np.log10(self.P_top))

        if sigma_P < 1e-10:
            return np.zeros_like(pressure)

        tau = self.cloud_opacity * np.exp(-((logP - logP_c) / sigma_P)**2)
        tau = np.where((pressure >= self.P_top) & (pressure <= self.P_base), tau, 0.0)
        return tau
