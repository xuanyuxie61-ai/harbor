# -*- coding: utf-8 -*-
"""
physical_constants.py
=====================
计算材料物理常数与材料参数模块

本模块定义位错运动与塑性变形模拟所需的物理常数和材料参数。
所有单位采用国际单位制(SI)。

核心物理量:
-----------
- μ: 剪切模量 (Pa)
- ν: 泊松比 (无量纲)
- b: Burgers矢量大小 (m)
- a: 晶格常数 (m)
- ρ: 质量密度 (kg/m³)
- k_B: Boltzmann常数 (J/K)
- T: 温度 (K)

位错理论基本关系:
----------------
线能量: E_line = μ b² / (4π) * ln(R/r₀)
Peierls应力: σ_P = 2μ/(1-ν) * exp(-2πa/(b(1-ν)))
声子拖曳系数: B_phonon ∝ k_B T / (b * v_s)
"""

import math

# ============================================================================
# 基本物理常数 (NIST CODATA 2018)
# ============================================================================

BOLTZMANN_CONSTANT = 1.380649e-23        # k_B, J/K
PLANCK_CONSTANT = 6.62607015e-34         # h, J·s
HBAR = PLANCK_CONSTANT / (2.0 * math.pi) # ℏ, J·s
ELECTRON_CHARGE = 1.602176634e-19        # e, C
SPEED_OF_LIGHT = 2.99792458e8            # c, m/s
AVOGADRO_NUMBER = 6.02214076e23          # N_A, mol^{-1}
PI = math.pi
EULER_NUMBER = math.e
SQRT2 = math.sqrt(2.0)
SQRT3 = math.sqrt(3.0)
GOLDEN_RATIO = (1.0 + math.sqrt(5.0)) / 2.0

# ============================================================================
# 铝(Al) FCC晶体参数 (参考: Hirth & Lothe, Theory of Dislocations)
# ============================================================================

class AluminumParameters:
    """
    铝(Al) FCC晶体材料参数

    晶体结构: FCC (面心立方)
    晶格常数: a₀ = 4.05 × 10⁻¹⁰ m
    滑移系: {111}<110> (12个)
    Burgers矢量: b = a₀/√2 * <110>
    """

    # 晶格参数
    a_lattice = 4.05e-10                # 晶格常数, m
    crystal_structure = 'FCC'

    # 弹性常数 (室温 ~300K)
    mu = 2.6e10                         # 剪切模量, Pa (26 GPa)
    nu = 0.345                          # 泊松比 (无量纲)
    young_modulus = 2.0 * mu * (1.0 + 0.345)  # 杨氏模量, ~70 GPa
    bulk_modulus = 2.0 * mu * (1.0 + 0.345) / (3.0 * (1.0 - 2.0*0.345))  # 体积模量

    # 密度
    rho_mass = 2700.0                   # 质量密度, kg/m³
    atomic_mass = 26.98e-3 / AVOGADRO_NUMBER  # 原子质量, kg
    atomic_volume = a_lattice**3 / 4.0  # 原子体积 (FCC, 4 atoms/cell), m³

    # Burgers矢量 (FCC: b = a₀/√2 <110>)
    b_magnitude = a_lattice / SQRT2     # |b| = a₀/√2, m
    b_per_area = 1.0 / (b_magnitude * a_lattice)  # 单位面积Burgers矢量数

    # Peierls-Nabarro参数
    # 广义层错能 γ_us (不稳定层错能)
    gamma_usf = 0.160                   # 不稳定层错能, J/m² (160 mJ/m²)
    gamma_isf = 0.120                   # 稳定层错能, J/m² (120 mJ/m²)

    # Peierls势垒参数 (正弦近似)
    # γ(u) = (γ_usf/2) * (1 - cos(2πu/b))
    gamma_amplitude = gamma_usf / 2.0  # γ势垒振幅 = γ_usf/2, J/m²

    # 位错核心宽度 (Peierls-Nabarro模型)
    # ζ = a / (1 - ν) 对于螺型位错
    zeta_screw = a_lattice / (1.0 - nu)           # 螺型位错核心宽度, m
    zeta_edge = a_lattice / (1.0 - nu) * 1.0     # 刃型位错核心宽度, m

    # 位错线能量 (每单位长度)
    # E_line = μ b² / (4π) * ln(R/r₀) * K_factor
    # K_screw = 1, K_edge = 1/(1-ν)
    K_screw = 1.0
    K_edge = 1.0 / (1.0 - nu)
    core_cutoff = b_magnitude           # 核心截断半径 r₀ ≈ b
    outer_radius = 1.0e-6               # 外径 R ~ 1 μm
    ln_ratio = math.log(outer_radius / core_cutoff)

    E_line_screw = mu * b_magnitude**2 / (4.0 * PI) * K_screw * ln_ratio  # J/m
    E_line_edge = mu * b_magnitude**2 / (4.0 * PI) * K_edge * ln_ratio    # J/m

    # 声速 (用于拖曳系数计算)
    v_shear = math.sqrt(mu / rho_mass)        # 横波声速, m/s
    v_long = math.sqrt(young_modulus / rho_mass)  # 纵波声速, m/s

    # 拖曳系数 (室温)
    B_phonon_ref = 1.0e-4               # 参考声子拖曳系数, Pa·s
    B_electron_ref = 5.0e-6             # 参考电子拖曳系数, Pa·s
    T_ref = 300.0                       # 参考温度, K

    # 热激活参数
    # Gibbs自由能垒: ΔG(τ) = ΔF₀ * (1 - (τ/τ_P)^p)^q
    delta_F0 = 1.5 * BOLTZMANN_CONSTANT * 300.0  # 零应力激活能, J
    activation_p = 1.0                  # 激活轮廓参数 p
    activation_q = 2.0                  # 激活轮廓参数 q
    activation_volume = b_magnitude**3 / 10.0  # 激活体积, m³

    # 数值参数
    grid_spacing = a_lattice * 10.0     # 网格间距, ~10a₀
    time_step_ref = 1.0e-13             # 参考时间步长, s (0.1 ps)

    def peierls_stress(self):
        """
        计算Peierls-Nabarro应力

        σ_P = 2μ/(1-ν) * exp(-2πζ/b)

        其中 ζ = a/(1-ν) 是位错核心宽度

        Returns:
            float: Peierls应力 (Pa)
        """
        exponent = -2.0 * PI * self.zeta_screw / self.b_magnitude
        sigma_p = 2.0 * self.mu / (1.0 - self.nu) * math.exp(exponent)
        return sigma_p

    def thermal_velocity(self, temperature):
        """
        计算热振动速度

        v_th = sqrt(3 k_B T / m_atom)

        Args:
            temperature: 温度 (K)

        Returns:
            float: 热速度 (m/s)
        """
        return math.sqrt(3.0 * BOLTZMANN_CONSTANT * temperature / self.atomic_mass)

    def drag_coefficient(self, temperature):
        """
        计算温度相关的拖曳系数

        B(T) = B_phonon * (T/T_ref) + B_electron

        Args:
            temperature: 温度 (K)

        Returns:
            float: 拖曳系数 (Pa·s)
        """
        return self.B_phonon_ref * (temperature / self.T_ref) + self.B_electron_ref

    def debye_frequency(self):
        """
        计算Debye频率

        ω_D = v_shear * (6π² n)^(1/3)

        其中 n = 1/Ω_atom 是原子数密度

        Returns:
            float: Debye角频率 (rad/s)
        """
        n_density = 1.0 / self.atomic_volume
        omega_D = self.v_shear * (6.0 * PI**2 * n_density)**(1.0/3.0)
        return omega_D

    def burgers_tensor_norm(self, direction):
        """
        计算给定方向的Burgers矢量模

        对于FCC: b = a/2 <110>
        |b| = a/2 * sqrt(h² + k² + l²)

        Args:
            direction: Miller指数元组 (h,k,l)

        Returns:
            float: Burgers矢量模 (m)
        """
        h, k, l = direction
        norm_sq = h**2 + k**2 + l**2
        return self.a_lattice / 2.0 * math.sqrt(norm_sq)


# ============================================================================
# 铜(Cu) FCC晶体参数 (对比用)
# ============================================================================

class CopperParameters(AluminumParameters):
    """
    铜(Cu) FCC晶体材料参数

    与铝类似但参数不同:
    - 更高的剪切模量 (48 GPa vs 26 GPa)
    - 更低的层错能 (~45 mJ/m² vs 120 mJ/m²)
    """

    a_lattice = 3.61e-10                # m
    mu = 4.8e10                         # Pa (48 GPa)
    nu = 0.34                           # 泊松比
    rho_mass = 8960.0                   # kg/m³
    atomic_mass = 63.55e-3 / AVOGADRO_NUMBER  # kg
    gamma_usf = 0.200                   # J/m²
    gamma_isf = 0.045                   # J/m² (Cu层错能低)
    gamma_amplitude = gamma_usf / 2.0

    def __init__(self):
        # 重新计算派生参数
        self.b_magnitude = self.a_lattice / SQRT2
        self.zeta_screw = self.a_lattice / (1.0 - self.nu)
        self.zeta_edge = self.a_lattice / (1.0 - self.nu)
        self.young_modulus = 2.0 * self.mu * (1.0 + self.nu)
        self.bulk_modulus = 2.0 * self.mu * (1.0 + self.nu) / (3.0 * (1.0 - 2.0*self.nu))
        self.atomic_volume = self.a_lattice**3 / 4.0
        self.K_edge = 1.0 / (1.0 - self.nu)
        self.ln_ratio = math.log(self.outer_radius / self.core_cutoff)
        self.E_line_screw = self.mu * self.b_magnitude**2 / (4.0*PI) * self.K_screw * self.ln_ratio
        self.E_line_edge = self.mu * self.b_magnitude**2 / (4.0*PI) * self.K_edge * self.ln_ratio
        self.v_shear = math.sqrt(self.mu / self.rho_mass)
        self.v_long = math.sqrt(self.young_modulus / self.rho_mass)
        self.grid_spacing = self.a_lattice * 10.0


# ============================================================================
# 钨(W) BCC晶体参数 (体心立方)
# ============================================================================

class TungstenParameters:
    """
    钨(W) BCC晶体材料参数

    BCC金属的位错核心结构比FCC复杂得多:
    - 螺型位错具有非平面核心扩展
    - Peierls应力显著高于FCC
    - 滑移面不确定定性 (pencil glide)
    """

    a_lattice = 3.16e-10                # m
    crystal_structure = 'BCC'
    mu = 1.61e11                        # Pa (161 GPa)
    nu = 0.28                           # 泊松比
    rho_mass = 19250.0                  # kg/m³
    atomic_mass = 183.84e-3 / AVOGADRO_NUMBER  # kg

    # BCC: b = a/2 <111>
    b_magnitude = SQRT3 / 2.0 * a_lattice

    # BCC层错能 (极高, 导致全位错不分解)
    gamma_usf = 1.50                    # J/m² (远高于FCC)
    gamma_isf = 0.80                    # J/m²
    gamma_amplitude = gamma_usf / 2.0

    # BCC螺型位错核心宽度 (极窄)
    zeta_screw = a_lattice * 1.5        # 窄核心, ~1.5a
    zeta_edge = a_lattice * 3.0         # 刃型较宽

    K_screw = 1.0
    K_edge = 1.0 / (1.0 - nu)
    core_cutoff = b_magnitude
    outer_radius = 1.0e-6
    ln_ratio = math.log(outer_radius / core_cutoff)

    E_line_screw = mu * b_magnitude**2 / (4.0*PI) * K_screw * ln_ratio
    E_line_edge = mu * b_magnitude**2 / (4.0*PI) * K_edge * ln_ratio

    v_shear = math.sqrt(mu / rho_mass)
    v_long = math.sqrt(2.0 * mu * (1.0 + nu) / rho_mass / (1.0 - 2.0*nu))

    atomic_volume = a_lattice**3 / 2.0  # BCC: 2 atoms/cell
    grid_spacing = a_lattice * 5.0

    def peierls_stress(self):
        """BCC的Peierls应力 (显著高于FCC)"""
        exponent = -2.0 * PI * self.zeta_screw / self.b_magnitude
        sigma_p = 2.0 * self.mu / (1.0 - self.nu) * math.exp(exponent)
        return sigma_p


# ============================================================================
# 默认材料参数 (使用铝)
# ============================================================================

DEFAULT_MATERIAL = AluminumParameters()


def compute_elisington_tensor(mu, nu):
    """
    计算各向同性弹性刚度张量 (Voigt记号 6×6)

    C_ijkl 的Voigt表示:
    C = [[2μ+λ, λ, λ, 0, 0, 0],
         [λ, 2μ+λ, λ, 0, 0, 0],
         [λ, λ, 2μ+λ, 0, 0, 0],
         [0, 0, 0, μ, 0, 0],
         [0, 0, 0, 0, μ, 0],
         [0, 0, 0, 0, 0, μ]]

    其中 λ = 2μν/(1-2ν) 是Lamé第一参数

    Args:
        mu: 剪切模量 (Pa)
        nu: 泊松比

    Returns:
        list: 6×6刚度矩阵
    """
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)  # Lamé第一参数

    C = [[0.0]*6 for _ in range(6)]
    C[0][0] = C[1][1] = C[2][2] = 2.0*mu + lam
    C[0][1] = C[0][2] = C[1][0] = C[1][2] = C[2][0] = C[2][1] = lam
    C[3][3] = C[4][4] = C[5][5] = mu

    return C


def compute_compliance_tensor(mu, nu):
    """
    计算各向同性弹性柔度张量 (Voigt记号 6×6)

    S = C^{-1}

    S = [[1/E, -ν/E, -ν/E, 0, 0, 0],
         [-ν/E, 1/E, -ν/E, 0, 0, 0],
         [-ν/E, -ν/E, 1/E, 0, 0, 0],
         [0, 0, 0, 1/μ, 0, 0],
         [0, 0, 0, 0, 1/μ, 0],
         [0, 0, 0, 0, 0, 1/μ]]

    Args:
        mu: 剪切模量 (Pa)
        nu: 泊松比

    Returns:
        list: 6×6柔度矩阵
    """
    E = 2.0 * mu * (1.0 + nu)

    S = [[0.0]*6 for _ in range(6)]
    S[0][0] = S[1][1] = S[2][2] = 1.0 / E
    S[0][1] = S[0][2] = S[1][0] = S[1][2] = S[2][0] = S[2][1] = -nu / E
    S[3][3] = S[4][4] = S[5][5] = 1.0 / mu

    return S


def image_force_stress(mu, b, nu, x, h):
    """
    计算自由表面镜像力产生的应力

    位错距表面距离 h 时的镜像力:
    F_image = -μ b² / (4π h) * K(ν)

    其中 K_screw = 1, K_edge = 1/(1-ν)

    Args:
        mu: 剪切模量 (Pa)
        b: Burgers矢量大小 (m)
        nu: 泊松比
        x: 位错位置 (m) - 用于判断边界
        h: 距表面距离 (m)

    Returns:
        float: 镜像力应力 (Pa)
    """
    if h < b:
        h = b  # 防止奇异
    K = 1.0 / (1.0 - nu)  # 保守估计 (刃型)
    F_image = -mu * b**2 / (4.0 * PI * h) * K
    return F_image


if __name__ == '__main__':
    al = AluminumParameters()
    cu = CopperParameters()
    w = TungstenParameters()

    print("=" * 70)
    print("计算材料物理常数验证")
    print("=" * 70)
    print(f"Al Peierls应力: {al.peierls_stress():.4e} Pa")
    print(f"Al 剪切波速: {al.v_shear:.2f} m/s")
    print(f"Al 螺型位错线能量: {al.E_line_screw:.4e} J/m")
    print(f"Al Debye频率: {al.debye_frequency():.4e} rad/s")
    print(f"Cu Burgers矢量: {cu.b_magnitude:.4e} m")
    print(f"W Peierls应力: {w.peierls_stress():.4e} Pa")
    print(f"W Burgers矢量: {w.b_magnitude:.4e} m")
