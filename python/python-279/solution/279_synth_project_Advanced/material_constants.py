"""
material_constants.py
=====================
物理与材料科学常数集合。所有常数均采用 SI 单位制，数值来源于
CODATA 2018 推荐值及Materials Project数据库。

物理背景:
  - 玻尔兹曼常数 kB = 1.380649e-23 J/K (2019 重新定义后的精确值)
  - 基本电荷 e = 1.602176634e-19 C
  - 真空介电常数 eps0 = 8.8541878128e-12 F/m
  - 气体常数 R = kB * NA = 8.314462618 J/(mol*K)
  - 法拉第常数 F = e * NA = 96485.33212 C/mol
"""

import numpy as np


# ============================================================================
# 基本物理常数 (CODATA 2018)
# ============================================================================
KB = 1.380649e-23              # Boltzmann constant [J/K]
ELEMENTARY_CHARGE = 1.602176634e-19  # Elementary charge [C]
EPSILON_0 = 8.8541878128e-12   # Vacuum permittivity [F/m]
MU_0 = 1.25663706212e-6        # Vacuum permeability [H/m]
AVOGADRO = 6.02214076e23       # Avogadro number [1/mol]
PLANCK = 6.62607015e-34        # Planck constant [J*s]
HBAR = PLANCK / (2.0 * np.pi)  # Reduced Planck constant [J*s]
SPEED_OF_LIGHT = 299792458.0   # Speed of light [m/s]


# ============================================================================
# 导出电化学常数
# ============================================================================
R_GAS = KB * AVOGADRO          # Universal gas constant [J/(mol*K)]
FARADAY = ELEMENTARY_CHARGE * AVOGADRO  # Faraday constant [C/mol]
THERMAL_VOLTAGE_300K = KB * 300.0 / ELEMENTARY_CHARGE  # ~25.85 mV


# ============================================================================
# LLZO 立方相典型参数 (cubic Li7La3Zr2O12)
# ============================================================================
LLZO_LATTICE_PARAM = 12.97e-10        # Lattice parameter [m]
LLZO_FORMULA_UNITS = 8                # Z (formula units per unit cell)
LLZO_MOLAR_MASS = 8.818e-4            # [kg/mol]
LLZO_DENSITY = 5120.0                 # [kg/m^3]
LLZO_RELATIVE_PERMITTIVITY = 50.0     # Dimensionless (cubic phase ~40-60)
LLZO_DIFFUSION_COEFF = 4.5e-12        # D_Li [m^2/s] at 300K
LLZO_ACTIVATION_ENERGY = 0.37         # E_a [eV] for Li migration
LLZO_IONIC_CONDUCTIVITY = 1.1e-3      # sigma [S/m] at 300K

# ============================================================================
# 热学参数
# ============================================================================
LLZO_THERMAL_CONDUCTIVITY = 1.8       # kappa [W/(m*K)]
LLZO_SPECIFIC_HEAT = 580.0            # c_p [J/(kg*K)]
LLZO_THERMAL_DIFFUSIVITY = (
    LLZO_THERMAL_CONDUCTIVITY / (LLZO_DENSITY * LLZO_SPECIFIC_HEAT)
)                                     # alpha [m^2/s]

# ============================================================================
# 材料基因组筛选参数空间
# ============================================================================
DOPANT_CONCENTRATIONS = np.array(
    [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    dtype=np.float64,
)  # 掺杂浓度 x in Li_{7-3x}Al_xLa3Zr2O12

STRAIN_RANGE = np.linspace(-0.02, 0.02, 9)   # +/-2% 应变范围
TEMPERATURE_RANGE = np.array(
    [250.0, 275.0, 300.0, 325.0, 350.0], dtype=np.float64
)  # 温度扫描 [K]

# ============================================================================
# 数值安全阈值
# ============================================================================
MACHINE_EPS = np.finfo(np.float64).eps     # ~2.22e-16
SMALL_NUMBER = 1e-30                       # 防除零保护
MAX_STABILITY_ITER = 5000                  # 稳定性分析最大迭代
CONVERGENCE_TOL = 1e-12                    # 收敛容差


def arrhenius_conductivity(sigma_0, E_a_eV, T_kelvin):
    """
    Arrhenius 型离子电导率。

    物理公式:
        sigma(T) = sigma_0 * exp(-E_a / (kB * T))

    其中 E_a [eV] 为迁移活化能，kB 需要转换为 eV/K 单位:
        kB_eV = kB / e = 8.617333e-5 eV/K

    返回:
        sigma [S/m]
    """
    kB_eV = KB / ELEMENTARY_CHARGE
    arg = -E_a_eV / (kB_eV * T_kelvin)
    arg = np.clip(arg, -500.0, 500.0)  # 防止溢出
    return sigma_0 * np.exp(arg)


def nernst_einstein_diffusion(sigma, T_kelvin, z_val, c_carrier):
    """
    Nernst-Einstein 关系由电导率求扩散系数:

        D = sigma * kB * T / (z^2 * e^2 * c)

    其中 z 为载流子价态 (Li+: z=1), c 为载流子数密度 [1/m^3]
    """
    num = sigma * KB * T_kelvin
    denom = (z_val ** 2) * (ELEMENTARY_CHARGE ** 2) * max(c_carrier, SMALL_NUMBER)
    return num / denom


def debye_length(eps_r, T_kelvin, c_mol_per_m3):
    """
    Debye 屏蔽长度:

        lambda_D = sqrt(eps_0 * eps_r * kB * T / (2 * e^2 * NA * c))

    其中 c 为摩尔浓度 [mol/m^3]
    """
    num = EPSILON_0 * eps_r * KB * T_kelvin
    denom = 2.0 * (ELEMENTARY_CHARGE ** 2) * AVOGADRO * max(c_mol_per_m3, SMALL_NUMBER)
    return np.sqrt(num / denom)


def peierls_nabarro_stress(G, nu, a, d):
    """
    Peierls-Nabarro 晶格阻力应力:

        tau_PN = (2*G / (1-nu)) * exp(-2*pi*a / (d*(1-nu)))

    用于评估离子在晶格中迁移的临界剪切应力。

    参数:
        G: 剪切模量 [Pa]
        nu: 泊松比
        a: 滑移面间距 [m]
        d: Burgers 矢量大小 [m]
    """
    prefactor = 2.0 * G / max(1.0 - nu, SMALL_NUMBER)
    exponent = -2.0 * np.pi * a / max(d * (1.0 - nu), SMALL_NUMBER)
    return prefactor * np.exp(exponent)
