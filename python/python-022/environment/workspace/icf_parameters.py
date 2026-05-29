"""
icf_parameters.py
=================
惯性约束聚变（ICF）内爆模拟的物理常数、靶丸参数与数值控制参数。

本模块定义了所有计算所需的物理常数、靶丸几何参数、材料属性及数值控制参数，
确保全项目参数统一、可溯源。
"""

import numpy as np

# ========================================================================
# 基本物理常数（SI单位制）
# ========================================================================
class PhysicalConstants:
    """基本物理常数，全部以国际单位制（SI）给出。"""
    BOLTZMANN: float = 1.380649e-23     # k_B [J/K]
    ELECTRON_MASS: float = 9.10938356e-31   # m_e [kg]
    PROTON_MASS: float = 1.6726219e-27      # m_p [kg]
    NEUTRON_MASS: float = 1.6749275e-27     # m_n [kg]
    ELEMENTARY_CHARGE: float = 1.602176634e-19  # e [C]
    SPEED_OF_LIGHT: float = 2.99792458e8    # c [m/s]
    VACUUM_PERMITTIVITY: float = 8.854187817e-12  # epsilon_0 [F/m]
    PLANCK: float = 6.62607015e-34          # h [J*s]
    AVOGADRO: float = 6.02214076e23         # N_A [mol^-1]
    STEFAN_BOLTZMANN: float = 5.670374419e-8    # sigma_SB [W/(m^2*K^4)]


# ========================================================================
# DT靶丸参数
# ========================================================================
class TargetParameters:
    """
    DT冰靶丸参数。
    靶丸为球形，由外层烧蚀层（CH塑料）与内层DT冰层构成。
    """
    # 几何参数
    R_ABLATION: float = 1.1e-3          # 烧蚀层外半径 [m] = 1.1 mm
    R_DT_ICE: float = 0.95e-3           # DT冰层外半径 [m] = 0.95 mm
    R_GAS: float = 0.4e-3               # 充气腔内半径 [m] = 0.4 mm

    # 材料参数
    RHO_CH: float = 1350.0              # CH塑料密度 [kg/m^3]
    RHO_DT: float = 250.0               # DT冰密度 [kg/m^3]
    RHO_GAS: float = 0.3                # 腔内气体密度 [kg/m^3]

    # 质量分数
    A_C: float = 12.0                   # 碳原子质量数
    A_H: float = 1.0                    # 氢原子质量数
    A_D: float = 2.0                    # 氘原子质量数
    A_T: float = 3.0                    # 氚原子质量数

    # 烧蚀层化学式 C_x H_y，取 x=1, y=1.5（近似聚苯乙烯）
    X_C: float = 1.0
    Y_H: float = 1.5

    @property
    def ablator_average_atomic_mass(self) -> float:
        """烧蚀层平均原子质量 [u]"""
        return (self.X_C * self.A_C + self.Y_H * self.A_H) / (self.X_C + self.Y_H)

    @property
    def ablator_atomic_number(self) -> float:
        """烧蚀层平均原子序数 Z"""
        return (6.0 * self.X_C + 1.0 * self.Y_H) / (self.X_C + self.Y_H)


# ========================================================================
# 激光参数（NIF-like）
# ========================================================================
class LaserParameters:
    """激光驱动参数，基于国家点火装置（NIF）典型参数。"""
    NUM_BEAMS: int = 192                # 激光束数量
    WAVELENGTH: float = 351.0e-9        # 三倍频波长 [m]
    TOTAL_ENERGY: float = 1.8e6         # 总激光能量 [J]
    PULSE_DURATION: float = 15.0e-9     # 脉冲持续时间 [s]
    POWER_PEAK: float = 350.0e12        # 峰值功率 [W]

    # 激光能量时间分布（高斯型）
    @staticmethod
    def power_profile(t: float, t0: float = 7.5e-9, sigma: float = 3.0e-9) -> float:
        """
        激光功率时间分布 P(t) = P_peak * exp(-(t-t0)^2 / (2*sigma^2))
        并施加边界约束保证非负。
        """
        if t < 0.0 or t > 20.0e-9:
            return 0.0
        p = LaserParameters.POWER_PEAK * np.exp(-(t - t0)**2 / (2.0 * sigma**2))
        return max(p, 0.0)


# ========================================================================
# 数值控制参数
# ========================================================================
class NumericalParameters:
    """数值模拟控制参数。"""
    N_RADIAL: int = 200                 # 径向网格数
    T_MAX: float = 20.0e-9              # 最大模拟时间 [s]
    CFL: float = 0.3                    # CFL数
    MAX_DT: float = 1.0e-12             # 最大时间步 [s]
    MIN_DT: float = 1.0e-16             # 最小时间步 [s]
    ADAPTIVE_TOL: float = 1.0e-6        # RKF45自适应容差

    # 热传导限制
    FLUX_LIMITER: float = 0.06          # 热流限制因子 f
    MAX_FLUX_MULTIPLIER: float = 5.0    # 最大热流倍乘

    # 不稳定性参数
    PERTURBATION_MODE: int = 12         # 初始扰动模式数 l
    PERTURBATION_AMPLITUDE: float = 1.0e-7  # 相对扰动幅度

    # 蒙特卡洛中子采样
    MC_NEUTRON_SAMPLES: int = 5000      # 中子MC采样数


# ========================================================================
# 状态方程参数
# ========================================================================
class EOSParameters:
    """
    状态方程（Equation of State）参数。
    采用理想气体+简并修正+库仑修正的多项式近似。
    """
    GAMMA_IDEAL: float = 5.0 / 3.0      # 理想气体绝热指数
    DEGENERACY_COEFF: float = 2.0       # 电子简并修正系数
    COULOMB_CORRECTION: float = 0.3     # 库仑修正系数


# ========================================================================
# 聚变反应参数（DT反应）
# ========================================================================
class FusionParameters:
    """
    DT聚变反应参数。
    主反应: D + T -> He-4 (3.5 MeV) + n (14.1 MeV)
    """
    Q_ALPHA: float = 3.5e6 * PhysicalConstants.ELEMENTARY_CHARGE  # alpha粒子能量 [J]
    Q_NEUTRON: float = 14.1e6 * PhysicalConstants.ELEMENTARY_CHARGE # 中子能量 [J]
    REACTIVITY_COEFF: np.ndarray = np.array([
        6.6610e-21, 2.4120e-14, 1.0290e-11,
        1.5630e-10, 1.6900e-9, 1.0200e-8,
        2.9750e-8, 4.7680e-8, 3.6970e-8,
        1.0400e-8
    ])  # <sigma*v> 拟合系数 (Bosch-Hale)

    @staticmethod
    def reactivity_dt(T_ion_kev: float) -> float:
        """
        Bosch-Hale parametrization for DT reactivity <sigma*v> [m^3/s].
        T_ion in keV.
        """
        if T_ion_kev <= 0.0:
            return 0.0
        theta = T_ion_kev / (1.0 - (T_ion_kev * (0.0642 + 0.0149 * T_ion_kev))
                             / (1.0 + 0.0642 * T_ion_kev + 0.0149 * T_ion_kev**2))
        xi = (0.2396 * theta)**(1.0 / 3.0)
        # 防止溢出
        if xi > 50.0:
            return 0.0
        sigma_v = 1.0e-6 * 1.17302e-9 * theta * np.sqrt(xi / (0.2396 * T_ion_kev**3)) \
            * np.exp(-3.0 * xi)
        return max(sigma_v, 0.0)


# ========================================================================
# 便捷实例
# ========================================================================
PC = PhysicalConstants()
TP = TargetParameters()
LP = LaserParameters()
NP = NumericalParameters()
EOS = EOSParameters()
FP = FusionParameters()
