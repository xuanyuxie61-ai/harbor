"""
physical_constants.py — 等离子体物理常数与无量纲化参数
======================================================

本项目采用 **无量纲单位制** (Heaviside-Lorentz + 归一化):
  速度归一化: x = v / v_th, 其中 v_th = sqrt(2 k_B T_0 / m_e)
  时间归一化: τ = t / τ_c, 其中 τ_c = 3 sqrt(m_e) (k_B T_0)^{3/2} /
                              (4 π n_e e^4 ln Λ)
  分布函数归一化: f̃ = f / (n_e v_th^{-3})

物理参考:
  Helander & Sigmar, "Collisional Transport in Magnetized Plasmas", CUP 2002.
  Chang & Cooper, JCP 6, 1 (1970).
  Chandrasekhar, ApJ 97, 255 (1943).
"""

import math
import numpy as np


# ===========================================================================
#  §1  SI 基本物理常数 (CODATA 2018)
# ===========================================================================
ELECTRON_MASS = 9.1093837015e-31       # m_e  [kg]
ELEMENTARY_CHARGE = 1.602176634e-19    # e    [C]
BOLTZMANN_CONST = 1.380649e-23         # k_B  [J/K]
VACUUM_PERMITTIVITY = 8.8541878128e-12 # ε_0  [F/m]
PLANCK_CONST = 6.62607015e-34          # h    [J·s]
HBAR = PLANCK_CONST / (2.0 * math.pi) # ℏ    [J·s]
SPEED_OF_LIGHT = 2.99792458e8          # c    [m/s]
BOHR_RADIUS = 5.29177210903e-11        # a_0  [m]
FINE_STRUCTURE = 7.2973525693e-3       # α
PI = math.pi
SQRT_PI = math.sqrt(PI)
TWO_PI = 2.0 * PI
FOUR_PI = 4.0 * PI


# ===========================================================================
#  §2  典型托卡马克芯部等离子体参数 (ITER-like)
# ===========================================================================
class PlasmaParameters:
    """封装一个完整等离子体状态所需的物理参数.

    Attributes
    ----------
    T_e : float   电子温度 [eV]
    n_e : float   电子数密度 [m^{-3}]
    Z_eff : float 有效电荷数
    m_i_amu : float  离子质量 [原子质量单位]
    ln_lambda : float 库仑对数
    """

    def __init__(self, T_e_eV=1000.0, n_e=1.0e19, Z_eff=1.0, m_i_amu=2.014,
                 ln_lambda=None):
        # 基本参数
        self.T_e_eV = float(T_e_eV)
        self.T_e = self.T_e_eV * ELEMENTARY_CHARGE          # [J]
        self.n_e = float(n_e)
        self.Z_eff = float(Z_eff)
        self.m_i = m_i_amu * 1.66053906660e-27              # [kg]
        self.m_e = ELECTRON_MASS

        # 导出量
        self.v_th = math.sqrt(2.0 * self.T_e / self.m_e)    # 热速度 [m/s]
        self.omega_pe = math.sqrt(                            # 等离子体频率 [rad/s]
            self.n_e * ELEMENTARY_CHARGE**2 /
            (self.m_e * VACUUM_PERMITTIVITY)
        )
        self.lambda_D = math.sqrt(                            # Debye 长度 [m]
            VACUUM_PERMITTIVITY * self.T_e /
            (self.n_e * ELEMENTARY_CHARGE**2)
        )
        # 经典最近接近距离
        self.b_classical = ELEMENTARY_CHARGE**2 / (
            FOUR_PI * VACUUM_PERMITTIVITY * self.T_e
        )
        # 量子距离 (de Broglie)
        self.b_quantum = HBAR / math.sqrt(2.0 * self.m_e * self.T_e)
        # 库仑对数
        if ln_lambda is not None:
            self.ln_lambda = float(ln_lambda)
        else:
            self.ln_lambda = self._compute_coulomb_logarithm()
        # 碰撞时间 [s]
        self.tau_c = (
            3.0 * math.sqrt(self.m_e) * self.T_e**1.5 /
            (FOUR_PI * self.n_e * ELEMENTARY_CHARGE**4 * self.ln_lambda)
        )
        # 碰撞频率 [1/s]
        self.nu_c = 1.0 / self.tau_c

    def _compute_coulomb_logarithm(self):
        """计算库仑对数 ln Λ = max(ln(b_D/b_min)).

        分别计算经典和量子截止距离，取较大者.
        """
        b_min = max(self.b_classical, self.b_quantum)
        if self.lambda_D <= b_min:
            return 2.0  # 强耦合极限
        val = 0.5 * math.log(1.0 + (self.lambda_D / b_min)**2)
        return max(val, 2.0)

    def summary(self):
        lines = [
            "=" * 70,
            "等离子体参数摘要",
            "=" * 70,
            f"  T_e           = {self.T_e_eV:.2f} eV  "
            f"({self.T_e:.4e} J)",
            f"  n_e           = {self.n_e:.3e} m^-3",
            f"  Z_eff         = {self.Z_eff:.1f}",
            f"  v_th          = {self.v_th:.4e} m/s",
            f"  omega_pe      = {self.omega_pe:.4e} rad/s",
            f"  lambda_D      = {self.lambda_D:.4e} m",
            f"  b_classical   = {self.b_classical:.4e} m",
            f"  b_quantum     = {self.b_quantum:.4e} m",
            f"  ln(Lambda)    = {self.ln_lambda:.4f}",
            f"  tau_c         = {self.tau_c:.4e} s",
            f"  nu_c          = {self.nu_c:.4e} s^-1",
            "=" * 70,
        ]
        return "\n".join(lines)


# ===========================================================================
#  §3  无量纲辅助函数
# ===========================================================================
def maxwellian_1d(x, T_ratio=1.0):
    """归一化 1D (各向同性) Maxwellian: f_0(x) = (pi T)^{-3/2} exp(-x^2/T).

    Parameters
    ----------
    x : float or ndarray  无量纲速度 x = v/v_th
    T_ratio : float  T / T_0 (温度比)
    """
    return (PI * T_ratio)**(-1.5) * np.exp(-x**2 / T_ratio)


def maxwellian_shell(x, T_ratio=1.0):
    """Maxwellian 乘以球壳体积元 4 pi x^2.

    n(x) dx = 4 pi x^2 f(x) dx
    """
    return FOUR_PI * x**2 * maxwellian_1d(x, T_ratio)


def energy_density(x, T_ratio=1.0):
    """能量密度核: (1/2) m v^2 f(x) * 4 pi x^2  (无量纲)."""
    return 0.5 * x**2 * maxwellian_shell(x, T_ratio)


def thermal_speed_from_T(T_ratio):
    """v_th(T) / v_th(T_0) = sqrt(T/T_0)."""
    return math.sqrt(max(T_ratio, 1.0e-30))


def number_density_from_T(T_ratio):
    """归一化数密度 (应等于 1 当 T_ratio=1)."""
    return 1.0


def entropy_maxwellian(T_ratio):
    """Maxwellian 的 Boltzmann 熵 (无量纲，到常数):
    S = -∫ f ln f d^3v = (3/2)(1 + ln(π T)) + const.
    """
    return 1.5 * (1.0 + math.log(PI * max(T_ratio, 1.0e-30)))
