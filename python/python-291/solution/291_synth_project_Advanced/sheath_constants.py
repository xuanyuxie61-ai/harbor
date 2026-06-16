"""
sheath_constants.py
===================
物理常数、等离子体参数与无量纲化体系。

本模块定义用于等离子体鞘层计算的物理常数和归一化参数。
采用无量纲化体系：
    - 长度尺度：电子德拜长度 λ_De
    - 速度尺度：离子声速 c_s = sqrt(k_B T_e / m_i)
    - 时间尺度：离子等离子体频率倒数 ω_pi^{-1}
    - 电势尺度：k_B T_e / e

物理模型：
    1. 一维平面鞘层（x 方向，壁面位于 x = L）
    2. 冷离子流体（T_i << T_e）
    3. 电子满足修正 Boltzmann 关系（含二次电子发射）
    4. 壁面为电介质/金属，具有三角化粗糙表面
    5. 外加磁场 B_0（用于磁前鞘计算）
"""

import math
import cmath
from typing import Dict, Tuple, Optional


# =============================================================================
# SI 物理常数
# =============================================================================
EPSILON_0: float = 8.8541878128e-12    # 真空介电常数 [F/m]
E_CHARGE: float = 1.602176634e-19      # 基本电荷 [C]
K_BOLTZMANN: float = 1.380649e-23      # 玻尔兹曼常数 [J/K]
M_PROTON: float = 1.67262192369e-27    # 质子质量 [kg]
M_ELECTRON: float = 9.1093837015e-31   # 电子质量 [kg]
PI: float = math.pi
EULER_GAMMA: float = 0.5772156649015329  # Euler-Mascheroni 常数
SQRT2: float = math.sqrt(2.0)
SQRT_PI: float = math.sqrt(PI)
LN2: float = math.log(2.0)


# =============================================================================
# 等离子体物理参数（默认 Argon 等离子体）
# =============================================================================
class PlasmaParams:
    """
    封装等离子体物理参数。

    属性：
        T_e: 电子温度 [eV]
        T_i: 离子温度 [eV]
        n_0: 参考等离子体密度 [m^{-3}]
        m_i: 离子质量 [kg]
        Z_eff: 有效电荷数
        gamma_e: 二次电子发射系数
        E_see_threshold: SEE 阈值能量 [eV]
        E_see_max: SEE 最大产额对应能量 [eV]
        delta_max: 最大 SEE 产额
        B_0: 外加磁场强度 [T]
        theta_B: 磁场入射角 [rad]
        V_wall: 壁面偏压 [V]
        L_domain: 无量纲计算域长度
        N_grid: 网格点数
        FD_order: 有限差分阶数
    """

    def __init__(
        self,
        T_e: float = 3.0,
        T_i: float = 0.03,
        n_0: float = 1.0e16,
        species: str = "argon",
        gamma_e: float = 0.05,
        B_0: float = 0.01,
        theta_B: float = 0.2618,  # ~15 degrees
        V_wall: float = -30.0,
        L_domain: float = 20.0,
        N_grid: int = 128,
        FD_order: int = 4,
    ):
        # 基本物理量
        self.T_e = T_e
        self.T_i = T_i
        self.n_0 = n_0
        self.gamma_e = gamma_e
        self.B_0 = B_0
        self.theta_B = theta_B
        self.V_wall = V_wall
        self.L_domain = L_domain
        self.N_grid = N_grid
        self.FD_order = FD_order

        # 离子质量选择
        if species == "argon":
            self.m_i = 39.948 * M_PROTON   # Ar
            self.Z_eff = 1
            self.species = "Ar"
        elif species == "helium":
            self.m_i = 4.0026 * M_PROTON   # He
            self.Z_eff = 1
            self.species = "He"
        elif species == "neon":
            self.m_i = 20.180 * M_PROTON   # Ne
            self.Z_eff = 1
            self.species = "Ne"
        elif species == "deuterium":
            self.m_i = 2.014 * M_PROTON    # D
            self.Z_eff = 1
            self.species = "D"
        else:
            raise ValueError(f"未知等离子体种类: {species}")

        # SEE 参数 (通用经验参数)
        self.E_see_threshold = 15.0   # eV
        self.E_see_max = 300.0        # eV
        self.delta_max_see = 1.2      # 最大产额

    # =========================================================================
    # 导出量计算
    # =========================================================================

    @property
    def T_e_J(self) -> float:
        """电子温度 [J]"""
        return self.T_e * E_CHARGE

    @property
    def T_i_J(self) -> float:
        """离子温度 [J]"""
        return self.T_i * E_CHARGE

    @property
    def lambda_De(self) -> float:
        """
        电子德拜长度 [m]
            λ_De = sqrt(ε₀ k_B T_e / (n₀ e²))
        """
        return math.sqrt(
            EPSILON_0 * self.T_e_J / (self.n_0 * E_CHARGE**2)
        )

    @property
    def c_s(self) -> float:
        """
        离子声速 (Bohm 速度) [m/s]
            c_s = sqrt(k_B T_e / m_i)    (冷离子极限)
        考虑有限离子温度修正：
            c_s = sqrt((Z T_e + γ_i T_i) / m_i)  其中 γ_i = 3 (1D)
        """
        gamma_i = 3.0  # 一维绝热指数
        return math.sqrt(
            self.Z_eff * self.T_e_J + gamma_i * self.T_i_J
        ) / math.sqrt(self.m_i)

    @property
    def omega_pi(self) -> float:
        """
        离子等离子体频率 [rad/s]
            ω_pi = sqrt(n₀ Z² e² / (ε₀ m_i))
        """
        return math.sqrt(
            self.n_0 * (self.Z_eff * E_CHARGE)**2
            / (EPSILON_0 * self.m_i)
        )

    @property
    def omega_pe(self) -> float:
        """
        电子等离子体频率 [rad/s]
            ω_pe = sqrt(n₀ e² / (ε₀ m_e))
        """
        return math.sqrt(
            self.n_0 * E_CHARGE**2 / (EPSILON_0 * M_ELECTRON)
        )

    @property
    def omega_ce(self) -> float:
        """
        电子回旋频率 [rad/s]
            ω_ce = e B₀ / m_e
        """
        return E_CHARGE * self.B_0 / M_ELECTRON

    @property
    def omega_ci(self) -> float:
        """
        离子回旋频率 [rad/s]
            ω_ci = Z e B₀ / m_i
        """
        return self.Z_eff * E_CHARGE * self.B_0 / self.m_i

    @property
    def mass_ratio(self) -> float:
        """电子-离子质量比 μ = m_e/m_i"""
        return M_ELECTRON / self.m_i

    @property
    def debye_ratio(self) -> float:
        """德拜长度与系统尺度比 ε = λ_De / L"""
        physical_L = self.L_domain * self.lambda_De
        return self.lambda_De / physical_L

    @property
    def v_bohm(self) -> float:
        """
        Bohm 速度 [m/s]
        Bohm 判据：u_i(x_s) ≥ c_s
            u_B = sqrt(Z k_B T_e / m_i)  (冷离子极限)
        """
        return math.sqrt(self.Z_eff * self.T_e_J / self.m_i)

    @property
    def rho_s(self) -> float:
        """
        声速拉莫尔半径 [m]
            ρ_s = c_s / ω_ci
        """
        if self.omega_ci < 1e-30:
            return float('inf')
        return self.c_s / self.omega_ci

    @property
    def chi_see(self) -> float:
        """
        有效电子温度降低因子 (含二次电子发射)
            χ = (1 - γ_eff) / (1 - γ_eff * sqrt(m_e / (2π m_i)))
        其中 γ_eff 为有效 SEE 系数
        """
        mu = self.mass_ratio
        gamma_eff = min(self.gamma_e, 0.99)  # 防止发散
        sqrt_factor = math.sqrt(mu / (2.0 * PI))
        denominator = 1.0 - gamma_eff * sqrt_factor
        if abs(denominator) < 1e-15:
            return 1.0
        return (1.0 - gamma_eff) / denominator

    def see_yield(self, E_impact: float) -> float:
        """
        二次电子发射产额 (Vaughan 模型)
            δ(E) = δ_max * (E/E_max)^0.7    当 E < E_max
            δ(E) = δ_max * (E/E_max)^{-0.5}  当 E ≥ E_max

        参数：
            E_impact: 入射电子能量 [eV]

        返回：
            δ: SEE 产额
        """
        if E_impact <= 0.0:
            return 0.0
        E_max = self.E_see_max
        delta_max = self.delta_max_see

        ratio = E_impact / E_max
        if ratio < 1.0:
            return delta_max * (ratio ** 0.7)
        else:
            return delta_max * (ratio ** (-0.5))

    def dimensionless_wall_potential(self) -> float:
        """
        无量纲壁面电势
            φ_wall = e V_wall / (k_B T_e)
        """
        return E_CHARGE * self.V_wall / self.T_e_J

    def bohman_criterion_check(self, u_i_edge: float) -> Tuple[bool, float]:
        """
        检查 Bohm 判据
            u_i(x_s) ≥ c_s
        返回 (是否满足, 比值 u_i/c_s)
        """
        ratio = u_i_edge / self.v_bohm
        return (ratio >= 1.0 - 1e-10, ratio)

    def summary(self) -> str:
        """输出等离子体参数摘要"""
        lines = [
            "=" * 70,
            "等离子体鞘层参数摘要",
            "=" * 70,
            f"  等离子体种类        : {self.species}",
            f"  电子温度 T_e        : {self.T_e:.2f} eV",
            f"  离子温度 T_i        : {self.T_i:.3f} eV",
            f"  参考密度 n₀         : {self.n_0:.3e} m^-3",
            f"  离子质量 m_i        : {self.m_i:.4e} kg",
            f"  质量比 μ=m_e/m_i    : {self.mass_ratio:.4e}",
            "-" * 70,
            f"  德拜长度 λ_De       : {self.lambda_De:.4e} m",
            f"  离子声速 c_s        : {self.c_s:.4e} m/s",
            f"  Bohm 速度 u_B       : {self.v_bohm:.4e} m/s",
            f"  离子等离子体频率 ω_pi: {self.omega_pi:.4e} rad/s",
            f"  电子等离子体频率 ω_pe: {self.omega_pe:.4e} rad/s",
            f"  电子回旋频率 ω_ce   : {self.omega_ce:.4e} rad/s",
            f"  离子回旋频率 ω_ci   : {self.omega_ci:.4e} rad/s",
            f"  声速拉莫尔半径 ρ_s  : {self.rho_s:.4e} m",
            "-" * 70,
            f"  SEE 系数 γ_eff      : {self.gamma_e:.3f}",
            f"  SEE 降低因子 χ      : {self.chi_see:.4f}",
            f"  壁面电势 (无量纲)   : {self.dimensionless_wall_potential():.3f}",
            f"  计算域长度 (无量纲) : {self.L_domain:.1f}",
            f"  网格点数 N          : {self.N_grid}",
            f"  有限差分阶数        : {self.FD_order}",
            "=" * 70,
        ]
        return "\n".join(lines)


# =============================================================================
# 方便函数
# =============================================================================

def plasma_frequency(n: float, m: float, Z: int = 1) -> float:
    """
    等离子体频率通用计算 [rad/s]
        ω_p = sqrt(n Z² e² / (ε₀ m))
    """
    return math.sqrt(n * (Z * E_CHARGE)**2 / (EPSILON_0 * m))


def debye_length(T_e_eV: float, n_e: float) -> float:
    """
    德拜长度 [m]
        λ_De = sqrt(ε₀ k_B T_e / (n_e e²))
    """
    T_e_J = T_e_eV * E_CHARGE
    return math.sqrt(EPSILON_0 * T_e_J / (n_e * E_CHARGE**2))


def larmor_radius(v_perp: float, B: float, m: float, q: float) -> float:
    """
    拉莫尔半径 [m]
        r_L = m v_⊥ / (|q| B)
    """
    if abs(B) < 1e-30:
        return float('inf')
    return m * abs(v_perp) / (abs(q) * abs(B))


def coulomb_logarithm(T_e_eV: float, n_e: float) -> float:
    """
    库仑对数 (等离子体)
        ln Λ = 14.9 - 0.5 ln(n_e / 10^20) + ln(T_e / 100)
    适用范围：T_e in eV, n_e in m^-3, 热等离子体
    更精确形式：
        ln Λ = ln(12 π sqrt(ε₀³ (k_B T_e)³ / (n_e e⁶)))
    """
    T_e_J = T_e_eV * E_CHARGE
    if n_e <= 0.0 or T_e_J <= 0.0:
        return 10.0  # 典型值
    arg = 12.0 * PI * math.sqrt(
        EPSILON_0**3 * T_e_J**3 / (n_e * E_CHARGE**6)
    )
    if arg <= 1.0:
        return 10.0
    ln_L = math.log(arg)
    return max(ln_L, 2.0)  # 下限保护
