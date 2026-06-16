"""
config.py — 全局配置与物理参数
============================================
科学领域: 数学优化 — 分布式优化与 ADMM 方法
项目: 分布式 ADMM 多物理场 PDE 约束优化框架

本模块定义所有物理常数、数值参数、ADMM 超参数和实验配置。
所有参数均基于严格的无量纲化处理。
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List


# ============================================================
#  数学常量
# ============================================================
PI = np.pi
EULER_GAMMA = 0.5772156649015329  # Euler-Mascheroni constant
SQRT2 = np.sqrt(2.0)
SQRT3 = np.sqrt(3.0)
EPS_NUM = 1.0e-14  # 数值容限 (避免除零)
MACHINE_EPS = np.finfo(np.float64).eps


# ============================================================
#  Gray-Scott 反应扩散系统参数
#  参考: Gray & Scott (1984), Autocatalytic reactions
#  dU/dt = Du * Laplacian(U) - U*V^2 + gamma*(1 - U)
#  dV/dt = Dv * Laplacian(V) + U*V^2 - (gamma + kappa)*V
# ============================================================
@dataclass
class GrayScottConfig:
    """Gray-Scott 反应扩散参数 (无量纲)"""
    Du: float = 0.16        # 物种 U 的扩散系数
    Dv: float = 0.08        # 物种 V 的扩散系数 (Du/Dv = 2 为典型比值)
    gamma: float = 0.024    # 供给率 (feed rate)
    kappa: float = 0.056    # 杀灭率 (kill rate)
    # 典型模式参数:
    #   gamma=0.024, kappa=0.056 → 斑点模式 (spots)
    #   gamma=0.030, kappa=0.054 → 条纹模式 (stripes)
    #   gamma=0.022, kappa=0.051 → 环状模式 (rings)
    initial_U: float = 1.0  # U 的初始均匀浓度
    initial_V: float = 0.0  # V 的初始均匀浓度
    perturbation: float = 0.01  # 中心扰动幅度
    nx: int = 40
    ny: int = 40
    Lx: float = 2.5        # 空间域 [0, Lx]
    Ly: float = 2.5        # 空间域 [0, Ly]
    T_final: float = 5.0   # 最终时间
    cfl_safety: float = 0.8  # CFL 安全系数

    @property
    def dx(self) -> float:
        return self.Lx / self.nx

    @property
    def dy(self) -> float:
        return self.Ly / self.ny

    @property
    def dt_max(self) -> float:
        """最大允许时间步长 (扩散稳定性条件)
        dt <= dx^2 * dy^2 / (2 * D_max * (dx^2 + dy^2))
        """
        D_max = max(self.Du, self.Dv)
        return self.cfl_safety * (self.dx**2 * self.dy**2) / (
            2.0 * D_max * (self.dx**2 + self.dy**2)
        )

    def __post_init__(self):
        if self.Du <= 0 or self.Dv <= 0:
            raise ValueError(f"扩散系数必须为正: Du={self.Du}, Dv={self.Dv}")
        if self.gamma < 0 or self.kappa < 0:
            raise ValueError(f"反应速率必须非负: gamma={self.gamma}, kappa={self.kappa}")


# ============================================================
#  Hodgkin-Huxley 神经轴突模型参数
#  参考: Hodgkin & Huxley (1952), J. Physiol.
#  C dV/dt = I_ext - g_Na*m^3*h*(V-E_Na) - g_K*n^4*(V-E_K) - g_L*(V-E_L)
# ============================================================
@dataclass
class HodgkinHuxleyConfig:
    """Hodgkin-Huxley 膜电位模型参数"""
    C_m: float = 1.0          # 膜电容 (muF/cm^2)
    g_Na: float = 120.0       # 钠离子最大电导 (mS/cm^2)
    g_K: float = 36.0         # 钾离子最大电导 (mS/cm^2)
    g_L: float = 0.3          # 漏电流电导 (mS/cm^2)
    E_Na: float = 50.0        # 钠离子反转电位 (mV)
    E_K: float = -77.0        # 钾离子反转电位 (mV)
    E_L: float = -54.387      # 漏电反转电位 (mV)
    V0: float = -65.0         # 初始膜电位 (mV)
    T_final: float = 50.0     # 模拟时长 (ms)
    dt: float = 0.01          # 时间步长 (ms)
    I_ext: float = 10.0       # 外部注入电流 (muA/cm^2)

    def alpha_n(self, V: np.ndarray) -> np.ndarray:
        """n 门控变量正向速率: alpha_n(V) = 0.01*(10-V)/(exp((10-V)/10)-1)"""
        dV = 10.0 - V
        return np.where(
            np.abs(dV) < EPS_NUM,
            0.1,  # L'Hopital 极限
            0.01 * dV / (np.exp(dV / 10.0) - 1.0 + EPS_NUM)
        )

    def beta_n(self, V: np.ndarray) -> np.ndarray:
        """n 门控变量反向速率: beta_n(V) = 0.125*exp(-V/80)"""
        return 0.125 * np.exp(-V / 80.0)

    def alpha_m(self, V: np.ndarray) -> np.ndarray:
        """m 门控变量正向速率: alpha_m(V) = 0.1*(25-V)/(exp((25-V)/10)-1)"""
        dV = 25.0 - V
        return np.where(
            np.abs(dV) < EPS_NUM,
            1.0,
            0.1 * dV / (np.exp(dV / 10.0) - 1.0 + EPS_NUM)
        )

    def beta_m(self, V: np.ndarray) -> np.ndarray:
        """m 门控变量反向速率: beta_m(V) = 4.0*exp(-V/18)"""
        return 4.0 * np.exp(-V / 18.0)

    def alpha_h(self, V: np.ndarray) -> np.ndarray:
        """h 门控变量正向速率: alpha_h(V) = 0.07*exp(-V/20)"""
        return 0.07 * np.exp(-V / 20.0)

    def beta_h(self, V: np.ndarray) -> np.ndarray:
        """h 门控变量反向速率: beta_h(V) = 1/(exp((30-V)/10)+1)"""
        return 1.0 / (np.exp((30.0 - V) / 10.0) + 1.0)


# ============================================================
#  大气臭氧化学模型参数
#  参考: Chapman mechanism + NOx cycle
#  4 物种: O(^3P), O(^1D), O3, NO2
# ============================================================
@dataclass
class OzoneChemistryConfig:
    """大气臭氧-氮氧化物化学动力学参数"""
    k2: float = 1.0e-3        # O + O3 -> 2O2 速率常数
    k3: float = 1.0e-4        # NO + O3 -> NO2 + O2 速率常数
    sigma2: float = 1.0e-11   # O2 光解源项
    y0: Tuple[float, ...] = (1.0e6, 1.0e9, 1.0e12, 1.0e9)  # [O, NO, NO2, O3]
    T_final: float = 24.0     # 24 小时模拟
    dt: float = 0.01          # 时间步长 (小时)

    def k1_diurnal(self, t: float) -> float:
        """日变化光解速率 k1(t)
        白天 (4h <= t <= 20h): k1 = 1e-5 * exp(7 * sin(pi*(t-4)/16)^0.2)
        夜间: k1 = 1e-40 (近似为零)
        """
        t_mod = t % 24.0
        if 4.0 <= t_mod <= 20.0:
            arg = PI * (t_mod - 4.0) / 16.0
            return 1.0e-5 * np.exp(7.0 * np.sin(arg) ** 0.2)
        return 1.0e-40


# ============================================================
#  粘性 Burgers 方程参数
#  du/dt + u*du/dx = nu * d^2u/dx^2
#  精确解: Hopf-Cole 变换
# ============================================================
@dataclass
class BurgersConfig:
    """粘性 Burgers 方程参数"""
    nu: float = 0.01          # 运动粘度 (Re = 1/nu)
    nx: int = 100             # 空间网格点数
    L: float = 2.0            # 域长 [-L, L]
    T_final: float = 1.0      # 最终时间
    cfl_safety: float = 0.4   # CFL 安全系数
    bc_type: str = "dirichlet"  # 边界条件类型: dirichlet, periodic, neumann
    ic_type: str = "shock"    # 初始条件: shock, gaussian, sine

    @property
    def dx(self) -> float:
        return 2.0 * self.L / (self.nx - 1)

    @property
    def dt_max(self) -> float:
        """稳定性条件: dt <= min(dx^2/(2*nu), dx/u_max)"""
        dt_visc = self.dx**2 / (2.0 * self.nu + EPS_NUM)
        dt_adv = self.dx / (1.0 + EPS_NUM)  # u_max ~ 1
        return self.cfl_safety * min(dt_visc, dt_adv)


# ============================================================
#  ADMM 分布式优化参数
#  标准 ADMM 形式:
#    min f(x) + g(z)  s.t.  Ax + Bz = c
#  增广 Lagrangian:
#    L_rho = f(x) + g(z) + (rho/2)||Ax + Bz - c + u||^2
# ============================================================
@dataclass
class ADMMConfig:
    """ADMM 分布式优化超参数"""
    rho: float = 1.0              # 初始罚参数 rho
    rho_max: float = 1.0e6        # 罚参数上界
    rho_min: float = 1.0e-4       # 罚参数下界
    abs_tol: float = 1.0e-5       # 绝对容差
    rel_tol: float = 1.0e-3       # 相对容差
    max_iter: int = 500           # 最大迭代次数
    mu: float = 10.0              # 残差比率阈值 (自适应惩罚)
    tau_incr: float = 2.0         # 罚参数增长因子
    tau_decr: float = 2.0         # 罚参数衰减因子
    alpha_overrelax: float = 1.5  # 外推松弛因子 (Boyd et al. 2011)
    warm_start: bool = True       # 是否使用热启动
    use_adaptive_penalty: bool = True  # 是否自适应调参
    use_nesterov_accel: bool = False   # Nesterov 加速标志


# ============================================================
#  全局实验配置
# ============================================================
@dataclass
class ExperimentConfig:
    """实验总配置"""
    # 全局网格
    nx_global: int = 24
    ny_global: int = 24
    n_subdomains: int = 4       # 子域数量 (2x2 分解)

    # 物理场选择
    primary_physics: str = "diffusion"  # diffusion, gray_scott, burgers

    # 噪声水平 (观测数据)
    noise_level: float = 0.01

    # 正则化参数
    alpha_reg: float = 1.0e-3   # Tikhonov 正则化

    # 子配置
    gs_config: GrayScottConfig = field(default_factory=GrayScottConfig)
    hh_config: HodgkinHuxleyConfig = field(default_factory=HodgkinHuxleyConfig)
    ozone_config: OzoneChemistryConfig = field(default_factory=OzoneChemistryConfig)
    burgers_config: BurgersConfig = field(default_factory=BurgersConfig)
    admm_config: ADMMConfig = field(default_factory=ADMMConfig)

    def validate(self):
        """验证配置一致性"""
        if self.nx_global < 4 or self.ny_global < 4:
            raise ValueError("全局网格至少 4x4")
        sqrt_n = int(np.sqrt(self.n_subdomains))
        if sqrt_n * sqrt_n != self.n_subdomains:
            raise ValueError("子域数量必须为完全平方数")
        if self.nx_global % sqrt_n != 0 or self.ny_global % sqrt_n != 0:
            raise ValueError("全局网格必须被子域数整除")
        if self.noise_level < 0:
            raise ValueError("噪声水平必须非负")
        if self.alpha_reg < 0:
            raise ValueError("正则化参数必须非负")
