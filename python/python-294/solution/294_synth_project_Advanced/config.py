"""
config.py - 物理常数与仿真参数配置

本模块定义了激光等离子体相互作用仿真中使用的全部物理常数和数值参数。
所有量均采用无量纲归一化单位制，便于数值计算。

归一化方案:
  - 长度单位: c / omega_p0 (等离子体趋肤深度)
  - 时间单位: 1 / omega_p0
  - 速度单位: c (光速)
  - 能量单位: m_e * c^2 (电子静止能量)

核心物理常数 (SI 单位):
  c          = 2.998e8 m/s       光速
  e          = 1.602e-19 C       基本电荷
  m_e        = 9.109e-31 kg      电子质量
  m_p        = 1.673e-27 kg      质子质量
  epsilon_0  = 8.854e-12 F/m     真空介电常数
  mu_0       = 1.257e-6 H/m      真空磁导率
  k_B        = 1.381e-23 J/K     玻尔兹曼常数
  hbar       = 1.055e-34 J*s     约化普朗克常数
  epsilon_m  = m_e*c^2 = 0.511 MeV  电子静止能量
"""

import numpy as np


class PhysicalConstants:
    """SI 单位制下的基本物理常数。

    这些常数用于无量纲化过程中的尺度转换，以及
    从仿真无量纲单位到物理单位的后处理转换。

    等离子体频率定义:
        omega_p = sqrt(n_e * e^2 / (m_e * epsilon_0))

    德拜长度定义:
        lambda_D = sqrt(epsilon_0 * k_B * T_e / (n_e * e^2))

    等离子体 beta 参数:
        beta = 2 * mu_0 * n_e * k_B * T_e / B^2
    """

    c_light = 2.998e8          # 光速 [m/s]
    e_charge = 1.602e-19       # 基本电荷 [C]
    m_electron = 9.109e-31     # 电子质量 [kg]
    m_proton = 1.673e-27       # 质子质量 [kg]
    epsilon_0 = 8.854e-12      # 真空介电常数 [F/m]
    mu_0 = 1.257e-6            # 真空磁导率 [H/m]
    k_B = 1.381e-23            # 玻尔兹曼常数 [J/K]
    hbar = 1.055e-34           # 约化普朗克常数 [J*s]
    pi = np.pi
    c2 = c_light ** 2          # 光速平方
    me_c2 = m_electron * c_light ** 2  # 电子静止能量 [J]


class NormalizationScales:
    """从物理参考值计算归一化尺度。

    参考等离子体参数:
        n_0   = 1e25 m^-3  (参考电子数密度)
        T_e   = 1.0 keV    (电子温度)
        lambda_L = 1.053 um (激光波长, Nd:Glass)

    导出量:
        omega_p0 = sqrt(n_0 * e^2 / (m_e * eps_0))  等离子体频率
        skin_depth = c / omega_p0                      趋肤深度
        v_th = sqrt(k_B * T_e / m_e)                  热速度
        lambda_D = v_th / omega_p0                     德拜长度
    """

    def __init__(self, n0=1.0e25, Te_keV=1.0):
        pc = PhysicalConstants
        self.n0 = n0
        self.Te_J = Te_keV * 1.602e-16  # keV -> J
        self.omega_p0 = np.sqrt(
            n0 * pc.e_charge ** 2 / (pc.m_electron * pc.epsilon_0)
        )
        self.skin_depth = pc.c_light / self.omega_p0
        self.v_th = np.sqrt(self.Te_J / pc.m_electron)
        self.lambda_D = self.v_th / self.omega_p0
        self.tau_norm = 1.0 / self.omega_p0
        self.lambda_L_um = 1.053
        self.omega_L = 2.0 * np.pi * pc.c_light / (self.lambda_L_um * 1.0e-6)


class SimulationConfig:
    """仿真运行的完整参数集。

    空间域:
        L_x = 20 * c/omega_p0     计算域长度
        N_x = 256                  空间网格点数

    时间域:
        T_end = 50 / omega_p0     终止时间 (约 8 个等离子体周期)
        CFL    = 0.5               库朗数 (CFL 稳定条件)

    等离子体密度剖面:
        n(x) = n_max * exp(-(x - L/2)^2 / (2 * sigma_n^2))
        形成高斯密度峰，模拟激光与密度梯度等离子体的相互作用

    激光驱动参数:
        a_0 = e * A / (m_e * c^2)  归一化矢势 (相对论参数)
        omega_0 = 1.1 * omega_p0   激光频率 (略高于等离子体频率)

    速度空间:
        N_v = 128                  速度格点数
        v_max = 6 * v_th           速度截断
    """

    def __init__(self):
        self.norm = NormalizationScales()

        # === 空间网格参数 ===
        self.L_x = 20.0                   # 域长度 [c/omega_p0]
        self.N_x = 256                     # 空间格点数
        self.dx = self.L_x / (self.N_x - 1)  # 空间步长

        # === 时间步进参数 ===
        self.CFL = 0.5                     # 库朗数
        self.dt = self.CFL * self.dx       # 时间步长 [1/omega_p0]
        self.T_end = 50.0                  # 终止时间
        self.N_t = int(self.T_end / self.dt)  # 总步数

        # === 等离子体密度剖面参数 ===
        self.n_max = 2.0                   # 峰值密度 [n_0]
        self.sigma_n = 3.0                 # 密度剖面宽度 [c/omega_p0]

        # === 激光驱动参数 ===
        self.a0 = 0.5                      # 归一化矢势 (非线性 regimes)
        self.omega_0 = 1.1                 # 激光频率 [omega_p0]
        self.tau_pulse = 5.0               # 脉冲半宽度 [1/omega_p0]
        self.t_rise = 1.0                  # 脉冲上升时间

        # === 速度空间参数 ===
        self.N_v = 128                     # 速度格点数
        self.v_max = 6.0                   # 最大速度 [v_th]
        self.dv = 2.0 * self.v_max / (self.N_v - 1)

        # === 高阶有限差分参数 ===
        self.fd_order = 4                  # 空间差分精度阶数 (2,4,6,8)
        self.dg_order = 3                  # DG 多项式阶数

        # === 碰撞算子参数 ===
        self.nu_ei_base = 0.01             # 基准电子-离子碰撞频率 [omega_p0]
        self.log_normal_mu = -1.0          # 超热电子 log-normal mu 参数
        self.log_normal_sigma = 0.8        # 超热电子 log-normal sigma 参数

        # === 贝叶斯诊断参数 ===
        self.n_mcmc_walkers = 32           # MCMC 行走器数量
        self.n_mcmc_burn = 500             # 燃烧步数
        self.n_mcmc_prod = 1500            # 生产步数
        self.mcmc_seed = 42                # 随机种子 (可复现性)

        # === 数值容差 ===
        self.tol_newton = 1.0e-12          # Newton 迭代容差
        self.max_iter_newton = 200         # Newton 最大迭代
        self.tol_orthogonality = 1.0e-10   # 正交性检验容差
        self.tol_stability = 1.0e-14       # 稳定性判断容差

        # === 空间网格坐标 (自动构建) ===
        self.x = np.linspace(0, self.L_x, self.N_x)
        self.v = np.linspace(-self.v_max, self.v_max, self.N_v)

    def plasma_density_profile(self, x=None):
        """计算等离子体密度剖面 n(x)。

        采用高斯密度分布:
            n(x) = n_max * exp(-(x - L_x/2)^2 / (2 * sigma_n^2))

        这种剖面模拟了激光与有限尺寸等离子体靶的相互作用,
        其中密度从边缘的日冕区平滑过渡到中心的过密度区。

        Parameters
        ----------
        x : ndarray, optional
            空间坐标数组。如果为 None，使用配置中的默认网格。

        Returns
        -------
        n : ndarray
            归一化电子密度 n(x)/n_0
        """
        if x is None:
            x = self.x
        xc = self.L_x / 2.0
        return self.n_max * np.exp(-(x - xc) ** 2 / (2.0 * self.sigma_n ** 2))

    def laser_envelope(self, t):
        """计算激光脉冲时间包络。

        采用 sin^2 包络 (有限持续时间的平滑脉冲):
            E(t) = a_0 * sin^2(pi * t / tau_total)  for 0 < t < tau_total
            E(t) = 0                                  otherwise

        其中 tau_total = t_rise + tau_pulse + t_rise

        Parameters
        ----------
        t : float or ndarray
            时间坐标 [1/omega_p0]

        Returns
        -------
        envelope : float or ndarray
            归一化激光包络振幅
        """
        tau_total = 2.0 * self.t_rise + self.tau_pulse
        t = np.asarray(t, dtype=float)
        envelope = np.zeros_like(t)
        mask = (t > 0) & (t < tau_total)
        envelope[mask] = self.a0 * np.sin(np.pi * t[mask] / tau_total) ** 2
        return envelope

    def critical_density_position(self):
        """计算临界密度面位置 (omega_L = omega_p)。

        临界密度条件: n(x_c) = (omega_L / omega_p0)^2 * n_0
        对于高斯密度剖面:
            x_c = L_x/2 +/- sigma_n * sqrt(-2 * ln(n_c / n_max))

        Returns
        -------
        x_c : float or None
            临界密度面位置。如果激光频率过高 (无临界面), 返回 None。
        """
        n_critical = self.omega_0 ** 2  # omega_0 已归一化
        if n_critical >= self.n_max:
            return None
        xc = self.L_x / 2.0
        arg = -2.0 * self.sigma_n ** 2 * np.log(n_critical / self.n_max)
        if arg < 0:
            return None
        return xc - np.sqrt(arg)  # 返回左侧临界面 (激光入射侧)
