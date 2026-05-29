"""
nanoparticle_dynamics.py
========================
纳米颗粒朗之万动力学与包裹动力学模块（源自 seed 488_grazing_ode）

本模块将 Thompson & Stewart 的放牧模型（带有饱和型 II 功能反应的捕食-被捕食 ODE）
改造为**纳米颗粒与生物膜相互作用的随机动力学模型**。核心思想：

    - 纳米颗粒（NP）的运动类比于食草动物（herbivore）；
    - 膜表面的受体/配体结合位点类比于植被（plant）；
    - Type-II 功能响应 c*v*(1-exp(-d*u)) 描述受体结合饱和动力学。

随机微分方程（ overdamped Langevin in 1D normal direction z ）：

    gamma * dz/dt = F_elec(z) + F_vdw(z) + F_bend(z) + F_bind(z) + xi(t)

其中：
    - gamma = 6*pi*eta*R_np  （Stokes 阻力系数，单位：kg/s -> kJ*ns/(nm^2*mol)）
    - F_elec(z)  来自 Poisson-Boltzmann 的电场力（屏蔽库仑）
    - F_vdw(z)   范德华相互作用（Lennard-Jones 型）
    - F_bend(z)  膜弯曲能产生的回复力
    - F_bind(z)  受体-配体结合力（Type-II 饱和响应）
    - xi(t)      高斯白噪声，满足 <xi(t) xi(t')> = 2*gamma*k_B*T*delta(t-t')

Type-II 结合力公式（改编自 grazing_ode）：
    设 u 为膜表面未结合受体密度，v 为已结合配体密度（与 NP 位置 z 耦合）。
    在准稳态近似下，结合力表示为：
        F_bind(z) = F_max * (1 - exp(-kappa_bind * max(z_cutoff - z, 0)))
    其中 z_cutoff 为有效作用距离，kappa_bind 为结合强度系数。
"""

import numpy as np
from typing import Tuple, Callable


class NanoparticleLangevinDynamics:
    """
    一维（膜法向）纳米颗粒朗之万动力学积分器。
    """

    def __init__(self,
                 R_np: float = 2.5,
                 eta: float = 0.89e-9,  # 水粘度，单位 kJ*ns/(nm^3*mol) 尺度下的有效值
                 T: float = 300.0,
                 k_B: float = 8.314e-3,  # kJ/(mol*K)
                 z0: float = 8.0,
                 z_cutoff: float = 1.0,
                 F_max_bind: float = 50.0,
                 kappa_bind: float = 2.0,
                 epsilon_LJ: float = 4.0,
                 sigma_LJ: float = 3.0,
                 k_spring_bend: float = 10.0):
        """
        Parameters
        ----------
        R_np : float
            颗粒半径（nm）。
        eta : float
            有效粘度（用于计算摩擦系数）。
        T : float
            温度（K）。
        z0 : float
            初始膜-颗粒表面距离（nm）。
        z_cutoff : float
            受体-配体有效作用距离（nm）。
        F_max_bind : float
            最大结合力（kJ/(mol*nm)）。
        kappa_bind : float
            结合衰减系数（1/nm）。
        epsilon_LJ : float
            LJ 势阱深度（kJ/mol）。
        sigma_LJ : float
            LJ 特征长度（nm）。
        k_spring_bend : float
            膜弯曲回复力等效弹簧常数（kJ/(mol*nm^2)）。
        """
        self.R_np = float(R_np)
        self.eta = float(eta)
        self.T = float(T)
        self.k_B = float(k_B)
        self.z = float(z0)
        self.z_cutoff = float(z_cutoff)
        self.F_max_bind = float(F_max_bind)
        self.kappa_bind = float(kappa_bind)
        self.epsilon_LJ = float(epsilon_LJ)
        self.sigma_LJ = float(sigma_LJ)
        self.k_spring_bend = float(k_spring_bend)
        # Stokes 摩擦系数（使用有效值保证数值稳定性）
        # 物理上 gamma = 6*pi*eta*R_np，但在粗粒化单位下需调整
        self.gamma = max(6.0 * np.pi * self.eta * self.R_np, 10.0)
        # 时间步长（overdamped 极限需要较小 dt 以保证稳定性）
        self.dt = 1e-5  # ns

    def force_electrostatic(self, z: float, debye_length: float = 1.0,
                            zeta_np: float = -0.05,
                            zeta_mem: float = -0.03) -> float:
        """
        简化的屏蔽库仑力（Derjaguin 近似）：

            F_elec(z) = (2*pi*epsilon*R_np) * (zeta_np * zeta_mem / lambda_D)
                        * exp(-z / lambda_D)

        其中 zeta 为 zeta 电势（V），lambda_D 为德拜长度（nm）。
        """
        eps_rel = 80.0
        eps0 = 8.854e-12  # F/m
        # 单位换算因子：转换为 kJ/(mol*nm) 量级
        # 这里采用无量纲标度形式以保证数值稳定
        prefactor = 10.0  # 有效标度
        F = prefactor * (zeta_np * zeta_mem / debye_length) * np.exp(-z / debye_length)
        return float(F)

    def force_vdw(self, z: float) -> float:
        """
        范德华相互作用力（Lennard-Jones 势的导数）：

            V_LJ(z) = 4*epsilon * [(sigma/z)^12 - (sigma/z)^6]
            F_vdw(z) = -dV/dz = 24*epsilon/z * [2*(sigma/z)^12 - (sigma/z)^6]

        为避免 z->0 时的奇异性，在 z < 0.1 nm 时截断为谐波排斥。
        """
        # [HOLE 2] 请补全 Lennard-Jones 范德华力计算：
        # 1. 对 z 做安全截断：z_safe = max(z, 0.1)
        # 2. 若 z < 0.1，返回谐波排斥力 1000*(0.1 - z)
        # 3. 否则计算 LJ 力：F = 24*epsilon/z_safe * (2*(sigma/z_safe)^12 - (sigma/z_safe)^6)
        # TODO: 实现上述公式
        raise NotImplementedError("HOLE 2: 请补全 force_vdw 的 LJ 力计算")

    def force_bending(self, z: float) -> float:
        """
        膜弯曲产生的回复力（简谐近似）：

            F_bend(z) = -k_spring_bend * (z - z_eq)

        平衡距离 z_eq 设为 0.5 nm（轻微接触）。
        """
        z_eq = 0.5
        return -self.k_spring_bend * (z - z_eq)

    def force_binding(self, z: float) -> float:
        """
        Type-II 功能响应改造的结合力（改编自 grazing_ode）：

            F_bind(z) = F_max * [1 - exp(-kappa_bind * max(z_cutoff - z, 0))]

        物理意义：当颗粒距离膜表面小于 z_cutoff 时，受体-配体开始结合；
        随着距离进一步减小，结合位点逐渐饱和，力趋近于 F_max。
        """
        dz = max(self.z_cutoff - z, 0.0)
        # 负号表示吸引力（将颗粒拉向膜表面）
        F = -self.F_max_bind * (1.0 - np.exp(-self.kappa_bind * dz))
        return float(F)

    def total_force(self, z: float, debye_length: float = 1.0) -> float:
        """
        合力 = 静电 + 范德华 + 弯曲 + 结合
        """
        return (self.force_electrostatic(z, debye_length)
                + self.force_vdw(z)
                + self.force_bending(z)
                + self.force_binding(z))

    def step_euler_maruyama(self, debye_length: float = 1.0) -> float:
        """
        Euler-Maruyama 积分过阻尼朗之万方程：

            z_{n+1} = z_n + (dt/gamma) * F_total(z_n)
                      + sqrt(2*k_B*T*dt/gamma) * N(0,1)

        其中 N(0,1) 为标准正态随机数。
        为数值稳定性，对合力进行截断（±5000 kJ/(mol*nm)），
        并在 z < 0.1 nm 处施加反射边界。
        """
        F = self.total_force(self.z, debye_length)
        F_capped = np.clip(F, -1000.0, 1000.0)
        drift = self.dt / self.gamma * F_capped
        diffusion = np.sqrt(2.0 * self.k_B * self.T * self.dt / self.gamma)
        noise = diffusion * np.random.randn()
        self.z = self.z + drift + noise
        # 反射边界
        if self.z < 0.1:
            self.z = 0.1 + (0.1 - self.z)
        return float(self.z)

    def simulate(self, n_steps: int = 50000,
                 debye_length: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        运行轨迹积分。

        Returns
        -------
        t : ndarray
            时间轴（ns）。
        z_traj : ndarray
            距离轨迹（nm）。
        F_traj : ndarray
            合力轨迹（kJ/(mol*nm)）。
        """
        t = np.arange(n_steps) * self.dt
        z_traj = np.zeros(n_steps, dtype=np.float64)
        F_traj = np.zeros(n_steps, dtype=np.float64)
        for i in range(n_steps):
            z_traj[i] = self.z
            F_traj[i] = self.total_force(self.z, debye_length)
            self.step_euler_maruyama(debye_length)
        return t, z_traj, F_traj
