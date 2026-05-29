"""
integrator.py
分子动力学积分器与稳定性分析模块

本模块实现脂质双分子层粗粒化 MD 的数值积分，并对积分器的
绝对稳定区域进行分析（受种子项目 104_boundary_locus 启发）。

核心算法:
    - 速度 Verlet 积分器（辛积分，保持能量）
    - Langevin 热浴（ thermostat，控制温度）
    - 积分器稳定区域分析（复平面上的 |R(z)| ≤ 1 区域）

参考种子项目: 104_boundary_locus (ODE 方法稳定区域)
                322_duffing_ode (非线性 ODE 驱动项)
"""

import numpy as np


class MDIntegrator:
    """
    分子动力学积分器。

    采用速度 Verlet 算法（Velocity Verlet）:
        r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt²
        v(t+dt/2) = v(t) + 0.5*a(t)*dt
        计算 a(t+dt) = F(t+dt)/m
        v(t+dt) = v(t+dt/2) + 0.5*a(t+dt)*dt

    对于取向自由度 (θ, φ)，采用类似的角速度 Verlet。

    Langevin 热浴:
        m dv/dt = F - γ v + √(2γ k_B T / dt) * ξ(t)
    其中 ξ(t) 为高斯白噪声，<ξ(t)ξ(t')> = δ(t-t')。
    """

    def __init__(self, system, friction_gamma=0.5, seed=None):
        """
        Parameters
        ----------
        system : LipidBilayerSystem
            被积分的双层系统。
        friction_gamma : float
            Langevin 摩擦系数 γ（单位: ps⁻¹）。
        seed : int or None
            随机数种子。
        """
        if system is None:
            raise ValueError("system 不能为 None。")
        if friction_gamma < 0:
            raise ValueError("摩擦系数必须非负。")

        self.sys = system
        self.gamma = friction_gamma
        self.rng = np.random.default_rng(seed)

    def step(self):
        """
        执行一个 MD 时间步（速度 Verlet + Langevin）。

        数值稳定性要求:
            dt < 2/ω_max，其中 ω_max 为系统最高振动频率。
            对于取向耦合 J=2.5 kJ/mol，特征频率 ~ √(2J/I) ≈ 2.2 ps⁻¹，
            因此 dt ≤ 0.5 ps 为安全范围。本模型取 dt=0.002（约 2 fs 的缩放单位）。
        """
        dt = self.sys.dt
        m = self.sys.mass
        nx, ny = self.sys.nx, self.sys.ny

        # 计算当前力
        self.sys.compute_forces()

        # 噪声幅度
        T_local = self.sys.temperature_field
        sigma = np.sqrt(2.0 * self.gamma * self.sys.kb * T_local / dt)
        noise_theta = self.rng.normal(0.0, 1.0, (nx, ny)) * sigma
        noise_phi = self.rng.normal(0.0, 1.0, (nx, ny)) * sigma

        # 角速度半更新 (Langevin 修正)
        # v' = v + 0.5*dt*(F - γv + noise)/m
        #    = v*(1 - 0.5*γ*dt/m) + 0.5*dt*(F + noise)/m
        damping = 1.0 - 0.5 * self.gamma * dt / m
        damping = max(damping, 0.0)  # 数值安全

        self.sys.omega_theta = (damping * self.sys.omega_theta +
                                0.5 * dt * (self.sys.torque_theta + noise_theta) / m)
        self.sys.omega_phi = (damping * self.sys.omega_phi +
                              0.5 * dt * (self.sys.torque_phi + noise_phi) / m)

        # 坐标全更新
        self.sys.theta = self.sys.theta + dt * self.sys.omega_theta
        self.sys.phi = self.sys.phi + dt * self.sys.omega_phi

        # 周期性边界: theta ∈ [0, π], phi ∈ [0, 2π)
        self.sys.theta = np.mod(self.sys.theta, 2.0 * np.pi)
        self.sys.phi = np.mod(self.sys.phi, 2.0 * np.pi)

        # 面积更新（一阶 Euler，因面积弛豫较慢）
        self.sys.area = self.sys.area + dt * self.sys.force_area / m
        self.sys.area = np.clip(self.sys.area, 0.1 * self.sys.area0, 5.0 * self.sys.area0)

        # 重新计算力
        self.sys.compute_forces()

        # 角速度后半更新
        noise_theta2 = self.rng.normal(0.0, 1.0, (nx, ny)) * sigma
        noise_phi2 = self.rng.normal(0.0, 1.0, (nx, ny)) * sigma

        self.sys.omega_theta = (damping * self.sys.omega_theta +
                                0.5 * dt * (self.sys.torque_theta + noise_theta2) / m)
        self.sys.omega_phi = (damping * self.sys.omega_phi +
                              0.5 * dt * (self.sys.torque_phi + noise_phi2) / m)

    def run_equilibration(self, n_steps=2000):
        """
        运行 n_steps 步的平衡化模拟。

        Returns
        -------
        energy_trace : ndarray
            每 10 步记录的能量轨迹。
        s2_trace : ndarray
            每 10 步记录的全局序参数轨迹。
        """
        energy_trace = []
        s2_trace = []
        for step in range(n_steps):
            self.step()
            if step % 10 == 0:
                e = self.sys.compute_total_energy()
                s2 = self.sys.global_order_parameter()
                energy_trace.append(e)
                s2_trace.append(s2)
        return np.array(energy_trace), np.array(s2_trace)


class IntegratorStability:
    """
    MD 积分器绝对稳定性分析。

    受种子项目 104_boundary_locus 启发，分析数值积分器应用于
    线性测试方程 y' = λy 时的稳定区域:
        y_{n+1} = R(z) y_n,  z = hλ
        稳定条件: |R(z)| ≤ 1

    对于本系统的耦合谐振子近似，有效方程:
        d²θ/dt² + γ dθ/dt + ω_0² θ = 0
    转化为复平面上的二阶系统，其特征值 λ = (-γ ± √(γ² - 4ω_0²))/2。
    """

    def __init__(self, integrator_type='verlet'):
        if integrator_type not in ('verlet', 'euler', 'rk4'):
            raise ValueError("integrator_type 必须是 'verlet', 'euler' 或 'rk4'")
        self.integrator_type = integrator_type

    def amplification_factor(self, z):
        """
        计算积分器的放大因子 R(z)。

        Parameters
        ----------
        z : complex ndarray
            z = h*λ，复平面上的测试点。

        Returns
        -------
        R : complex ndarray
            |R(z)| 决定稳定性。
        """
        z = np.asarray(z, dtype=complex)
        if self.integrator_type == 'euler':
            # 显式 Euler: R(z) = 1 + z
            return 1.0 + z
        elif self.integrator_type == 'rk4':
            # RK4: R(z) = 1 + z + z²/2 + z³/6 + z⁴/24
            return 1.0 + z + z**2 / 2.0 + z**3 / 6.0 + z**4 / 24.0
        elif self.integrator_type == 'verlet':
            # 速度 Verlet 对于 y'' = -ω² y 的放大矩阵特征值
            # 令 z = i*ω*h，特征方程 r² - (2 + z²)r + 1 = 0
            # 特征值 r = 1 + z²/2 ± 0.5*z*√(z² + 4)
            disc = z**2 + 4.0
            sqrt_disc = np.sqrt(disc)
            lambda1 = 1.0 + z**2 / 2.0 + 0.5 * z * sqrt_disc
            lambda2 = 1.0 + z**2 / 2.0 - 0.5 * z * sqrt_disc
            return np.maximum(np.abs(lambda1), np.abs(lambda2))
        else:
            raise NotImplementedError

    def stability_region_mask(self, xlim=(-3.0, 3.0), ylim=(-3.0, 3.0),
                               npts=401):
        """
        在复平面上生成稳定区域掩码 |R(z)| ≤ 1。

        Returns
        -------
        X, Y : 2D ndarray
            复平面网格坐标。
        mask : 2D ndarray (bool)
            True 表示该点位于稳定区域内。
        """
        x = np.linspace(xlim[0], xlim[1], npts)
        y = np.linspace(ylim[0], ylim[1], npts)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y
        Rval = self.amplification_factor(Z)
        mask = Rval <= 1.0
        return X, Y, mask

    def check_system_stability(self, omega_max=2.5, gamma=0.5, dt=None):
        """
        检验当前系统参数是否在积分器稳定区域内。

        系统的特征值:
            λ_{1,2} = (-γ ± √(γ² - 4ω_max²)) / 2

        Parameters
        ----------
        omega_max : float
            系统最大本征频率。
        gamma : float
            阻尼系数。
        dt : float or None
            时间步长；若为 None 则使用系统默认值。

        Returns
        -------
        stable : bool
            若所有特征值对应的 z=dt*λ 都在稳定区域内，返回 True。
        z_points : list of complex
            特征值对应的 z 值。
        """
        if dt is None:
            dt = 0.002

        # 对于速度 Verlet，实用稳定性判据为 ω_max * dt < 2.0
        if self.integrator_type == 'verlet':
            stable = omega_max * dt < 2.0
            disc = gamma**2 - 4.0 * omega_max**2
            if disc >= 0:
                lambda1 = (-gamma + np.sqrt(disc)) / 2.0
                lambda2 = (-gamma - np.sqrt(disc)) / 2.0
            else:
                lambda1 = (-gamma + 1j * np.sqrt(-disc)) / 2.0
                lambda2 = (-gamma - 1j * np.sqrt(-disc)) / 2.0
            z_points = [dt * lambda1, dt * lambda2]
            return stable, z_points

        disc = gamma**2 - 4.0 * omega_max**2
        if disc >= 0:
            lambda1 = (-gamma + np.sqrt(disc)) / 2.0
            lambda2 = (-gamma - np.sqrt(disc)) / 2.0
        else:
            lambda1 = (-gamma + 1j * np.sqrt(-disc)) / 2.0
            lambda2 = (-gamma - 1j * np.sqrt(-disc)) / 2.0

        z_points = [dt * lambda1, dt * lambda2]
        stable = all(np.abs(self.amplification_factor(z)) <= 1.0 + 1e-10
                     for z in z_points)
        return stable, z_points
