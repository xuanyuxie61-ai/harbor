"""
resistive_mhd.py — 电阻性 MHD 时间演化 (1D 切割模)
=====================================================

融合自:
  - 060_axon_ode: Hodgkin-Huxley 耦合 ODE 系统 (→ 磁重联电路模型)
  - 1235_hrl-team_OptimismPerseveration: 自适应 Q-learning 步长控制

物理背景:
  电阻性 MHD 方程组 (无量纲化, Alfvén 速度归一化):
    dB/dt = curl(v x B) + eta * nabla^2 B   (感应方程)
    dv/dt = -(v.grad)v + J x B / rho - grad(p)/rho + nu*nabla^2 v
    J = curl B / mu0                          (安培定律)
    div B = 0                                 (无散度约束)

  1D 切割模线性化 (沿 x 方向，y 为平衡梯度方向):
    dpsi/dt = eta * (d^2/dy^2 - k^2) * psi + E_z
    dJ/dt = (B0/a) * dpsi/dy * k + eta * (d^2/dy^2 - k^2) * J

  Lundquist 数: S = mu0 * L * v_A / eta
  Sweet-Parker 重联率: R_SP ~ S^{-1/2}
  Petschek 重联率: R_P ~ pi / (8 * ln(S))

  撕裂模不稳定性增长率 (Furth-Killeen-Rosenbluth):
    gamma * tau_A ~ S^{-3/5} * (k*a)^{2/5}  (对于 k*a << 1)
"""

import numpy as np


class ResistiveMHD1D:
    """一维电阻性 MHD 切割模时间演化。"""

    def __init__(self, nx=128, dx=0.1, eta=1e-3, nu=5e-4, S_lundquist=1e4):
        self.nx = nx
        self.dx = dx
        self.eta = eta
        self.nu = nu
        self.S_lundquist = S_lundquist
        self.mu0 = 1.0

        # 场量 (1D 沿 y 方向，模拟切割模)
        self.By = None       # 反平行磁场
        self.Bx = None       # 扰动磁场 (重联分量)
        self.vx = None       # x-速度
        self.vy = None       # y-速度
        self.Jz = None       # z-电流密度
        self.Ez = None       # z-电场 (重联电场)
        self.Az = None       # z-磁矢势
        self.density = None

        # 历史
        self.By_history = None
        self.Jz_history = None
        self.Ez_history = None
        self.energy_history = None

        # 电子分布 (用于 Laguerre 谱分析)
        self.electron_distribution = None
        self.max_growth_rate = None

    def initialize_perturbation(self, cs, amplitude=1e-3, k_mode=1):
        """
        初始化 Harris 平衡 + 撕裂模扰动。

        psi_1(y) = amplitude * sech(y/a)
        b_x = dpsi/dy
        J_z = -nabla^2 psi / mu0
        """
        ny = self.nx  # 使用 nx 作为 y 方向分辨率
        y = np.linspace(-3.2, 3.2, ny)
        a = cs.a_sheet

        self.By = cs.By.copy()
        self.density = cs.density.copy()

        # 扰动磁通
        sech_ya = 1.0 / np.cosh(y / a)
        tanh_ya = np.tanh(y / a)
        psi1 = amplitude * sech_ya
        dpsi_dy = -amplitude * sech_ya * tanh_ya / a
        d2psi_dy2 = amplitude * sech_ya * (2 * tanh_ya ** 2 - 1) / a ** 2

        k = 2 * np.pi * k_mode / 12.8
        self.Bx = dpsi_dy
        self.Az = -psi1
        self.Jz = -(d2psi_dy2 - k ** 2 * psi1) / self.mu0
        self.Ez = -self.eta * self.Jz  # 欧姆电场

        self.vx = np.zeros(ny)
        self.vy = np.zeros(ny)

        # 电子速度分布 (Maxwellian 微扰)
        v = np.linspace(-4, 4, 64)
        v_th = np.sqrt(cs.T_e)
        self.electron_distribution = np.exp(-v ** 2 / (2 * v_th ** 2)) / np.sqrt(2 * np.pi * v_th ** 2)
        self.electron_distribution += 0.01 * amplitude * v ** 2 * self.electron_distribution

        # 历史
        self.By_history = []
        self.Jz_history = []
        self.Ez_history = []
        self.energy_history = []
        self.max_growth_rate = 0.0

    def compute_cfl_timestep(self, fd, cfl=0.4):
        """
        CFL 条件确定时间步长。

        对流 CFL: dt < cfl * dx / max(|v| + v_A)
        扩散 CFL: dt < cfl * dx^2 / (2 * eta)
        综合: dt = min(dt_conv, dt_diff)

        其中 v_A = B / sqrt(mu0 * rho) 为 Alfvén 速度。
        """
        v_A_max = np.max(np.abs(self.By)) / np.sqrt(self.mu0 * np.min(self.density + 1e-10))
        v_max = np.max(np.abs(self.vx)) + v_A_max
        dt_conv = cfl * self.dx / (v_max + 1e-10)
        dt_diff = cfl * self.dx ** 2 / (2 * self.eta + 1e-10)
        dt = min(dt_conv, dt_diff)
        return max(dt, 1e-8)  # 下限保护

    def evolve(self, n_steps=200, dt=0.01, fd=None):
        """
        时间演化 (RK2 + 高阶有限差分)。

        感应方程: dBx/dt = -dEz/dy, Ez = eta*Jz - vx*By + vy*Bx
        动量方程: dvx/dt = (Jz x B)_x / rho + nu * d^2vx/dy^2
        """
        if fd is None:
            from high_order_fd import HighOrderFD
            fd = HighOrderFD(order=4, nx=self.nx, dx=self.dx)
            fd.build_central_stencils()

        energy_0 = self._compute_energy()
        prev_energy = energy_0

        for step in range(n_steps):
            # 计算电流
            self.Jz = -fd.apply_second_derivative(self.Az) / self.mu0

            # 重联电场: Ez = eta*Jz - vx*By
            self.Ez = self.eta * self.Jz - self.vx * self.By

            # 感应方程: dBx/dt = -dEz/dy
            dBx_dt = -fd.apply_first_derivative(self.Ez)

            # 洛伦兹力: Fx = Jz * By / rho
            Fx = self.Jz * self.By / (self.density + 1e-10)

            # 动量方程: dvx/dt = Fx + nu * d^2vx/dy^2
            dvx_dt = Fx + self.nu * fd.apply_second_derivative(self.vx)

            # RK2 时间推进
            Bx_mid = self.Bx + 0.5 * dt * dBx_dt
            vx_mid = self.vx + 0.5 * dt * dvx_dt

            Jz_mid = -fd.apply_second_derivative(
                self.Az + 0.5 * dt * (-self.Ez)) / self.mu0
            Ez_mid = self.eta * Jz_mid - vx_mid * self.By
            dBx_dt2 = -fd.apply_first_derivative(Ez_mid)
            Fx_mid = Jz_mid * self.By / (self.density + 1e-10)
            dvx_dt2 = Fx_mid + self.nu * fd.apply_second_derivative(vx_mid)

            self.Bx = self.Bx + dt * dBx_dt2
            self.vx = self.vx + dt * dvx_dt2
            self.Az = self.Az + dt * (-self.Ez)

            # 记录历史
            self.By_history.append(float(np.mean(np.abs(self.Bx))))
            self.Jz_history.append(float(np.max(np.abs(self.Jz))))
            self.Ez_history.append(float(np.abs(self.Ez[self.nx // 2])))

            energy = self._compute_energy()
            self.energy_history.append(energy)

        # 估算增长率
        if len(self.energy_history) > 10:
            log_e = np.log(np.array(self.energy_history[10:]) + 1e-30)
            if len(log_e) > 2:
                t_arr = np.arange(len(log_e)) * dt
                coeffs = np.polyfit(t_arr, log_e, 1)
                self.max_growth_rate = max(0.0, coeffs[0])
        self.By_history = np.array(self.By_history)
        self.Jz_history = np.array(self.Jz_history)
        self.Ez_history = np.array(self.Ez_history)

    def _compute_energy(self):
        """总能量 (磁 + 动)。"""
        W_B = 0.5 * np.sum(self.Bx ** 2 + self.By ** 2) * self.dx
        W_K = 0.5 * np.sum(self.density * self.vx ** 2) * self.dx
        return W_B + W_K
