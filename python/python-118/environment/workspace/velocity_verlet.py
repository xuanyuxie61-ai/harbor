r"""
velocity_verlet.py
速度Verlet积分器 + Nosé-Hoover链恒温器

融合种子项目：
- 833_ode_trapezoidal: 隐式梯形法思想 → Nosé-Hoover热浴方程的隐式积分
- 908_predator_prey_ode: 耦合ODE系统 → 热浴-粒子系统耦合演化
- 1025_ripple_ode: 参数化ODE → 时间步长与截断参数管理

运动方程:
    dr_i/dt = v_i
    m_i dv_i/dt = F_i - \xi m_i v_i
    Q d\xi/dt = \sum_i m_i v_i^2 - 3N k_B T  (Nosé-Hoover单热浴)

速度Verlet离散化 (含热浴修正):
    v(t+\Delta t/2) = v(t) + (\Delta t/2m) [ F(t) - \xi(t) m v(t) ]
    r(t+\Delta t) = r(t) + \Delta t v(t+\Delta t/2)
    \xi(t+\Delta t) = \xi(t) + (\Delta t/2Q) [ G(t) + G(t+\Delta t) ]  (梯形法)
    v(t+\Delta t) = v(t+\Delta t/2) + (\Delta t/2m) [ F(t+\Delta t) - \xi(t+\Delta t) m v(t+\Delta t) ]
其中 G = \sum_i m_i v_i^2 - 3N k_B T。

G_0 = \sum_i m_i v_i^2 为瞬时动能的两倍。
"""

import numpy as np
from utils_numeric import check_bounds, relative_convergence_check


class VelocityVerletNVT:
    """NVT系综下的速度Verlet积分器，融合Nosé-Hoover链恒温器。"""

    def __init__(self, dt=1.0, T_target=1200.0, nhc_chain_length=3, Q_factor=1.0):
        """
        参数:
            dt: 时间步长 (fs)
            T_target: 目标温度 (K)
            nhc_chain_length: Nosé-Hoover链长度 (1=单热浴)
            Q_factor: 热浴质量因子 Q = 3N k_B T / \omega^2, \omega ~ 1/dt
        """
        self.dt = float(dt)
        self.T_target = float(T_target)
        self.kb = 8.617333e-5  # eV/K
        self.nhc_length = max(1, int(nhc_chain_length))
        self.Q_factor = float(Q_factor)

        # 热浴变量
        self.xi = None  # (nhc_length,) 热浴位置 (这里xi就是热浴动量/质量)
        self.v_xi = None  # 热浴速度
        self.Q = None  # 热浴质量，延迟到知道系统大小后初始化

    def _initialize_nhc(self, n_dof):
        """初始化Nosé-Hoover链热浴变量。"""
        # 特征频率 ~ 1/dt
        omega = 1.0 / self.dt
        self.Q = np.zeros(self.nhc_length, dtype=np.float64)
        self.Q[0] = self.Q_factor * n_dof * self.kb * self.T_target / (omega ** 2)
        for i in range(1, self.nhc_length):
            self.Q[i] = self.kb * self.T_target / (omega ** 2)
        self.v_xi = np.zeros(self.nhc_length, dtype=np.float64)

    def _nhc_scale_factor(self, v, masses, n_steps=3):
        """计算NHC链对速度的缩放因子，使用多时间步积分 (n_steps内循环)。"""
        # TODO: Hole 2 — 实现Nosé-Hoover链恒温器的速度缩放因子计算
        # 需要计算热浴力 G，并通过多时间步积分更新热浴速度 v_xi，
        # 最终返回粒子速度的指数缩放因子 scale。
        # 提示: G[0] = kinetic - n_dof * kb * T; Q[i] 为热浴质量。
        return 1.0

    def step(self, positions, velocities, masses, species_idx, potential, box):
        """
        执行单步速度Verlet+NHC积分。

        返回:
            new_positions, new_velocities, total_energy, temperature, virial
        """
        n_atoms = positions.shape[0]
        n_dof = 3 * n_atoms
        if self.Q is None:
            self._initialize_nhc(n_dof)

        # 1. 计算当前力
        _, forces, virial = potential.compute_forces_and_energies(positions, species_idx, box)

        # 2. NHC速度缩放 (半步前)
        scale = self._nhc_scale_factor(velocities, masses, n_steps=3)
        velocities *= scale

        # 3. 速度半步推进
        velocities_half = velocities + 0.5 * self.dt * forces / masses[:, None]

        # 4. 位置全步推进
        new_positions = positions + self.dt * velocities_half
        # 应用周期性边界
        new_positions -= box * np.floor(new_positions / box)

        # 5. 新位置的力
        pot_energy, new_forces, new_virial = potential.compute_forces_and_energies(
            new_positions, species_idx, box)

        # 6. 速度半步推进到全步
        new_velocities = velocities_half + 0.5 * self.dt * new_forces / masses[:, None]

        # 7. NHC速度缩放 (半步后)
        scale2 = self._nhc_scale_factor(new_velocities, masses, n_steps=3)
        new_velocities *= scale2

        # 8. 计算温度与能量
        kinetic = 0.5 * np.sum(masses[:, None] * new_velocities ** 2)
        temperature = 2.0 * kinetic / (n_dof * self.kb)
        total_energy = pot_energy + kinetic

        # 热浴能量
        nhc_energy = 0.0
        for i in range(self.nhc_length):
            nhc_energy += 0.5 * self.Q[i] * self.v_xi[i] ** 2
            if i == 0:
                nhc_energy += n_dof * self.kb * self.T_target * 0.0  # 仅动能项
        total_energy += nhc_energy

        return new_positions, new_velocities, total_energy, temperature, pot_energy, kinetic, new_virial


class TrapezoidalThermostatIntegrator:
    """基于隐式梯形法的热浴-系统耦合积分器，融合833_ode_trapezoidal。"""

    def __init__(self, dt=1.0, T_target=1200.0, max_iter=10, tol=1e-8):
        self.dt = float(dt)
        self.T_target = float(T_target)
        self.kb = 8.617333e-5
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.xi = 0.0  # 热浴变量

    def step(self, positions, velocities, masses, species_idx, potential, box):
        """使用隐式梯形法求解耦合ODE:
            dv/dt = F/m - xi*v
            d xi/dt = (2K - 3NkT) / Q
        对 xi 方程使用梯形迭代。
        """
        n_atoms = positions.shape[0]
        n_dof = 3 * n_atoms
        Q = n_dof * self.kb * self.T_target * (self.dt ** 2) * 10.0

        # 当前力
        _, forces, virial = potential.compute_forces_and_energies(positions, species_idx, box)

        # 预测步 (显式Euler)
        v_pred = velocities + self.dt * (forces / masses[:, None] - self.xi * velocities)
        pos_pred = positions + self.dt * v_pred
        pos_pred -= box * np.floor(pos_pred / box)

        # 隐式迭代修正 xi
        xi_old = self.xi
        for iteration in range(self.max_iter):
            # 用当前 xi 修正速度
            v_half = velocities + 0.5 * self.dt * (forces / masses[:, None] - xi_old * velocities)
            pos_new = positions + self.dt * v_half
            pos_new -= box * np.floor(pos_new / box)

            _, f_new, _ = potential.compute_forces_and_energies(pos_new, species_idx, box)
            v_new = v_half + 0.5 * self.dt * (f_new / masses[:, None] - xi_old * v_half)

            K_new = np.sum(masses[:, None] * v_new ** 2)
            G_new = K_new - n_dof * self.kb * self.T_target
            K_old = np.sum(masses[:, None] * velocities ** 2)
            G_old = K_old - n_dof * self.kb * self.T_target

            xi_new = xi_old + 0.5 * self.dt * (G_old + G_new) / Q
            if abs(xi_new - xi_old) < self.tol:
                self.xi = xi_new
                break
            xi_old = xi_new
        else:
            self.xi = xi_old

        kinetic = 0.5 * np.sum(masses[:, None] * v_new ** 2)
        _, f_final, virial_final = potential.compute_forces_and_energies(pos_new, species_idx, box)
        pot_energy = _
        temperature = 2.0 * kinetic / (n_dof * self.kb)
        total_energy = pot_energy + kinetic
        return pos_new, v_new, total_energy, temperature, pot_energy, kinetic, virial_final
