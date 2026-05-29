"""
md_engine.py
============
分子动力学模拟引擎，基于 Velocity Verlet 积分方案。

核心物理原理
------------
牛顿运动方程：
    m_i · d²r_i/dt² = F_i = -∇_{r_i} V

Velocity Verlet 算法（辛积分器，保持相空间体积守恒）：

    1.  r(t+Δt) = r(t) + v(t)·Δt + ½·a(t)·Δt²
    2.  v(t+Δt) = v(t) + ½·[a(t) + a(t+Δt)]·Δt
    3.  a(t+Δt) = F(t+Δt) / m

能量守恒误差：O(Δt²) 每步，全局 O(Δt²) 在有限时间内。
对于长时间模拟，可采用更高阶的辛积分器，但 Verlet 在
计算效率和稳定性之间提供了最佳平衡。

温度控制（Berendsen 弱耦合 thermostat）：
    dT/dt = (T_bath - T) / τ
    v_new = v · √(1 + Δt/τ · (T_bath/T - 1))

其中 T = (2/(d·N·k_B)) · K 为瞬时动能温度，
K = ½ Σ_i m_i v_i² 为总动能。

压强计算（维里定理）：
    P = (N k_B T)/V + (1/(d·V)) Σ_{i<j} r_{ij}·F_{ij}
"""

import numpy as np
from typing import Callable, Tuple, Optional
from potential_models import total_forces_lj, total_potential_lj


class MDEngine:
    """
    分子动力学模拟引擎。
    
    支持：
    - Velocity Verlet 时间积分
    - Berendsen 温度控制
    - 周期性边界条件（PBC）
    - 能量/温度/压强追踪
    """

    def __init__(self, n_particles: int, dim: int = 2,
                 mass: float = 1.0, dt: float = 0.001,
                 box_size: float = 10.0,
                 epsilon: float = 1.0, sigma: float = 1.0,
                 rcut: float = 2.5,
                 temperature: float = 1.0,
                 tau_thermostat: float = 0.1):
        """
        初始化 MD 引擎。
        
        参数:
            n_particles: 粒子数
            dim: 空间维度（2 或 3）
            mass: 粒子质量（统一质量）
            dt: 时间步长
            box_size: 模拟盒子边长（立方体/正方形）
            epsilon, sigma: LJ 参数
            rcut: LJ 截断半径
            temperature: 目标温度（k_B 单位制）
            tau_thermostat: Berendsen 耦合时间常数
        """
        self.n = n_particles
        self.dim = dim
        self.mass = mass
        self.dt = dt
        self.box = box_size
        self.epsilon = epsilon
        self.sigma = sigma
        self.rcut = rcut
        self.target_temp = temperature
        self.tau = tau_thermostat

        # 状态变量
        self.pos = np.zeros((n_particles, dim))
        self.vel = np.zeros((n_particles, dim))
        self.acc = np.zeros((n_particles, dim))
        self.force = np.zeros((n_particles, dim))

        # 追踪量
        self.time_history = []
        self.potential_history = []
        self.kinetic_history = []
        self.total_energy_history = []
        self.temperature_history = []
        self.pressure_history = []

    def initialize_positions_lattice(self, lattice_type: str = "square"):
        """
        在晶格上初始化粒子位置。
        
        支持：
        - square: 正方晶格
        - hexagonal: 六方密堆积（2D）
        """
        if lattice_type == "square":
            n_side = int(np.ceil(np.sqrt(self.n)))
            spacing = self.box / n_side
            idx = 0
            for i in range(n_side):
                for j in range(n_side):
                    if idx >= self.n:
                        break
                    self.pos[idx, 0] = (i + 0.5) * spacing
                    self.pos[idx, 1] = (j + 0.5) * spacing
                    if self.dim == 3:
                        self.pos[idx, 2] = self.box / 2.0
                    idx += 1
                if idx >= self.n:
                    break
        elif lattice_type == "hexagonal":
            # 2D 六方密堆积
            spacing = self.sigma * 1.12  # 略大于平衡间距
            n_x = int(np.ceil(self.box / spacing))
            n_y = int(np.ceil(self.box / (spacing * np.sqrt(3.0) / 2.0)))
            idx = 0
            for j in range(n_y):
                offset = 0.0 if j % 2 == 0 else spacing * 0.5
                for i in range(n_x):
                    if idx >= self.n:
                        break
                    x = offset + i * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    if x < self.box and y < self.box:
                        self.pos[idx, 0] = x
                        self.pos[idx, 1] = y
                        idx += 1
            # 如果未填满，随机填充剩余
            while idx < self.n:
                self.pos[idx] = np.random.rand(self.dim) * self.box
                idx += 1
        else:
            # 随机初始化
            self.pos = np.random.rand(self.n, self.dim) * self.box

    def initialize_velocities_maxwell_boltzmann(self):
        """
        按 Maxwell-Boltzmann 分布初始化速度。
        
        概率密度：
            P(v) ∝ exp(-m v² / (2 k_B T))
        
        每个分量服从 N(0, k_B T / m)。
        """
        std = np.sqrt(self.target_temp / self.mass)
        self.vel = np.random.normal(0.0, std, (self.n, self.dim))
        # 去除质心速度（保持动量守恒）
        v_cm = np.mean(self.vel, axis=0)
        self.vel -= v_cm

    def apply_periodic_boundary(self):
        """应用周期性边界条件：粒子穿过边界时在对面重现。"""
        self.pos -= self.box * np.floor(self.pos / self.box)

    def compute_forces_and_energies(self) -> Tuple[float, float]:
        """
        计算力、势能和动能。
        
        返回:
            (potential_energy, kinetic_energy)
        """
        self.force = total_forces_lj(self.pos, self.epsilon,
                                      self.sigma, self.rcut, self.box)
        potential = total_potential_lj(self.pos, self.epsilon,
                                       self.sigma, self.rcut, self.box)
        kinetic = 0.5 * self.mass * np.sum(self.vel ** 2)
        return potential, kinetic

    def compute_temperature(self, kinetic: float) -> float:
        """
        从动能计算瞬时温度。
        
        T = 2K / (d · N · k_B)
        
        这里取 k_B = 1（自然单位制）。
        """
        dof = self.n * self.dim
        if dof < 1:
            return 0.0
        return 2.0 * kinetic / dof

    def compute_pressure(self, potential: float, kinetic: float) -> float:
        """
        计算瞬时压强（理想气体 + 维里修正）。
        
        P = (N k_B T)/V + (1/(d·V)) Σ r_{ij}·F_{ij}
        """
        volume = self.box ** self.dim
        n_kb_t = 2.0 * kinetic / self.dim  # = N k_B T / d * d = N k_B T ?
        # 更准确的：P = (2K + W) / (d·V)，其中 W = Σ r·F
        from potential_models import virial_stress_lj
        stress = virial_stress_lj(self.pos, self.epsilon, self.sigma,
                                   self.rcut, volume, self.box)
        virial = np.trace(stress) * volume  # = Σ r·F
        pressure = (2.0 * kinetic + virial) / (self.dim * volume)
        return pressure

    def berendsen_thermostat(self, current_temp: float):
        """
        Berendsen 弱耦合 thermostat。
        
        v_new = v · λ，其中 λ = √(1 + Δt/τ · (T_target/T - 1))
        """
        if current_temp < 1e-12:
            return
        lam = np.sqrt(1.0 + self.dt / self.tau * (self.target_temp / current_temp - 1.0))
        lam = np.clip(lam, 0.8, 1.2)  # 稳定性约束
        self.vel *= lam

    def velocity_verlet_step(self, apply_thermostat: bool = True):
        """
        执行一个 Velocity Verlet 时间步。
        
        算法流程：
            1. pos(t+dt) = pos(t) + vel(t)*dt + 0.5*acc(t)*dt²
            2. 应用 PBC
            3. 计算新力 force(t+dt) 和 acc(t+dt)
            4. vel(t+dt) = vel(t) + 0.5*(acc(t)+acc(t+dt))*dt
            5. 可选 thermostat
        """
        # TODO: 实现 Velocity Verlet 积分单步
        # 提示: 辛积分器，保持相空间体积守恒
        raise NotImplementedError("Hole_2: 请补全 velocity_verlet_step 积分方案")

    def run(self, n_steps: int, equilibration_steps: int = 100,
            apply_thermostat: bool = True) -> dict:
        """
        运行 MD 模拟。
        
        参数:
            n_steps: 总步数
            equilibration_steps: 平衡步数（仅 thermostat）
            apply_thermostat: 是否应用温度控制
        
        返回:
            包含时间序列数据的字典
        """
        # 初始化加速度
        _, _ = self.compute_forces_and_energies()
        self.acc = self.force / self.mass

        self.time_history.clear()
        self.potential_history.clear()
        self.kinetic_history.clear()
        self.total_energy_history.clear()
        self.temperature_history.clear()
        self.pressure_history.clear()

        for step in range(n_steps):
            pot, kin, temp, press, etot = self.velocity_verlet_step(
                apply_thermostat=(apply_thermostat and step < equilibration_steps)
            )
            t = step * self.dt
            self.time_history.append(t)
            self.potential_history.append(pot)
            self.kinetic_history.append(kin)
            self.total_energy_history.append(etot)
            self.temperature_history.append(temp)
            self.pressure_history.append(press)

        return {
            'time': np.array(self.time_history),
            'potential': np.array(self.potential_history),
            'kinetic': np.array(self.kinetic_history),
            'total_energy': np.array(self.total_energy_history),
            'temperature': np.array(self.temperature_history),
            'pressure': np.array(self.pressure_history),
        }

    def get_final_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回最终的 (positions, velocities, accelerations)。"""
        return self.pos.copy(), self.vel.copy(), self.acc.copy()
