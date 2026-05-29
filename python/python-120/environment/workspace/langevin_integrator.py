"""
langevin_integrator.py
Langevin 随机分子动力学积分器

整合原项目:
  - 1063_sde: 随机微分方程 Euler-Maruyama 方法及其强/弱收敛性分析
  - 819_normal01_multivariate_distance: 多元正态分布随机采样

科学背景:
  表面催化体系中，吸附原子/分子在势能面上运动受环境热涨落影响。
  Langevin 方程描述了这种带阻尼和随机力的运动:
  
    m * d²r/dt² = -∇V(r) - γ * m * dr/dt + √(2γm k_B T) * η(t)
  
  其中:
    -∇V(r)      : 保守力 (来自势能面)
    -γm dr/dt   : 摩擦阻尼力 (与表面声子耦合)
    √(2γm k_B T) η(t) : 随机力 (涨落-耗散定理)
    η(t)        : 高斯白噪声, ⟨η(t)η(t')⟩ = δ(t-t')
  
  等价的一阶形式 (SDE):
    dr = v dt
    dv = (-∇V/m - γv) dt + √(2γ k_B T / m) dW
  
  其中 W(t) 为标准 Wiener 过程
"""

import numpy as np
from typing import Callable, Tuple, Optional, List


class LangevinIntegrator:
    """
    Langevin 动力学积分器
    
    采用 BAOAB 分裂积分方案 (Leimkuhler & Matthews, 2013):
      B: v ← v - (∇V/m) * (dt/2)
      A: r ← r + v * (dt/2)
      O: v ← c * v + √(m k_B T (1-c²)) * ξ  (c = exp(-γ dt))
      A: r ← r + v * (dt/2)
      B: v ← v - (∇V/m) * (dt/2)
    
    BAOAB 为二阶精度辛积分器，在温和阻尼条件下保持 Boltzmann 分布
    """

    def __init__(self, mass_amu: np.ndarray, gamma_ps: float,
                 temperature_k: float, dt_fs: float = 1.0,
                 n_dims: int = 3):
        """
        参数:
          mass_amu: 粒子质量 (amu)
          gamma_ps: 摩擦系数 (ps^-1)
          temperature_k: 温度 (K)
          dt_fs: 时间步长 (fs)
          n_dims: 每个粒子的空间维度
        """
        from utils import AMU_TO_KG, FS_TO_S, BOLTZMANN_KB
        self.mass = np.asarray(mass_amu, dtype=float) * AMU_TO_KG
        self.gamma = gamma_ps / (1.0e12)  # 转换为 s^-1
        self.temperature = temperature_k
        self.dt = dt_fs * FS_TO_S
        self.n_dims = n_dims
        self.kb = BOLTZMANN_KB
        self.amu_to_kg = AMU_TO_KG

        # BAOAB 系数
        self.c1 = np.exp(-self.gamma * self.dt)
        self.c2 = np.sqrt(self.kb * self.temperature * (1.0 - self.c1 ** 2))

        self.positions = None
        self.velocities = None
        self.n_particles = None

    def initialize(self, positions: np.ndarray, velocities: Optional[np.ndarray] = None):
        """
        初始化位置和速度
        
        若未提供速度，则从 Maxwell-Boltzmann 分布采样:
          P(v) ∝ exp(-m v² / (2 k_B T))
        """
        positions = np.asarray(positions, dtype=float)
        if positions.ndim == 1:
            positions = positions.reshape(-1, self.n_dims)
        self.n_particles = positions.shape[0]
        self.positions = positions.copy()

        if velocities is None:
            # Maxwell-Boltzmann 速度分布
            sigma_v = np.sqrt(self.kb * self.temperature / self.mass)
            self.velocities = np.random.normal(0.0, sigma_v[:, None],
                                               size=(self.n_particles, self.n_dims))
        else:
            self.velocities = np.asarray(velocities, dtype=float).copy()

    def step(self, force_func: Callable[[np.ndarray], np.ndarray]):
        """
        执行一个 BAOAB 时间步
        """
        if self.positions is None:
            raise RuntimeError("必须先调用 initialize()")

        # B step (half)
        forces = force_func(self.positions)
        acc = forces / self.mass[:, None]
        self.velocities += acc * (self.dt * 0.5)

        # A step (half)
        self.positions += self.velocities * (self.dt * 0.5)

        # O step (Ornstein-Uhlenbeck)
        # 多元正态采样 (整合原项目 819_normal01_multivariate_distance)
        noise = np.random.normal(0.0, 1.0, size=(self.n_particles, self.n_dims))
        mass_factor = 1.0 / np.sqrt(self.mass[:, None])
        self.velocities = (self.c1 * self.velocities +
                           self.c2 * mass_factor * noise)

        # A step (half)
        self.positions += self.velocities * (self.dt * 0.5)

        # B step (half)
        forces = force_func(self.positions)
        acc = forces / self.mass[:, None]
        self.velocities += acc * (self.dt * 0.5)

    def run(self, n_steps: int, force_func: Callable[[np.ndarray], np.ndarray],
            callback: Optional[Callable[[int, np.ndarray, np.ndarray], None]] = None):
        """
        运行 n_steps 步 Langevin 动力学
        """
        if n_steps < 0:
            raise ValueError("n_steps >= 0")
        for step in range(n_steps):
            self.step(force_func)
            if callback is not None:
                callback(step, self.positions, self.velocities)

    def kinetic_energy(self) -> float:
        """
        计算体系总动能
        
        公式:
          E_kin = Σ_i (1/2) m_i v_i²
        """
        if self.velocities is None:
            return 0.0
        return float(np.sum(0.5 * self.mass[:, None] * self.velocities ** 2))

    def temperature_instantaneous(self) -> float:
        """
        计算瞬时温度
        
        公式:
          T_inst = (2 E_kin) / (N_f k_B)
        
        其中 N_f = n_particles * n_dims 为自由度
        """
        e_kin = self.kinetic_energy()
        n_dof = self.n_particles * self.n_dims
        return (2.0 * e_kin) / (n_dof * self.kb)

    def compute_mean_square_displacement(self, traj: List[np.ndarray]) -> np.ndarray:
        """
        计算均方位移 (Mean Square Displacement, MSD)
        
        公式:
          MSD(t) = ⟨|r(t) - r(0)|²⟩
        
        用于计算表面扩散系数 (Einstein 关系):
          D = lim_{t→∞} MSD(t) / (2 d t)
        """
        if len(traj) == 0:
            return np.array([])
        r0 = traj[0]
        msd = np.zeros(len(traj))
        for i, ri in enumerate(traj):
            dr = ri - r0
            msd[i] = np.mean(np.sum(dr ** 2, axis=1))
        return msd

    def diffusion_coefficient_from_msd(self, traj: List[np.ndarray],
                                       time_interval_fs: float) -> float:
        """
        由 MSD 拟合扩散系数
        
        Einstein 关系:
          D = MSD / (2 * d * t)
        
        其中 d = n_dims 为空间维度
        """
        msd = self.compute_mean_square_displacement(traj)
        n_frames = len(msd)
        if n_frames < 10:
            return 0.0
        # 取后 50% 数据线性拟合
        start_idx = n_frames // 2
        t_vals = np.arange(start_idx, n_frames) * time_interval_fs * 1e-15
        msd_vals = msd[start_idx:]
        # 线性拟合: MSD = 2 * d * D * t
        A = np.vstack([t_vals, np.ones_like(t_vals)]).T
        slope, _ = np.linalg.lstsq(A, msd_vals, rcond=None)[0]
        D = slope / (2.0 * self.n_dims)
        return float(D)


class StochasticReactionDynamics:
    """
    表面催化反应的随机反应动力学
    
    将 Langevin 动力学与 Gillespie 算法结合，
    模拟吸附、扩散、反应和脱附的耦合过程
    """

    def __init__(self, surface, pes, temperature_k: float = 500.0):
        self.surface = surface
        self.pes = pes
        self.temperature = temperature_k
        self.time = 0.0
        self.event_log = []

    def compute_event_rates(self, species_map: np.ndarray) -> np.ndarray:
        """
        计算所有可能事件 (吸附/脱附/扩散/反应) 的速率
        
        公式 (Arrhenius 形式):
          k_i = ν * exp(-E_a,i / (k_B T))
        
        其中 E_a,i 由势能面局部曲率估计
        """
        # TODO Hole_2: 实现事件速率计算
        # 提示:
        #   1. 遍历每个表面位点，获取位点坐标和 PES 能量 (self.pes.evaluate)
        #   2. 根据 species_map[i] 判断位点状态:
        #      - 空位 (0): 计算吸附活化能 E_a = max(0, -e_site)，速率 = ν * exp(-E_a / kBT)
        #      - 占据态: 计算脱附活化能 E_a = max(0, e_site + 1.0)
        #                以及扩散势垒 (约 0.3 eV)
        #   3. 返回长度为 n_sites * 3 的速率数组
        # 注意: 本方法依赖 potential_surface.PotentialEnergySurface.evaluate()
        raise NotImplementedError("Hole_2: 请实现 compute_event_rates 方法")

    def gillespie_step(self, species_map: np.ndarray) -> Tuple[float, int, int]:
        """
        执行一步 Gillespie 随机模拟算法
        
        算法:
          1. 计算所有事件速率 r_i
          2. 总速率 R = Σ r_i
          3. 采样等待时间 τ = -ln(u_1) / R
          4. 按概率 r_i / R 选择事件
        
        返回:
          τ: 等待时间
          event_type: 事件类型
          site_idx: 位点索引
        """
        rates = self.compute_event_rates(species_map)
        R_total = np.sum(rates)
        if R_total < 1e-300:
            return 1.0e10, -1, -1

        u1 = np.random.random()
        tau = -np.log(u1) / R_total

        # 选择事件
        cumsum = np.cumsum(rates) / R_total
        u2 = np.random.random()
        event_idx = int(np.searchsorted(cumsum, u2))
        site_idx = event_idx // 3
        event_type = event_idx % 3  # 0=吸附, 1=脱附, 2=扩散

        return tau, event_type, site_idx
