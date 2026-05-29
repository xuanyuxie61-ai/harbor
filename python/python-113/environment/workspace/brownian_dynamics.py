"""
brownian_dynamics.py
布朗动力学模拟模块

基于种子项目 613_jumping_bean_simulation 的核心算法：
- 温度驱动的随机跳跃过程
- 粒子在二维/三维空间中的离散时间随机行走

在离子通道问题中的应用：
将 jumping bean 的热激活跳跃模型映射为离子在通道势能面中的
Langevin 动力学：

    dr = (D / k_B T) F(r) dt + sqrt(2D dt) R(t)

其中 F(r) = -∇U(r) 为平均力，D 为扩散系数，R(t) 为标准高斯白噪声。

选择性机制：离子在滤器中的跳跃概率受温度、局部电场和配位环境调制，
K+ 由于与羰基氧的完美几何匹配，其跳跃能垒远低于 Na+。
"""

import numpy as np


class IonParticle:
    """
    单个离子粒子的状态。
    """
    def __init__(self, pos, charge, radius, mass_amu=39.0983):
        self.pos = np.array(pos, dtype=float)  # nm
        self.charge = charge  # e
        self.radius = radius  # nm（Pauling 半径）
        self.mass = mass_amu * 1.66053906660e-27  # kg
        self.trajectory = [self.pos.copy()]


class BrownianDynamicsEngine:
    """
    布朗动力学引擎：Overdamped Langevin 方程的 Euler-Maruyama 离散化。

    方程：
        γ dr = F(r) dt + sqrt(2 γ k_B T) dW(t)

    离散形式（时间步长 Δt）：
        r_{n+1} = r_n + (D / k_B T) F(r_n) Δt + sqrt(2 D Δt) ξ_n

    其中 D = k_B T / γ 为 Einstein 关系，ξ_n ~ N(0, I)。
    """
    def __init__(self, temperature=300.0, dt=1e-15, friction=1e-11):
        """
        Parameters
        ----------
        temperature : float
            温度 (K)
        dt : float
            时间步长 (s)
        friction : float
            摩擦系数 γ (kg/s)
        """
        self.T = temperature
        self.dt = dt
        self.gamma = friction
        self.kB = 1.380649e-23
        self.kT = self.kB * temperature

    def _random_displacement(self, D):
        """
        随机位移项 sqrt(2 D dt) ξ，ξ 为标准 3D 高斯向量。
        """
        sigma = np.sqrt(2.0 * D * self.dt)
        return sigma * np.random.randn(3)

    def _force_field(self, particle, phi_field, grid_origin, grid_spacing):
        """
        从电势场计算电场力 F = -q ∇φ，并叠加简谐约束（通道壁）。

        额外力项：
            F_wall = -k_wall * max(0, r - r_channel(z))^2 * n_r
        """
        # 三线性插值获取局部电势梯度
        pos = particle.pos
        ix = int((pos[0] - grid_origin[0]) / grid_spacing[0])
        iy = int((pos[1] - grid_origin[1]) / grid_spacing[1])
        iz = int((pos[2] - grid_origin[2]) / grid_spacing[2])

        Nx, Ny, Nz = phi_field.shape
        ix = np.clip(ix, 1, Nx - 2)
        iy = np.clip(iy, 1, Ny - 2)
        iz = np.clip(iz, 1, Nz - 2)

        dx, dy, dz = grid_spacing
        # 中心差分近似电场
        Ex = -(phi_field[ix + 1, iy, iz] - phi_field[ix - 1, iy, iz]) / (2.0 * dx)
        Ey = -(phi_field[ix, iy + 1, iz] - phi_field[ix, iy - 1, iz]) / (2.0 * dy)
        Ez = -(phi_field[ix, iy, iz + 1] - phi_field[ix, iy, iz - 1]) / (2.0 * dz)
        E_field = np.array([Ex, Ey, Ez])

        # 电场力 (N)，注意单位转换
        e_charge = 1.602176634e-19
        F_electric = -particle.charge * e_charge * E_field * 1e9  # 转为 N (V/nm -> N/C)

        # 通道壁约束（简谐势）
        r_xy = np.sqrt(pos[0] ** 2 + pos[1] ** 2)
        z = pos[2]
        # 简化通道半径
        if 1.5 <= z <= 2.7:
            r_channel = 0.15
        elif 0.5 <= z < 1.5:
            r_channel = 0.5
        elif z < 0.5:
            r_channel = 0.2 + 0.4 * z
        else:
            r_channel = 0.6

        k_wall = 1e-8  # N/m
        if r_xy > r_channel:
            dr = r_xy - r_channel
            n_r = np.array([pos[0], pos[1], 0.0]) / (r_xy + 1e-12)
            F_wall = -k_wall * dr * n_r
        else:
            F_wall = np.zeros(3)

        return F_electric + F_wall

    def step(self, particle, phi_field, grid_origin, grid_spacing, D_coeff):
        """
        执行一个布朗动力学时间步。
        """
        F = self._force_field(particle, phi_field, grid_origin, grid_spacing)
        # Overdamped: dr = D/(kT) F dt + sqrt(2D dt) dW
        drift = (D_coeff / self.kT) * F * self.dt
        diffusion = self._random_displacement(D_coeff)
        particle.pos += drift + diffusion
        particle.trajectory.append(particle.pos.copy())
        return particle

    def run(self, particles, phi_field, grid_origin, grid_spacing,
            D_k=1.96e-9, D_na=1.33e-9, n_steps=1000):
        """
        运行多粒子布朗动力学模拟。

        Parameters
        ----------
        particles : list of IonParticle
        phi_field : ndarray
            电势场 (V)
        grid_origin : array-like
            网格原点坐标 (nm)
        grid_spacing : array-like
            网格间距 (nm)
        D_k, D_na : float
            K+ 和 Na+ 的扩散系数 (m^2/s)
        n_steps : int
            模拟步数
        """
        for step in range(n_steps):
            for p in particles:
                D = D_k if p.mass > 3e-26 else D_na  # 简单区分 K+ 和 Na+
                self.step(p, phi_field, grid_origin, grid_spacing, D)
        return particles


def compute_mean_square_displacement(trajectories, dt, max_lag=None):
    """
    计算均方位移（MSD）：
        MSD(τ) = < |r(t+τ) - r(t)|^2 >

    用于验证 Einstein 关系：
        MSD(τ) = 6 D τ  （三维）
    """
    if max_lag is None:
        max_lag = len(trajectories[0]) // 4

    msd = np.zeros(max_lag)
    counts = np.zeros(max_lag)

    for traj in trajectories:
        traj = np.array(traj)
        n = len(traj)
        for lag in range(1, max_lag):
            disp = traj[lag:] - traj[:-lag]
            sq = np.sum(disp ** 2, axis=1)
            msd[lag] += np.sum(sq)
            counts[lag] += len(sq)

    msd = msd / (counts + 1e-30)
    tau = np.arange(max_lag) * dt
    # 线性拟合求 D
    valid = (tau > 0) & (msd > 0)
    if np.sum(valid) > 5:
        slope = np.polyfit(tau[valid], msd[valid], 1)[0]
        D_estimated = slope / 6.0
    else:
        D_estimated = 0.0

    return tau, msd, D_estimated
