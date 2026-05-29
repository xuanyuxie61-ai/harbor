"""
fdtd_engine.py

时域有限差分(FDTD)核心引擎模块。
融合biharmonic_fd2d的高阶差分思想与backward_euler_fixed的隐式时间推进思想，
实现三维麦克斯韦方程组的数值求解。

核心数值方法:
--------------
1. 显式Yee算法（Leap-Frog格式）:
   H^{n+½} = H^{n-½} - (Δt/μ) · ∇ × E^n
   E^{n+1} = E^n + (Δt/ε) · (∇ × H^{n+½} - σE^n)

2. 隐式-显式混合格式（基于backward_euler思想）:
   对于高损耗区域，采用隐式处理电导率项:
   (ε + σΔt/2) E^{n+1} = (ε - σΔt/2) E^n + Δt · ∇ × H^{n+½}

3. 稳定性分析:
   CFL条件: Δt ≤ Δx / (c√3)
   数值色散: sin²(ωΔt/2) = (cΔt)² [sin²(kxΔx/2)/Δx² + ...]

4. 能量守恒监测（基于rigid_body_ode的守恒量思想）:
   dW_em/dt + P_loss + ∮S·n dA = 0
"""

import numpy as np
from physics_constants import (
    curl_electric_to_magnetic,
    curl_magnetic_to_electric,
    electromagnetic_energy_density,
    cfl_condition_3d,
)


class FDTD3DEngine:
    """
    三维FDTD仿真引擎。
    """

    def __init__(self, grid, epsilon, mu, sigma, source=None, dt=None, cfl_factor=0.95):
        """
        Parameters
        ----------
        grid : YeeGrid3D
            Yee网格对象
        epsilon, mu, sigma : ndarray
            介电常数、磁导率、电导率场
        source : callable or None
            激励源函数 source(t, E, H, grid)
        dt : float or None
            时间步长，None时自动按CFL条件计算
        cfl_factor : float
            CFL安全因子 (0 < factor ≤ 1)
        """
        self.grid = grid
        self.epsilon = epsilon
        self.mu = mu
        self.sigma = sigma
        self.source = source

        # 确定最大波速
        c_max = np.max(1.0 / np.sqrt(epsilon * mu))
        self.c_max = c_max

        # 计算时间步长
        if dt is None:
            dt_cfl = cfl_condition_3d(grid.dx, grid.dy, grid.dz, c_max)
            self.dt = cfl_factor * dt_cfl
        else:
            dt_cfl = cfl_condition_3d(grid.dx, grid.dy, grid.dz, c_max)
            if dt > dt_cfl:
                raise ValueError(f"时间步长{dt}超过CFL极限{dt_cfl}")
            self.dt = dt

        self.cfl_factor = cfl_factor
        self.time = 0.0
        self.step_count = 0

        # 初始化场
        nx, ny, nz = grid.nx, grid.ny, grid.nz
        shape = (nx, ny, nz)
        self.Ex = np.zeros(shape)
        self.Ey = np.zeros(shape)
        self.Ez = np.zeros(shape)
        self.Hx = np.zeros(shape)
        self.Hy = np.zeros(shape)
        self.Hz = np.zeros(shape)

        # 预计算更新系数（提高运行效率）
        self._compute_update_coefficients()

        # 能量历史
        self.energy_history = []
        self.time_history = []

    def _compute_update_coefficients(self):
        """预计算电场和磁场的更新系数。"""
        dt = self.dt
        eps = self.epsilon
        mu = self.mu
        sig = self.sigma

        # 磁场更新系数: H_new = H_old - (dt/μ) * curl(E)
        self.ch = dt / mu

        # 电场更新系数（显式）: E_new = E_old + (dt/ε) * (curl(H) - σE)
        # 隐式处理电导率:
        # (ε + σΔt/2) E^{n+1} = (ε - σΔt/2) E^n + Δt * curl(H)
        denom = eps + 0.5 * sig * dt
        denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
        self.ce1 = (eps - 0.5 * sig * dt) / denom
        self.ce2 = dt / denom

    def update_magnetic(self):
        """
        更新磁场（法拉第定律）。
        H^{n+½} = H^{n-½} - (Δt/μ) · ∇ × E^n
        """
        # TODO: Hole 2a — 实现磁场更新
        # 1. 调用curl_electric_to_magnetic计算电场旋度
        # 2. 使用预计算的更新系数self.ch更新Hx, Hy, Hz
        raise NotImplementedError("Hole 2a: 请实现update_magnetic的磁场更新逻辑")

    def update_electric(self):
        """
        更新电场（安培定律，含损耗的隐式处理）。
        (ε + σΔt/2) E^{n+1} = (ε - σΔt/2) E^n + Δt · ∇ × H^{n+½}
        """
        # TODO: Hole 2b — 实现电场更新
        # 1. 调用curl_magnetic_to_electric计算磁场旋度
        # 2. 使用预计算的更新系数self.ce1和self.ce2更新Ex, Ey, Ez
        raise NotImplementedError("Hole 2b: 请实现update_electric的电场更新逻辑")

    def apply_pec_boundary(self):
        """
        应用理想导体(PEC)边界条件。
        PEC边界: 切向电场 E_tangential = 0
        """
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz

        # x = 0 和 x = Lx 边界
        self.Ey[0, :, :] = 0.0
        self.Ez[0, :, :] = 0.0
        self.Ey[-1, :, :] = 0.0
        self.Ez[-1, :, :] = 0.0

        # y = 0 和 y = Ly 边界
        self.Ex[:, 0, :] = 0.0
        self.Ez[:, 0, :] = 0.0
        self.Ex[:, -1, :] = 0.0
        self.Ez[:, -1, :] = 0.0

        # z = 0 和 z = Lz 边界
        self.Ex[:, :, 0] = 0.0
        self.Ey[:, :, 0] = 0.0
        self.Ex[:, :, -1] = 0.0
        self.Ey[:, :, -1] = 0.0

    def apply_source(self):
        """应用激励源。"""
        if self.source is not None:
            self.source(self.time, self)

    def compute_energy(self):
        """计算当前总电磁能量。"""
        E = (self.Ex, self.Ey, self.Ez)
        H = (self.Hx, self.Hy, self.Hz)
        w = electromagnetic_energy_density(E, H, self.epsilon, self.mu)
        return np.sum(w) * self.grid.cell_volume()

    def compute_power_loss(self):
        """计算当前损耗功率 P_loss = ∫ σ|E|² dV。"""
        E_mag_sq = self.Ex**2 + self.Ey**2 + self.Ez**2
        p_loss = self.sigma * E_mag_sq
        return np.sum(p_loss) * self.grid.cell_volume()

    def step(self):
        """
        执行一个完整的FDTD时间步进。
        顺序: 更新H -> 更新E -> 边界条件 -> 激励源
        """
        self.update_magnetic()
        self.update_electric()
        self.apply_pec_boundary()
        self.time += self.dt
        self.step_count += 1
        self.apply_source()

    def run(self, n_steps, energy_sample_interval=10):
        """
        运行多个时间步。

        Parameters
        ----------
        n_steps : int
            时间步数
        energy_sample_interval : int
            能量采样间隔

        Returns
        -------
        dict
            包含能量历史、时间历史等结果
        """
        for i in range(n_steps):
            self.step()
            if i % energy_sample_interval == 0:
                W = self.compute_energy()
                P = self.compute_power_loss()
                self.energy_history.append(W)
                self.time_history.append(self.time)

        return {
            'time_history': np.array(self.time_history),
            'energy_history': np.array(self.energy_history),
            'final_E': (self.Ex.copy(), self.Ey.copy(), self.Ez.copy()),
            'final_H': (self.Hx.copy(), self.Hy.copy(), self.Hz.copy()),
            'dt': self.dt,
            'n_steps': n_steps,
        }


class HarmonicSource:
    """
    正弦调制高斯脉冲激励源。
    用于激励特定频率的电磁模式。

    时域形式:
    s(t) = A · exp(-(t-t0)²/(2τ²)) · sin(2πf_c t)
    """

    def __init__(self, amplitude, frequency, t0, tau, position, component='Ez'):
        """
        Parameters
        ----------
        amplitude : float
            幅度 [V/m]
        frequency : float
            中心频率 [Hz]
        t0 : float
            脉冲峰值时间 [s]
        tau : float
            脉冲宽度 [s]
        position : tuple
            (ix, iy, iz) 源位置索引
        component : str
            'Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'
        """
        self.amplitude = amplitude
        self.frequency = frequency
        self.t0 = t0
        self.tau = tau
        self.position = position
        self.component = component

    def __call__(self, t, engine):
        """将源加入场中。"""
        envelope = np.exp(-((t - self.t0) ** 2) / (2.0 * self.tau ** 2))
        value = self.amplitude * envelope * np.sin(2.0 * np.pi * self.frequency * t)

        ix, iy, iz = self.position
        if self.component == 'Ex':
            engine.Ex[ix, iy, iz] += value
        elif self.component == 'Ey':
            engine.Ey[ix, iy, iz] += value
        elif self.component == 'Ez':
            engine.Ez[ix, iy, iz] += value
        elif self.component == 'Hx':
            engine.Hx[ix, iy, iz] += value
        elif self.component == 'Hy':
            engine.Hy[ix, iy, iz] += value
        elif self.component == 'Hz':
            engine.Hz[ix, iy, iz] += value


def stability_analysis_2d_scalar(kx, ky, dx, dy, dt, c):
    """
    二维标量波动方程的数值色散分析。

    理论色散关系: ω² = c²(kx² + ky²)
    数值色散关系: sin²(ωΔt/2)/(cΔt)² = sin²(kxΔx/2)/Δx² + sin²(kyΔy/2)/Δy²

    Parameters
    ----------
    kx, ky : float
        波数分量
    dx, dy : float
        空间步长
    dt : float
        时间步长
    c : float
        波速

    Returns
    -------
    omega_numerical, omega_exact : float
        数值频率与理论频率
    """
    rhs = (np.sin(kx * dx / 2.0) / dx) ** 2 + (np.sin(ky * dy / 2.0) / dy) ** 2
    arg = (c * dt) ** 2 * rhs
    arg = min(arg, 1.0)  # 稳定性限制
    omega_numerical = 2.0 / dt * np.arcsin(np.sqrt(arg))
    omega_exact = c * np.sqrt(kx ** 2 + ky ** 2)
    return omega_numerical, omega_exact
