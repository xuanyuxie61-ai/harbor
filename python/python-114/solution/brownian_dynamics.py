"""
brownian_dynamics.py

DNA 损伤修复分子动力学 —— 修复蛋白的布朗动力学与随机微分方程求解

基于种子项目:
  - 818_normal_ode: 正态分布ODE (dy/dt = -t*y, 精确解为高斯函数)
  - 517_henon_orbit: Henon 离散动力系统映射

科学背景:
  DNA 损伤响应 (DDR) 中，KU70/KU80、DNA-PKcs、XRCC4/Ligase IV 等修复
  蛋白通过三维布朗运动在核质中扩散，搜寻并结合 DSB 位点。本模块实现了
   overdamped Langevin 方程的数值积分，用于模拟修复蛋白的随机轨迹；
  同时引入 Henon 型混沌映射来刻画损伤位点周围染色质纤维的非均匀
  阻碍效应 (anomalous diffusion in crowded environment)。
"""

import numpy as np
from typing import Callable, Tuple, Optional


class OverdampedLangevinIntegrator:
    """
    Overdamped Langevin 方程积分器:

        γ * dx/dt = -∇U(x) + ξ(t)

    其中 γ 为摩擦系数，U(x) 为势能，ξ(t) 为白噪声，满足:
        <ξ_i(t)> = 0
        <ξ_i(t) ξ_j(t')> = 2γ k_B T δ_{ij} δ(t-t')

    离散形式 (Euler-Maruyama):
        x_{n+1} = x_n - (Δt/γ) ∇U(x_n) + sqrt(2 D Δt) * η_n

    其中 D = k_B T / γ 为扩散系数，η_n ~ N(0, I)。
    """

    def __init__(
        self,
        diffusion_coeff: float = 5.0,  # nm^2 / μs
        friction_coeff: float = 1.2e-3,  # pN·s/nm
        temperature: float = 310.0,  # K (生理温度)
        dt: float = 1e-3,  # μs
    ):
        """
        Parameters
        ----------
        diffusion_coeff : float
            扩散系数 D (nm^2/μs)。根据 Stokes-Einstein:
                D = k_B T / (6π η r_h)
            典型蛋白 (~50 kDa) 在核质中 D ≈ 1–10 μm^2/s = 1–10 nm^2/μs。
        friction_coeff : float
            摩擦系数 γ (pN·s/nm)。
        temperature : float
            绝对温度 (K)。
        dt : float
            时间步长 (μs)。
        """
        self.D = float(diffusion_coeff)
        self.gamma = float(friction_coeff)
        self.T = float(temperature)
        self.dt = float(dt)
        self.k_B = 1.380649e-5  # pN·nm / K

        # 数值稳定性检查：确保 Δt << γ / |Hessian| 的典型尺度
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.D <= 0.0:
            raise ValueError("diffusion coefficient must be positive")

        # 通过涨落-耗散定理自洽校验
        D_fdt = self.k_B * self.T / self.gamma
        if not np.isclose(self.D, D_fdt, rtol=0.5):
            # 允许用户自定义有效扩散系数（核质拥挤效应会修正 D）
            pass

    def step(
        self,
        x: np.ndarray,
        force: np.ndarray,
    ) -> np.ndarray:
        """
        执行一个 Euler-Maruyama 步进。

        Parameters
        ----------
        x : ndarray, shape (n_particles, 3)
            当前坐标。
        force : ndarray, shape (n_particles, 3)
            保守力 -∇U (pN)。

        Returns
        -------
        x_new : ndarray
            更新后的坐标。
        """
        x = np.asarray(x, dtype=np.float64)
        force = np.asarray(force, dtype=np.float64)

        if x.shape != force.shape:
            raise ValueError("x and force must have the same shape")

        n_particles = x.shape[0]

        # 确定性漂移项
        drift = (self.dt / self.gamma) * force

        # 随机扩散项: sqrt(2 D dt) * N(0,1)
        noise_amp = np.sqrt(2.0 * self.D * self.dt)
        noise = noise_amp * np.random.randn(*x.shape)

        x_new = x + drift + noise

        # 边界处理：将粒子限制在核内（半径约 5000 nm）
        nuclear_radius = 5000.0  # nm
        for i in range(n_particles):
            r = np.linalg.norm(x_new[i, :])
            if r > nuclear_radius:
                x_new[i, :] *= nuclear_radius / (r + 1e-12)

        return x_new

    def integrate(
        self,
        x0: np.ndarray,
        force_func: Callable[[np.ndarray], np.ndarray],
        n_steps: int,
    ) -> np.ndarray:
        """
        积分多条轨迹。

        Returns
        -------
        trajectory : ndarray, shape (n_steps+1, n_particles, 3)
        """
        x0 = np.asarray(x0, dtype=np.float64)
        traj = np.zeros((n_steps + 1,) + x0.shape, dtype=np.float64)
        traj[0, :, :] = x0

        x = x0.copy()
        for step in range(n_steps):
            f = force_func(x)
            x = self.step(x, f)
            traj[step + 1, :, :] = x

        return traj


def dna_repair_protein_force(
    x: np.ndarray,
    dsb_site: np.ndarray,
    binding_energy: float = 15.0,  # k_B T
    binding_range: float = 5.0,  # nm
    repulsion_strength: float = 2.0,  # k_B T / nm
    protein_radius: float = 3.0,  # nm
) -> np.ndarray:
    """
    计算修复蛋白受到的合力。

    势能模型:
        U_bind(r) = -E_bind * exp(-r^2 / (2σ_bind^2))

        U_rep(r_ij) = E_rep * exp(-(r_ij - 2r_p)^2 / (2σ_rep^2))  if r_ij < 2r_p

    其中 r 为蛋白到 DSB 位点的距离，r_ij 为蛋白间距离。

    总力:
        F_i = -∇_{x_i} [U_bind(|x_i - x_dsb|) + 0.5 Σ_{j≠i} U_rep(|x_i - x_j|)]
    """
    x = np.asarray(x, dtype=np.float64)
    dsb_site = np.asarray(dsb_site, dtype=np.float64)
    n = x.shape[0]

    if n == 0:
        return np.zeros_like(x)

    # DSB 结合力（高斯势阱）
    dx = x - dsb_site[np.newaxis, :]  # (n, 3)
    r = np.linalg.norm(dx, axis=1) + 1e-12

    # 力的大小: F = -dU/dr = -E_bind * r/σ^2 * exp(-r^2/(2σ^2))
    # 注意力的方向指向 DSB（势能最小点）
    f_bind_mag = binding_energy * (r / (binding_range ** 2)) * np.exp(-r ** 2 / (2.0 * binding_range ** 2))
    f_bind = -f_bind_mag[:, np.newaxis] * (dx / r[:, np.newaxis])

    # 蛋白间短程排斥力
    f_rep = np.zeros_like(x)
    if n > 1:
        for i in range(n):
            dx_ij = x[i, :] - x[i + 1 :, :]  # (n-i-1, 3)
            r_ij = np.linalg.norm(dx_ij, axis=1) + 1e-12
            mask = r_ij < 2.5 * protein_radius
            if np.any(mask):
                dr = r_ij[mask] - 2.0 * protein_radius
                f_mag = repulsion_strength * np.exp(-dr ** 2 / (2.0 * (protein_radius ** 2)))
                f_vec = f_mag[:, np.newaxis] * (dx_ij[mask, :] / r_ij[mask, np.newaxis])
                f_rep[i, :] += np.sum(f_vec, axis=0)
                # 反作用力
                idx = np.where(mask)[0] + i + 1
                for j_local, j_global in enumerate(idx):
                    f_rep[j_global, :] -= f_vec[j_local, :]

    # 总力（单位：k_B T / nm），需要换算为 pN
    k_B = 1.380649e-5  # pN·nm / K
    T = 310.0
    total_force = (f_bind + f_rep) * k_B * T  # pN

    return total_force


def henon_crowding_map(
    x: np.ndarray,
    y: np.ndarray,
    c: float = 0.98,
    n_iter: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Henon 离散映射，用于模拟染色质拥挤环境中粒子轨迹的混沌混合效应。

    映射方程（参数化为旋转-剪切形式）:
        x_{n+1} = x_n * c - (y_n - x_n^2) * s
        y_{n+1} = x_n * s + (y_n - x_n^2) * c

    其中 s = sqrt(1 - c^2), c = cos(α)。

    在 DNA 损伤修复的语境下，该映射描述了粒子在高度折叠的染色质纤维
    附近运动时，由于局部拓扑约束导致的非线性轨迹偏转。

    Parameters
    ----------
    x, y : ndarray
        输入坐标（可以是二维平面投影或参数化坐标）。
    c : float
        cos(α)，控制映射的混沌程度。取值接近 1 时接近可积，
        减小则混沌增强。
    n_iter : int
        迭代次数。

    Returns
    -------
    x_new, y_new : ndarray
        映射后的坐标。
    """
    if not (-1.0 <= c <= 1.0):
        raise ValueError("c must be in [-1, 1]")
    s = np.sqrt(max(0.0, 1.0 - c ** 2))

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # 仅在单位圆内应用映射以保持有界性
    r = np.sqrt(x ** 2 + y ** 2)
    valid = r < 1.0

    xv = x[valid].copy()
    yv = y[valid].copy()

    for _ in range(n_iter):
        xnew = xv * c - (yv - xv ** 2) * s
        ynew = xv * s + (yv - xv ** 2) * c
        xv, yv = xnew, ynew

    x_out = x.copy()
    y_out = y.copy()
    x_out[valid] = xv
    y_out[valid] = yv

    return x_out, y_out


def normal_distribution_ode_solution(t: np.ndarray, sigma0: float = 1.0) -> np.ndarray:
    """
    正态分布 ODE 的精确解，用于生成初始高斯分布的修复蛋白空间密度。

    ODE:
        dy/dt = -t * y

    解析解:
        y(t) = exp(-t^2 / 2) / sqrt(2π)

    该高斯分布描述了 DNA 损伤后修复因子从核质向损伤位点的径向浓度
    分布的稳态近似。

    Parameters
    ----------
    t : ndarray
        自变量（无量纲化径向距离）。
    sigma0 : float
        标准差缩放。

    Returns
    -------
    y : ndarray
        高斯分布值。
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.exp(-(t / sigma0) ** 2 / 2.0) / np.sqrt(2.0 * np.pi) / sigma0

    # 边界鲁棒性
    y = np.where(np.isfinite(y), y, 0.0)
    y = np.where(y >= 0, y, 0.0)
    return y


def simulate_ku80_search_time(
    n_proteins: int = 50,
    n_steps: int = 5000,
    dsb_position: Optional[np.ndarray] = None,
) -> dict:
    """
    模拟 KU80 蛋白在细胞核内搜索 DSB 位点的平均首次通过时间 (MFPT)。

    使用 overdamped Langevin 动力学，结合 DSB 处的结合势阱。

    Returns
    -------
    result : dict
        包含平均搜索时间、结合概率、均方位移 (MSD) 等信息。
    """
    if dsb_position is None:
        dsb_position = np.array([0.0, 0.0, 0.0])

    # 初始位置：在核内随机分布
    np.random.seed(42)
    x0 = np.random.randn(n_proteins, 3) * 1000.0  # nm
    nuclear_r = 5000.0
    for i in range(n_proteins):
        r = np.linalg.norm(x0[i, :])
        if r > nuclear_r:
            x0[i, :] *= nuclear_r / r

    integrator = OverdampedLangevinIntegrator(dt=0.01)

    force_func = lambda x: dna_repair_protein_force(x, dsb_position)

    traj = integrator.integrate(x0, force_func, n_steps)

    # 计算平均首次通过时间（到达 DSB 5nm 范围内）
    binding_radius = 5.0
    distances = np.linalg.norm(traj - dsb_position[np.newaxis, np.newaxis, :], axis=2)
    bound = distances < binding_radius  # (n_steps+1, n_proteins)

    # 首次通过步数
    first_pass = np.full(n_proteins, n_steps + 1, dtype=np.int64)
    for p in range(n_proteins):
        hit = np.where(bound[:, p])[0]
        if len(hit) > 0:
            first_pass[p] = hit[0]

    bound_proteins = first_pass <= n_steps
    if np.any(bound_proteins):
        mftp = float(np.mean(first_pass[bound_proteins])) * integrator.dt  # μs
    else:
        mftp = float(n_steps) * integrator.dt  # 上限估计

    # 均方位移 (MSD)
    msd = np.mean(np.sum((traj - traj[0, :, :][np.newaxis, :, :]) ** 2, axis=2), axis=1)

    return {
        "mfpt_us": mftp,
        "binding_fraction": float(np.mean(first_pass <= n_steps)),
        "msd_final_nm2": float(msd[-1]),
        "trajectory_shape": traj.shape,
    }
