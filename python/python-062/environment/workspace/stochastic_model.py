"""
stochastic_model.py
================================================================================
随机拉格朗日扩散模型模块 —— 基于种子项目 175_chuckaluck_simulation（蒙特卡洛思想）

在边界层研究中，拉格朗日粒子扩散模型（LPDM）用于追踪标量（如污染物、
水汽、气溶胶）的传输与扩散。本模块实现随机 Langevin 方程的数值积分，
模拟粒子在湍流场中的随机游走。

核心物理公式
--------------------------------------------------------------------------------
Langevin 方程（Well-Mixed Condition, Thomson 1987）：
    dX_i = U_i dt
    dU_i = a_i(X, U, t) dt + b_{ij}(X, t) dW_j(t)

其中 W_j(t) 为维纳过程，b_{ij} = √(2 C_0 ε) δ_{ij} 为扩散矩阵，
C_0 ≈ 2.1 为 Kolmogorov 常数。

对于边界层中的垂直扩散，漂移系数 a_w 需满足 well-mixed 条件：
    a_w = - (C_0 ε / σ_w²) w' + 1/2 ∂σ_w²/∂z

其中 σ_w² = ⟨w'²⟩ 为垂直速度方差。

蒙特卡洛估计：通过大量粒子轨迹的系综平均，可估计浓度场：
    C(x,t) = (1/N_p) Σ_{n=1}^{N_p} δ(x - X^{(n)}(t))
"""

import numpy as np


def langevin_step_euler(particles, velocities, sigma_w, epsilon, dt, C0=2.1):
    """
    执行一个 Euler-Maruyama 步的 Langevin 扩散。

    参数
    ----------
    particles : np.ndarray, shape (n, 3)
        粒子位置 [x, y, z]
    velocities : np.ndarray, shape (n, 3)
        粒子速度 [u, v, w]
    sigma_w : np.ndarray, shape (n,)
        当地垂直速度标准差
    epsilon : np.ndarray, shape (n,)
        当地耗散率
    dt : float
        时间步长
    C0 : float
        Kolmogorov 常数

    返回
    -------
    new_particles, new_velocities : np.ndarray
    """
    n_particle = particles.shape[0]

    # 漂移项
    w = velocities[:, 2]
    sigma_w_safe = np.maximum(sigma_w, 1e-6)

    # 简化的 Langevin 漂移（中性边界层近似）
    a_w = - (C0 * epsilon / sigma_w_safe**2) * w

    # 扩散系数
    b = np.sqrt(C0 * epsilon)

    # 随机增量
    dW = np.random.randn(n_particle) * np.sqrt(dt)

    # 速度更新
    new_velocities = np.copy(velocities)
    new_velocities[:, 2] = w + a_w * dt + b * dW

    # 位置更新
    new_particles = particles + new_velocities * dt

    # 下边界反射（地表）
    for i in range(n_particle):
        if new_particles[i, 2] < 0:
            new_particles[i, 2] = -new_particles[i, 2]
            new_velocities[i, 2] = -abs(new_velocities[i, 2])

    return new_particles, new_velocities


def initialize_particles(n_particles, domain_x, domain_y, domain_z, release_height=10.0):
    """
    初始化粒子群。

    参数
    ----------
    n_particles : int
    domain_x, domain_y, domain_z : tuple
        (xmin, xmax) 等
    release_height : float
        释放高度

    返回
    -------
    particles, velocities : np.ndarray
    """
    np.random.seed(123)

    particles = np.zeros((n_particles, 3), dtype=np.float64)
    particles[:, 0] = np.random.uniform(domain_x[0], domain_x[1], n_particles)
    particles[:, 1] = np.random.uniform(domain_y[0], domain_y[1], n_particles)
    particles[:, 2] = release_height + np.random.exponential(2.0, n_particles)
    particles[:, 2] = np.clip(particles[:, 2], 0.1, domain_z[1])

    velocities = np.zeros((n_particles, 3), dtype=np.float64)
    velocities[:, 0] = 5.0 + np.random.randn(n_particles) * 0.5
    velocities[:, 2] = np.random.randn(n_particles) * 0.3

    return particles, velocities


def ensemble_concentration(particles, grid_x, grid_y, grid_z):
    """
    将粒子位置转换为网格浓度（箱计数法）。

    参数
    ----------
    particles : np.ndarray, shape (n, 3)
    grid_x, grid_y, grid_z : np.ndarray
        网格边界

    返回
    -------
    conc : np.ndarray, shape (nx-1, ny-1, nz-1)
    """
    nx, ny, nz = len(grid_x) - 1, len(grid_y) - 1, len(grid_z) - 1
    conc = np.zeros((nx, ny, nz), dtype=np.float64)

    dx = grid_x[1] - grid_x[0]
    dy = grid_y[1] - grid_y[0]
    dz = grid_z[1] - grid_z[0]
    vol = dx * dy * dz

    for p in particles:
        ix = int((p[0] - grid_x[0]) / dx)
        iy = int((p[1] - grid_y[0]) / dy)
        iz = int((p[2] - grid_z[0]) / dz)

        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
            conc[ix, iy, iz] += 1.0 / vol

    return conc
