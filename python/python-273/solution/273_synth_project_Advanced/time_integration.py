"""
time_integration.py — 晶格动力学时间积分方案
=============================================

融合种子项目:
  - 746_md_parfor: Velocity-Verlet 时间积分 + 对势分子动力学
    x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
    v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
  - 818_normal_ode: 一阶 ODE dy/dt = -t*y 的精确解 (正态分布)
  - 1149_BanerjeeLab: Euler 前向积分 + 自适应时间步

物理背景:
  晶格动力学的运动方程:
    m_i * d^2 u_i / dt^2 = -sum_j Phi_{ij} * u_j

  等价于一阶系统:
    du/dt = v
    dv/dt = -D * u  (D = 动力学矩阵 / 质量)

  声子模式是上述方程的简正模解:
    u_i(t) = A * e_i * exp(i*(k*R - omega*t))

  辛积分器 (Velocity-Verlet) 保能量长期守恒,
  比 Runge-Kutta 更适合哈密顿系统。
"""

import numpy as np
from typing import Tuple, Dict, Callable, Optional


def velocity_verlet_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    accelerations: np.ndarray,
    force_func: Callable,
    masses: np.ndarray,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Velocity-Verlet 单步 (融合 md_parfor)。

    算法:
      1. x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
      2. 计算新加速度 a(t+dt) = F(x(t+dt)) / m
      3. v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt

    辛性质: 相空间体积守恒, 能量误差不漂移 (仅振荡)。

    参数:
        positions: (N, 3) 位置
        velocities: (N, 3) 速度
        accelerations: (N, 3) 当前加速度
        force_func: 力函数 F(x) -> (N, 3)
        masses: (N,) 质量
        dt: 时间步长

    返回:
        new_pos, new_vel, new_acc: 更新后的状态
        kinetic_energy: 动能
        potential_energy: 势能
    """
    inv_masses = 1.0 / masses[:, np.newaxis]

    # Step 1: 更新位置
    new_pos = positions + velocities * dt + 0.5 * accelerations * dt ** 2

    # Step 2: 计算新力和加速度
    forces = force_func(new_pos)
    new_acc = forces * inv_masses

    # Step 3: 更新速度
    new_vel = velocities + 0.5 * (accelerations + new_acc) * dt

    # 能量计算
    ke = 0.5 * np.sum(masses[:, np.newaxis] * new_vel ** 2)
    pe = _compute_harmonic_pe(new_pos, positions, forces)

    return new_pos, new_vel, new_acc, ke, pe


def _compute_harmonic_pe(
    pos_new: np.ndarray, pos_old: np.ndarray, forces: np.ndarray,
) -> float:
    """
    简谐势能近似: PE = -0.5 * sum_i F_i . (x_i - x_eq)
    简化估计。
    """
    displacement = pos_new - pos_old
    return -0.5 * np.sum(forces * displacement)


def velocity_verlet_trajectory(
    initial_positions: np.ndarray,
    initial_velocities: np.ndarray,
    force_func: Callable,
    masses: np.ndarray,
    dt: float,
    n_steps: int,
    sample_interval: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Velocity-Verlet 轨迹演化 (融合 md_parfor 主循环)。

    返回:
        results: 包含 positions, velocities, energies, times 的字典
    """
    n_atoms = len(masses)
    n_samples = n_steps // sample_interval + 1

    pos_traj = np.zeros((n_samples, n_atoms, 3))
    vel_traj = np.zeros((n_samples, n_atoms, 3))
    ke_traj = np.zeros(n_samples)
    pe_traj = np.zeros(n_samples)
    te_traj = np.zeros(n_samples)
    time_traj = np.zeros(n_samples)

    pos = initial_positions.copy()
    vel = initial_velocities.copy()
    forces = force_func(pos)
    acc = forces / masses[:, np.newaxis]

    sample_idx = 0
    ke = 0.5 * np.sum(masses[:, np.newaxis] * vel ** 2)
    pe = _compute_harmonic_pe(pos, pos, forces)

    pos_traj[0] = pos
    vel_traj[0] = vel
    ke_traj[0] = ke
    pe_traj[0] = pe
    te_traj[0] = ke + pe
    time_traj[0] = 0.0

    for step in range(1, n_steps + 1):
        pos, vel, acc, ke, pe = velocity_verlet_step(
            pos, vel, acc, force_func, masses, dt
        )

        if step % sample_interval == 0:
            sample_idx += 1
            if sample_idx < n_samples:
                pos_traj[sample_idx] = pos
                vel_traj[sample_idx] = vel
                ke_traj[sample_idx] = ke
                pe_traj[sample_idx] = pe
                te_traj[sample_idx] = ke + pe
                time_traj[sample_idx] = step * dt

    # 截断到实际采样数
    actual = sample_idx + 1
    return {
        'positions': pos_traj[:actual],
        'velocities': vel_traj[:actual],
        'kinetic_energy': ke_traj[:actual],
        'potential_energy': pe_traj[:actual],
        'total_energy': te_traj[:actual],
        'times': time_traj[:actual],
    }


def nose_hoover_thermostat_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    accelerations: np.ndarray,
    force_func: Callable,
    masses: np.ndarray,
    dt: float,
    xi: float,
    target_ke: float,
    Q: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Nose-Hoover 热浴 Velocity-Verlet 步。

    扩展哈密顿量:
      H_ext = sum p_i^2/(2*m_i) + V(x) + p_xi^2/(2*Q) + g*k_B*T*xi

    运动方程:
      dx/dt = v
      dv/dt = F/m - xi*v  (摩擦项)
      dxi/dt = (sum m*v^2 - g*k_B*T) / Q

    其中 Q 为热浴质量参数, g 为自由度数目。

    参数:
        xi: Nose-Hoover 摩擦变量
        target_ke: 目标动能 = (g/2)*k_B*T
        Q: 热浴惯量
    """
    n_atoms = len(masses)
    inv_masses = 1.0 / masses[:, np.newaxis]
    g_dof = 3 * n_atoms

    # 半步速度更新 (含摩擦)
    vel_half = velocities + 0.5 * dt * (accelerations - xi * velocities)

    # 位置全步
    new_pos = positions + dt * vel_half

    # 新力
    forces = force_func(new_pos)
    new_acc = forces * inv_masses

    # 动能
    ke = 0.5 * np.sum(masses[:, np.newaxis] * vel_half ** 2)

    # 更新摩擦变量
    new_xi = xi + dt * (2.0 * ke - target_ke) / Q

    # 速度全步
    new_vel = (vel_half + 0.5 * dt * new_acc) / (1.0 + 0.5 * dt * new_xi)

    return new_pos, new_vel, new_acc, new_xi, ke


def euler_adaptive_step(
    state: np.ndarray,
    rhs_func: Callable,
    t: float,
    dt: float,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, float, float]:
    """
    自适应 Euler 步 (融合 BanerjeeLab 的自适应时间步)。

    使用步长加倍法估计误差:
      y_h = Euler(y, h)
      y_hh = Euler(Euler(y, h/2), h/2)
      error ≈ |y_hh - y_h|

    若 error > tol: dt -> dt/2
    若 error < tol/4: dt -> 2*dt

    参数:
        state: 当前状态向量
        rhs_func: dy/dt = f(t, y)
        t: 当前时间
        dt: 时间步长
        tol: 误差容限

    返回:
        new_state, error, recommended_dt
    """
    # 半步两次
    k1 = rhs_func(t, state)
    y_half = state + 0.5 * dt * k1
    k2 = rhs_func(t + 0.5 * dt, y_half)
    y_hh = y_half + 0.5 * dt * k2

    # 全步一次
    y_h = state + dt * k1

    error = np.linalg.norm(y_hh - y_h)
    error = max(error, 1e-30)

    # 自适应步长推荐
    if error > 1e-30:
        dt_new = dt * min(2.0, max(0.1, 0.9 * (tol / error) ** 0.5))
    else:
        dt_new = dt * 2.0

    return y_hh, error, dt_new


def initialize_thermal_velocities(
    masses: np.ndarray, temperature: float, seed: int = 42,
) -> np.ndarray:
    """
    初始化 Maxwell-Boltzmann 热速度分布。

    v_i ~ N(0, sqrt(k_B*T / m_i))

    参数:
        masses: (N,) 原子质量
        temperature: 温度 (K)
        seed: 随机种子

    返回:
        velocities: (N, 3) 初始速度
    """
    k_B = 1.3806e-23  # J/K
    amu_to_kg = 1.6605e-27  # kg

    rng = np.random.RandomState(seed)
    n_atoms = len(masses)
    velocities = np.zeros((n_atoms, 3))

    for i in range(n_atoms):
        sigma = np.sqrt(k_B * temperature / (masses[i] * amu_to_kg))
        velocities[i] = rng.normal(0, sigma, 3)

    # 去除质心运动
    total_mass = np.sum(masses)
    v_cm = np.sum(masses[:, np.newaxis] * velocities, axis=0) / total_mass
    velocities -= v_cm

    return velocities


def normal_ode_reference(t: np.ndarray) -> np.ndarray:
    """
    正态分布 ODE 参考解 (融合 normal_ode)。
    dy/dt = -t*y, 精确解: y(t) = exp(-t^2/2) / sqrt(2*pi)
    用于验证时间积分器的精度。
    """
    return np.exp(-t ** 2 / 2.0) / np.sqrt(2.0 * np.pi)
