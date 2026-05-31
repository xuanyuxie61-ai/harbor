"""
particle_dynamics.py
生物质颗粒运动学模拟模块
基于分子动力学（Molecular Dynamics）的 Velocity Verlet 算法，
模拟生物质颗粒在反应器流场中的运动、碰撞与传热。
原项目映射:
  - 745_md_fast (快速分子动力学模拟)
"""

import numpy as np
from utils import check_bounds


def sinsq_potential(r, r0=1.0):
    """
    修正的 sin² 势能函数（平滑截断）。
    V(r) = sin²(min(r, π/2))
    
    用于描述颗粒间短程排斥相互作用。
    映射自 md_fast.m 中的 sinsq_potential。
    """
    r = np.asarray(r, dtype=np.float64)
    pi2 = np.pi / 2.0
    r_trunc = np.where(r <= pi2, r, pi2)
    return np.sin(r_trunc) ** 2


def sinsq_force(r, r0=1.0):
    """
    sin² 势能对应的力: F = -dV/dr = -sin(2r) (for r <= π/2)。
    """
    r = np.asarray(r, dtype=np.float64)
    pi2 = np.pi / 2.0
    r_trunc = np.where(r <= pi2, r, pi2)
    return -np.sin(2.0 * r_trunc)


def compute_forces_and_energies(pos, vel, mass, box, interaction_type='sinsq'):
    """
    计算颗粒间作用力与系统能量。
    映射自 md_fast.m 中的 compute 函数。
    
    参数:
        pos: 位置 (nd, np)
        vel: 速度 (nd, np)
        mass: 质量标量
        box: 盒子尺寸 (nd,)
    返回:
        force: 力 (nd, np)
        potential: 总势能
        kinetic: 总动能
    """
    nd, np_particles = pos.shape
    force = np.zeros_like(pos, dtype=np.float64)
    potential = 0.0

    for i in range(np_particles):
        # 到粒子 i 的向量
        Ri = pos - pos[:, i:i + 1]
        # 最小镜像约定（周期性边界）
        for d in range(nd):
            Ri[d, :] -= box[d] * np.round(Ri[d, :] / box[d])
        # 距离
        D = np.sqrt(np.sum(Ri ** 2, axis=0))
        mask = D > 1e-10
        Ri_valid = Ri[:, mask]
        D_valid = D[mask]

        if interaction_type == 'sinsq':
            pi2 = np.pi / 2.0
            D2 = np.where(D_valid <= pi2, D_valid, pi2)
            potential += 0.5 * np.sum(np.sin(D2) ** 2)
            # 力: F = Ri * (sin(2D)/D)
            force_factor = np.sin(2.0 * D2) / D_valid
            force[:, i] += np.sum(Ri_valid * force_factor, axis=1)
        elif interaction_type == 'lennard_jones':
            # Lennard-Jones 势: V = 4ε[(σ/r)^12 - (σ/r)^6]
            sigma = 0.3
            epsilon = 1.0
            sr = sigma / D_valid
            sr6 = sr ** 6
            sr12 = sr6 ** 2
            potential += 0.5 * np.sum(4.0 * epsilon * (sr12 - sr6))
            force_factor = 24.0 * epsilon * (2.0 * sr12 - sr6) / D_valid
            force[:, i] += np.sum(Ri_valid * force_factor, axis=1)

    kinetic = 0.5 * mass * np.sum(vel ** 2)
    return force, potential, kinetic


def initialize_particles(np_particles, nd, box, temperature=300.0, mass=1.0):
    """
    初始化颗粒位置、速度与加速度。
    映射自 md_fast.m 中的 initialize 函数。
    
    位置在盒子内均匀随机分布，
    速度按 Maxwell-Boltzmann 分布采样（正态分布）。
    """
    pos = np.random.rand(nd, np_particles).astype(np.float64)
    for d in range(nd):
        pos[d, :] *= box[d]
    # 根据温度初始化速度: σ_v = sqrt(kB T / m)
    kB = 1.380649e-23
    sigma_v = np.sqrt(kB * temperature / mass)
    vel = np.random.randn(nd, np_particles).astype(np.float64) * sigma_v
    acc = np.zeros((nd, np_particles), dtype=np.float64)
    return pos, vel, acc


def velocity_verlet_step(pos, vel, acc, force, mass, dt, box):
    """
    Velocity Verlet 时间积分步进。
    映射自 md_fast.m 中的 update 函数。
    
    公式:
        r(t+Δt) = r(t) + v(t) Δt + 0.5 a(t) Δt²
        v(t+Δt) = v(t) + 0.5 (a(t) + a(t+Δt)) Δt
        a(t+Δt) = f(t+Δt) / m
    
    同时施加周期性边界条件。
    """
    rmass = 1.0 / mass
    pos_new = pos + vel * dt + 0.5 * acc * dt * dt
    # 周期性边界
    for d in range(pos_new.shape[0]):
        pos_new[d, :] -= box[d] * np.floor(pos_new[d, :] / box[d])

    acc_new = force * rmass
    vel_new = vel + 0.5 * dt * (acc_new + acc)

    return pos_new, vel_new, acc_new


def simulate_particle_transport(np_particles, nd, box, dt, n_steps, mass=1.0,
                                 temperature=300.0, interaction_type='sinsq'):
    """
    完整颗粒运动模拟。
    
    返回:
        trajectory: 位置历史 (n_steps+1, nd, np)
        energy_history: (n_steps+1, 3) [potential, kinetic, total]
    """
    pos, vel, acc = initialize_particles(np_particles, nd, box, temperature, mass)
    trajectory = np.zeros((n_steps + 1, nd, np_particles), dtype=np.float64)
    energy_history = np.zeros((n_steps + 1, 3), dtype=np.float64)

    force, pot, kin = compute_forces_and_energies(pos, vel, mass, box, interaction_type)
    e0 = pot + kin
    trajectory[0, :, :] = pos
    energy_history[0, :] = [pot, kin, e0]

    for step in range(n_steps):
        pos, vel, acc = velocity_verlet_step(pos, vel, acc, force, mass, dt, box)
        force, pot, kin = compute_forces_and_energies(pos, vel, mass, box, interaction_type)
        trajectory[step + 1, :, :] = pos
        energy_history[step + 1, :] = [pot, kin, pot + kin]

    return trajectory, energy_history


def compute_local_temperature_from_kinetic(vel, mass, kB=1.380649e-23):
    """
    由颗粒动能计算局部温度:
        T = (m <v²>) / (d * kB)
    其中 d 为空间维度。
    """
    nd, np_particles = vel.shape
    v2_mean = np.mean(np.sum(vel ** 2, axis=0))
    T_local = mass * v2_mean / (nd * kB)
    return T_local
