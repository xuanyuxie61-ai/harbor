"""
粒子动力学模块
整合自：745_md_fast（快速分子动力学模拟）

在吸积盘模拟中用于：
  1. 吸积盘微观粒子碰撞模型（尘埃/离子碰撞）
  2. 速度 Verlet 积分器（辛积分，能量守恒好）
  3. 计算粒子系统的动能、势能和总能量
"""
import numpy as np


def sinsq_potential(r, cutoff=np.pi / 2):
    """
    正弦平方势（truncated）：
        V(r) = sin^2(min(r, cutoff))

    力：F = -dV/dr = -sin(2*r)  (r < cutoff)
    """
    r = np.asarray(r, dtype=np.float64)
    r_eff = np.minimum(r, cutoff)
    return np.sin(r_eff) ** 2


def sinsq_force(r_vec, cutoff=np.pi / 2):
    """
    计算正弦平方势对应的力。

    对于粒子对 (i, j)：
        F_ij = -dV/dr * (r_j - r_i) / |r_j - r_i|
             = sin(2*r_eff) / r * r_vec

    参数:
        r_vec: 相对位移向量
        cutoff: 截断距离

    返回:
        force: 力向量
    """
    r = np.linalg.norm(r_vec)
    if r < 1e-15:
        return np.zeros_like(r_vec)

    r_eff = min(r, cutoff)
    # dV/dr = 2*sin(r)*cos(r) = sin(2r)
    # F = -dV/dr * r_vec / r
    force_magnitude = np.sin(2.0 * r_eff)
    return force_magnitude * r_vec / r


def lennard_jones_potential(r, epsilon=1.0, sigma=1.0):
    """
    Lennard-Jones 势：
        V(r) = 4*epsilon * [(sigma/r)^12 - (sigma/r)^6]

    在吸积盘微观粒子模型中，可表示尘埃粒子间的范德华相互作用。
    """
    r = np.asarray(r, dtype=np.float64)
    sr = sigma / r
    sr6 = sr ** 6
    sr12 = sr6 ** 2
    return 4.0 * epsilon * (sr12 - sr6)


def velocity_verlet_step(positions, velocities, forces, dt, mass=1.0,
                          force_func=None, box_size=None):
    """
    单步速度 Verlet 积分。

    算法：
        x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
        v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
        a(t+dt) = F(t+dt)/m

    这是辛积分器，对长时间模拟能量守恒非常好。

    参数:
        positions: (n, dim) 位置
        velocities: (n, dim) 速度
        forces: (n, dim) 当前力
        dt: 时间步长
        mass: 质量
        force_func: 力计算函数 force_func(positions) -> forces
        box_size: 周期性边界盒子大小（可选）

    返回:
        new_positions, new_velocities, new_forces
    """
    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    forces = np.asarray(forces, dtype=np.float64)

    # 更新位置
    new_positions = positions + velocities * dt + 0.5 * forces * dt * dt / mass

    # 周期性边界条件
    if box_size is not None:
        new_positions = new_positions % box_size

    # 计算新力
    if force_func is not None:
        new_forces = force_func(new_positions)
    else:
        new_forces = np.zeros_like(forces)

    # 更新速度
    new_velocities = velocities + 0.5 * (forces + new_forces) * dt / mass

    return new_positions, new_velocities, new_forces


def compute_pair_forces(positions, potential='sinsq', cutoff=None, box_size=None):
    """
    计算所有粒子对相互作用力（O(N^2) 实现，适用于小系统）。

    参数:
        positions: (n, dim) 位置
        potential: 'sinsq' 或 'lj'
        cutoff: 截断距离
        box_size: 周期性边界

    返回:
        forces: (n, dim) 力
        potential_energy: 总势能
    """
    positions = np.asarray(positions, dtype=np.float64)
    n, dim = positions.shape

    forces = np.zeros_like(positions)
    pe = 0.0

    if cutoff is None:
        cutoff = np.pi / 2 if potential == 'sinsq' else 2.5

    for i in range(n):
        for j in range(i + 1, n):
            r_vec = positions[j] - positions[i]

            # 最小镜像约定
            if box_size is not None:
                r_vec = r_vec - box_size * np.rint(r_vec / box_size)

            r = np.linalg.norm(r_vec)
            if r < 1e-15 or r > cutoff:
                continue

            if potential == 'sinsq':
                r_eff = min(r, cutoff)
                f_mag = np.sin(2.0 * r_eff)
                f_vec = f_mag * r_vec / r
                pe += np.sin(r_eff) ** 2
            elif potential == 'lj':
                sr = 1.0 / r
                sr6 = sr ** 6
                sr12 = sr6 ** 2
                f_mag = 24.0 * (2.0 * sr12 - sr6) / r
                f_vec = f_mag * r_vec / r
                pe += 4.0 * (sr12 - sr6)
            else:
                continue

            forces[i] += f_vec
            forces[j] -= f_vec

    return forces, pe


def compute_kinetic_energy(velocities, mass=1.0):
    """
    动能：KE = 0.5 * m * sum(v^2)
    """
    velocities = np.asarray(velocities, dtype=np.float64)
    return 0.5 * mass * np.sum(velocities ** 2)


def run_particle_simulation(n_particles, dim, n_steps, dt, box_size=10.0,
                             mass=1.0, potential='sinsq', seed=None):
    """
    运行简化的粒子动力学模拟。

    参数:
        n_particles: 粒子数
        dim: 维度
        n_steps: 步数
        dt: 时间步长
        box_size: 盒子大小
        mass: 质量
        potential: 势函数类型
        seed: 随机种子

    返回:
        history: dict 包含位置和能量历史
    """
    if seed is not None:
        np.random.seed(seed)

    # 初始化
    positions = np.random.rand(n_particles, dim) * box_size
    velocities = np.random.randn(n_particles, dim)
    # 减去质心速度
    velocities -= np.mean(velocities, axis=0)

    # 初始力
    forces, pe = compute_pair_forces(positions, potential=potential,
                                      cutoff=np.pi / 2 if potential == 'sinsq' else 2.5,
                                      box_size=box_size)

    ke_history = []
    pe_history = []
    pos_history = []

    for step in range(n_steps):
        ke = compute_kinetic_energy(velocities, mass)
        ke_history.append(ke)
        pe_history.append(pe)
        pos_history.append(positions.copy())

        positions, velocities, forces = velocity_verlet_step(
            positions, velocities, forces, dt, mass=mass,
            force_func=lambda p: compute_pair_forces(p, potential=potential,
                                                      cutoff=np.pi / 2 if potential == 'sinsq' else 2.5,
                                                      box_size=box_size)[0],
            box_size=box_size
        )

        # 更新势能（用于下一步记录）
        _, pe = compute_pair_forces(positions, potential=potential,
                                     cutoff=np.pi / 2 if potential == 'sinsq' else 2.5,
                                     box_size=box_size)

    return {
        'positions': pos_history,
        'kinetic_energy': np.array(ke_history),
        'potential_energy': np.array(pe_history),
        'total_energy': np.array(ke_history) + np.array(pe_history)
    }


def accretion_disk_particle_model(n_dust, r_in, r_out, z_scale,
                                   n_steps, dt, seed=None):
    """
    吸积盘尘埃粒子碰撞模型。

    在环形区域内初始化尘埃粒子，模拟碰撞和凝聚过程。

    参数:
        n_dust: 尘埃粒子数
        r_in, r_out: 径向范围
        z_scale: 垂直尺度
        n_steps: 步数
        dt: 时间步长
        seed: 随机种子

    返回:
        history: 模拟历史
    """
    if seed is not None:
        np.random.seed(seed)

    dim = 3
    # 在环形盘内初始化
    phi = np.random.uniform(0, 2 * np.pi, n_dust)
    r = np.random.uniform(r_in, r_out, n_dust)
    z = np.random.normal(0, z_scale, n_dust)

    positions = np.zeros((n_dust, dim))
    positions[:, 0] = r * np.cos(phi)
    positions[:, 1] = r * np.sin(phi)
    positions[:, 2] = z

    # 开普勒速度
    G = 1.0
    M_bh = 1.0
    v_kep = np.sqrt(G * M_bh / r)
    velocities = np.zeros((n_dust, dim))
    velocities[:, 0] = -v_kep * np.sin(phi)
    velocities[:, 1] = v_kep * np.cos(phi)

    # 添加随机热运动
    velocities += 0.01 * np.random.randn(n_dust, dim)

    # 简化的力函数（只包含中心引力和近距离排斥）
    def disk_force_func(pos):
        n = pos.shape[0]
        forces = np.zeros_like(pos)

        # 中心引力
        dists = np.linalg.norm(pos, axis=1)
        dists = np.where(dists < 0.01, 0.01, dists)
        forces = -G * M_bh * pos / (dists.reshape(-1, 1) ** 3)

        # 近距离排斥（简化碰撞）
        cutoff = 0.05
        for i in range(n):
            for j in range(i + 1, n):
                r_vec = pos[j] - pos[i]
                r = np.linalg.norm(r_vec)
                if r < cutoff and r > 1e-15:
                    f_rep = -0.01 * (1.0 / r - 1.0 / cutoff) * r_vec / r
                    forces[i] += f_rep
                    forces[j] -= f_rep

        return forces

    forces = disk_force_func(positions)

    pos_history = []
    for step in range(n_steps):
        pos_history.append(positions.copy())
        positions, velocities, forces = velocity_verlet_step(
            positions, velocities, forces, dt,
            force_func=disk_force_func
        )

    return {
        'positions': pos_history,
        'final_positions': positions,
        'final_velocities': velocities
    }
