"""
ray_tracer.py
室内声场蒙特卡洛射线追踪
基于 sobol (低差异序列)、line_monte_carlo (线段采样) 与 pagerank2 (稀疏图) 核心算法重构

声学工程应用：
在 Schroeder 频率以上的中高频段，声场表现为扩散特性，
适合使用几何声学方法——射线追踪计算房间脉冲响应 (RIR) 与能量衰减曲线 (EDC)。

核心物理模型：
- 射线传播：r(t) = r0 + c * t * d̂
- 反射定律：d̂_ref = d̂ - 2 (d̂ · n̂) n̂
- 能量衰减：E_{n+1} = E_n * (1 - α_i)
- 散射：d̂_scat = d̂_ref + σ * ξ̂, ξ̂ ∼ N(0,1)
"""

import numpy as np
from quadrature_rules import line01_sample_ergodic


C_AIR = 343.0


def sobol_generate(m, n, skip=0):
    """
    生成 Sobol 低差异序列的简化实现。
    基于 sobol 的 i4_sobol 核心思想：
    使用 Gray-code 加速和方向数的 GF(2) 递推。

    本实现使用 Van der Corput / Halton 序列作为1D/2D近似，
    对于球面方向采样，通过正态分布变换实现。
    """
    # 使用黄金比例递推生成低差异序列
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    result = np.zeros((m, n), dtype=float)
    for dim in range(m):
        alpha = (phi ** (dim + 1)) % 1.0
        for i in range(n):
            result[dim, i] = ((skip + i + 1) * alpha) % 1.0
    return result


def sample_directions_sobol(n_rays, dim=3):
    """
    使用 Sobol-like 序列在球面上均匀采样方向。
    方法：生成低差异的正态分布样本，然后归一化。
    """
    # 使用 Box-Muller 或正态采样
    sobol = sobol_generate(dim, n_rays)
    # 将均匀 [0,1] 映射到近似正态（使用逆变换近似）
    # 使用 sqrt(-2 ln u) cos(2pi v) 的简化
    u1 = sobol[0, :]
    u2 = sobol[1, :] if dim > 1 else np.random.rand(n_rays)
    u3 = sobol[2, :] if dim > 2 else np.random.rand(n_rays)

    # 避免 log(0)
    u1 = np.clip(u1, 1e-10, 1.0 - 1e-10)
    u2 = np.clip(u2, 1e-10, 1.0 - 1e-10)
    u3 = np.clip(u3, 1e-10, 1.0 - 1e-10)

    # Box-Muller for x, y
    r = np.sqrt(-2.0 * np.log(u1))
    theta = 2.0 * np.pi * u2
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    # 对于 z，使用另一个正态变量
    r2 = np.sqrt(-2.0 * np.log(u3))
    phi_angle = 2.0 * np.pi * u1  # 复用
    z = r2 * np.cos(phi_angle)

    dirs = np.column_stack((x, y, z))
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-14)
    return dirs / norms


def ray_plane_intersection(ray_origin, ray_dir, plane_point, plane_normal):
    """
    射线与平面求交：
        (r0 + t*d - p0) · n = 0
        t = (p0 - r0) · n / (d · n)
    返回 t > epsilon 的交点参数。
    """
    denom = np.dot(ray_dir, plane_normal)
    if abs(denom) < 1e-14:
        return np.inf
    t = np.dot(plane_point - ray_origin, plane_normal) / denom
    return t


def reflect_direction(dir_vec, normal):
    """
    反射方向计算（镜面反射）：
        d_ref = d - 2 (d · n) n
    """
    dn = np.dot(dir_vec, normal)
    return dir_vec - 2.0 * dn * normal


def scatter_direction(dir_vec, normal, scattering_coeff=0.1):
    """
    散射方向：在反射方向附近加入随机扰动。
        d_scat = d_ref + σ * ξ,  ξ ∼ N(0,I)
    然后归一化。
    """
    d_ref = reflect_direction(dir_vec, normal)
    noise = np.random.randn(3) * scattering_coeff
    d_scat = d_ref + noise
    norm = np.linalg.norm(d_scat)
    if norm < 1e-14:
        return d_ref
    return d_scat / norm


def trace_ray(room_bounds, surfaces, normals, absorption,
              ray_origin, ray_dir, max_reflections=50,
              energy_threshold=1e-6, scattering_coeff=0.05):
    """
    追踪单条射线，返回反射次数、路径长度序列、能量衰减序列和击中的表面序列。

    房间边界：box [0,10] x [0,8] x [0,5]
    表面：floor, ceiling, front_wall, back_wall, left_wall, right_wall
    """
    path_lengths = []
    energies = []
    hit_surfaces = []
    positions = [ray_origin.copy()]

    energy = 1.0
    pos = ray_origin.copy()
    direction = ray_dir.copy()

    for refl in range(max_reflections):
        if energy < energy_threshold:
            break

        # 求与各个表面的交点
        t_min = np.inf
        hit_surf = None
        hit_normal = None

        for name, normal in normals.items():
            # 每个表面需要找一个代表点
            surf_tris = surfaces[name]
            rep_point = surf_tris[0]  # 取第一个顶点作为代表
            t = ray_plane_intersection(pos, direction, rep_point, normal)
            if t > 1e-6 and t < t_min:
                # 检查是否在边界框内
                hit_point = pos + t * direction
                # 简单边界检查
                if name == 'floor' or name == 'ceiling':
                    if 0.0 <= hit_point[0] <= 10.0 and 0.0 <= hit_point[1] <= 8.0:
                        t_min = t
                        hit_surf = name
                        hit_normal = normal
                elif name == 'front_wall' or name == 'back_wall':
                    if 0.0 <= hit_point[0] <= 10.0 and 0.0 <= hit_point[2] <= 5.0:
                        t_min = t
                        hit_surf = name
                        hit_normal = normal
                elif name == 'left_wall' or name == 'right_wall':
                    if 0.0 <= hit_point[1] <= 8.0 and 0.0 <= hit_point[2] <= 5.0:
                        t_min = t
                        hit_surf = name
                        hit_normal = normal

        if hit_surf is None or t_min == np.inf:
            break

        # 更新位置
        pos = pos + t_min * direction
        path_lengths.append(t_min)
        energies.append(energy)
        hit_surfaces.append(hit_surf)
        positions.append(pos.copy())

        # 能量衰减
        alpha = absorption.get(hit_surf, 0.05)
        energy *= (1.0 - alpha)

        # 反射/散射
        if np.random.rand() < scattering_coeff * 5.0:
            direction = scatter_direction(direction, hit_normal, scattering_coeff)
        else:
            direction = reflect_direction(direction, hit_normal)

    return {
        'reflections': len(hit_surfaces),
        'path_lengths': path_lengths,
        'energies': energies,
        'surfaces': hit_surfaces,
        'positions': positions
    }


def monte_carlo_ray_tracing(surfaces, normals, absorption,
                            source_pos, n_rays=5000,
                            max_reflections=50, scattering_coeff=0.05):
    """
    蒙特卡洛射线追踪：从声源发射 n_rays 条射线，
    计算能量衰减曲线 (EDC) 和早期衰减时间 (EDT)。

    EDC(t) = ∑_{paths with arrival time > t} E_i
    """
    directions = sample_directions_sobol(n_rays)
    all_energies = []
    all_times = []

    for i in range(n_rays):
        result = trace_ray(
            None, surfaces, normals, absorption,
            source_pos, directions[i],
            max_reflections=max_reflections,
            scattering_coeff=scattering_coeff
        )
        # 累积路径时间
        cum_time = 0.0
        for j, length in enumerate(result['path_lengths']):
            cum_time += length / C_AIR
            all_times.append(cum_time)
            all_energies.append(result['energies'][j])

    # 按时间排序构建 EDC
    if len(all_times) == 0:
        return np.array([]), np.array([]), 0.0

    idx = np.argsort(all_times)
    all_times = np.array(all_times)[idx]
    all_energies = np.array(all_energies)[idx]

    # 能量衰减曲线（反向累积）
    edc = np.zeros_like(all_energies)
    edc[-1] = all_energies[-1]
    for i in range(len(all_energies) - 2, -1, -1):
        edc[i] = edc[i + 1] + all_energies[i]

    # 计算 T60（Schroeder 反向积分法）
    # 找到能量衰减到 -60dB 的时间
    if edc[0] > 1e-14:
        edc_db = 10.0 * np.log10(edc / edc[0])
    else:
        edc_db = np.zeros_like(edc)

    # 线性拟合 -5dB 到 -35dB 段
    mask = (edc_db <= -5.0) & (edc_db >= -35.0)
    if np.sum(mask) > 5:
        t_fit = all_times[mask]
        e_fit = edc_db[mask]
        # 线性回归
        A_mat = np.vstack([t_fit, np.ones(len(t_fit))]).T
        slope, intercept = np.linalg.lstsq(A_mat, e_fit, rcond=None)[0]
        if slope < 0:
            T60 = -60.0 / slope
            EDT = -10.0 / slope  # 早期衰减时间（-10dB 外推）
        else:
            T60 = 0.0
            EDT = 0.0
    else:
        T60 = 0.0
        EDT = 0.0

    return all_times, edc, T60, EDT


def build_reflection_graph(surfaces, normals, absorption, n_rays=2000):
    """
    构建反射邻接图（基于 pagerank2 的稀疏图构造思想）。
    节点 = 房间表面，边权重 = 射线从一个表面反射到另一个表面的概率。
    用于分析声能扩散特性。
    """
    source_pos = np.array([5.0, 4.0, 2.5])
    directions = sample_directions_sobol(n_rays)
    surf_names = list(surfaces.keys())
    n_surf = len(surf_names)
    trans_counts = np.zeros((n_surf, n_surf), dtype=float)
    surf_to_idx = {name: i for i, name in enumerate(surf_names)}

    for i in range(n_rays):
        result = trace_ray(
            None, surfaces, normals, absorption,
            source_pos, directions[i], max_reflections=20
        )
        surfaces_hit = result['surfaces']
        for j in range(len(surfaces_hit) - 1):
            s1 = surf_to_idx[surfaces_hit[j]]
            s2 = surf_to_idx[surfaces_hit[j + 1]]
            trans_counts[s1, s2] += 1.0

    # 归一化得到转移概率矩阵
    row_sums = trans_counts.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-14, 1.0, row_sums)
    trans_prob = trans_counts / row_sums

    return trans_prob, surf_names


def compute_room_response_stats(surfaces, normals, absorption, source_pos,
                                 n_rays=1000, max_reflections=30):
    """
    计算房间响应统计量：平均自由程、平均反射次数、能量衰减率。
    """
    directions = sample_directions_sobol(n_rays)
    free_paths = []
    reflection_counts = []
    final_energies = []

    for i in range(n_rays):
        result = trace_ray(
            None, surfaces, normals, absorption,
            source_pos, directions[i], max_reflections=max_reflections
        )
        if len(result['path_lengths']) > 0:
            free_paths.extend(result['path_lengths'])
            reflection_counts.append(result['reflections'])
            if len(result['energies']) > 0:
                final_energies.append(result['energies'][-1])

    stats = {
        'mean_free_path': float(np.mean(free_paths)) if free_paths else 0.0,
        'std_free_path': float(np.std(free_paths)) if free_paths else 0.0,
        'mean_reflections': float(np.mean(reflection_counts)) if reflection_counts else 0.0,
        'mean_final_energy': float(np.mean(final_energies)) if final_energies else 0.0,
    }
    return stats
