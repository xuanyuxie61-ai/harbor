"""
sensor_network.py
================================================================================
海洋监测传感器网络优化：CVT 最优布点 + TSP 采样路径规划

融合项目：
    - 255_cvt_corn        : 圆盘上质心 Voronoi 镶嵌
    - 1366_tsp_moler      : 旅行商问题随机启发式

核心科学问题：
    (1) 在二维海洋 basin 上，使用 CVT（Centroidal Voronoi Tessellation）
        将 N 个监测传感器最优分布，使得每个传感器覆盖的区域面积
        与其重要性权重（如碳通量梯度）成正比。
    (2) 给定一组需要采样的站点坐标，使用 TSP 启发式规划最短巡航路径。

科学背景：
    CVT 定义：给定区域 Ω 和密度函数 ρ(x)，CVT 将 Ω 划分为 N 个 Voronoi
    单元 V_i，使得生成元 g_i 恰好是 V_i 的质心：
        g_i = (∫_{V_i} x·ρ(x) dx) / (∫_{V_i} ρ(x) dx)
    
    Lloyd 算法迭代：
        1. 在 Ω 中随机采样大量点
        2. 对每个采样点，找到最近的生成元
        3. 将每个生成元更新为其 Voronoi 单元的质心
        4. 重复直到收敛
    
    TSP：给定 N 个站点的距离矩阵 D，寻找最短的哈密顿回路：
        min_{π} Σ_{i=1}^{N} D(π_i, π_{i+1})
        其中 π_{N+1} = π_1
    
    Moler 启发式：从随机回路出发，进行 2-opt 反转和单点插入的局部搜索。
================================================================================
"""

import numpy as np


# =============================================================================
# CVT 在二维海洋 Basin 上的实现 (来自 cvt_corn)
# =============================================================================

def disk_sample_uniform(n, r=1.0):
    """
    在圆盘内均匀随机采样 n 个点。
    
    使用极坐标变换：
        r = R·√u,  θ = 2π·v
    其中 u,v ~ Uniform(0,1)。
    """
    u = np.random.rand(n)
    v = np.random.rand(n)
    radius = r * np.sqrt(u)
    theta = 2.0 * np.pi * v
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.column_stack([x, y])


def disk_sample_nonuniform(n, r=1.0, density_power=1.0):
    """
    在圆盘内按密度 ρ(r) = r^{density_power} 非均匀采样。
    用于模拟近岸高梯度区域需要更多传感器的情形。
    """
    u = np.random.rand(n)
    v = np.random.rand(n)
    # 累积分布 F(r) = r^{2+density_power}
    radius = r * u**(1.0 / (2.0 + density_power))
    theta = 2.0 * np.pi * v
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.column_stack([x, y])


def find_closest_samples(samples, generators):
    """
    对每个采样点，找到最近的生成元索引。
    
    参数:
        samples    : ndarray, shape (n_s, 2)
        generators : ndarray, shape (n_g, 2)
    
    返回:
        labels : ndarray, shape (n_s,), 最近生成元索引
    """
    n_s = samples.shape[0]
    labels = np.zeros(n_s, dtype=int)
    for i in range(n_s):
        dists = np.sum((generators - samples[i])**2, axis=1)
        labels[i] = np.argmin(dists)
    return labels


def cvt_on_disk(n_generators, r=1.0, n_samples=10000, n_iterations=50,
                boundary_generators=None, density_power=0.0):
    """
    在圆盘上计算 Centroidal Voronoi Tessellation。
    
    参数:
        n_generators   : int, 内部生成元数量
        r              : float, 圆盘半径
        n_samples      : int, 每次迭代的 Monte Carlo 采样数
        n_iterations   : int, Lloyd 迭代次数
        boundary_generators : int or None, 边界上的固定生成元数
        density_power  : float, 采样密度幂次 (0=均匀)
    
    返回:
        generators : ndarray, shape (n_total, 2), 生成元坐标
        types      : ndarray, shape (n_total,), 0=内部, 1=边界
    """
    if boundary_generators is None:
        boundary_generators = max(3, n_generators // 3)
    
    n_total = n_generators + boundary_generators
    
    # 初始化：内部均匀随机，边界均匀分布在圆周上
    generators = np.zeros((n_total, 2))
    types = np.zeros(n_total, dtype=int)
    
    # 内部点
    generators[:n_generators, :] = disk_sample_uniform(n_generators, r)
    
    # 边界点
    angles = np.linspace(0, 2*np.pi, boundary_generators, endpoint=False)
    generators[n_generators:, 0] = r * np.cos(angles)
    generators[n_generators:, 1] = r * np.sin(angles)
    types[n_generators:] = 1
    
    for it in range(n_iterations):
        # Monte Carlo 采样
        if density_power > 0:
            samples = disk_sample_nonuniform(n_samples, r, density_power)
        else:
            samples = disk_sample_uniform(n_samples, r)
        
        labels = find_closest_samples(samples, generators)
        
        # 更新内部生成元到质心
        for g in range(n_generators):
            mask = (labels == g)
            if np.sum(mask) > 0:
                generators[g, :] = np.mean(samples[mask, :], axis=0)
        
        # 边界生成元投影回圆周
        for g in range(n_generators, n_total):
            dx, dy = generators[g, 0], generators[g, 1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 1e-10:
                generators[g, 0] = r * dx / dist
                generators[g, 1] = r * dy / dist
    
    return generators, types


def cvt_on_rectangle(n_generators, Lx, Ly, n_samples=10000, n_iterations=50):
    """
    在矩形区域 [0,Lx]×[0,Ly] 上计算 CVT。
    
    边界处理：在四条边上放置固定生成元。
    """
    n_boundary = max(4, 2 * (n_generators // 5))
    n_total = n_generators + n_boundary
    
    generators = np.zeros((n_total, 2))
    types = np.zeros(n_total, dtype=int)
    
    # 内部随机初始化
    generators[:n_generators, 0] = np.random.rand(n_generators) * Lx
    generators[:n_generators, 1] = np.random.rand(n_generators) * Ly
    
    # 边界：均匀分布在四条边上
    nb_per_side = n_boundary // 4
    idx = n_generators
    for side in range(4):
        for k in range(nb_per_side):
            t = k / max(1, nb_per_side - 1)
            if side == 0:  # 底边
                x, y = t * Lx, 0.0
            elif side == 1:  # 右边
                x, y = Lx, t * Ly
            elif side == 2:  # 顶边
                x, y = (1-t) * Lx, Ly
            else:  # 左边
                x, y = 0.0, (1-t) * Ly
            if idx < n_total:
                generators[idx, :] = [x, y]
                types[idx] = 1
                idx += 1
    
    for it in range(n_iterations):
        samples = np.column_stack([
            np.random.rand(n_samples) * Lx,
            np.random.rand(n_samples) * Ly
        ])
        
        labels = find_closest_samples(samples, generators)
        
        for g in range(n_generators):
            mask = (labels == g)
            if np.sum(mask) > 0:
                generators[g, :] = np.mean(samples[mask, :], axis=0)
        
        # 边界投影
        for g in range(n_generators, n_total):
            x, y = generators[g, :]
            # 投影到最近的边
            d_left = x
            d_right = Lx - x
            d_bottom = y
            d_top = Ly - y
            d_min = min(d_left, d_right, d_bottom, d_top)
            if d_min == d_left:
                generators[g, 0] = 0.0
            elif d_min == d_right:
                generators[g, 0] = Lx
            elif d_min == d_bottom:
                generators[g, 1] = 0.0
            else:
                generators[g, 1] = Ly
    
    return generators, types


# =============================================================================
# TSP 路径规划 (来自 tsp_moler)
# =============================================================================

def compute_distance_matrix(coords):
    """
    计算站点间的欧氏距离矩阵。
    
    参数:
        coords : ndarray, shape (n, 2), 站点坐标
    
    返回:
        D : ndarray, shape (n, n)
    """
    n = coords.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            D[i, j] = d
            D[j, i] = d
    return D


def path_length(p, D):
    """
    计算路径 p 的总长度。
    
    p : ndarray, 路径节点序列（0-based）
    D : ndarray, 距离矩阵
    """
    n = len(p)
    total = 0.0
    for i in range(n - 1):
        total += D[p[i], p[i+1]]
    return total


def solve_tsp_moler(D, n_iterations=10000, seed=None):
    """
    使用 Moler 随机启发式求解 TSP。
    
    算法：
        1. 随机初始化回路
        2. 重复 n_iterations 次：
           a. 2-opt 反转：随机选两点，反转中间子路径，若更短则接受
           b. 单点插入：随机移除一点并插入新位置，若更短则接受
        3. 返回闭合回路
    
    参数:
        D             : ndarray, shape (n, n), 距离矩阵
        n_iterations  : int
        seed          : int, 随机种子
    
    返回:
        best_len  : float, 最短回路长度
        best_path : ndarray, 最短回路（闭合，即包含首尾相同节点）
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = D.shape[0]
    if n <= 1:
        return 0.0, np.array([0, 0])
    
    # 随机初始回路
    p = np.random.permutation(n)
    best_path = p.copy()
    best_len = path_length(np.append(p, p[0]), D)
    
    for it in range(n_iterations):
        # ---- 2-opt 反转 ----
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if i > j:
            i, j = j, i
        if i == j:
            continue
        
        p_new = p.copy()
        p_new[i:j+1] = p_new[i:j+1][::-1]
        len_new = path_length(np.append(p_new, p_new[0]), D)
        if len_new < best_len:
            best_len = len_new
            best_path = p_new.copy()
            p = p_new.copy()
            continue
        
        # ---- 单点插入 ----
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if i == j:
            continue
        
        p_new = p.copy()
        node = p_new[i]
        p_new = np.delete(p_new, i)
        p_new = np.insert(p_new, j, node)
        len_new = path_length(np.append(p_new, p_new[0]), D)
        if len_new < best_len:
            best_len = len_new
            best_path = p_new.copy()
            p = p_new.copy()
    
    return best_len, np.append(best_path, best_path[0])


def plan_ocean_sampling_route(station_coords, seed=None):
    """
    规划海洋采样站点的最优巡航路径。
    
    参数:
        station_coords : ndarray, shape (n, 2), 站点经纬度或平面坐标 (km)
        seed           : int
    
    返回:
        dict: {'total_distance': 总距离, 'path': 路径序列,
               'distance_matrix': 距离矩阵}
    """
    D = compute_distance_matrix(station_coords)
    best_len, best_path = solve_tsp_moler(D, n_iterations=20000, seed=seed)
    
    return {
        'total_distance': best_len,
        'path': best_path,
        'distance_matrix': D,
    }


# =============================================================================
# 传感器网络综合部署
# =============================================================================

def deploy_sensor_network(domain_type, domain_params, n_sensors, seed=None):
    """
    综合传感器部署：CVT 布点 + TSP 路径规划。
    
    参数:
        domain_type   : str, 'disk' or 'rectangle'
        domain_params : dict, {'r': radius} 或 {'Lx': Lx, 'Ly': Ly}
        n_sensors     : int, 内部传感器数量
        seed          : int
    
    返回:
        dict: 包含传感器坐标、TSP 路径、总巡航距离
    """
    if seed is not None:
        np.random.seed(seed)
    
    if domain_type == 'disk':
        r = domain_params.get('r', 1.0)
        generators, types = cvt_on_disk(n_sensors, r=r, n_samples=15000,
                                         n_iterations=60, density_power=0.5)
    elif domain_type == 'rectangle':
        Lx = domain_params.get('Lx', 1.0)
        Ly = domain_params.get('Ly', 1.0)
        generators, types = cvt_on_rectangle(n_sensors, Lx, Ly,
                                              n_samples=15000, n_iterations=60)
    else:
        raise ValueError(f"不支持的域类型: {domain_type}")
    
    # 仅使用内部传感器进行 TSP 规划
    interior_mask = (types == 0)
    interior_coords = generators[interior_mask]
    
    route = plan_ocean_sampling_route(interior_coords, seed=seed)
    
    return {
        'all_coords': generators,
        'sensor_types': types,
        'interior_coords': interior_coords,
        'route': route,
    }
