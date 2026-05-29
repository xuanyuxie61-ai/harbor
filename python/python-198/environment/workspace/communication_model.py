"""
communication_model.py
======================
通信模型与延迟分析模块（融合 1365_tsp_greedy + 292_disk_distance）

功能：
- 建模分布式内存系统中的通信延迟
- 使用TSP贪心算法优化处理器间的通信路径
- 基于圆盘距离统计建模网络延迟分布
- 量化通信避免算法带来的加速比

数学公式：
- 通信延迟模型: T_comm = α + β * m
  α: 启动延迟(latency), β: 每字节传输时间, m: 消息大小
- 通信避免s-step方法的理论加速比:
  S_CA = (s * T_comp + T_comm) / (s * T_comp + T_comm/s)
       ≈ s  (当通信主导时)
- 圆盘上两点距离期望值: E[D] = 128 / (45π) ≈ 0.9054
"""

import numpy as np


def compute_distance_matrix(points):
    """
    计算欧氏距离矩阵，融合 1365_tsp_greedy 的核心数据结构。
    """
    n = points.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            D[i, j] = d
            D[j, i] = d
    return D


def tsp_greedy_path(n, distance, start=0):
    """
    贪心算法构造TSP路径（单起点）。
    从start出发，每次选择最近的未访问节点。
    返回长度为n的路径（节点索引列表）。
    """
    visited = np.zeros(n, dtype=bool)
    path = []
    current = start
    
    for _ in range(n):
        path.append(current)
        visited[current] = True
        # 找最近的未访问节点
        min_dist = np.inf
        next_node = current
        for j in range(n):
            if not visited[j] and distance[current, j] < min_dist:
                min_dist = distance[current, j]
                next_node = j
        current = next_node
    
    return path


def tsp_multi_start(n, distance):
    """
    多起点贪心TSP，选择总距离最短的路径。
    返回最优路径和总距离。
    """
    best_cost = np.inf
    best_path = list(range(n))
    
    for start in range(n):
        path = tsp_greedy_path(n, distance, start)
        cost = sum(distance[path[i], path[(i + 1) % n]] for i in range(n))
        if cost < best_cost:
            best_cost = cost
            best_path = path
    
    return best_path, best_cost


def model_communication_latency(msg_size, alpha=1e-4, beta=1e-8, noise_level=0.05):
    """
    基于圆盘距离统计的通信延迟模型。
    
    T = α + β * m + ε
    其中 ε ~ TruncatedNormal(0, σ_noise, -3σ, 3σ) 模拟网络抖动。
    参考 292_disk_distance 的单位圆盘距离统计思想。
    """
    base = alpha + beta * msg_size
    noise = np.random.normal(0, noise_level * base)
    noise = np.clip(noise, -3 * noise_level * base, 3 * noise_level * base)
    return base + noise


def estimate_disk_distance_mean(n_samples=10000):
    """
    蒙特卡洛估计单位圆盘上两点距离的均值。
    理论值: 128 / (45π) ≈ 0.9054
    融合 292_disk_distance 的核心思想。
    """
    # 在单位圆盘上均匀采样
    theta1 = np.random.uniform(0, 2 * np.pi, n_samples)
    r1 = np.sqrt(np.random.uniform(0, 1, n_samples))
    theta2 = np.random.uniform(0, 2 * np.pi, n_samples)
    r2 = np.sqrt(np.random.uniform(0, 1, n_samples))
    
    p1 = np.column_stack((r1 * np.cos(theta1), r1 * np.sin(theta1)))
    p2 = np.column_stack((r2 * np.cos(theta2), r2 * np.sin(theta2)))
    
    distances = np.linalg.norm(p1 - p2, axis=1)
    return float(np.mean(distances)), float(np.var(distances))


def ca_speedup_theory(s, t_comp_per_step, t_comm_per_step):
    """
    通信避免s-step算法的理论加速比。
    
    标准方法每步通信: T_std = T_comp + T_comm
    CA方法每s步通信一次: T_ca = s*T_comp + T_comm/s （聚合通信）
    每步等价时间: T_ca_step = T_comp + T_comm/s²
    
    加速比: S = (T_comp + T_comm) / (T_comp + T_comm/s²)
    
    参数:
        s: 聚合步数
        t_comp_per_step: 每步计算时间
        t_comm_per_step: 每步通信时间
    """
    if s < 1:
        return 1.0
    t_std = t_comp_per_step + t_comm_per_step
    t_ca = t_comp_per_step + t_comm_per_step / (s ** 2)
    if t_ca < 1e-15:
        return 1.0
    return t_std / t_ca


def optimize_s_parameter(t_comp, t_comm, s_max=20):
    """
    选择最优的s参数以最大化加速比。
    """
    best_s = 1
    best_speedup = 1.0
    for s in range(1, s_max + 1):
        sp = ca_speedup_theory(s, t_comp, t_comm)
        if sp > best_speedup:
            best_speedup = sp
            best_s = s
    return best_s, best_speedup


def processor_communication_schedule(n_procs, topology='ring'):
    """
    生成处理器间的通信调度表。
    使用TSP路径优化通信顺序。
    
    返回:
        schedule: 列表，每个元素为 (src, dst, msg_size) 的通信对
    """
    if topology == 'ring':
        schedule = []
        for i in range(n_procs):
            src = i
            dst = (i + 1) % n_procs
            schedule.append((src, dst, 1.0))
        return schedule
    elif topology == 'tsp_optimized':
        # 将处理器放在圆上，用TSP优化路径
        angles = np.linspace(0, 2 * np.pi, n_procs, endpoint=False)
        points = np.column_stack((np.cos(angles), np.sin(angles)))
        D = compute_distance_matrix(points)
        path, _ = tsp_multi_start(n_procs, D)
        schedule = []
        for i in range(n_procs):
            src = path[i]
            dst = path[(i + 1) % n_procs]
            schedule.append((src, dst, 1.0))
        return schedule
    else:
        # 默认全连接
        schedule = []
        for i in range(n_procs):
            for j in range(n_procs):
                if i != j:
                    schedule.append((i, j, 1.0))
        return schedule
