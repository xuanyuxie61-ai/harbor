"""
optimal_path.py
海洋内波能量传播的最优路径追踪

融合项目:
- 287_dijkstra: Dijkstra最短路径算法
- 696_locker_simulation: 置换循环结构 → 内波波包传播路径中的循环

核心科学:
内波能量在密度分层海洋中的传播遵循群速度方向。
利用Dijkstra算法可以在离散化的海洋分层网络中寻找
从源点到目标点的最优能量传播路径。

数学模型:
1. 能量传播图:
    G = (V, E, w)
    V: 海洋分层中的离散点 (深度, 水平位置)
    E: 可能的能量传播方向
    w: 传播代价 (传播时间的倒数)

2. 最优路径:
    min_γ ∫_γ ds / |c_g|

3. 置换循环结构:
    内波模态间的能量交换可建模为置换循环，
    每个循环对应一种能量传播模式。
"""

import numpy as np


def dijkstra_shortest_path(graph, source):
    """
    Dijkstra最短路径算法
    
    参数:
        graph: 邻接矩阵 (n x n)，graph[i,j] 为边权重
        source: 源节点索引
    
    返回:
        distances: 到各节点的最短距离
        previous: 前驱节点数组
    """
    n = len(graph)
    visited = np.zeros(n, dtype=bool)
    distances = np.full(n, np.inf)
    previous = np.full(n, -1, dtype=int)
    
    distances[source] = 0.0
    
    for _ in range(n):
        # 找到未访问节点中距离最小的
        min_dist = np.inf
        u = -1
        for i in range(n):
            if not visited[i] and distances[i] < min_dist:
                min_dist = distances[i]
                u = i
        
        if u == -1:
            break
        
        visited[u] = True
        
        # 松弛操作
        for v in range(n):
            if not visited[v] and graph[u, v] > 0:
                alt = distances[u] + graph[u, v]
                if alt < distances[v]:
                    distances[v] = alt
                    previous[v] = u
    
    return distances, previous


def reconstruct_path(previous, target):
    """
    从Dijkstra结果重建最短路径
    
    参数:
        previous: 前驱节点数组
        target: 目标节点索引
    
    返回:
        path: 节点索引列表
    """
    path = []
    u = target
    
    while u != -1:
        path.append(u)
        u = previous[u]
    
    path.reverse()
    return path


def build_energy_propagation_graph(depths, horizontal_positions,
                                   N_profile, f=1.0e-4):
    """
    构建内波能量传播图
    
    节点: (z_i, x_j) 离散化的海洋分层点
    边权重: 传播时间 dt = dx / |c_g|
    
    参数:
        depths: 深度数组 [m]
        horizontal_positions: 水平位置数组 [m]
        N_profile: 浮力频率剖面 [rad/s]
        f: 科里奥利参数 [rad/s]
    
    返回:
        graph: 邻接矩阵
        node_coords: 节点坐标列表 [(z, x)]
    """
    n_z = len(depths)
    n_x = len(horizontal_positions)
    n_nodes = n_z * n_x
    
    # 节点坐标
    node_coords = []
    for i in range(n_z):
        for j in range(n_x):
            node_coords.append((depths[i], horizontal_positions[j]))
    
    graph = np.full((n_nodes, n_nodes), np.inf)
    np.fill_diagonal(graph, 0.0)
    
    # 构建边 (连接相邻节点)
    dz = np.abs(depths[1] - depths[0]) if n_z > 1 else 1.0
    dx = np.abs(horizontal_positions[1] - horizontal_positions[0]) if n_x > 1 else 1.0
    
    for i in range(n_z):
        for j in range(n_x):
            node_idx = i * n_x + j
            N = N_profile[i] if i < len(N_profile) else 0.01
            
            # 水平传播 (向右)
            if j + 1 < n_x:
                neighbor_idx = i * n_x + (j + 1)
                # 群速度近似
                kh = 2.0 * np.pi / 1000.0
                m = np.pi / max(np.abs(depths[i]), 1.0)
                denom = kh**2 + m**2
                cgx = kh * (N**2 - f**2) * m**2 / (denom**2 + 1.0e-12)
                cgx = max(abs(cgx), 0.01)
                weight = dx / cgx
                graph[node_idx, neighbor_idx] = weight
            
            # 垂向传播 (向下)
            if i + 1 < n_z:
                neighbor_idx = (i + 1) * n_x + j
                kh = 2.0 * np.pi / 1000.0
                m = np.pi / max(np.abs(depths[i]), 1.0)
                denom = kh**2 + m**2
                cgz = m * (N**2 - f**2) * kh**2 / (denom**2 + 1.0e-12)
                cgz = max(abs(cgz), 0.001)
                weight = dz / cgz
                graph[node_idx, neighbor_idx] = weight
            
            # 对角传播
            if i + 1 < n_z and j + 1 < n_x:
                neighbor_idx = (i + 1) * n_x + (j + 1)
                weight = np.sqrt(dx**2 + dz**2) / max(cgx, cgz)
                graph[node_idx, neighbor_idx] = weight
    
    # 对称化
    graph = np.minimum(graph, graph.T)
    
    return graph, node_coords


def permutation_cycle_analysis(n_lockers=100, n_tries=50):
    """
    置换循环分析用于内波模态能量交换
    
    受locker_simulation启发，将内波模态间的能量交换建模为置换循环:
    - 每个"locker"对应一个内波模态
    - "follow the chain"对应能量沿模态链传递
    
    参数:
        n_lockers: 模态数量
        n_tries: 最大追踪步数
    
    返回:
        cycles: 循环结构列表
        cycle_lengths: 循环长度数组
        success_rate: 能量成功传递率
    """
    # 随机置换
    permutation = np.random.permutation(n_lockers)
    
    # 循环分解
    visited = np.zeros(n_lockers, dtype=bool)
    cycles = []
    cycle_lengths = []
    
    for start in range(n_lockers):
        if visited[start]:
            continue
        
        cycle = []
        current = start
        
        while not visited[current]:
            visited[current] = True
            cycle.append(current)
            current = permutation[current]
        
        if len(cycle) > 0:
            cycles.append(cycle)
            cycle_lengths.append(len(cycle))
    
    # 能量传递成功率 (循环长度 <= n_tries)
    n_success = sum(1 for cl in cycle_lengths if cl <= n_tries)
    success_rate = n_success / len(cycles) if len(cycles) > 0 else 0.0
    
    return cycles, np.array(cycle_lengths), success_rate


def ray_tracing_cycle(wave_frequency, N_profile, z, x0=0.0,
                       theta0=np.pi/4, max_steps=500):
    """
    内波射线追踪 (考虑群速度和循环结构)
    
    射线方程:
        dx/dt = c_gx
        dz/dt = c_gz
        dθ/dt = -∂ω/∂z · (∂ω/∂k)^{-1}
    
    参数:
        wave_frequency: 波频率 [rad/s]
        N_profile: 浮力频率剖面 [rad/s]
        z: 深度坐标 [m]
        x0: 初始水平位置 [m]
        theta0: 初始传播角度 [rad]
        max_steps: 最大步数
    
    返回:
        x_path: 水平路径
        z_path: 垂向路径
        theta_path: 角度路径
    """
    dz = np.abs(z[1] - z[0]) if len(z) > 1 else 1.0
    dt = dz / 0.5  # 时间步
    
    x_path = np.zeros(max_steps)
    z_path = np.zeros(max_steps)
    theta_path = np.zeros(max_steps)
    
    x = x0
    z_current = np.mean(z)
    theta = theta0
    
    for step in range(max_steps):
        x_path[step] = x
        z_path[step] = z_current
        theta_path[step] = theta
        
        # 插值浮力频率
        N = np.interp(z_current, z, N_profile)
        N = max(N, 1.0e-6)
        
        # 群速度 (基于角度)
        c_g = 0.5 * N * np.cos(theta) * np.sin(theta)
        c_gx = c_g * np.cos(theta)
        c_gz = c_g * np.sin(theta)
        
        # 更新位置
        x += c_gx * dt
        z_current += c_gz * dt
        
        # 反射边界
        if z_current > np.max(z):
            z_current = 2 * np.max(z) - z_current
            theta = -theta
        elif z_current < np.min(z):
            z_current = 2 * np.min(z) - z_current
            theta = -theta
        
        # 角度演化
        dN_dz = 0.0
        if step > 0:
            idx = np.argmin(np.abs(z - z_current))
            idx = max(1, min(idx, len(N_profile) - 2))
            dN_dz = (N_profile[idx+1] - N_profile[idx-1]) / (z[idx+1] - z[idx-1])
        
        theta += dN_dz * np.sin(theta) * dt
        
        # 边界处理
        theta = np.clip(theta, -np.pi/2 + 0.01, np.pi/2 - 0.01)
    
    return x_path, z_path, theta_path
