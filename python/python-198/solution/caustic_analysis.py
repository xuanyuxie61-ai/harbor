"""
caustic_analysis.py
===================
焦散几何与特征线拓扑分析模块（融合 140_caustic）

功能：
- 圆内焦散曲线的参数化生成
- 将焦散几何用于Burgers方程的特征线分析
- 特征速度场的拓扑奇点检测（激波形成位置预测）

数学公式：
- 焦散参数方程: z_j = exp(2πi * j/n), j=0..n
  连线: z(j) → z(mod(j*m, n))
- Burgers方程的特征线: dx/dt = u(x,t)
  激波形成条件: ∂u/∂x → -∞
  破裂时间: t_b = -1 / min(∂u₀/∂x)
- 模运算映射 f(j) = (m*j) mod n 定义了圆上的多值覆盖
"""

import numpy as np


def caustic_mapping(n, m):
    """
    生成圆上焦散映射的边列表。
    对 j=0..n，连接 j → (m*j) mod n。
    
    返回:
        edges: (n+1, 2) 起点-终点索引对
        points: (n+1, 2) 圆上对应点的笛卡尔坐标
    """
    j = np.arange(n + 1)
    k = np.mod(m * j, n)
    
    theta_j = 2.0 * np.pi * j / n
    theta_k = 2.0 * np.pi * k / n
    
    points_j = np.column_stack((np.cos(theta_j), np.sin(theta_j)))
    points_k = np.column_stack((np.cos(theta_k), np.sin(theta_k)))
    
    edges = np.column_stack((j, k))
    return edges, points_j, points_k


def characteristic_burgers_1d(x0, u0, t):
    """
    无粘Burgers方程的特征线。
    x(t) = x0 + u0(x0) * t
    
    参数:
        x0: 初始位置
        u0: 初始速度
        t: 时间
    
    返回:
        x_t: 特征线位置
    """
    return x0 + u0 * t


def shock_formation_time(u0_func, x_grid):
    """
    估计Burgers方程的激波形成时间。
    t_b = -1 / min(du0/dx)
    
    参数:
        u0_func: 可调用函数，返回速度
        x_grid: 空间网格
    
    返回:
        t_b: 破裂时间（若为inf则表示无激波）
        x_shock: 激波初始位置
    """
    u0 = u0_func(x_grid)
    du = np.gradient(u0, x_grid)
    min_du = np.min(du)
    idx = np.argmin(du)
    
    if min_du >= 0:
        return np.inf, x_grid[idx]
    
    t_b = -1.0 / min_du
    return t_b, x_grid[idx]


def caustic_inspired_topology_field(nodes, elements, m=5, n=20):
    """
    基于焦散映射的思想，在三角网格上构造一个拓扑速度场。
    速度在圆盘边界上按照焦刻映射的模运算模式变化。
    
    返回:
        velocity: (n_elem,) 每个单元的特征速度
    """
    # 计算重心
    p1 = nodes[elements[:, 0]]
    p2 = nodes[elements[:, 1]]
    p3 = nodes[elements[:, 2]]
    centroid = (p1 + p2 + p3) / 3.0
    
    r = np.sqrt(centroid[:, 0] ** 2 + centroid[:, 1] ** 2)
    theta = np.arctan2(centroid[:, 1], centroid[:, 0])
    
    # 焦刻启发：将角度映射通过模运算扭曲
    # theta -> theta * m mod 2π，平滑版本
    theta_mod = np.mod(theta * m, 2.0 * np.pi)
    
    # 构造速度场：径向速度随焦刻模式振荡
    velocity = np.sin(n * theta_mod) * np.exp(-2.0 * r ** 2)
    
    return velocity


def detect_gradient_catastrophe(x_history, u_history, t_array):
    """
    检测数值解中的梯度灾变（激波形成）。
    使用有限差分避免np.gradient的零间距问题。
    """
    grad_max_history = []
    for i, t in enumerate(t_array):
        u = u_history[i]
        if len(u) < 2:
            grad_max_history.append(0.0)
            continue
        # 使用间距有限差分，避免零间距导致除零
        x_sorted = np.sort(x_history)
        dx_min = np.min(np.diff(x_sorted))
        if dx_min < 1e-12:
            dx_min = 1e-12
        grad = np.gradient(u, dx_min)
        grad_max_history.append(np.max(np.abs(grad)))
    
    grad_max_history = np.array(grad_max_history)
    threshold = 10.0 * grad_max_history[0] if grad_max_history[0] > 0 else 100.0
    
    catastrophic = np.where(grad_max_history > threshold)[0]
    if len(catastrophic) > 0:
        return t_array[catastrophic[0]], grad_max_history
    return None, grad_max_history
