"""
unicycle_boundary.py
================================================================================
非完整约束边界执行器模型与离散序列优化模块

本模块融合以下种子项目的核心算法：
  - 1372_unicycle      : 单轮车（unicycle）运动学模型、排列生成与随机化
  - 668_levenshtein_distance : 编辑距离动态规划算法

科学背景
--------
在最优控制的工程应用中，控制作用往往由在边界上移动的物理执行器实现。
单轮车（unicycle）模型是机器人学中最经典的非完整约束系统：
    ẋ = v cos θ
    ẏ = v sin θ
    θ̇ = ω
    (x,y) : 位置，θ : 朝向，v : 线速度，ω : 角速度

非完整约束意味着系统的速度被限制在特定方向上（不能侧移），
这导致配置空间中的路径规划具有非平凡的几何结构。

在最优边界控制中，我们将边界 ∂Ω 参数化为一维曲线，
执行器沿边界移动并施加局部控制作用 q(s,t)。
由于执行器数量有限且存在动力学约束，控制变量 q 的时空分布
受限于非完整运动学。这引入了额外的耦合约束，使问题达到博士级难度。

编辑距离（Levenshtein Distance）在此用于：
  - 评估不同优化迭代中执行器路径序列的相似性
  - 自适应网格细化时，边界控制节点序列的匹配与重排序
  - 多执行器协同控制中的路径规划一致性检验

关键公式
--------
1. Unicycle 运动学：
   ṡ = v,  θ̇ = ω/v · tan(φ)  （简化为一维边界参数 s）
   更简化：执行器沿参数化边界以速度 v(t) 移动，
   位置 s(t) ∈ [0, L]，L 为边界周长。

2. 边界控制参数化：
   q(s,t) = Σ_{k=1}^{N_a} q_k(t) · ψ(s − s_k(t))
   其中 ψ 是局部形状函数，s_k(t) 是第 k 个执行器的位置。

3. Levenshtein 距离动态规划：
   d(i,j) = min( d(i-1,j) + 1, d(i,j-1) + 1, d(i-1,j-1) + cost )
   cost = 0 若 a_i = b_j，否则为 1。
"""

import numpy as np


def unicycle_dynamics(state, control):
    """
    单轮车运动学右端函数。
    state = [x, y, theta]
    control = [v, omega]

    返回导数 [dx/dt, dy/dt, dtheta/dt]
    """
    x, y, theta = state
    v, omega = control
    return np.array([v * np.cos(theta), v * np.sin(theta), omega], dtype=float)


def unicycle_integrate_rk4(state0, control_trajectory, dt):
    """
    使用四阶 Runge-Kutta 积分单轮车运动学方程。

    参数
    ----
    state0            : 初始状态 [x, y, theta]
    control_trajectory: (N, 2) 控制序列
    dt                : 时间步长

    返回
    ----
    states : (N+1, 3) 状态轨迹
    """
    N = control_trajectory.shape[0]
    states = np.zeros((N + 1, 3), dtype=float)
    states[0] = state0
    for n in range(N):
        u = control_trajectory[n]
        k1 = unicycle_dynamics(states[n], u)
        k2 = unicycle_dynamics(states[n] + 0.5 * dt * k1, u)
        k3 = unicycle_dynamics(states[n] + 0.5 * dt * k2, u)
        k4 = unicycle_dynamics(states[n] + dt * k3, u)
        states[n + 1] = states[n] + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return states


def parametric_ellipse_boundary(a, b, n_segments):
    """
    将椭圆边界参数化为 n_segments 段。
    返回每段的起止参数 theta 和弧长近似。
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n_segments + 1)
    # 椭圆弧长微元：ds = sqrt(a² sin²θ + b² cos²θ) dθ
    dtheta = 2.0 * np.pi / n_segments
    arc_lengths = []
    for i in range(n_segments):
        th = theta[i]
        ds = np.sqrt((a * np.sin(th)) ** 2 + (b * np.cos(th)) ** 2) * dtheta
        arc_lengths.append(ds)
    return theta, np.array(arc_lengths)


def boundary_actuator_positions(a, b, n_acts, t, speeds, theta0=None):
    """
    计算 n_acts 个执行器在椭圆边界上的位置。
    假设执行器以恒定角速度 speeds[i] 沿参数 θ 移动。

    参数
    ----
    a, b    : 椭圆半轴
    n_acts  : 执行器数量
    t       : 当前时间
    speeds  : 每个执行器的角速度数组
    theta0  : 初始角度数组

    返回
    ----
    positions : (n_acts, 2) 的 (x,y) 坐标
    theta     : 当前角度
    """
    if theta0 is None:
        theta0 = np.linspace(0.0, 2.0 * np.pi, n_acts, endpoint=False)
    theta = (theta0 + speeds * t) % (2.0 * np.pi)
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    return np.column_stack((x, y)), theta


def actuator_control_to_boundary(a, b, n_boundary_nodes, boundary_nodes_coords,
                                 actuator_positions, actuator_values,
                                 sigma=0.1):
    """
    将执行器的局部控制作用扩散到边界节点上。
    使用高斯型径向基函数（RBF）核：
        q(node) = Σ_k val_k · exp( −|node − pos_k|² / (2σ²) )

    参数
    ----
    a, b                : 椭圆半轴
    n_boundary_nodes    : 边界节点数
    boundary_nodes_coords: (n_boundary_nodes, 2) 坐标
    actuator_positions  : (n_acts, 2) 执行器位置
    actuator_values     : (n_acts,) 执行器控制强度
    sigma               : 扩散宽度
    """
    q = np.zeros(n_boundary_nodes, dtype=float)
    for pos, val in zip(actuator_positions, actuator_values):
        dist2 = (boundary_nodes_coords[:, 0] - pos[0]) ** 2 + (boundary_nodes_coords[:, 1] - pos[1]) ** 2
        q += val * np.exp(-dist2 / (2.0 * sigma ** 2))
    return q


def levenshtein_distance(s, t):
    """
    Levenshtein 编辑距离。
    计算将序列 s 转换为序列 t 所需的最少插入、删除、替换操作数。
    融合 668_levenshtein_distance 的动态规划算法。

    参数
    ----
    s, t : 可迭代序列（字符串、列表、数组）

    返回
    ----
    distance : 整数距离
    """
    m = len(s)
    n = len(t)
    # 使用一维 DP 优化空间
    prev = np.arange(n + 1, dtype=int)
    curr = np.zeros(n + 1, dtype=int)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1,      # 插入
                          prev[j] + 1,          # 删除
                          prev[j - 1] + cost)   # 替换
        prev, curr = curr, prev

    return int(prev[n])


def sequence_similarity_score(seq1, seq2):
    """
    基于编辑距离的序列相似度评分：
    score = 1 − distance / max(len(seq1), len(seq2))
    score ∈ [0, 1]，1 表示完全相同。
    """
    dist = levenshtein_distance(seq1, seq2)
    maxlen = max(len(seq1), len(seq2))
    if maxlen == 0:
        return 1.0
    return 1.0 - dist / maxlen


def rank_boundary_control_sequence(q_values, n_bins=10):
    """
    将连续边界控制值离散化为符号序列，用于编辑距离比较。
    使用分箱（binning）将连续值映射为离散符号。
    """
    q_min = np.min(q_values)
    q_max = np.max(q_values)
    if abs(q_max - q_min) < 1.0e-15:
        return ['0'] * len(q_values)
    bins = np.linspace(q_min, q_max, n_bins + 1)
    symbols = np.digitize(q_values, bins) - 1
    symbols = np.clip(symbols, 0, n_bins - 1)
    return [str(s) for s in symbols]


def random_unicycle_path(a, b, T, n_steps, rng=None):
    """
    生成一条随机单轮车轨迹，起点在椭圆边界上，
    控制输入 v 和 ω 为随机过程。
    用于初始化边界执行器的随机路径。
    """
    if rng is None:
        rng = np.random.default_rng(42)
    dt = T / n_steps
    v = 0.5 + 0.3 * rng.random(n_steps)
    omega = 0.5 * (rng.random(n_steps) - 0.5)
    controls = np.column_stack((v, omega))
    theta0 = rng.random() * 2.0 * np.pi
    state0 = np.array([a * np.cos(theta0), b * np.sin(theta0), theta0])
    return unicycle_integrate_rk4(state0, controls, dt)
