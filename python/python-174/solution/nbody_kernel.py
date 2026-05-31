"""
nbody_kernel.py
N体相互作用核函数与直接求和模块

融合种子项目:
- 923_pwc_plot_1d (分段常数函数思想, 用于核函数分段近似)
- 1026_risk_matrix (转移矩阵, 用于粒子状态转移分析)
- 777_monomial_value (单项式求值, 用于展开项)

科学背景:
N体问题描述N个粒子在相互作用下(如引力或库仑力)的动力学/静力学行为。
直接求和需要O(N^2)次计算, 是FMM加速的基准。

核心公式:
- 库仑/引力势:
    Phi(x_i) = sum_{j != i} q_j / |x_i - x_j|
    
- 相互作用力:
    F(x_i) = sum_{j != i} q_j * (x_i - x_j) / |x_i - x_j|^3
    
- 多极展开 (远场近似):
    1/|x - x_j| = sum_{l=0}^{inf} sum_{m=-l}^{l} (r_j^l / r^{l+1}) * Y_l^m(theta_j, phi_j) * conj(Y_l^m(theta, phi))
    其中 r > r_j (展开点在粒子外部)
    
- 局部展开 (近场近似):
    1/|x - x_j| = sum_{l=0}^{inf} sum_{m=-l}^{l} (r^l / r_j^{l+1}) * Y_l^m(theta, phi) * conj(Y_l^m(theta_j, phi_j))
    其中 r < r_j (展开点在粒子内部)

- 分段常数核近似 (PWC):
    将距离区间 [r_min, r_max] 划分为K段, 每段内用常数近似 1/r
    用于粗粒度预计算
"""

import numpy as np


def monomial_value(exponents, points):
    """
    计算单项式值 (融合777_monomial_value)
    
    公式:
        value = prod_{d=1}^{D} x_d^{e_d}
    
    参数:
        exponents: iterable of int, 各维度指数
        points: ndarray (N, D), 坐标点
    
    返回:
        ndarray (N,), 单项式值
    """
    points = np.atleast_2d(points)
    N, D = points.shape
    value = np.ones(N)
    for j in range(D):
        e = int(exponents[j])
        if e != 0:
            value *= np.power(points[:, j], e)
    return value


def coulomb_potential_direct(points, charges, epsilon=1e-10):
    """
    直接计算库仑势能 (O(N^2))
    
    公式:
        Phi_i = sum_{j != i} q_j / |r_i - r_j|
    
    数值稳定性:
        - 添加软化参数 epsilon 避免零距离奇点
        - |r_i - r_j| < epsilon 时, 势能被截断为 q_j / epsilon
    
    参数:
        points: ndarray (N, 3), 粒子位置
        charges: ndarray (N,), 电荷/质量
        epsilon: float, 软化长度
    
    返回:
        ndarray (N,), 各点势能
    """
    points = np.asarray(points, dtype=float)
    charges = np.asarray(charges, dtype=float)
    N = points.shape[0]
    if charges.shape[0] != N:
        raise ValueError("charges长度必须等于points行数")

    potential = np.zeros(N)
    for i in range(N):
        diff = points[i] - points
        dist = np.linalg.norm(diff, axis=1)
        # 避免自相互作用和除零
        dist[i] = np.inf
        # 边界处理
        dist = np.where(dist < epsilon, epsilon, dist)
        potential[i] = np.sum(charges / dist)
    return potential


def coulomb_force_direct(points, charges, epsilon=1e-10):
    """
    直接计算库仑力 (O(N^2))
    
    公式:
        F_i = sum_{j != i} q_j * (r_i - r_j) / |r_i - r_j|^3
    
    参数:
        points: ndarray (N, 3)
        charges: ndarray (N,)
        epsilon: float, 软化长度
    
    返回:
        ndarray (N, 3), 各点受力
    """
    points = np.asarray(points, dtype=float)
    charges = np.asarray(charges, dtype=float)
    N = points.shape[0]
    forces = np.zeros((N, 3))
    for i in range(N):
        diff = points[i] - points
        dist = np.linalg.norm(diff, axis=1)
        dist[i] = np.inf
        dist = np.where(dist < epsilon, epsilon, dist)
        inv_r3 = 1.0 / (dist ** 3)
        forces[i] = np.sum((charges * inv_r3)[:, None] * diff, axis=0)
    return forces


def pwc_kernel_approx(r_min, r_max, n_segments):
    """
    构造分段常数核函数近似 (融合923_pwc_plot_1d)
    
    将区间 [r_min, r_max] 划分为n_segments段,
    每段内用该段中点的 1/r 值作为常数近似
    
    公式:
        K_k = 1 / r_k^*,  r_k^* = (r_{k-1} + r_k) / 2
    
    参数:
        r_min: float, 最小距离
        r_max: float, 最大距离
        n_segments: int, 分段数
    
    返回:
        breaks: ndarray (n_segments+1,), 断点
        values: ndarray (n_segments,), 各段常数值
    """
    if r_min <= 0 or r_max <= r_min or n_segments <= 0:
        raise ValueError("参数非法")
    breaks = np.linspace(r_min, r_max, n_segments + 1)
    centers = 0.5 * (breaks[:-1] + breaks[1:])
    values = 1.0 / centers
    return breaks, values


def evaluate_pwc_kernel(r, breaks, values):
    """
    查询分段常数核函数值
    
    参数:
        r: float 或 ndarray, 距离
        breaks: ndarray, 断点
        values: ndarray, 常数值
    
    返回:
        float 或 ndarray, 近似核函数值
    """
    r = np.asarray(r)
    result = np.zeros_like(r, dtype=float)
    for k in range(len(values)):
        mask = (r >= breaks[k]) & (r < breaks[k + 1])
        result[mask] = values[k]
    # 边界处理: 超出范围的用端点值
    result[r < breaks[0]] = values[0]
    result[r >= breaks[-1]] = values[-1]
    return result


def build_transition_matrix_from_neighbors(neighbor_counts, n_states):
    """
    根据邻居计数构建转移矩阵 (融合1026_risk_matrix)
    
    在FMM中, 可用于描述粒子从一个空间区域(状态)转移到另一个区域的概率
    
    公式:
        T_{ij} = N_{ij} / sum_k N_{ik}  (行归一化)
        若行和为零, 则 T_{ii} = 1 / n_states (均匀分布)
    
    参数:
        neighbor_counts: ndarray (n_states, n_states), 邻居计数矩阵
        n_states: int, 状态数
    
    返回:
        ndarray (n_states, n_states), 转移概率矩阵
    """
    neighbor_counts = np.asarray(neighbor_counts, dtype=float)
    if neighbor_counts.shape != (n_states, n_states):
        raise ValueError("neighbor_counts形状不匹配")
    T = np.zeros((n_states, n_states))
    for i in range(n_states):
        row_sum = np.sum(neighbor_counts[i, :])
        if row_sum < 1e-15:
            T[i, :] = 1.0 / n_states
        else:
            T[i, :] = neighbor_counts[i, :] / row_sum
    return T


def kernel_gradient_laplacian(points, charges, target, epsilon=1e-10):
    """
    计算核函数在目标点的梯度和拉普拉斯量
    
    公式:
        grad(1/r) = -r_vec / r^3
        Laplacian(1/r) = 0  (r != 0, 满足泊松方程)
    
    参数:
        points: ndarray (N, 3)
        charges: ndarray (N,)
        target: ndarray (3,), 目标点
        epsilon: float
    
    返回:
        potential, gradient, laplacian
    """
    diff = target - points
    dist = np.linalg.norm(diff, axis=1)
    dist = np.where(dist < epsilon, epsilon, dist)
    inv_r = 1.0 / dist
    inv_r3 = inv_r ** 3
    potential = np.sum(charges * inv_r)
    gradient = np.sum((charges * inv_r3)[:, None] * diff, axis=0)
    # Laplacian在源外为0
    laplacian = 0.0
    return float(potential), gradient, float(laplacian)
