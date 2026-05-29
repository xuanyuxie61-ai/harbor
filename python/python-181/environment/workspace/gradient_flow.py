"""
gradient_flow.py
中心差分梯度计算与捕食者-猎物竞争动力学特征选择
融合原项目: 279_diff_center, 350_fd_predator_prey

核心科学思想:
1. 在流形上使用中心差分计算梯度场，驱动数据点沿梯度流演化；
2. 引入捕食者-猎物(Lotka-Volterra)动力学模型，
   将不同特征维度视为竞争物种，模拟特征选择过程。

数学模型:
中心差分:
    ∂f/∂x ≈ (f(x+h) - f(x-h)) / (2h) + O(h²)

Lotka-Volterra竞争系统:
    dx_i/dt = x_i (r_i - Σ_j a_{ij} x_j)
其中 x_i 为第i个特征的"丰度"(重要性)，
r_i 为内禀增长率，a_{ij} 为竞争系数矩阵。

在降维中，我们将各嵌入维度视为竞争物种，
通过动力学演化筛选出最具代表性的维度。
"""

import numpy as np
from typing import Tuple, Callable


def centered_difference(f: Callable, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """
    中心差分计算标量函数梯度
    ∇f(x) = [∂f/∂x_1, ..., ∂f/∂x_D]^T
    """
    D = len(x)
    grad = np.zeros(D, dtype=np.float64)
    for d in range(D):
        x_plus = x.copy()
        x_minus = x.copy()
        # 自适应步长
        step = max(abs(x[d]) * h, h)
        x_plus[d] += step
        x_minus[d] -= step
        grad[d] = (f(x_plus) - f(x_minus)) / (2.0 * step)
    return grad


def hessian_approximation(f: Callable, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """
    中心差分近似Hessian矩阵
    H_{ij} = ∂²f / ∂x_i ∂x_j
    """
    D = len(x)
    H = np.zeros((D, D), dtype=np.float64)
    for i in range(D):
        for j in range(i, D):
            x_pp = x.copy()
            x_pm = x.copy()
            x_mp = x.copy()
            x_mm = x.copy()
            hi = max(abs(x[i]) * h, h)
            hj = max(abs(x[j]) * h, h)
            x_pp[i] += hi; x_pp[j] += hj
            x_pm[i] += hi; x_pm[j] -= hj
            x_mp[i] -= hi; x_mp[j] += hj
            x_mm[i] -= hi; x_mm[j] -= hj
            H[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4.0 * hi * hj)
            H[j, i] = H[i, j]
    return H


def gradient_flow_descent(f: Callable, x0: np.ndarray, dt: float = 0.01,
                          max_steps: int = 1000, tol: float = 1e-8) -> np.ndarray:
    """
    梯度流下降: dx/dt = -∇f(x)
    使用显式Euler离散化
    """
    x = x0.copy()
    for step in range(max_steps):
        grad = centered_difference(f, x)
        x_new = x - dt * grad
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x


def predator_prey_dynamics(x0: np.ndarray, r: np.ndarray,
                            A: np.ndarray, t_span: Tuple[float, float],
                            n_steps: int = 5000) -> np.ndarray:
    """
    Lotka-Volterra竞争动力学
    dx/dt = x * (r - A x)
    使用有限差分法求解
    """
    n = len(x0)
    t_start, t_stop = t_span
    dt = (t_stop - t_start) / n_steps
    x = x0.copy()
    trajectory = np.zeros((n_steps + 1, n))
    trajectory[0] = x
    for i in range(n_steps):
        # 显式Euler + 投影保证非负
        dx = x * (r - A @ x)
        x = x + dt * dx
        x = np.maximum(x, 0.0)  # 生物量非负
        trajectory[i + 1] = x
    return trajectory


def feature_selection_by_competition(feature_scores: np.ndarray,
                                      interaction_matrix: np.ndarray = None,
                                      n_selected: int = 5) -> np.ndarray:
    """
    利用捕食者-猎物竞争动力学进行特征选择
    将特征重要性视为种群丰度，通过竞争演化筛选优势特征
    """
    n = len(feature_scores)
    if interaction_matrix is None:
        # 构造竞争矩阵: 特征间相关性越高竞争越强
        A = np.ones((n, n)) * 0.5
        np.fill_diagonal(A, 1.0)
    else:
        A = interaction_matrix
    # 内禀增长率与初始重要性成正比
    r = feature_scores / (np.max(feature_scores) + 1e-15)
    x0 = feature_scores / (np.sum(feature_scores) + 1e-15)
    # 演化到稳态
    trajectory = predator_prey_dynamics(x0, r, A, (0.0, 10.0), n_steps=2000)
    steady_state = trajectory[-1]
    # 选择稳态丰度最高的特征
    idx = np.argsort(steady_state)[::-1][:n_selected]
    return idx


def diffusion_map_gradient(data: np.ndarray, embedding: np.ndarray,
                            target_point: int, sigma: float = 1.0) -> np.ndarray:
    """
    计算扩散映射上的梯度
    在嵌入空间中，梯度指示了数据流形的局部变化方向
    """
    N = len(data)
    # 计算核权重
    dists = np.linalg.norm(data - data[target_point], axis=1)
    weights = np.exp(-dists ** 2 / (2.0 * sigma ** 2))
    weights[target_point] = 0.0
    weights = weights / (np.sum(weights) + 1e-15)
    # 嵌入空间中的加权平均位移
    grad = np.zeros(embedding.shape[1])
    for i in range(N):
        grad += weights[i] * (embedding[i] - embedding[target_point])
    return grad


def conserved_quantity_prey_predator(prey: float, predator: float) -> float:
    """
    捕食者-猎物系统的守恒量 (Hamiltonian-like):
        E(r, f) = c1 * r + c2 * f - a1 * ln(r) - a2 * ln(f)
    在精确解中此量守恒
    """
    c1 = 0.003
    c2 = 0.004
    a1 = 10.0
    a2 = 2.0
    if prey <= 0 or predator <= 0:
        return np.inf
    return c1 * prey + c2 * predator - a1 * np.log(prey) - a2 * np.log(predator)
