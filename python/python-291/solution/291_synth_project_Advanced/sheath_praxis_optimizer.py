"""
sheath_praxis_optimizer.py
==========================
PRAXIS 主轴优化方法。

本模块融合种子项目 907_praxis 的 PRAXIS 主轴优化算法
（Brent 1973），用于鞘层参数的最优化。

物理背景：
    在鞘层稳定性分析中，需要优化多个物理参数：
        - Bohm 速度 u_0 (满足 Bohm 判据的临界值)
        - SEE 系数 γ_eff (影响鞘层电势分布)
        - 磁场入射角 θ_B (影响磁前鞘结构)

    目标函数：
        f(u_0, γ_eff, θ_B) = max{Re(λ)} - 稳定性裕度

核心算法：
    PRAXIS (Principal Axis method):
        1. 构造主方向集（无需梯度）
        2. 沿各方向进行线搜索
        3. 更新方向集（Gram-Schmidt 正交化）
        4. 收敛判断
"""

import numpy as np
from scipy import linalg as la
from typing import Tuple, Callable, Optional
import math


def praxis_minimize(
    func: Callable[[np.ndarray], float],
    x0: np.ndarray,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    tol: float = 1e-8,
    max_eval: int = 1000,
    step_size: float = 1.0,
) -> Tuple[np.ndarray, float, int]:
    """
    PRAXIS 主轴优化方法
    （源自种子项目 907_praxis: praxis, minny, flin）

    Brent (1973) 的主轴方法 (Principal Axis method)：
    无需梯度的直接搜索方法，通过构造共轭方向集实现超线性收敛。

    算法步骤：
        1. 初始化方向集 V = {e_1, e_2, ..., e_n}
        2. 沿各方向执行线搜索
        3. 计算总下降量，判断是否更新方向集
        4. 新方向 = x_final - x_initial
        5. 用新方向替换下降最大的旧方向
        6. 重复直到收敛

    参数：
        func: 目标函数 f(x) -> float
        x0: 初始点 shape (n,)
        bounds: (lower, upper) 各维度界限
        tol: 收敛容差
        max_eval: 最大函数评估次数
        step_size: 初始步长

    返回：
        x_opt: 最优点
        f_opt: 最优值
        n_eval: 函数评估次数
    """
    n = len(x0)
    x = x0.copy().astype(float)
    f_val = func(x)
    n_eval = 1

    # 方向集初始化
    V = np.eye(n)

    # 步长
    step = np.full(n, step_size)

    best_x = x.copy()
    best_f = f_val

    for iteration in range(max_eval // n):
        x_start = x.copy()
        f_start = f_val

        # 记录各方向下降量
        decreases = np.zeros(n)

        for j in range(n):
            d = V[:, j]

            # 沿方向 d 进行线搜索
            x_new, f_new, n_line = _line_search(
                func, x, d, f_val, step[j], bounds
            )
            n_eval += n_line

            decreases[j] = f_val - f_new

            x = x_new
            f_val = f_new

            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()

        # 检查收敛
        total_decrease = f_start - f_val
        if abs(total_decrease) < tol * (abs(f_val) + tol):
            break

        # 更新方向集
        new_dir = x - x_start
        norm_dir = np.linalg.norm(new_dir)

        if norm_dir > 1e-15:
            new_dir /= norm_dir

            # 找下降最大的方向
            j_max = np.argmax(decreases)

            # 替换
            V[:, j_max] = new_dir

            # 正交化 (Gram-Schmidt)
            V = _gram_schmidt(V)

        # 自适应步长
        for j in range(n):
            if decreases[j] > 0:
                step[j] *= 1.5
            else:
                step[j] *= 0.5
            step[j] = np.clip(step[j], 1e-12, 100.0)

        if n_eval >= max_eval:
            break

    return best_x, best_f, n_eval


def _line_search(
    func: Callable,
    x: np.ndarray,
    d: np.ndarray,
    f_current: float,
    initial_step: float,
    bounds: Optional[Tuple],
) -> Tuple[np.ndarray, float, int]:
    """
    沿方向 d 的线搜索 (二次插值)

    使用三点二次插值找极小值
    """
    n_eval = 0

    # 尝试三个点
    alpha1 = 0.0
    alpha2 = initial_step
    alpha3 = 2.0 * initial_step

    # 应用边界限制
    if bounds is not None:
        lower, upper = bounds
        for alpha in [alpha2, alpha3]:
            x_test = x + alpha * d
            for i in range(len(x)):
                if x_test[i] < lower[i]:
                    alpha = max(1e-10, (lower[i] - x[i]) / (d[i] + 1e-30))
                if x_test[i] > upper[i]:
                    alpha = min(alpha, (upper[i] - x[i]) / (d[i] + 1e-30))

    f1 = f_current
    x2 = x + alpha2 * d
    if bounds is not None:
        x2 = np.clip(x2, bounds[0], bounds[1])
    f2 = func(x2)
    n_eval += 1

    x3 = x + alpha3 * d
    if bounds is not None:
        x3 = np.clip(x3, bounds[0], bounds[1])
    f3 = func(x3)
    n_eval += 1

    # 二次插值
    denom = (alpha1 - alpha2) * (alpha1 - alpha3) * (alpha2 - alpha3)
    if abs(denom) < 1e-30:
        if f2 < f1:
            return x2, f2, n_eval
        return x, f1, n_eval

    A = (alpha3 * (f2 - f1) + alpha2 * (f1 - f3) + alpha1 * (f3 - f2)) / denom

    if A <= 1e-15:
        # 无法形成上凸二次型，取最小值点
        if f2 < f1 and f2 < f3:
            return x2, f2, n_eval
        elif f3 < f1:
            return x3, f3, n_eval
        return x, f1, n_eval

    B = (alpha3**2 * (f1 - f2) + alpha2**2 * (f3 - f1) + alpha1**2 * (f2 - f3)) / denom
    alpha_min = -B / (2 * A)

    if alpha_min < 0:
        alpha_min = 0.0

    x_min = x + alpha_min * d
    if bounds is not None:
        x_min = np.clip(x_min, bounds[0], bounds[1])
    f_min = func(x_min)
    n_eval += 1

    # 选择最优点
    candidates = [(x, f1), (x2, f2), (x3, f3), (x_min, f_min)]
    best = min(candidates, key=lambda c: c[1])
    return best[0], best[1], n_eval


def _gram_schmidt(V: np.ndarray) -> np.ndarray:
    """修正 Gram-Schmidt 正交化"""
    n = V.shape[1]
    Q = V.copy()

    for j in range(n):
        for i in range(j):
            proj = np.dot(Q[:, i], Q[:, j])
            Q[:, j] -= proj * Q[:, i]

        norm = np.linalg.norm(Q[:, j])
        if norm > 1e-15:
            Q[:, j] /= norm
        else:
            Q[:, j] = V[:, j]

    return Q


def sheath_parameter_optimization(
    phi_wall: float,
    chi_see: float,
    mass_ratio: float,
) -> dict:
    """
    鞘层参数优化

    优化目标：最小化最大增长率（使系统更稳定）

    优化变量：
        x = [u_bohm, gamma_eff, theta_B]

    约束：
        u_bohm ≥ 1 (Bohm 判据)
        0 ≤ gamma_eff ≤ 0.5
        0 ≤ theta_B ≤ π/2

    参数：
        phi_wall: 壁面电势
        chi_see: SEE 因子
        mass_ratio: 质量比

    返回：
        优化结果字典
    """
    def objective(x):
        u_bohm = max(x[0], 1.0)
        gamma_eff = np.clip(x[1], 0.0, 0.5)
        theta_B = np.clip(x[2], 0.0, math.pi / 2)

        # 简化目标函数：稳定性裕度
        # 基于 Bohm 判据和 SEE 的稳定性指标
        bohman_margin = u_bohm - 1.0

        # SEE 引起的不稳定性
        see_destabilization = gamma_eff**2 * math.cos(theta_B)

        # 鞘层厚度效应
        sheath_stabilization = abs(phi_wall) / (1.0 + u_bohm)

        # 综合目标（最小化）
        f = see_destabilization - sheath_stabilization * 0.1 + 0.01 * (u_bohm - 1.0)**2
        return f

    x0 = np.array([1.5, 0.1, 0.3])
    lower = np.array([1.0, 0.0, 0.0])
    upper = np.array([5.0, 0.5, math.pi / 2])

    x_opt, f_opt, n_eval = praxis_minimize(
        objective, x0,
        bounds=(lower, upper),
        tol=1e-6,
        max_eval=200,
    )

    return {
        'optimal_x': x_opt,
        'optimal_u_bohm': x_opt[0],
        'optimal_gamma': x_opt[1],
        'optimal_theta': x_opt[2],
        'objective_value': f_opt,
        'n_evaluations': n_eval,
    }
