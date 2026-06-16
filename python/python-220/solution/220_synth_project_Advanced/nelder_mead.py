"""
nelder_mead.py — Nelder-Mead 单纯形法 (无导数优化)
============================================
来源项目: 797_nelder_mead

本模块实现 Nelder-Mead 单纯形直接搜索法, 用于无导数优化.
在分布式 ADMM 中用作局部子问题的无导数求解器
(当目标函数不可微或梯度计算代价过高时).

Nelder-Mead 算法 (来源: 797_nelder_mead):
  在 n 维空间中维护 n+1 个顶点构成的单纯形.
  每步迭代执行以下操作之一:
    1. 反射 (Reflection):     x_r = x̄ + ρ(x̄ - x_worst)
    2. 扩展 (Expansion):      x_e = x̄ + ρξ(x̄ - x_worst)
    3. 外收缩 (Outside cont.): x_c = x̄ + ργ(x̄ - x_worst)
    4. 内收缩 (Inside cont.):  x_c = x̄ + γ(x_worst - x̄)
    5. 缩缩 (Shrink):         x_i = x_best + σ(x_i - x_best)

  标准参数:
    ρ = 1 (反射系数)
    ξ = 2 (扩展系数)
    γ = 1/2 (收缩系数)
    σ = 1/2 (缩缩系数)

收敛性:
  - 不保证全局收敛 (可能停滞在非驻点)
  - 对低维问题 (n < 10) 通常有效
  - 在 ADMM 中用作局部接口子问题的鲁棒求解器

在分布式 ADMM 中的角色:
  接口共识子问题可能是非光滑的 (L1 范数, indicator function).
  Nelder-Mead 作为无导数替代方案处理这类情况.
"""

import numpy as np
from typing import Callable, Optional, Tuple, List
from config import EPS_NUM


# ============================================================
#  Nelder-Mead 核心实现 (来源: 797_nelder_mead)
# ============================================================
def nelder_mead(f: Callable, x0: np.ndarray,
                initial_simplex: Optional[np.ndarray] = None,
                max_iter: int = 1000, tol: float = 1e-8,
                rho: float = 1.0, xi: float = 2.0,
                gamma: float = 0.5, sigma: float = 0.5) -> dict:
    """Nelder-Mead 单纯形法求解 min f(x)

    算法流程 (来源: 797):
      1. 排序: f(x_1) ≤ f(x_2) ≤ ... ≤ f(x_{n+1})
      2. 计算质心: x̄ = (1/n) Σ_{i=1}^{n} x_i (排除最差点)
      3. 反射: x_r = x̄ + ρ(x̄ - x_{n+1})
         若 f(x_1) ≤ f(x_r) < f(x_n): 接受反射
      4. 扩展: 若 f(x_r) < f(x_1):
           x_e = x̄ + ρξ(x̄ - x_{n+1})
           若 f(x_e) < f(x_r): 接受扩展
           否则: 接受反射
      5. 收缩: 若 f(x_r) ≥ f(x_n):
           外收缩: 若 f(x_r) < f(x_{n+1}):
             x_c = x̄ + ργ(x̄ - x_{n+1})
             若 f(x_c) ≤ f(x_r): 接受外收缩
           内收缩: 否则:
             x_c = x̄ + γ(x_{n+1} - x̄)
             若 f(x_c) < f(x_{n+1}): 接受内收缩
      6. 缩缩: 若以上均失败:
           x_i = x_1 + σ(x_i - x_1)  for i = 2, ..., n+1

    Args:
        f: 目标函数 f(x) → float
        x0: 初始点 (n,)
        initial_simplex: 初始单纯形 (n+1, n) (可选)
        max_iter: 最大迭代次数
        tol: 收敛容差
        rho, xi, gamma, sigma: NM 参数

    Returns:
        dict: x_opt, f_opt, iterations, simplex_history, converged
    """
    n = len(x0)

    # 初始化单纯形
    if initial_simplex is not None:
        simplex = initial_simplex.copy()
    else:
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0.copy()
        for i in range(n):
            simplex[i + 1] = x0.copy()
            step = max(abs(x0[i]) * 0.05, 0.01)
            simplex[i + 1, i] += step

    # 初始函数值
    f_values = np.array([f(simplex[i]) for i in range(n + 1)])

    history = []
    converged = False

    for iteration in range(max_iter):
        # 排序
        order = np.argsort(f_values)
        simplex = simplex[order]
        f_values = f_values[order]

        # 记录
        history.append({
            'iteration': iteration,
            'f_best': f_values[0],
            'f_worst': f_values[-1],
            'simplex_size': np.max(np.linalg.norm(
                simplex[1:] - simplex[0], axis=1
            )),
        })

        # 收敛判据
        f_range = abs(f_values[-1] - f_values[0])
        simplex_size = np.max(np.linalg.norm(simplex[1:] - simplex[0], axis=1))

        if f_range < tol and simplex_size < tol:
            converged = True
            break

        # 质心 (排除最差点)
        x_bar = np.mean(simplex[:-1], axis=0)

        # 1. 反射
        x_r = x_bar + rho * (x_bar - simplex[-1])
        f_r = f(x_r)

        if f_values[0] <= f_r < f_values[-2]:
            # 接受反射
            simplex[-1] = x_r
            f_values[-1] = f_r
            continue

        if f_r < f_values[0]:
            # 2. 扩展
            x_e = x_bar + rho * xi * (x_bar - simplex[-1])
            f_e = f(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_values[-1] = f_e
            else:
                simplex[-1] = x_r
                f_values[-1] = f_r
            continue

        # 3. 收缩
        if f_r < f_values[-1]:
            # 外收缩
            x_c = x_bar + rho * gamma * (x_bar - simplex[-1])
        else:
            # 内收缩
            x_c = x_bar + gamma * (simplex[-1] - x_bar)

        f_c = f(x_c)

        if f_c < min(f_r, f_values[-1]):
            simplex[-1] = x_c
            f_values[-1] = f_c
            continue

        # 4. 缩缩
        x_best = simplex[0]
        f_best = f_values[0]
        for i in range(1, n + 1):
            simplex[i] = x_best + sigma * (simplex[i] - x_best)
            f_values[i] = f(simplex[i])

    # 最终排序
    order = np.argsort(f_values)
    simplex = simplex[order]
    f_values = f_values[order]

    return {
        'x_opt': simplex[0],
        'f_opt': f_values[0],
        'iterations': min(iteration + 1, max_iter),
        'history': history,
        'converged': converged,
        'final_simplex_size': simplex_size if iteration > 0 else float('inf'),
        'final_f_range': f_range if iteration > 0 else float('inf'),
    }


# ============================================================
#  ADMM 接口子问题的 NM 求解器
# ============================================================
def solve_interface_subproblem(objective: Callable, x_init: np.ndarray,
                                bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                                max_iter: int = 500, tol: float = 1e-6) -> dict:
    """使用 Nelder-Mead 求解 ADMM 接口子问题

    接口子问题形式:
      min_x  f(x) + (ρ/2)||x - z + u||²

    其中 z 为共识变量, u 为对偶变量, ρ 为罚参数.

    当目标函数不可微 (如 L1 范数) 时, NM 是鲁棒的替代方案.

    Args:
        objective: 目标函数 (含增广 Lagrangian 项)
        x_init: 初始点
        bounds: 变量上下界 (lower, upper)
        max_iter: 最大迭代
        tol: 收敛容差

    Returns:
        dict: x_opt, f_opt, converged
    """
    n = len(x_init)

    # 添加边界惩罚
    if bounds is not None:
        lb, ub = bounds
        def bounded_objective(x):
            # 违反边界的惩罚
            penalty = 0.0
            for i in range(n):
                if x[i] < lb[i]:
                    penalty += 1e6 * (lb[i] - x[i])**2
                elif x[i] > ub[i]:
                    penalty += 1e6 * (x[i] - ub[i])**2
            return objective(x) + penalty
    else:
        bounded_objective = objective

    result = nelder_mead(bounded_objective, x_init, max_iter=max_iter, tol=tol)

    # 投影到可行域
    if bounds is not None:
        result['x_opt'] = np.clip(result['x_opt'], bounds[0], bounds[1])

    return result


# ============================================================
#  Rosenbrock 测试函数 (用于验证 NM 求解器)
# ============================================================
def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock 函数 (香蕉函数)
    f(x, y) = (a - x)² + b(y - x²)²
    全局最小值: f(a, a²) = 0, 标准参数 a=1, b=100
    """
    result = 0.0
    for i in range(len(x) - 1):
        result += 100.0 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return result


def sphere(x: np.ndarray) -> float:
    """球面函数: f(x) = ||x||²"""
    return np.sum(x**2)


def beale(x: np.ndarray) -> float:
    """Beale 函数 (2D)
    f(x,y) = (1.5 - x + xy)² + (2.25 - x + xy²)² + (2.625 - x + xy³)²
    最小值: f(3, 0.5) = 0
    """
    if len(x) != 2:
        return float('inf')
    x_, y_ = x[0], x[1]
    return ((1.5 - x_ + x_ * y_)**2
            + (2.25 - x_ + x_ * y_**2)**2
            + (2.625 - x_ + x_ * y_**3)**2)
