"""
cobweb_iterator.py
================================================================================
不动点迭代与流固耦合收敛模块 (来源于 194_cobweb_plot 项目)
================================================================================
本模块将蛛网图迭代思想应用于流固耦合 (FSI) 问题的不动点迭代。
在潮汐能提取系统中，流体载荷与结构变形相互依赖，需要通过
迭代找到自洽解。蛛网迭代提供了一种直观的收敛性分析工具。

核心公式:
    不动点迭代:
        x_{k+1} = f(x_k)

    流固耦合映射 (简化模型):
        u_{k+1} = G(u_k)
        其中 G 表示 "给定流速场→计算结构变形→反推等效流速" 的复合映射

    Aitken 加速:
        x* = x_k - (x_{k+1} - x_k)² / (x_{k+2} - 2x_{k+1} + x_k)
"""

import numpy as np
from typing import Callable, Tuple


def cobweb_iterate(
    f: Callable[[float], float],
    x0: float,
    n_iter: int,
    tol: float = 1e-10,
) -> Tuple[float, np.ndarray, bool]:
    """
    执行不动点迭代并追踪收敛轨迹。

    参数:
        f: 迭代函数 x_{k+1} = f(x_k)
        x0: 初始猜测
        n_iter: 最大迭代次数
        tol: 收敛容差

    返回:
        (x_star, history, converged)
        x_star: 不动点估计
        history: 迭代历史数组
        converged: 是否收敛
    """
    history = np.zeros(n_iter + 1)
    history[0] = x0
    for k in range(n_iter):
        history[k + 1] = f(history[k])
        if abs(history[k + 1] - history[k]) < tol:
            return history[k + 1], history[:k + 2], True
    return history[-1], history, False


def aitken_acceleration(
    f: Callable[[float], float],
    x0: float,
    n_iter: int,
    tol: float = 1e-10,
) -> Tuple[float, np.ndarray, bool]:
    """
    Aitken Δ² 加速不动点迭代。

    公式:
        x*_k = x_k - (x_{k+1} - x_k)² / (x_{k+2} - 2x_{k+1} + x_k)

    参数:
        f: 迭代函数
        x0: 初始猜测
        n_iter: 最大迭代次数
        tol: 收敛容差

    返回:
        (x_star, history, converged)
    """
    history = np.zeros(n_iter + 1)
    history[0] = x0
    k = 0
    while k + 2 <= n_iter:
        xk = history[k]
        xk1 = f(xk)
        xk2 = f(xk1)
        denom = xk2 - 2.0 * xk1 + xk
        if abs(denom) < 1e-14:
            history[k + 1] = xk1
            k += 1
            continue
        x_star = xk - (xk1 - xk) ** 2 / denom
        history[k + 1] = x_star
        if abs(x_star - xk) < tol:
            return x_star, history[:k + 2], True
        k += 1
    return history[k], history[:k + 1], False


def fsi_fixed_point_solver(
    fluid_solver: Callable[[np.ndarray], np.ndarray],
    structure_solver: Callable[[np.ndarray], np.ndarray],
    initial_guess: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-6,
    relaxation: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray, bool, int]:
    """
    流固耦合分区迭代求解器。

    物理模型:
        步骤 1: 给定结构变形 δ_k，求解流体得到载荷 F_k = Fluid(δ_k)
        步骤 2: 给定载荷 F_k，求解结构得到新变形 δ_{k+1} = Structure(F_k)
        步骤 3: 松弛更新: δ_{k+1} = ω·δ_{new} + (1-ω)·δ_k

    参数:
        fluid_solver: 流体求解器，输入变形，输出载荷
        structure_solver: 结构求解器，输入载荷，输出变形
        initial_guess: 初始变形场
        max_iter: 最大迭代次数
        tol: 收敛容差 (L2范数)
        relaxation: 松弛因子 (0 < ω ≤ 1)

    返回:
        (deformation, load, converged, iterations)
    """
    delta = np.asarray(initial_guess, dtype=float).copy()
    history = np.zeros(max_iter + 1)
    history[0] = np.linalg.norm(delta)

    for it in range(max_iter):
        load = fluid_solver(delta)
        delta_new = structure_solver(load)
        delta = relaxation * delta_new + (1.0 - relaxation) * delta
        history[it + 1] = np.linalg.norm(delta)
        residual = np.linalg.norm(delta_new - delta) / (np.linalg.norm(delta) + 1e-12)
        if residual < tol:
            return delta, load, True, it + 1

    return delta, load, False, max_iter


def contraction_factor_estimate(history: np.ndarray) -> float:
    """
    从迭代历史估计收缩因子 L (Lipschitz 常数)。

    公式:
        L ≈ ||x_{k+1} - x_k|| / ||x_k - x_{k-1}||

    参数:
        history: 范数历史数组

    返回:
        收缩因子估计
    """
    if len(history) < 3:
        return 1.0
    diffs = np.abs(np.diff(history))
    valid = diffs[1:] / (diffs[:-1] + 1e-14)
    return float(np.median(valid[valid < 1.0])) if np.any(valid < 1.0) else 1.0
