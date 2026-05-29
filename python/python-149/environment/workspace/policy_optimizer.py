"""
policy_optimizer.py
策略优化器：基于Nelder-Mead与策略梯度的混合优化

融合种子项目:
  - 797_nelder_mead: 单纯形法无梯度优化
  - 766_midpoint_explicit: 显式中点积分器用于策略评估

科学背景:
  对于参数化策略 π_θ(x)，目标是优化参数 θ 使得期望总代价最小:

      J(θ) = E_{x0~ρ0} [ ∫_0^T L(x(t), π_θ(x(t))) dt + Φ(x(T)) ]

  其中 x(t) 服从受控SDE。由于SDE导致目标函数非光滑，
  采用Nelder-Mead单纯形法进行无梯度优化:

      单纯形: S = {θ_0, θ_1, ..., θ_m} ⊂ R^m
      反射:  θ_r = θ̄ + ρ(θ̄ - θ_worst)
      扩展:  θ_e = θ̄ + ξ(θ̄ - θ_worst)
      外收缩: θ_c = θ̄ + γ(θ̄ - θ_worst)
      内收缩: θ_ci = θ̄ - γ(θ̄ - θ_worst)
      收缩:  θ_i = θ_best + σ(θ_i - θ_best)

  参数通常取: ρ=1, ξ=2, γ=0.5, σ=0.5
"""

import numpy as np
from typing import Callable, Optional, Tuple


def evaluate_policy_cost(
    theta: np.ndarray,
    policy_fn: Callable[[np.ndarray, np.ndarray], float],
    sde_integrator: Callable,
    cost_fn: Callable,
    n_mc: int = 50,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    蒙特卡洛估计策略参数θ对应的期望总代价。

        J(θ) ≈ (1/N_mc) Σ_{k=1}^{N_mc} [ Σ_j L(x_j, u_j) Δt + Φ(x_N) ]

    Parameters
    ----------
    theta : ndarray
        策略参数
    policy_fn : callable
        u = policy_fn(theta, x)
    sde_integrator : callable
        SDE积分器，返回 (t, y)
    cost_fn : callable
        代价计算函数
    n_mc : int
        Monte Carlo样本数

    Returns
    -------
    cost : float
        平均总代价
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)

    costs = []
    for _ in range(n_mc):
        t, y = sde_integrator(rng=rng)
        c = cost_fn(t, y, policy_fn, theta)
        costs.append(c)

    costs_arr = np.array(costs)
    # 去除极端异常值（鲁棒性）
    q_low, q_high = np.percentile(costs_arr, [5, 95])
    mask = (costs_arr >= q_low) & (costs_arr <= q_high)
    if np.sum(mask) > 0:
        return float(np.mean(costs_arr[mask]))
    return float(np.mean(costs_arr))


def nelder_mead_optimize(
    obj_fn: Callable[[np.ndarray], float],
    x0: np.ndarray,
    rho: float = 1.0,
    xi: float = 2.0,
    gam: float = 0.5,
    sig: float = 0.5,
    tolerance: float = 1e-6,
    max_feval: int = 500,
) -> Tuple[np.ndarray, int]:
    """
    Nelder-Mead单纯形优化算法。

    Parameters
    ----------
    obj_fn : callable
        目标函数 f(x) → float
    x0 : ndarray, shape (m,)
        初始猜测
    rho, xi, gam, sig : float
        NM参数
    tolerance : float
        函数值收敛容差
    max_feval : int
        最大函数评估次数

    Returns
    -------
    x_opt : ndarray
        最优参数
    n_feval : int
        函数评估次数
    """
    m = len(x0)
    # 构造初始单纯形
    simplex = np.zeros((m + 1, m))
    simplex[0, :] = x0
    # 其他顶点沿坐标轴偏移
    scale = np.maximum(np.abs(x0), 0.1)
    for i in range(m):
        simplex[i + 1, :] = x0.copy()
        simplex[i + 1, i] += 0.05 * scale[i]

    # 评估单纯形各顶点
    f_vals = np.zeros(m + 1)
    for i in range(m + 1):
        f_vals[i] = obj_fn(simplex[i, :])

    n_feval = m + 1

    converged = False
    diverged = False

    while not converged and not diverged:
        # 按函数值排序
        order = np.argsort(f_vals)
        simplex = simplex[order, :]
        f_vals = f_vals[order]

        # 最优与最差
        f_best = f_vals[0]
        f_worst = f_vals[-1]
        f_second_worst = f_vals[-2]

        # 重心（除最差外）
        x_bar = np.mean(simplex[:m, :], axis=0)

        # 反射
        x_r = x_bar + rho * (x_bar - simplex[-1, :])
        f_r = obj_fn(x_r)
        n_feval += 1

        if f_best <= f_r < f_second_worst:
            simplex[-1, :] = x_r
            f_vals[-1] = f_r
        elif f_r < f_best:
            # 扩展
            x_e = x_bar + xi * (x_bar - simplex[-1, :])
            f_e = obj_fn(x_e)
            n_feval += 1
            if f_e < f_r:
                simplex[-1, :] = x_e
                f_vals[-1] = f_e
            else:
                simplex[-1, :] = x_r
                f_vals[-1] = f_r
        elif f_second_worst <= f_r < f_worst:
            # 外收缩
            x_c = x_bar + gam * (x_bar - simplex[-1, :])
            f_c = obj_fn(x_c)
            n_feval += 1
            if f_c <= f_r:
                simplex[-1, :] = x_c
                f_vals[-1] = f_c
            else:
                # 收缩
                for i in range(1, m + 1):
                    simplex[i, :] = simplex[0, :] + sig * (simplex[i, :] - simplex[0, :])
                    f_vals[i] = obj_fn(simplex[i, :])
                    n_feval += 1
        else:
            # 内收缩
            x_ci = x_bar - gam * (x_bar - simplex[-1, :])
            f_ci = obj_fn(x_ci)
            n_feval += 1
            if f_ci < f_worst:
                simplex[-1, :] = x_ci
                f_vals[-1] = f_ci
            else:
                # 收缩
                for i in range(1, m + 1):
                    simplex[i, :] = simplex[0, :] + sig * (simplex[i, :] - simplex[0, :])
                    f_vals[i] = obj_fn(simplex[i, :])
                    n_feval += 1

        # 检查收敛
        converged = abs(f_vals[-1] - f_vals[0]) < tolerance
        diverged = n_feval > max_feval

        # 边界保护：若单纯形坍塌，添加扰动
        simplex_spread = np.max(np.std(simplex, axis=0))
        if simplex_spread < 1e-12 and not converged:
            for i in range(1, m + 1):
                noise = np.random.randn(m) * 1e-4
                simplex[i, :] += noise
                f_vals[i] = obj_fn(simplex[i, :])
                n_feval += 1

    x_opt = simplex[0, :]
    return x_opt, n_feval


def linear_feedback_policy(
    theta: np.ndarray,
    x: np.ndarray,
    x_eq: Optional[np.ndarray] = None,
) -> float:
    """
    线性反馈策略:

        u(t) = -K · (x(t) - x_eq)

    其中 K 为增益向量（包含在 theta 中）。

    Parameters
    ----------
    theta : ndarray
        策略参数（一维反馈增益K）
    x : ndarray
        当前状态
    x_eq : ndarray or None
        平衡点

    Returns
    -------
    u : float
        标量控制输入
    """
    if x_eq is None:
        x_eq = np.zeros_like(x)
    dx = x - x_eq
    K = theta[: len(dx)]
    u = -np.dot(K, dx)
    # 饱和约束
    return float(np.clip(u, -5.0, 5.0))


def quadratic_policy(
    theta: np.ndarray,
    x: np.ndarray,
    x_eq: Optional[np.ndarray] = None,
) -> float:
    """
    二次反馈策略（非线性控制）:

        u(t) = -K_1^T (x-x_eq) - (x-x_eq)^T K_2 (x-x_eq)

    其中 theta = [K_1, vec(K_2)]。
    """
    dim = len(x)
    if x_eq is None:
        x_eq = np.zeros(dim)
    dx = x - x_eq

    K1 = theta[:dim]
    # K2 为对称矩阵，只存储上三角
    k2_vals = theta[dim:]
    K2 = np.zeros((dim, dim))
    idx = 0
    for i in range(dim):
        for j in range(i, dim):
            if idx < len(k2_vals):
                K2[i, j] = k2_vals[idx]
                K2[j, i] = k2_vals[idx]
                idx += 1

    u = -np.dot(K1, dx) - float(dx @ K2 @ dx)
    return float(np.clip(u, -5.0, 5.0))
