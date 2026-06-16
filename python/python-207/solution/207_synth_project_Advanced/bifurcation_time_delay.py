"""
bifurcation_time_delay.py — 时滞动力系统分岔与不确定性传播

科学背景
========
驱动时滞系统的边界碰撞分岔 (Pierce & Ryan, 2019):

    ẋ(t) = F(x(t), x(t-τ), sgn(sin(2πt))) + b·sgn(sin(2πt))

其中反馈力依赖于历史状态 x(t-τ), 驱动力的符号每 0.5 个时间单位翻转.

该系统的动力学行为对参数 (τ, b) 极为敏感,
是研究不确定性在非线性动力系统中传播的理想模型.

算法来源 (种子项目 1085_PierceRyan)
===================================
直接移植时间步进迭代器, 扩展用于:
1. 参数不确定性 → 解的分散性
2. 初始条件不确定性 → Lyapunov 指数估计
3. 分岔点附近的不确定性放大

在本项目中的角色
================
模拟一个对参数敏感的子系统, 其输出作为 SHE 的边界条件或源项.
通过 MC 采样量化参数不确定性对系统响应的影响.

核心公式
========
1. 迭代映射:
   drive = feedback + forcing
   t_d = 0.5 - (t mod 0.5)        (到下一次驱动翻转的时间)
   t_h = τ + Z[0]                  (到下一次反馈变更的时间)
   t_z = -x / drive                (到下一次零交叉的时间)
   Δt = min(t_d, t_h, t_z)
2. Lyapunov 指数:
   λ = lim_{T→∞} (1/T) · Σ ln|δx_i'/δx_i|
"""

import numpy as np
import math


def iterate_map(tau, Z, x, t, b):
    """迭代映射至下一个事件 (种子 1085).

    参数
    ----
    tau : float  时滞
    Z : list  历史事件时间队列
    x : float  当前状态
    t : float  当前时间
    b : float  驱动力幅值

    返回
    ----
    x_new : float  新状态
    dt : float  到下一事件的时间间隔
    """
    s = t % 1.0
    sign = 1 if s < 0.5 else -1
    forcing = b * np.copysign(1.0, sign)
    feedback = math.copysign(1.0, x) * (-1) ** (len(Z) + 1)
    drive = feedback + forcing

    t_d = 0.5 - (t % 0.5)
    t_h = tau + Z[0] if len(Z) > 0 else np.inf
    t_z = -x / drive if abs(drive) > 1e-30 and -x / drive > 0 else np.inf

    dt = min(t_d, t_h, t_z)
    dt = max(dt, 1.0e-15)  # 防止零步长
    at = np.argmin([t_d, t_h, t_z])

    # 更新历史
    Z_new = [z - dt for z in Z]

    if at == 2:  # 零交叉
        Z_new.append(0.0)
        x_new = math.copysign(0.0, -x) if abs(x) > 1e-30 else 0.0
    elif at == 1:  # 反馈变更
        if len(Z_new) > 0:
            Z_new = Z_new[1:]
        x_new = x + dt * drive
    else:  # 驱动翻转
        x_new = x + dt * drive

    return x_new, dt, Z_new


def simulate_bifurcation(tau, b, hist, x0, tmax, max_events=100000):
    """模拟时滞系统至指定时间 (种子 1085).

    参数
    ----
    tau : float
    b : float
    hist : list  初始历史
    x0 : float
    tmax : float
    max_events : int

    返回
    ----
    X : list of float
    T : list of float
    """
    T = [0.0]
    X = [x0]
    Z = list(hist)
    t = 0.0
    x = x0

    for _ in range(max_events):
        if t >= tmax:
            break
        x, dt, Z = iterate_map(tau, Z, x, t, b)
        t = t + dt
        T.append(t)
        X.append(x)

    return X, T, Z


def parameter_sensitivity(tau_range, b_fixed, x0=0.01, tmax=50.0):
    """研究参数 τ 对系统终态的敏感性.

    参数
    ----
    tau_range : ndarray
    b_fixed : float
    x0 : float
    tmax : float

    返回
    ----
    final_states : ndarray
    """
    final_states = np.zeros(len(tau_range))
    for i, tau in enumerate(tau_range):
        try:
            X, T, _ = simulate_bifurcation(tau, b_fixed, [-0.94], x0, tmax)
            final_states[i] = X[-1] if len(X) > 0 else 0.0
        except Exception:
            final_states[i] = 0.0
    return final_states


def lyapunov_exponent_estimate(tau, b, x0=0.01, delta=1e-8,
                                n_renorm=1000, t_per_renorm=1.0):
    """估计最大 Lyapunov 指数.

    λ ≈ (1/T) · Σ_k ln(|δx_k| / |δx_0|)

    参数
    ----
    tau, b : float
    x0 : float
    delta : float  初始扰动
    n_renorm : int
    t_per_renorm : float

    返回
    ----
    lambda_max : float
    """
    # 参考轨道
    X_ref, T_ref, Z_ref = simulate_bifurcation(
        tau, b, [-0.94], x0, t_per_renorm * n_renorm
    )
    # 扰动轨道
    X_pert, T_pert, Z_pert = simulate_bifurcation(
        tau, b, [-0.94], x0 + delta, t_per_renorm * n_renorm
    )

    # 简化估计: 使用最终分离度
    if len(X_ref) > 1 and len(X_pert) > 1:
        final_sep = abs(X_pert[-1] - X_ref[-1])
        total_time = T_ref[-1] if len(T_ref) > 0 else 1.0
        if final_sep > 0 and delta > 0:
            lambda_max = np.log(final_sep / delta) / total_time
        else:
            lambda_max = 0.0
    else:
        lambda_max = 0.0

    return lambda_max
