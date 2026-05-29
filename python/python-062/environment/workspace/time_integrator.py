"""
time_integrator.py
================================================================================
时间积分模块 —— 基于种子项目 826_ode_euler_backward 与 346_exp_ode

LES 中的时间推进需要兼顾稳定性与精度：
- 对流项：显式处理（CFL 限制）
- 扩散项：隐式处理（后向欧拉，无条件稳定）
- 压力项：投影法（分步求解）

本模块提供后向欧拉法（Backward Euler）与指数积分参考解。

核心物理公式
--------------------------------------------------------------------------------
后向欧拉离散（对于 y' = f(t,y)）：
    y^{n+1} = y^n + Δt * f(t^{n+1}, y^{n+1})

对于线性扩散方程 y' = -λ y，后向欧拉给出：
    y^{n+1} = y^n / (1 + λ Δt)

其放大因子 |G| = 1 / |1 + λ Δt| < 1，无条件稳定。

指数积分参考解（用于验证和 Richardson 外推）：
    y(t) = y0 * exp(α (t - t0))

在 LES 中，时间步长需满足：
    Δt < CFL * min(Δx / |u|, Δy / |v|, Δz / |w|)
    Δt < 0.5 * min(Δx²/ν, Δy²/ν, Δz²/ν)   （扩散限制）
"""

import numpy as np


def backward_euler_step(y_n, f_func, dt, max_picard_iter=10, tol=1e-10):
    """
    执行一个后向欧拉时间步。

    参数
    ----------
    y_n : np.ndarray
        当前时刻解
    f_func : callable
        右端项函数 f(t, y)
    dt : float
        时间步长
    max_picard_iter : int
        Picard 迭代次数
    tol : float
        收敛容差

    返回
    -------
    y_np1 : np.ndarray
        下一时刻解
    """
    y_np1 = np.copy(y_n)

    for _ in range(max_picard_iter):
        y_new = y_n + dt * f_func(y_np1)
        diff = np.linalg.norm(y_new - y_np1)
        y_np1 = y_new
        if diff < tol:
            break

    return y_np1


def backward_euler_linear(A_mat, y_n, dt, b_vec=None):
    """
    对线性系统 dy/dt = A y + b 执行后向欧拉步。

    参数
    ----------
    A_mat : np.ndarray
        系统矩阵
    y_n : np.ndarray
        当前解
    dt : float
        时间步长
    b_vec : np.ndarray, optional
        常数源项

    返回
    -------
    y_np1 : np.ndarray
    """
    n = len(y_n)
    I = np.eye(n)

    lhs = I - dt * A_mat
    rhs = np.copy(y_n)
    if b_vec is not None:
        rhs = rhs + dt * b_vec

    # 边界鲁棒性：检查矩阵条件数
    cond = np.linalg.cond(lhs)
    if cond > 1e15:
        # 正则化
        lhs = lhs + 1e-12 * I

    y_np1 = np.linalg.solve(lhs, rhs)
    return y_np1


def exp_exact_solution(t, alpha, t0, y0):
    """
    指数方程 y' = α y 的精确解。

    参数
    ----------
    t : float 或 np.ndarray
        时间
    alpha : float
        增长率
    t0 : float
        初始时间
    y0 : float
        初始值

    返回
    -------
    y : float 或 np.ndarray
    """
    return y0 * np.exp(alpha * (t - t0))


def exp_deriv(t, y, alpha):
    """
    指数方程的右端项。

    参数
    ----------
    t : float
    y : float 或 np.ndarray
    alpha : float

    返回
    -------
    dydt : float 或 np.ndarray
    """
    return alpha * y


def compute_cfl_limit(u, v, w, dx, dy, dz, cfl_number=0.5):
    """
    计算 CFL 时间步长限制。

    参数
    ----------
    u, v, w : np.ndarray
        速度分量
    dx, dy, dz : float
        网格间距
    cfl_number : float
        CFL 数

    返回
    -------
    dt_cfl : float
        最大允许时间步长
    """
    u_max = np.max(np.abs(u)) + 1e-12
    v_max = np.max(np.abs(v)) + 1e-12
    w_max = np.max(np.abs(w)) + 1e-12

    dt_cfl = cfl_number / (u_max / dx + v_max / dy + w_max / dz)
    return dt_cfl


def compute_diffusion_limit(nu, dx, dy, dz, safety=0.5):
    """
    计算扩散稳定性限制。

    参数
    ----------
    nu : float
        粘性系数
    dx, dy, dz : float
        网格间距
    safety : float
        安全系数

    返回
    -------
    dt_diff : float
    """
    dt_diff = safety / (nu * (1.0 / dx**2 + 1.0 / dy**2 + 1.0 / dz**2))
    return dt_diff


def adaptive_timestep(u, v, w, dx, dy, dz, nu_eff, cfl=0.5):
    """
    根据 CFL 与扩散限制自适应选择时间步长。

    参数
    ----------
    u, v, w : np.ndarray
    dx, dy, dz : float
    nu_eff : float
    cfl : float

    返回
    -------
    dt : float
    """
    dt_cfl = compute_cfl_limit(u, v, w, dx, dy, dz, cfl)
    dt_diff = compute_diffusion_limit(nu_eff, dx, dy, dz)
    dt = min(dt_cfl, dt_diff)
    # 限制极端值
    dt = max(1e-8, min(dt, 10.0))
    return dt
