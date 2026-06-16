"""
lax_wendroff_advection.py — Lax-Wendroff 对流方程求解 (不确定性传播)

科学背景
========
在对流-扩散-反应系统中, 纯对流分量描述物质的输运:

    ∂u/∂t + c · ∂u/∂x = 0

Lax-Wendroff 方法为二阶精度的显式格式:

    u_i^{n+1} = u_i^n - (ν/2)·(u_{i+1}^n - u_{i-1}^n)
               + (ν²/2)·(u_{i+1}^n - 2·u_i^n + u_{i-1}^n)

其中 ν = c·Δt/Δx 为 CFL 数.

算法来源 (种子项目 355_fd1d_advection_lax_wendroff)
==================================================
直接移植 Lax-Wendroff 格式, 并扩展用于:
1. 不确定性在纯对流下的传播
2. 初始条件随机扰动的演化
3. 与隐式扩散求解器的算子分裂耦合

在本项目中的角色
================
模拟不确定性波包的传播: 初始随机温度扰动以速度 c 向右传播.
结合隐式扩散求解器实现算子分裂:
    u^{n+1} = L_diff(L_adv(u^n))

核心公式
========
1. CFL 条件:  ν = c·Δt/Δx ≤ 1 (稳定性)
2. Lax-Wendroff 步:
   u_i' = u_i - (ν/2)·(u_{i+1} - u_{i-1})
          + (ν²/2)·(u_{i+1} - 2·u_i + u_{i-1})
3. 数值耗散:  有效扩散系数 ν_eff = c·Δx·(1-ν²)/6
4. 色散关系:  ω = c·k - c·k³·Δx²·(1-ν²)/6 + O(Δx⁴)
"""

import numpy as np


def lax_wendroff_step(u, c, dx, dt):
    """执行一步 Lax-Wendroff 更新.

    参数
    ----
    u : ndarray, shape (nx,)
        当前时刻的场
    c : float
        对流速度
    dx : float
        空间步长
    dt : float
        时间步长

    返回
    ----
    u_new : ndarray, shape (nx,)
        更新后的场

    算法 (种子 355)
    ===============
    ν = c·dt/dx
    c1 = 0.5·ν
    c2 = 0.5·ν²
    u_i' = u_i - c1·(u_{i+1} - u_{i-1}) + c2·(u_{i+1} - 2·u_i + u_{i-1})
    """
    nx = len(u)
    nu = c * dt / dx

    # CFL 检查
    if abs(nu) > 1.0:
        # 自动调整 dt (但返回警告标志)
        dt_safe = 0.9 * dx / abs(c) if abs(c) > 1e-30 else dt
        nu = c * dt_safe / dx
        cfl_warning = True
    else:
        cfl_warning = False

    c1 = 0.5 * nu
    c2 = 0.5 * nu ** 2

    # 周期边界索引
    im1 = np.arange(nx) - 1
    ip1 = np.arange(nx) + 1
    im1[0] = nx - 1
    ip1[-1] = 0

    u_new = np.zeros_like(u)
    u_new[1:-1] = (u[1:-1]
                    - c1 * (u[ip1[1:-1]] - u[im1[1:-1]])
                    + c2 * (u[ip1[1:-1]] - 2.0 * u[1:-1] + u[im1[1:-1]]))

    # Dirichlet 边界 (不更新边界值)
    u_new[0] = u[0]
    u_new[-1] = u[-1]

    return u_new, cfl_warning


def solve_advection_lw(u_init, c, dx, dt, nt, boundary='dirichlet'):
    """Lax-Wendroff 方法求解一维对流方程.

    ∂u/∂t + c·∂u/∂x = 0

    参数
    ----
    u_init : ndarray, shape (nx,)
        初始条件
    c : float
        对流速度 (常数)
    dx, dt : float
    nt : int
        时间步数
    boundary : str
        'dirichlet' 或 'periodic'

    返回
    ----
    u_history : ndarray, shape (nt+1, nx)
    cfl_number : float
    """
    nx = len(u_init)
    nu = c * dt / dx

    u = u_init.copy()
    u_history = np.zeros((nt + 1, nx))
    u_history[0] = u.copy()

    for step in range(nt):
        if boundary == 'periodic':
            # 周期边界
            c1 = 0.5 * nu
            c2 = 0.5 * nu ** 2
            im1 = np.array([nx - 1] + list(range(nx - 1)))
            ip1 = np.array(list(range(1, nx)) + [0])
            u_new = (u - c1 * (u[ip1] - u[im1])
                     + c2 * (u[ip1] - 2.0 * u + u[im1]))
            u = u_new
        else:
            u, _ = lax_wendroff_step(u, c, dx, dt)

        u_history[step + 1] = u.copy()

    return u_history, nu


def operator_split_step(u, kappa_field, c_adv, dx, dt, params):
    """算子分裂: 先对流 (Lax-Wendroff), 后扩散 (隐式).

    Strang 分裂的一阶版本:
        u*  = L_adv(u^n)       (Lax-Wendroff)
        u^{n+1} = L_diff(u*)   (隐式)

    参数
    ----
    u : ndarray, shape (nx,)
    kappa_field : ndarray, shape (nx,)
    c_adv : float  对流速度
    dx, dt : float
    params : UQProblemParameters

    返回
    ----
    u_new : ndarray, shape (nx,)
    """
    from stochastic_heat_implicit import (
        build_implicit_system_matrix, thomas_solver
    )

    # Step 1: Lax-Wendroff 对流
    u_adv, _ = lax_wendroff_step(u, c_adv, dx, dt)

    # Step 2: 隐式扩散
    _, lower, main, upper = build_implicit_system_matrix(
        params.nx, dx, dt, kappa_field
    )
    rhs = u_adv.copy()
    rhs[0] = params.u_left
    rhs[-1] = params.u_right
    u_new = thomas_solver(lower, main, upper, rhs)
    u_new[0] = params.u_left
    u_new[-1] = params.u_right

    return u_new
