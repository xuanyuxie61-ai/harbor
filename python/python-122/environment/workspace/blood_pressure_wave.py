"""
脑血流动力学 — 血压波传播与压力场求解模块

整合 laplace_radial_exact（径向 Laplace 精确解）与 burgers_time_inviscid（无粘 Burgers 方程），
模拟脑动脉系统中的压力波传播与稳态压力场分布。

科学背景:
- 脑动脉血压波可用一维非线性波动方程描述。忽略粘性时，Burgers 方程 u_t + (u^2/2)_x = 0
  可描述压力脉冲在弹性血管壁中的传播。
- 在微血管层面，稳态血流满足 Laplace 方程 ∇²P = 0，其径向解 P(r) = a log(r) + b
  描述圆形血管截面内的压力分布。
- 结合 Poiseuille 定律: Q = (π r^4 ΔP) / (8 μ L)
"""

import numpy as np


def laplace_radial_2d_exact(x, y, a, b):
    """
    二维径向 Laplace 方程精确解。
    ∇²u = 0,  u(r) = a * log(r) + b

    返回: u, ux, uy, uxx, uxy, uyy
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r2 = x ** 2 + y ** 2
    # 避免 r=0 奇异
    r2 = np.where(r2 < 1e-14, 1e-14, r2)
    r = np.sqrt(r2)

    u = a * np.log(r) + b
    ux = a * x / r2
    uy = a * y / r2
    uxx = a * (-2.0 * x ** 2 / r2 + 1.0) / r2
    uxy = -2.0 * a * x * y / r2 ** 2
    uyy = a * (-2.0 * y ** 2 / r2 + 1.0) / r2
    return u, ux, uy, uxx, uxy, uyy


def laplace_radial_3d_exact(x, y, z, a, b):
    """
    三维径向 Laplace 方程精确解。
    ∇²u = 0,  u(r) = a / r + b  (球对称)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    r2 = x ** 2 + y ** 2 + z ** 2
    r2 = np.where(r2 < 1e-14, 1e-14, r2)
    r = np.sqrt(r2)

    u = a / r + b
    ux = -a * x / r ** 3
    uy = -a * y / r ** 3
    uz = -a * z / r ** 3
    return u, ux, uy, uz


def burgers_flux(u):
    """Burgers 守恒量通量: f(u) = 0.5 * u^2"""
    return 0.5 * u ** 2


def burgers_flux_derivative(u):
    """Burgers 通量导数: df/du = u"""
    return u


def godunov_nf(u_left, u_right):
    """
    Godunov 数值通量。
    求解 Riemann 问题以确定界面通量。
    """
    u_left = np.asarray(u_left, dtype=float)
    u_right = np.asarray(u_right, dtype=float)
    ustar = np.empty_like(u_left)

    mask1 = u_right <= u_left
    mask2 = ~mask1

    # 冲击波情况
    cond_a = mask1 & ((u_left + u_right) / 2.0 > 0)
    cond_b = mask1 & ((u_left + u_right) / 2.0 <= 0)
    ustar[cond_a] = u_left[cond_a]
    ustar[cond_b] = u_right[cond_b]

    # 稀疏波情况
    cond_c = mask2 & (u_left > 0)
    cond_d = mask2 & (u_right < 0)
    cond_e = mask2 & ~(cond_c | cond_d)
    ustar[cond_c] = u_left[cond_c]
    ustar[cond_d] = u_right[cond_d]
    ustar[cond_e] = 0.0

    return burgers_flux(ustar)


def burgers_time_inviscid_godunov(u0, nx, nt, t_max, bc_type='periodic'):
    """
    用 Godunov 格式求解无粘 Burgers 方程:
        u_t + (0.5 u^2)_x = 0,  x ∈ [-1, 1]

    参数:
        u0: 初始条件函数句柄
        nx: 空间节点数
        nt: 时间步数
        t_max: 终止时间
        bc_type: 'periodic' 或 'dirichlet'

    返回:
        U: (nt+1, nx) 的解矩阵
        x: 空间坐标
    """
    a = -1.0
    b = 1.0
    dx = (b - a) / nx
    x = np.linspace(a, b, nx)
    dt = t_max / nt

    U = np.zeros((nt + 1, nx))
    u = u0(x).astype(float)
    U[0, :] = u

    for i in range(nt):
        unew = np.empty_like(u)

        if bc_type == 'periodic':
            unew[0] = u[0] - dt / dx * (godunov_nf(u[0], u[1]) - godunov_nf(u[-1], u[0]))
            unew[1:-1] = u[1:-1] - dt / dx * (
                godunov_nf(u[1:-1], u[2:]) - godunov_nf(u[:-2], u[1:-1])
            )
            unew[-1] = u[-1] - dt / dx * (godunov_nf(u[-1], u[0]) - godunov_nf(u[-2], u[-1]))
        else:
            unew[0] = u[0]
            unew[1:-1] = u[1:-1] - dt / dx * (
                godunov_nf(u[1:-1], u[2:]) - godunov_nf(u[:-2], u[1:-1])
            )
            unew[-1] = u[-1]

        u = unew
        U[i + 1, :] = u

    return U, x


def windkessel_pressure_outflow(Q_in, R, C, dt, n_steps):
    """
    二元件 Windkessel 模型计算动脉出口压力:
        P(t) + R C dP/dt = R Q_in(t)
    解析解（离散化）:
        P_{n+1} = (P_n + R*Q_n*dt/C) / (1 + dt/(R*C))

    参数:
        Q_in: 入流速率数组
        R: 外周阻力 (Pa·s/m³)
        C: 动脉顺应性 (m³/Pa)
        dt: 时间步长
        n_steps: 步数
    """
    P = np.zeros(n_steps)
    P[0] = Q_in[0] * R
    alpha = dt / (R * C)
    for n in range(n_steps - 1):
        P[n + 1] = (P[n] + R * Q_in[n] * alpha) / (1.0 + alpha)
    return P


def poiseuille_flow_rate(radius, delta_p, length, mu=3.5e-3):
    """
    Poiseuille 定律计算圆管体积流量:
        Q = (π r⁴ ΔP) / (8 μ L)
    μ: 血液动力粘度 [Pa·s]，常温约 3.5e-3
    """
    if radius <= 0 or length <= 0:
        return 0.0
    return np.pi * radius ** 4 * delta_p / (8.0 * mu * length)


def compute_vascular_pressure_field(nodes, edges, radius, inflow_node, outflow_nodes, P_in, P_out_base):
    """
    基于 Laplace 稳态假设与 Poiseuille 阻力网络计算节点压力场。

    对每条边 e=(i,j)，水力阻力:
        R_e = (8 μ L_e) / (π r_e⁴)

    构建 conductance 矩阵 G，其中 G_ii = Σ(1/R_e), G_ij = -1/R_e (若 i,j 相连)。
    求解线性系统 G P = b，其中 b 在 inflow 节点施加 P_in，在 outflow 节点施加 P_out_base。
    """
    # HOLE_2: 实现基于 Poiseuille 阻力网络的节点压力场求解
    # 1. 遍历每条边，计算水力阻力 R_e = 8μL_e / (π r_e⁴)
    # 2. 构建 conductance 矩阵 G (对角线累加，非对角线相减)
    # 3. 对 inflow/outflow 节点施加 Dirichlet 边界条件
    # 4. 正则化后求解线性系统 G P = b
    raise NotImplementedError("HOLE_2: 血管网络压力场求解待实现")
