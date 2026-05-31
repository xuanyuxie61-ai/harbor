"""
tridiagonal_acoustics.py
一维管道声学模态分析与三对角求解器

融合原始项目:
  - 1355_tridiagonal_solver (Thomas算法)
  - 674_lindberg_exact (刚性ODE精确解与残差)

科学背景:
  在管道主动噪声控制中,声压满足一维Helmholtz方程:
      d^2p/dx^2 + k^2 p = -j rho_0 omega q(x)
  其中 k = omega/c_0 为波数, q(x) 为体积速度源.

  使用中心差分离散后得到三对角线性系统 Ax = d.
  本模块同时利用 Lindberg 刚性ODE测试函数验证数值积分器
  在声学瞬态分析中的稳定性.
"""

import numpy as np
import math


def tridiagonal_solver(a, b, c, d):
    """
    Thomas算法求解三对角线性系统.

    系统形式:
        b[0]*x[0] + c[0]*x[1] = d[0]
        a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i], 1 <= i < n-1
        a[n-1]*x[n-2] + b[n-1]*x[n-1] = d[n-1]

    参数:
        a: 下对角线, a[0] 未使用
        b: 主对角线
        c: 上对角线, c[-1] 未使用
        d: 右端项 (numpy数组)

    返回:
        x: 解向量
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    d = np.asarray(d, dtype=float)
    n = b.size

    if d.ndim == 1:
        d = d.reshape((n, 1))

    # 前向消元
    for i in range(1, n):
        if b[i - 1] == 0.0:
            raise ValueError(f"tridiagonal_solver: zero pivot at b[{i - 1}]")
        s = a[i] / b[i - 1]
        b[i] = b[i] - s * c[i - 1]
        d[i, :] = d[i, :] - s * d[i - 1, :]

    x = np.empty_like(d, dtype=float)
    # 回代
    for i in range(n - 1, -1, -1):
        if abs(b[i]) < 1e-15:
            raise ValueError(f"tridiagonal_solver: near-zero pivot at b[{i}]")
        if i == n - 1:
            x[i, :] = d[i, :] / b[i]
        else:
            x[i, :] = (d[i, :] - c[i] * x[i + 1, :]) / b[i]

    if x.shape[1] == 1:
        return x[:, 0]
    return x


def tridiagonal_mv(a, b, c, x):
    """
    三对角矩阵与向量乘积.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    rhs = np.zeros(n, dtype=float)
    rhs[:] = b * x
    rhs[1:] = rhs[1:] + a[1:] * x[:-1]
    rhs[:-1] = rhs[:-1] + c[:-1] * x[1:]
    return rhs


def pipe_helmholtz_solver(L, N, k, source_profile, rho0=1.225, c0=343.0):
    """
    求解一维管道Helmholtz方程的离散系统.

    方程:
        p''(x) + k^2 p(x) = -j * rho0 * omega * q(x)

    边界条件:
        p(0) = 0 (刚性端), p(L) = 0 (刚性端) => 两端封闭管

    离散化:
        h = L / (N+1)
        (p_{i-1} - 2p_i + p_{i+1})/h^2 + k^2 p_i = f_i
        => p_{i-1} + (k^2 h^2 - 2) p_i + p_{i+1} = h^2 f_i

    参数:
        L: 管道长度 [m]
        N: 内部离散点数
        k: 波数 [rad/m]
        source_profile: 长度为N的源分布数组
        rho0: 空气密度 [kg/m^3]
        c0: 声速 [m/s]

    返回:
        x: 空间坐标数组
        p: 复声压数组
    """
    if N <= 0:
        raise ValueError("N must be positive")
    h = L / (N + 1)
    x = np.linspace(h, L - h, N)

    omega = k * c0
    f = -1j * rho0 * omega * np.asarray(source_profile, dtype=complex)

    # 三对角系数
    a = np.ones(N, dtype=float)
    b = np.full(N, (k * h) ** 2 - 2.0, dtype=float)
    c = np.ones(N, dtype=float)
    d = (h ** 2) * f

    # 边界条件处理: 两端 p=0 意味着在离散方程中 ghost point=0,
    # 已自然体现在第一行和最后一行只有2个非零元.
    # 但Thomas算法需要 a[0]=c[-1]=0
    a[0] = 0.0
    c[-1] = 0.0

    # 分别求解实部和虚部
    p_real = tridiagonal_solver(a.copy(), b.copy(), c.copy(), d.real)
    p_imag = tridiagonal_solver(a.copy(), b.copy(), c.copy(), d.imag)
    p = p_real + 1j * p_imag
    return x, p


def lindberg_exact_solution(t):
    """
    Lindberg刚性ODE的精确解.

    系统:
        y1' = 1e4 * y1 * y3 + 1e4 * y2 * y4
        y2' = -1e4 * y1 * y4 + 1e4 * y2 * y3
        y3' = 1 - y3
        y4' = -0.5*y3 - y4 + 0.5

    精确解:
        g1 = 1e4*(t + 2*exp(-t) - 2)
        g2 = 1e4*(1 - exp(-t) - t*exp(-t))
        y1 = exp(g1)*(cos(g2) + sin(g2))
        y2 = exp(g1)*(cos(g2) - sin(g2))
        y3 = 1 - 2*exp(-t)
        y4 = t*exp(-t)

    声学意义:
        该ODE可视为简化的声学模态耦合系统,
        用于测试数值积分器在处理声学瞬态问题时的稳定性.
    """
    t = np.asarray(t, dtype=float)
    n = t.size
    y = np.zeros((n, 4), dtype=float)
    dydt = np.zeros((n, 4), dtype=float)

    g1 = 1.0e4 * (t + 2.0 * np.exp(-t) - 2.0)
    g2 = 1.0e4 * (1.0 - np.exp(-t) - t * np.exp(-t))

    dg1dt = 1.0e4 * (1.0 - 2.0 * np.exp(-t))
    dg2dt = 1.0e4 * (t * np.exp(-t))

    eg1 = np.exp(g1)
    cg2 = np.cos(g2)
    sg2 = np.sin(g2)

    y[:, 0] = eg1 * (cg2 + sg2)
    y[:, 1] = eg1 * (cg2 - sg2)
    y[:, 2] = 1.0 - 2.0 * np.exp(-t)
    y[:, 3] = t * np.exp(-t)

    dydt[:, 0] = eg1 * dg1dt * (cg2 + sg2) + eg1 * (-sg2 + cg2) * dg2dt
    dydt[:, 1] = eg1 * dg1dt * (cg2 - sg2) + eg1 * (-sg2 - cg2) * dg2dt
    dydt[:, 2] = 2.0 * np.exp(-t)
    dydt[:, 3] = (1.0 - t) * np.exp(-t)

    return y, dydt


def lindberg_residual(t, y, dydt):
    """
    计算Lindberg ODE的残差,用于验证数值解的精度.
    """
    t = np.asarray(t)
    y = np.asarray(y)
    dydt = np.asarray(dydt)
    n = t.size
    r = np.zeros((n, 4), dtype=float)
    r[:, 0] = dydt[:, 0] - (1.0e4 * y[:, 0] * y[:, 2] + 1.0e4 * y[:, 1] * y[:, 3])
    r[:, 1] = dydt[:, 1] - (-1.0e4 * y[:, 0] * y[:, 3] + 1.0e4 * y[:, 1] * y[:, 2])
    r[:, 2] = dydt[:, 2] - (1.0 - y[:, 2])
    r[:, 3] = dydt[:, 3] - (-0.5 * y[:, 2] - y[:, 3] + 0.5)
    return r
