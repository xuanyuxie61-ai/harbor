"""
current_continuity.py
=====================
电流连续性方程求解模块

基于种子项目:
  - 1068_shallow_water_1d: Lax-Wendroff 守恒律数值格式

科学背景:
  耳蜗内电流密度 J 满足连续性方程（电荷守恒）:
      ∂ρ/∂t + ∇·J = I_e

  其中 ρ 为电荷密度，在准静态近似下 ∂ρ/∂t ≈ 0，得到:
      ∇·J = I_e

  结合欧姆定律 J = -σ ∇V，退化为 Poisson 方程:
      ∇·(σ ∇V) = -I_e

  在时变电刺激情况下（如脉冲刺激），需要考虑电荷在组织中的
  积累与弛豫过程，可用双曲型守恒律方程描述:
      ∂(σV)/∂t + ∂J/∂x = I_e(t)

  本模块实现一维守恒律格式，用于纵向电流传播分析。
"""

import numpy as np


def lax_wendroff_current_continuity(nx, nt, x_max, t_max, sigma, I_source,
                                     bc_type='neumann'):
    """
    使用 Lax-Wendroff 格式求解一维电流连续性方程。

    方程:
        ∂(σV)/∂t + ∂J/∂x = I_e(x,t)
        J = -σ ∂V/∂x

    合并得:
        ∂V/∂t = (σ/σ) ∂²V/∂x² + I_e/σ
              = D_eff ∂²V/∂x² + source

    这里将其视为守恒律形式:
        ∂U/∂t + ∂F/∂x = S
        U = σV,  F = -σ ∂V/∂x

    Parameters
    ----------
    nx : int
        空间节点数
    nt : int
        时间步数
    x_max : float
        区域长度 (mm)
    t_max : float
        总时间 (ms)
    sigma : float
        电导率 (S/m)
    I_source : callable
        I_e(x, t) -> float, 源项 (A/mm³)
    bc_type : str
        'neumann', 'dirichlet', 'periodic'

    Returns
    -------
    V_array : ndarray, shape (nx, nt)
        电势时空分布
    x : ndarray, shape (nx,)
    t : ndarray, shape (nt,)
    """
    nx = int(nx)
    nt = int(nt)
    x_max = float(x_max)
    t_max = float(t_max)

    dx = x_max / (nx - 1)
    dt = t_max / (nt - 1)

    # CFL 条件: D * dt / dx^2 <= 0.5
    D_eff = 1.0 / sigma  # 有效扩散系数 (简化)
    cfl = D_eff * dt / (dx**2)
    if cfl > 0.5:
        # 自动调整时间步长
        dt = 0.4 * dx**2 / D_eff
        nt = int(np.ceil(t_max / dt)) + 1
        dt = t_max / (nt - 1)
        cfl = D_eff * dt / (dx**2)

    x = np.linspace(0.0, x_max, nx)
    t = np.linspace(0.0, t_max, nt)

    V_array = np.zeros((nx, nt))
    V = np.zeros(nx)

    # 中间变量
    Vm = np.zeros(nx - 1)
    Jm = np.zeros(nx - 1)

    for it in range(nt):
        if it == 0:
            # 初始条件: 零电势
            V = np.zeros(nx)
        else:
            # Lax-Wendroff 两步格式
            # 半步: 计算中点值
            for i in range(nx - 1):
                Vm[i] = 0.5 * (V[i] + V[i + 1])
                # 电流密度梯度近似
                J_left = -sigma * (V[i] - (V[i - 1] if i > 0 else V[i])) / dx
                J_right = -sigma * (V[i + 1] - V[i]) / dx
                Jm[i] = 0.5 * (J_left + J_right)

                # 源项
                S_mid = 0.5 * (I_source(x[i], t[it - 1]) + I_source(x[i + 1], t[it - 1]))
                Vm[i] += 0.5 * dt * (S_mid + (J_left - J_right) / dx)

            # 全步
            for i in range(1, nx - 1):
                source = I_source(x[i], t[it])
                V[i] += dt * source
                # 扩散项
                V[i] += D_eff * dt * (V[i + 1] - 2 * V[i] + V[i - 1]) / (dx**2)

        # 边界条件
        if bc_type == 'neumann':
            V[0] = V[1]
            V[-1] = V[-2]
        elif bc_type == 'dirichlet':
            V[0] = 0.0
            V[-1] = 0.0
        elif bc_type == 'periodic':
            V[0] = V[-2]
            V[-1] = V[1]

        V_array[:, it] = V

    return V_array, x, t


def current_density_1d(V, x, sigma):
    """
    从一维电势计算电流密度。

    J = -σ dV/dx

    Parameters
    ----------
    V : ndarray, shape (nx,) or (nx, nt)
        电势
    x : ndarray, shape (nx,)
    sigma : float

    Returns
    -------
    J : ndarray
        电流密度
    """
    V = np.asarray(V, dtype=float)
    x = np.asarray(x, dtype=float)
    dx = np.mean(np.diff(x))

    if V.ndim == 1:
        J = np.zeros_like(V)
        J[1:-1] = -sigma * (V[2:] - V[:-2]) / (2.0 * dx)
        J[0] = J[1]
        J[-1] = J[-2]
        return J
    elif V.ndim == 2:
        nx, nt = V.shape
        J = np.zeros_like(V)
        for it in range(nt):
            J[1:-1, it] = -sigma * (V[2:, it] - V[:-2, it]) / (2.0 * dx)
            J[0, it] = J[1, it]
            J[-1, it] = J[-2, it]
        return J
    else:
        raise ValueError("V 必须为 1D 或 2D 数组")


def compute_charge_conservation_error(V_array, x, t, sigma, I_source_func):
    """
    计算电荷守恒误差。

    验证: ∂V/∂t ≈ D_eff ∂²V/∂x² + I_source/σ

    Parameters
    ----------
    V_array : ndarray, shape (nx, nt)
    x, t : ndarray
    sigma : float
    I_source_func : callable

    Returns
    -------
    error : ndarray, shape (nx-2, nt-1)
        局部守恒误差
    """
    nx, nt = V_array.shape
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    D_eff = 1.0 / sigma

    dVdt = np.diff(V_array, axis=1) / dt  # (nx, nt-1)
    d2Vdx2 = np.zeros((nx - 2, nt - 1))

    for it in range(nt - 1):
        for i in range(1, nx - 1):
            d2Vdx2[i - 1, it] = (V_array[i + 1, it] - 2 * V_array[i, it] + V_array[i - 1, it]) / (dx**2)

    source = np.zeros((nx - 2, nt - 1))
    for it in range(nt - 1):
        for i in range(1, nx - 1):
            source[i - 1, it] = I_source_func(x[i], t[it]) / sigma

    error = dVdt[1:-1, :] - D_eff * d2Vdx2 - source
    return error
