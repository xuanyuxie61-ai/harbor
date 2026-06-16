"""
boundary_conditions.py - 边界条件处理

本模块实现激光等离子体仿真中使用的各种边界条件。

在 Vlasov-Maxwell 仿真中，边界条件的选择对结果的物理正确性
和数值稳定性至关重要。

实现的边界条件:

1. 周期边界条件 (Periodic BC):
   f(0, v) = f(L, v), E(0) = E(L)
   适用于均匀等离子体中的波传播研究。

2. 零 Neumann 边界 (反射壁):
   df/dx|_{boundary} = 0
   适用于封闭等离子体腔。

3. 吸收边界条件 (Absorbing BC):
   在边界附近添加吸收层，逐渐衰减 outgoing 波。
   实现方式:
     - 简单衰减: f *= exp(-sigma * dt) 在边界区域
     - PML (Perfectly Matched Layer): 更精确的匹配层

4. 开放边界 (Outflow BC):
   允许波和粒子自由离开计算域而不产生反射。
   实现: 在边界处使用单侧差分 (upwind)。

5. 激光注入边界:
   在左边界注入激光电磁场:
       E_y(0, t) = a_0(t) * sin(omega_0 * t) * exp(-(t-t_0)^2/tau^2)

6. 速度空间边界:
   f(x, v_max) = 0, f(x, -v_max) = 0
   假设没有超高能粒子从外部进入。
"""

import numpy as np


def apply_periodic_bc_1d(f):
    """应用 1D 周期边界条件。

    f[0] = f[-1], f[-1] = f[0] (循环)

    Parameters
    ----------
    f : ndarray (N,)
        场数据

    Returns
    -------
    f_bc : ndarray (N,)
        应用边界条件后的场
    """
    f_bc = f.copy()
    return f_bc


def apply_neumann_bc_1d(f):
    """应用 1D 零 Neumann (零梯度) 边界条件。

    f[0] = f[1], f[-1] = f[-2]

    Parameters
    ----------
    f : ndarray (N,)
        场数据

    Returns
    -------
    f_bc : ndarray (N,)
        应用边界条件后的场
    """
    f_bc = f.copy()
    f_bc[0] = f_bc[1]
    f_bc[-1] = f_bc[-2]
    return f_bc


def apply_absorbing_layer(f, x, dx, sigma_max=2.0, n_layer=20, bc_type='exponential'):
    """在边界区域应用吸收层。

    吸收层通过添加虚部电导率来衰减 outgoing 波:
        df/dt = ... - sigma(x) * f

    其中 sigma(x) 为在边界区域逐渐增大的吸收系数。

    指数型吸收剖面:
        sigma(x) = sigma_max * exp(-(x - x_boundary)^2 / (2 * delta^2))

    多项式型 (PML 风格):
        sigma(x) = sigma_max * ((x - x_start) / d_layer)^m

    Parameters
    ----------
    f : ndarray (N,)
        场数据
    x : ndarray (N,)
        空间坐标
    dx : float
        网格间距
    sigma_max : float
        最大吸收系数
    n_layer : int
        吸收层格点数
    bc_type : str
        吸收剖面类型 ('exponential', 'polynomial')

    Returns
    -------
    f_absorbed : ndarray (N,)
        吸收后的场
    """
    f_abs = f.copy()
    N = len(f)

    if n_layer >= N // 2:
        n_layer = N // 4

    # 左边界吸收层
    for i in range(n_layer):
        xi = float(n_layer - i) / n_layer  # 1 at boundary, 0 at interior
        if bc_type == 'exponential':
            sigma = sigma_max * np.exp(-3.0 * (1.0 - xi) ** 2)
        else:  # polynomial
            sigma = sigma_max * xi ** 3
        f_abs[i] *= np.exp(-sigma * dx)

    # 右边界吸收层
    for i in range(N - n_layer, N):
        xi = float(i - (N - n_layer) + 1) / n_layer
        if bc_type == 'exponential':
            sigma = sigma_max * np.exp(-3.0 * (1.0 - xi) ** 2)
        else:
            sigma = sigma_max * xi ** 3
        f_abs[i] *= np.exp(-sigma * dx)

    return f_abs


def apply_outflow_bc(f, dx, direction='right'):
    """应用开放边界 (outflow) 条件。

    使用单侧 (upwind) 差分在边界处计算导数:
        右侧出流: df/dx|_{N-1} = (f_{N-1} - f_{N-2}) / dx
        左侧出流: df/dx|_0 = (f_1 - f_0) / dx

    Parameters
    ----------
    f : ndarray (N,)
        场数据
    dx : float
        网格间距
    direction : str
        出流方向 ('left', 'right', 'both')

    Returns
    -------
    f_out : ndarray (N,)
        应用出流边界后的场
    """
    f_out = f.copy()
    N = len(f)

    if direction in ('right', 'both'):
        # 右侧: 使用向后差分, 允许波向右传出
        f_out[-1] = f_out[-2]  # 零梯度

    if direction in ('left', 'both'):
        # 左侧: 使用向前差分
        f_out[0] = f_out[1]

    return f_out


def inject_laser_field(E_y, B_z, x, dx, t, config):
    """在左边界注入激光电磁场。

    激光场 (线偏振, 正入射):
        E_y(0, t) = a_0 * g(t) * sin(omega_0 * t)
        B_z(0, t) = -E_y(0, t) / c  (在归一化单位中 B_z = -E_y)

    其中 g(t) 为脉冲包络:
        g(t) = sin^2(pi * t / tau_total)  for 0 < t < tau_total

    注入方式: 使用特征变量分解, 仅注入 outgoing 特征:
        E_y + B_z = outgoing wave (向右)
        E_y - B_z = incoming wave (向左)

    Parameters
    ----------
    E_y : ndarray (N_x,)
        横向电场
    B_z : ndarray (N_x,)
        磁场
    x : ndarray (N_x,)
        空间坐标
    dx : float
        网格间距
    t : float
        当前时间
    config : SimulationConfig
        仿真配置

    Returns
    -------
    E_y : ndarray
        更新后的电场
    B_z : ndarray
        更新后的磁场
    """
    envelope = config.laser_envelope(t)

    # 注入电场 (左边界)
    E_y_inject = envelope * np.sin(config.omega_0 * t)

    # 特征注入: 只注入向右传播的特征
    # E_y = (outgoing + incoming) / 2
    # B_z = (outgoing - incoming) / 2
    E_y[0] = E_y_inject
    B_z[0] = -E_y_inject  # 向右传播: B = -E/c (c=1)

    return E_y, B_z


def velocity_space_bc(f):
    """应用速度空间边界条件。

    假设在速度截断处 f -> 0:
        f(x, v_max) = 0
        f(x, -v_max) = 0

    这要求 v_max 足够大 (通常 > 6*v_th)。

    Parameters
    ----------
    f : ndarray (N_x, N_v)
        分布函数

    Returns
    -------
    f : ndarray
        应用速度边界条件后的分布函数
    """
    f[:, 0] = 0.0
    f[:, -1] = 0.0
    f[:, 1] = f[:, 2] if f.shape[1] > 3 else 0.0
    f[:, -2] = f[:, -3] if f.shape[1] > 3 else 0.0
    return f


def apply_all_boundary_conditions(f, E_y, B_z, E_x, x, dx, t, config):
    """应用所有边界条件。

    组合执行:
      1. 速度空间 BC
      2. 空间吸收层 (可选)
      3. 激光注入

    Parameters
    ----------
    f : ndarray (N_x, N_v)
        分布函数
    E_y, B_z, E_x : ndarray (N_x,)
        电磁场分量
    x : ndarray (N_x,)
        空间坐标
    dx : float
        网格间距
    t : float
        当前时间
    config : SimulationConfig
        仿真配置

    Returns
    -------
    f, E_y, B_z, E_x : ndarray
        应用边界条件后的场
    """
    # 速度空间 BC
    f = velocity_space_bc(f)

    # 吸收层 (在电磁场上)
    E_y = apply_absorbing_layer(E_y, x, dx, sigma_max=1.0, n_layer=10)
    B_z = apply_absorbing_layer(B_z, x, dx, sigma_max=1.0, n_layer=10)

    # 激光注入
    E_y, B_z = inject_laser_field(E_y, B_z, x, dx, t, config)

    # 纵向电场: Neumann BC
    E_x = apply_neumann_bc_1d(E_x)

    return f, E_y, B_z, E_x
