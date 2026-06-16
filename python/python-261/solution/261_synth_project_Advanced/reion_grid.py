"""
reion_grid.py
=============
计算网格生成与坐标变换

本模块实现再电离模拟所需的各类计算网格:
  - 一维均匀共动网格 (主工作网格)
  - 对数稀疏网格 (源聚集区高分辨率)
  - 三维周期性立方网格 (用于 3D 效应分析)
  - 球对称径向网格 (用于单源电离气泡解析)

核心功能:
  1. build_uniform_grid : 构造一维均匀共动网格, 返回物理/共动坐标
  2. build_log_grid     : 对数稀疏网格, 源附近分辨率高
  3. build_3d_grid      : 三维周期性立方网格
  4. build_radial_grid  : 球对称径向网格
  5. comoving_proper_pair : 共动-物理坐标互转
  6. build_redshift_array : 生成时间/红移数组

对应种子项目:
  - 493_grids_display (多维网格显示思想 → 多维网格生成)
"""

import numpy as np
from reion_cosmology import Hubble_parameter


def build_uniform_grid(L_comoving, N_cells, z_ref=8.0, periodic=True):
    """构造一维均匀共动网格.

    Parameters
    ----------
    L_comoving : float
        模拟盒共动尺寸 [cm (共动)]
    N_cells : int
        网格单元数
    z_ref : float
        参考红移 (用于计算物理尺寸)
    periodic : bool
        是否为周期边界

    Returns
    -------
    grid_info : dict
        x_comoving : 节点共动坐标 [N+1]
        x_proper   : 节点物理坐标 [N+1]
        x_cell     : 单元中心共动坐标 [N]
        dx_comoving: 共动网格间距
        dx_proper  : 物理网格间距 (在 z_ref)
        N          : 网格单元数
        L_comoving : 共动盒尺寸
        z_ref      : 参考红移
        periodic   : 边界类型
    """
    if N_cells < 4:
        raise ValueError("N_cells 必须 >= 4, 当前: %d" % N_cells)
    if L_comoving <= 0:
        raise ValueError("L_comoving 必须 > 0")

    dx_comoving = L_comoving / N_cells
    # 节点坐标 (含两端)
    x_comoving = np.linspace(0.0, L_comoving, N_cells + 1)
    # 单元中心坐标 (周期性: 不含最右端)
    if periodic:
        x_cell = (np.arange(N_cells) + 0.5) * dx_comoving
    else:
        x_cell = x_comoving[:-1] + 0.5 * dx_comoving
    # 物理坐标
    scale = 1.0 + z_ref
    x_proper = x_comoving / scale
    dx_proper = dx_comoving / scale

    return {
        "x_comoving": x_comoving,
        "x_proper": x_proper,
        "x_cell": x_cell,
        "dx_comoving": dx_comoving,
        "dx_proper": dx_proper,
        "N": N_cells,
        "L_comoving": L_comoving,
        "z_ref": z_ref,
        "periodic": periodic,
    }


def build_log_grid(r_min, r_max, N_cells, z_ref=8.0, alpha=5.0):
    """构造对数稀疏网格, 源附近分辨率高.

    映射函数:
        x = r_min * exp(alpha * s / N),   s in [0, N]
    其中 alpha 控制疏密比.

    Parameters
    ----------
    r_min, r_max : float
        最小/最大物理半径 [cm]
    N_cells : int
        网格单元数
    z_ref : float
        参考红移
    alpha : float
        疏密参数 (alpha > 0 越大越集中)

    Returns
    -------
    grid_info : dict
    """
    if r_min <= 0 or r_max <= r_min:
        raise ValueError("需要 0 < r_min < r_max")
    if N_cells < 4:
        raise ValueError("N_cells 必须 >= 4")

    s = np.linspace(0.0, 1.0, N_cells + 1)
    # 对数映射
    r_proper = r_min * np.exp(alpha * s)
    # 归一化使 r_proper[-1] = r_max
    r_proper = r_proper * (r_max / r_proper[-1])
    # 单元中心
    r_cell = 0.5 * (r_proper[:-1] + r_proper[1:])
    # 间距
    dr_proper = np.diff(r_proper)
    # 共动坐标
    scale = 1.0 + z_ref
    r_comoving = r_proper * scale
    r_cell_comoving = r_cell * scale
    dx_comoving = np.diff(r_comoving)

    return {
        "r_proper": r_proper,
        "r_comoving": r_comoving,
        "r_cell": r_cell_comoving,
        "dx_comoving": dx_comoving,
        "N": N_cells,
        "r_min": r_min,
        "r_max": r_max,
        "z_ref": z_ref,
        "periodic": False,
    }


def build_3d_grid(L_comoving, N_per_dim, z_ref=8.0):
    """构造三维周期性立方网格.

    Parameters
    ----------
    L_comoving : float
        共动盒尺寸 [cm]
    N_per_dim : int
        每个维度的网格单元数
    z_ref : float
        参考红移

    Returns
    -------
    grid_info : dict
        X, Y, Z : 三维网格坐标数组 (cell centers)
        dx : 共动间距
        N_total : 总单元数 N_per_dim^3
    """
    if N_per_dim < 2:
        raise ValueError("N_per_dim 必须 >= 2")
    dx = L_comoving / N_per_dim
    x1d = (np.arange(N_per_dim) + 0.5) * dx
    X, Y, Z = np.meshgrid(x1d, x1d, x1d, indexing="ij")
    return {
        "X": X,
        "Y": Y,
        "Z": Z,
        "x1d": x1d,
        "dx": dx,
        "N_per_dim": N_per_dim,
        "N_total": N_per_dim**3,
        "L_comoving": L_comoving,
        "z_ref": z_ref,
    }


def build_radial_grid(R_max, N_cells, z_ref=8.0, origin_shift=1.0e20):
    """构造球对称径向网格 (避免 r=0 奇点).

    Parameters
    ----------
    R_max : float
        最大物理半径 [cm]
    N_cells : int
        径向网格数
    z_ref : float
        参考红移
    origin_shift : float
        原点偏移 (避免 r=0)

    Returns
    -------
    grid_info : dict
    """
    if R_max <= 0 or N_cells < 4:
        raise ValueError("参数不合理")
    r_proper = np.linspace(origin_shift, R_max, N_cells + 1)
    r_cell = 0.5 * (r_proper[:-1] + r_proper[1:])
    dr = np.diff(r_proper)
    # 球壳体积权重
    V_shell = (4.0 / 3.0) * np.pi * (r_proper[1:]**3 - r_proper[:-1]**3)
    scale = 1.0 + z_ref
    return {
        "r_proper": r_proper,
        "r_cell": r_cell,
        "dr": dr,
        "V_shell": V_shell,
        "N": N_cells,
        "z_ref": z_ref,
        "periodic": False,
        "r_comoving": r_proper * scale,
    }


def comoving_proper_pair(x_comoving, z):
    """共动 ↔ 物理坐标对 (向量化).

    Returns
    -------
    x_proper : array
    """
    return np.asarray(x_comoving) / (1.0 + np.asarray(z, dtype=float))


def build_redshift_array(z_start, z_end, N_steps):
    """生成等间隔红移数组 (从 z_start 降到 z_end, 对应时间推进).

    Parameters
    ----------
    z_start : float
        起始红移 (较大, 对应较早时刻)
    z_end : float
        结束红移
    N_steps : int
        步数

    Returns
    -------
    z_arr : array [N_steps + 1]
        红移数组 (降序)
    """
    if z_start <= z_end:
        raise ValueError("需要 z_start > z_end")
    if N_steps < 1:
        raise ValueError("N_steps 必须 >= 1")
    return np.linspace(z_start, z_end, N_steps + 1)


def grid_vonNeumann_wavenumbers(N, L_comoving):
    """生成 von Neumann 稳定性分析所需波数.

    对于周期边界 N 点网格, 离散波数为:
        k_m = 2 * pi * m / L,   m = -N/2, ..., N/2 - 1

    Returns
    -------
    k_arr : array [N]
        波数 [1/cm (共动)]
    """
    m_arr = np.fft.fftfreq(N, d=1.0 / N)
    return 2.0 * np.pi * m_arr / L_comoving
