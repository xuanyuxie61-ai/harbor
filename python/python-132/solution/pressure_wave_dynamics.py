"""
pressure_wave_dynamics.py
=========================
精馏塔内压力波动传播模拟模块。

本模块基于一维波动方程有限差分法（源自项目 366_fd1d_wave），
模拟塔内压力扰动的传播过程。

科学背景
--------
精馏塔操作过程中，压力波动是常见的不稳定因素。一维波动方程描述
压力波在塔内轴向的传播：

    c^2 ∂²P/∂z² = ∂²P/∂t²

其中 c 为声速在气相混合物中的有效传播速度 [m/s]。

有限差分离散（中心差分）：
    P_{j}^{n+1} = 2(1 - α²) P_j^n + α²(P_{j+1}^n + P_{j-1}^n) - P_j^{n-1}

稳定性条件（CFL条件）：
    |α| = |c Δt / Δz| ≤ 1

初始条件：
    P(0,z) = P_0(z)
    ∂P/∂t(0,z) = 0

边界条件（Dirichlet）：
    P(t,0) = P_bottom(t)
    P(t,L) = P_top(t)
"""

import numpy as np
from utils import clip_with_warning


def fd1d_wave_solve(z_num, z1, z2, t_num, t1, t2, c, P_x1, P_x2, P_t1, Pt_t1):
    """
    一维波动方程有限差分求解（源自项目 366_fd1d_wave）。

    Parameters
    ----------
    z_num : int
        空间网格数。
    z1, z2 : float
        空间区间 [m]。
    t_num : int
        时间步数。
    t1, t2 : float
        时间区间 [s]。
    c : float
        波速 [m/s]。
    P_x1, P_x2 : callable
        边界函数 P(t) [Pa]。
    P_t1 : callable
        初始压力分布 P(z) [Pa]。
    Pt_t1 : callable
        初始压力时间导数 dP/dt(z) [Pa/s]。

    Returns
    -------
    P : ndarray, shape (t_num+1, z_num+1)
        压力场 [Pa]。
    alpha : float
        CFL 数。
    """
    if z_num < 1:
        z_num = 1
    if t_num < 1:
        t_num = 1

    t_delta = (t2 - t1) / t_num
    z_delta = (z2 - z1) / z_num
    alpha = c * t_delta / z_delta

    if abs(alpha) > 1.0:
        print(f"[WARN] fd1d_wave: CFL condition |alpha|={abs(alpha):.4f} > 1 violated.")
        # 自动调整时间步以满足稳定性
        t_delta = z_delta / abs(c) * 0.95
        t_num = int(np.ceil((t2 - t1) / t_delta))
        t_delta = (t2 - t1) / t_num
        alpha = c * t_delta / z_delta
        print(f"       Adjusted to t_num={t_num}, alpha={abs(alpha):.4f}")

    P = np.zeros((t_num + 1, z_num + 1), dtype=float)

    # 边界条件
    for n in range(t_num + 1):
        t = t1 + n * t_delta
        P[n, 0] = P_x1(t)
        P[n, z_num] = P_x2(t)

    # 初始条件
    z_grid = np.linspace(z1, z2, z_num + 1)
    P[0, :] = P_t1(z_grid)
    Pt0 = Pt_t1(z_grid)

    # 第一步使用初始导数信息
    for j in range(1, z_num):
        P[1, j] = (
            0.5 * alpha ** 2 * P[0, j + 1]
            + (1.0 - alpha ** 2) * P[0, j]
            + 0.5 * alpha ** 2 * P[0, j - 1]
            + t_delta * Pt0[j]
        )

    # 后续时间步
    for n in range(1, t_num):
        for j in range(1, z_num):
            P[n + 1, j] = (
                2.0 * (1.0 - alpha ** 2) * P[n, j]
                + alpha ** 2 * (P[n, j + 1] + P[n, j - 1])
                - P[n - 1, j]
            )

    return P, alpha


def pressure_wave_in_column(column_height, c_sound, P_bottom, P_top, P_initial,
                            disturbance_z, disturbance_amp, t_end, nz=50, nt=200):
    """
    模拟精馏塔内压力波动的传播。

    Parameters
    ----------
    column_height : float
        塔高 [m]。
    c_sound : float
        气相声速 [m/s]。
    P_bottom, P_top : float
        底部与顶部稳态压力 [Pa]。
    P_initial : float
        初始均匀压力 [Pa]。
    disturbance_z : float
        扰动位置 [m]。
    disturbance_amp : float
        扰动幅值 [Pa]。
    t_end : float
        模拟结束时间 [s]。
    nz, nt : int
        空间与时间离散数。

    Returns
    -------
    P_field : ndarray
        压力场。
    z_grid : ndarray
        空间网格。
    t_grid : ndarray
        时间网格。
    alpha : float
        CFL 数。
    """
    # 边界条件函数
    def P_x1(t):
        return P_bottom

    def P_x2(t):
        return P_top

    # 初始条件：含扰动
    def P_t1(z):
        z = np.asarray(z, dtype=float)
        sigma = column_height * 0.05
        dist = np.exp(-0.5 * ((z - disturbance_z) / sigma) ** 2)
        return P_initial + disturbance_amp * dist

    def Pt_t1(z):
        z = np.asarray(z, dtype=float)
        return np.zeros_like(z)

    P_field, alpha = fd1d_wave_solve(
        nz, 0.0, column_height, nt, 0.0, t_end, c_sound,
        P_x1, P_x2, P_t1, Pt_t1
    )

    z_grid = np.linspace(0.0, column_height, nz + 1)
    t_grid = np.linspace(0.0, t_end, nt + 1)

    return P_field, z_grid, t_grid, alpha


def pressure_stability_index(P_field):
    """
    计算压力稳定性指标：压力随时间的方差变化率。

    Returns
    -------
    stability_index : float
        稳定性指标（越小越稳定）。
    """
    nt = P_field.shape[0]
    var_t = np.var(P_field, axis=1)
    if nt > 1:
        dvar = np.abs(var_t[-1] - var_t[0]) / (nt - 1)
    else:
        dvar = 0.0
    return float(dvar)
