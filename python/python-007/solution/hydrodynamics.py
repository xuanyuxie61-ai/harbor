"""
流体力学时间积分模块
整合自：
  - 1034_rk3（三阶Runge-Kutta ODE求解器）
  - 213_contour_gradient_3d（梯度计算）

在吸积盘模拟中用于：
  1. 使用RK3显式积分器推进流体力学方程
  2. 计算压力梯度和粘性应力
  3. 追踪密度、动量和能量的时间演化
"""
import numpy as np


# ===========================
# RK3 Time Integrator
# ===========================

def rk3_step(f_func, t, y, dt):
    """
    单步三阶Runge-Kutta积分。

    Butcher 表（经典 RK3）：
        k1 = dt * f(t, y)
        k2 = dt * f(t + dt, y + k1)
        k3 = dt * f(t + dt/2, y + k1/4 + k2/4)
        y_{n+1} = y_n + (k1 + k2 + 4*k3) / 6

    参数:
        f_func: 右端函数 f(t, y)
        t: 当前时间
        y: 当前状态向量
        dt: 时间步长

    返回:
        y_new: 下一时刻状态
    """
    y = np.asarray(y, dtype=np.float64)

    k1 = dt * np.asarray(f_func(t, y), dtype=np.float64)
    k2 = dt * np.asarray(f_func(t + dt, y + k1), dtype=np.float64)
    k3 = dt * np.asarray(f_func(t + 0.5 * dt, y + 0.25 * k1 + 0.25 * k2), dtype=np.float64)

    y_new = y + (k1 + k2 + 4.0 * k3) / 6.0
    return y_new


def rk3_integrate(f_func, t_span, y0, n_steps):
    """
    使用固定步长RK3进行时间积分。

    参数:
        f_func: 右端函数 f(t, y)
        t_span: [t_start, t_end]
        y0: 初始状态
        n_steps: 步数

    返回:
        t_array: 时间序列
        y_array: 状态序列 (n_steps+1, len(y0))
    """
    y0 = np.asarray(y0, dtype=np.float64)
    t0, tf = t_span
    dt = (tf - t0) / n_steps

    t_array = np.zeros(n_steps + 1, dtype=np.float64)
    y_array = np.zeros((n_steps + 1, len(y0)), dtype=np.float64)

    t_array[0] = t0
    y_array[0] = y0

    for k in range(n_steps):
        y_array[k + 1] = rk3_step(f_func, t_array[k], y_array[k], dt)
        t_array[k + 1] = t_array[k] + dt

    return t_array, y_array


# ===========================
# Gradient Computation
# ===========================

def gradient_2d(field, dx, dy):
    """
    计算2D标量场的数值梯度。

    采用中心差分：
        df/dx(i,j) = [f(i+1,j) - f(i-1,j)] / (2*dx)
        df/dy(i,j) = [f(i,j+1) - f(i,j-1)] / (2*dy)

    边界采用一阶前差/后差。

    参数:
        field: (nx, ny) 标量场
        dx, dy: 网格步长

    返回:
        grad_x, grad_y: 两个方向的梯度分量
    """
    field = np.asarray(field, dtype=np.float64)
    nx, ny = field.shape

    grad_x = np.zeros_like(field)
    grad_y = np.zeros_like(field)

    # 内部：中心差分
    if nx > 2:
        grad_x[1:-1, :] = (field[2:, :] - field[:-2, :]) / (2.0 * dx)
    # 边界：前差/后差
    if nx > 1:
        grad_x[0, :] = (field[1, :] - field[0, :]) / dx
        grad_x[-1, :] = (field[-1, :] - field[-2, :]) / dx

    if ny > 2:
        grad_y[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2.0 * dy)
    if ny > 1:
        grad_y[:, 0] = (field[:, 1] - field[:, 0]) / dy
        grad_y[:, -1] = (field[:, -1] - field[:, -2]) / dy

    return grad_x, grad_y


def gradient_1d(field, dx):
    """1D梯度"""
    field = np.asarray(field, dtype=np.float64)
    n = len(field)
    grad = np.zeros_like(field)

    if n > 2:
        grad[1:-1] = (field[2:] - field[:-2]) / (2.0 * dx)
    if n > 1:
        grad[0] = (field[1] - field[0]) / dx
        grad[-1] = (field[-1] - field[-2]) / dx

    return grad


def laplacian_1d(field, dx):
    """
    1D拉普拉斯算子（二阶导数）的离散形式：
        d^2f/dx^2(i) = [f(i+1) - 2f(i) + f(i-1)] / dx^2
    """
    field = np.asarray(field, dtype=np.float64)
    n = len(field)
    lap = np.zeros_like(field)

    if n > 2:
        lap[1:-1] = (field[2:] - 2.0 * field[1:-1] + field[:-2]) / (dx * dx)
    if n > 1:
        # 边界一阶近似
        lap[0] = (field[1] - field[0]) / (dx * dx)
        lap[-1] = (field[-1] - field[-2]) / (dx * dx)

    return lap


def divergence_cylindrical(v_r, v_phi, v_z, r_grid, dr, dz):
    """
    柱坐标系下的散度：
        div(v) = (1/r) * d(r*v_r)/dr + (1/r) * dv_phi/dphi + dv_z/dz

    在轴对称假设下（d/dphi = 0）：
        div(v) = (1/r) * d(r*v_r)/dr + dv_z/dz

    参数:
        v_r, v_phi, v_z: (nr, nz) 速度分量
        r_grid: 径向坐标数组
        dr, dz: 步长

    返回:
        div: (nr, nz) 散度场
    """
    v_r = np.asarray(v_r, dtype=np.float64)
    v_z = np.asarray(v_z, dtype=np.float64)
    nr, nz = v_r.shape

    # dv_z/dz
    dv_z_dz = np.zeros_like(v_z)
    if nz > 2:
        dv_z_dz[:, 1:-1] = (v_z[:, 2:] - v_z[:, :-2]) / (2.0 * dz)
    if nz > 1:
        dv_z_dz[:, 0] = (v_z[:, 1] - v_z[:, 0]) / dz
        dv_z_dz[:, -1] = (v_z[:, -1] - v_z[:, -2]) / dz

    # d(r*v_r)/dr
    rvr = r_grid.reshape(-1, 1) * v_r
    d_rvr_dr = np.zeros_like(v_r)
    if nr > 2:
        d_rvr_dr[1:-1, :] = (rvr[2:, :] - rvr[:-2, :]) / (2.0 * dr)
    if nr > 1:
        d_rvr_dr[0, :] = (rvr[1, :] - rvr[0, :]) / dr
        d_rvr_dr[-1, :] = (rvr[-1, :] - rvr[-2, :]) / dr

    # 避免 r=0 除法
    div = np.zeros_like(v_r)
    for i in range(nr):
        r = r_grid[i]
        if r > 1e-15:
            div[i, :] = d_rvr_dr[i, :] / r + dv_z_dz[i, :]
        else:
            div[i, :] = dv_z_dz[i, :]

    return div


def compute_cfl_timestep(v_r, v_phi, v_z, cs, dr, dz, r_grid, cfl=0.3):
    """
    基于 CFL 条件计算允许的最大时间步长。

    CFL 条件：
        dt <= CFL * min(dx / (|v| + c_s))

    其中 c_s 为声速，对于吸积盘：
        c_s = sqrt(gamma * P / rho)

    参数:
        v_r, v_phi, v_z: 速度分量
        cs: 声速场 (nr, nz)
        dr, dz: 空间步长
        r_grid: 径向坐标
        cfl: CFL 数

    返回:
        dt: 最大时间步长
    """
    v_r = np.asarray(v_r, dtype=np.float64)
    v_z = np.asarray(v_z, dtype=np.float64)
    cs = np.asarray(cs, dtype=np.float64)

    # 局部速度模
    v_mag = np.sqrt(v_r ** 2 + v_phi ** 2 + v_z ** 2)

    # 有效波速
    v_eff = v_mag + cs
    v_eff = np.where(v_eff < 1e-15, 1e-15, v_eff)

    # 考虑柱坐标的最小尺度
    dt_r = np.zeros_like(v_eff)
    for i in range(len(r_grid)):
        dt_r[i, :] = dr / v_eff[i, :]

    dt_z = dz / v_eff

    dt_min = min(np.min(dt_r), np.min(dt_z))
    return cfl * dt_min
