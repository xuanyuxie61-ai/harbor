# -*- coding: utf-8 -*-
"""
time_marching.py
时间推进与 ODE 积分模块

融合来源:
- 472_glycolysis_ode: 非线性 ODE 系统的时间导数计算
- 860_pendulum_nonlinear_exact: 非线性系统的精确解与椭圆函数思想

功能:
- 提供 Adams-Bashforth、Runge-Kutta 4 阶等时间积分方案
- 求解 Navier-Stokes 方程的时间推进
- 非线性摆精确解的验证框架（用于验证时间积分器的精度）

数学背景:
  对于常微分方程 dy/dt = f(t, y)，

  RK4 方法:
    k1 = f(t_n, y_n)
    k2 = f(t_n + dt/2, y_n + dt*k1/2)
    k3 = f(t_n + dt/2, y_n + dt*k2/2)
    k4 = f(t_n + dt, y_n + dt*k3)
    y_{n+1} = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

  局部截断误差: O(dt^5)
  全局误差: O(dt^4)

  Adams-Bashforth 3 阶:
    y_{n+1} = y_n + dt/12 * (23*f_n - 16*f_{n-1} + 5*f_{n-2})

  对于不可压缩 NS 方程，采用分数步方法:
    1. 预测步: u* = u^n + dt * (-C(u^n) + nu*L(u^n) + f^n)
    2. 投影步: 求解压力 Poisson 方程得到 p^{n+1}
    3. 校正步: u^{n+1} = u* - dt * grad(p^{n+1}) / rho
"""

import numpy as np


def glycolysis_rhs(t, y, a=0.08, b=0.6):
    """
    Selkov 糖酵解模型的右端项。
    融合自 472_glycolysis_ode 的 glycolysis_deriv。

    化学模型:
      du/dt = -u + a*v + u^2*v
      dv/dt =  b - a*v - u^2*v

    参数:
      t: 时间
      y: [u, v] 浓度
      a, b: 反应参数

    返回:
      dydt: [du/dt, dv/dt]
    """
    u, v = y[0], y[1]
    dudt = -u + a * v + u ** 2 * v
    dvdt = b - a * v - u ** 2 * v
    return np.array([dudt, dvdt], dtype=float)


def pendulum_nonlinear_rhs(t, y, g=9.81, l=1.0):
    """
    非线性摆的右端项。
    融合自 860_pendulum_nonlinear_exact 的思想。

    物理模型:
      dtheta/dt = omega
      domega/dt = -(g/l) * sin(theta)

    参数:
      t: 时间
      y: [theta, omega]
      g: 重力加速度
      l: 摆长

    返回:
      dydt: [dtheta/dt, domega/dt]
    """
    theta, omega = y[0], y[1]
    dtheta = omega
    domega = -(g / l) * np.sin(theta)
    return np.array([dtheta, domega], dtype=float)


def pendulum_exact_solution(t, theta0=0.5, omega0=0.0, g=9.81, l=1.0):
    """
    非线性摆的近似精确解（使用小角度近似与椭圆积分的混合）。

    数学推导:
      对于小角度，周期 T = 2*pi*sqrt(l/g)。
      对于大角度，需使用椭圆积分:
        T = 4*sqrt(l/g) * K(k)
        其中 K(k) 为第一类完全椭圆积分，k = sin(theta0/2)。

    参数:
      t: 时间数组
      theta0: 初始角度
      omega0: 初始角速度

    返回:
      theta, omega: 精确解
    """
    k0 = np.sin(theta0 / 2.0)
    omega_freq = np.sqrt(g / l)
    ep = 4.0 * g / l
    e0 = omega0 ** 2 + ep * k0 ** 2
    k = np.sqrt(e0 / ep)

    # 小振幅修正
    if k < 1e-6:
        theta = theta0 * np.cos(omega_freq * t)
        omega = -theta0 * omega_freq * np.sin(omega_freq * t)
        return theta, omega

    # 使用近似公式
    chi = 1.0 / (k + 1e-15)
    sn_val = np.tanh(omega_freq * t * k)  # 近似椭圆正弦
    cn_val = 1.0 / np.cosh(omega_freq * t * k)

    theta = 2.0 * np.sign(cn_val) * np.arcsin(np.clip(np.abs(sn_val), 0, 1))
    omega = np.sign(omega0) * np.sqrt(e0) * cn_val

    return theta, omega


def rk4_step(f, t, y, dt, *args):
    """
    单步 RK4 积分。

    数学公式:
      k1 = f(t, y)
      k2 = f(t + dt/2, y + dt*k1/2)
      k3 = f(t + dt/2, y + dt*k2/2)
      k4 = f(t + dt, y + dt*k3)
      y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    """
    k1 = f(t, y, *args)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1, *args)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2, *args)
    k4 = f(t + dt, y + dt * k3, *args)
    y_new = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return y_new


def adams_bashforth_3_step(f, t, y, dt, history, *args):
    """
    三步 Adams-Bashforth 积分。

    数学公式:
      y_{n+1} = y_n + dt/12 * (23*f_n - 16*f_{n-1} + 5*f_{n-2})

    参数:
      history: 包含 [f_{n-2}, f_{n-1}, f_n] 的列表
    """
    if len(history) < 3:
        # 历史不足时退化为前向 Euler
        return y + dt * f(t, y, *args)

    fn = history[-1]
    fn1 = history[-2]
    fn2 = history[-3]
    y_new = y + (dt / 12.0) * (23.0 * fn - 16.0 * fn1 + 5.0 * fn2)
    return y_new


def fractional_step_ns_2d(u, v, p, dt, dx, dy, nu, forcing_u, forcing_v):
    """
    二维不可压缩 Navier-Stokes 方程的分数步时间推进。

    控制方程:
      du/dt + u*du/dx + v*du/dy = -dp/dx + nu*(d2u/dx2 + d2u/dy2) + f_u
      dv/dt + u*dv/dx + v*dv/dy = -dp/dy + nu*(d2v/dx2 + d2v/dy2) + f_v
      du/dx + dv/dy = 0

    参数:
      u, v: 当前速度场
      p: 当前压力场
      dt: 时间步长
      dx, dy: 空间网格间距
      nu: 运动粘度
      forcing_u, forcing_v: 体积力

    返回:
      u_new, v_new, p_new: 更新后的场
    """
    nx, ny = u.shape

    # 中心差分计算导数
    def ddx(f):
        result = np.zeros_like(f)
        result[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2.0 * dx)
        # 边界：单侧差分
        result[0, :] = (f[1, :] - f[0, :]) / dx
        result[-1, :] = (f[-1, :] - f[-2, :]) / dx
        return result

    def ddy(f):
        result = np.zeros_like(f)
        result[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2.0 * dy)
        result[:, 0] = (f[:, 1] - f[:, 0]) / dy
        result[:, -1] = (f[:, -1] - f[:, -2]) / dy
        return result

    def laplacian(f):
        result = np.zeros_like(f)
        result[1:-1, 1:-1] = (
            (f[2:, 1:-1] - 2 * f[1:-1, 1:-1] + f[:-2, 1:-1]) / dx ** 2
            + (f[1:-1, 2:] - 2 * f[1:-1, 1:-1] + f[1:-1, :-2]) / dy ** 2
        )
        return result

    # 1. 预测步: 求解中间速度 u*, v*（忽略压力梯度）
    conv_u = u * ddx(u) + v * ddy(u)
    conv_v = u * ddx(v) + v * ddy(v)

    u_star = u + dt * (-conv_u + nu * laplacian(u) + forcing_u)
    v_star = v + dt * (-conv_v + nu * laplacian(v) + forcing_v)

    # 2. 求解压力 Poisson 方程
    div_u_star = ddx(u_star) + ddy(v_star)

    # 简化的 Jacobi 迭代求解压力修正
    p_corr = np.zeros_like(p)
    for _ in range(50):
        p_new = np.zeros_like(p_corr)
        p_new[1:-1, 1:-1] = 0.25 * (
            p_corr[2:, 1:-1] + p_corr[:-2, 1:-1]
            + p_corr[1:-1, 2:] + p_corr[1:-1, :-2]
            - dx * dy * div_u_star[1:-1, 1:-1] / dt
        )
        p_corr = p_new

    # 3. 校正步
    dpdx = ddx(p_corr)
    dpdy = ddy(p_corr)

    u_new = u_star - dt * dpdx
    v_new = v_star - dt * dpdy
    p_new = p + p_corr

    return u_new, v_new, p_new


def fractional_step_ns_3d(u, v, w, p, dt, dx, dy, dz, nu,
                          forcing_u, forcing_v, forcing_w):
    """
    三维不可压缩 Navier-Stokes 方程的分数步时间推进。

    控制方程:
      du/dt + (u.grad)u = -grad(p)/rho + nu*Laplacian(u) + f
      div(u) = 0

    参数:
      u, v, w: 三个方向速度分量 (nx, ny, nz)
      p: 压力场
      dt: 时间步长
      dx, dy, dz: 网格间距
      nu: 运动粘度
      forcing_u, forcing_v, forcing_w: 体积力

    返回:
      u_new, v_new, w_new, p_new
    """
    def ddx(f):
        result = np.zeros_like(f)
        result[1:-1, :, :] = (f[2:, :, :] - f[:-2, :, :]) / (2.0 * dx)
        result[0, :, :] = (f[1, :, :] - f[0, :, :]) / dx
        result[-1, :, :] = (f[-1, :, :] - f[-2, :, :]) / dx
        return result

    def ddy(f):
        result = np.zeros_like(f)
        result[:, 1:-1, :] = (f[:, 2:, :] - f[:, :-2, :]) / (2.0 * dy)
        result[:, 0, :] = (f[:, 1, :] - f[:, 0, :]) / dy
        result[:, -1, :] = (f[:, -1, :] - f[:, -2, :]) / dy
        return result

    def ddz(f):
        result = np.zeros_like(f)
        result[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2.0 * dz)
        result[:, :, 0] = (f[:, :, 1] - f[:, :, 0]) / dz
        result[:, :, -1] = (f[:, :, -1] - f[:, :, -2]) / dz
        return result

    def laplacian(f):
        result = np.zeros_like(f)
        result[1:-1, 1:-1, 1:-1] = (
            (f[2:, 1:-1, 1:-1] - 2 * f[1:-1, 1:-1, 1:-1] + f[:-2, 1:-1, 1:-1]) / dx ** 2
            + (f[1:-1, 2:, 1:-1] - 2 * f[1:-1, 1:-1, 1:-1] + f[1:-1, :-2, 1:-1]) / dy ** 2
            + (f[1:-1, 1:-1, 2:] - 2 * f[1:-1, 1:-1, 1:-1] + f[1:-1, 1:-1, :-2]) / dz ** 2
        )
        return result

    # TODO(Hole 2): 实现三维不可压缩 Navier-Stokes 方程的分数步时间推进。
    # 要求:
    #   1. 预测步: 计算对流项 conv_u, conv_v, conv_w，然后计算中间速度 u*, v*, w*
    #      u* = u + dt * (-conv_u + nu * laplacian(u) + forcing_u)
    #   2. 压力 Poisson 方程: 求解 p' 使得 div(u*) - dt * nabla^2(p') = 0
    #      使用 Jacobi 迭代，离散 Laplacian 的系数需与三维网格匹配
    #   3. 校正步: u^{n+1} = u* - dt * grad(p')
    #   4. 注意 nu 参数的类型兼容性: main.py 中传入的 nu_eff 是标量均值，
    #      但 SGS 模型原始输出的是三维场。需要与 Hole 1 和 Hole 3 协同处理。
    raise NotImplementedError("Hole 2: Fractional step NS 3D time marching not implemented")
