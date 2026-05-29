"""
exact_benchmarks.py
===================
精确解验证与外部驱动模块（融合 646_laplace_radial_exact + 1059_sawtooth_ode）

功能：
- 2D/3D拉普拉斯方程的径向精确解及其导数
- 锯齿波ODE驱动系统（周期性激励）
- 用于验证数值解的正确性和收敛性

数学公式：
- 2D Laplace径向解: u(r) = a log(r) + b
  ∇²u = 0,  r = √(x²+y²)
  u_x = a x / r²,  u_xx = a (r² - 2x²) / r⁴
- 锯齿波驱动ODE:
  y₁' = y₂
  y₂' = -y₁ + saw(ω t)
  saw(t) = 2*(t/T - floor(t/T + 0.5))
"""

import numpy as np


def laplace_radial_2d_exact(x, y, a, b):
    """
    2D拉普拉斯方程的径向精确解。
    u(x,y) = a * log(r) + b
    
    返回:
        u, ux, uy, uxx, uxy, uyy
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r2 = x ** 2 + y ** 2
    r2 = np.clip(r2, 1e-15, None)
    r = np.sqrt(r2)
    
    u = a * np.log(r) + b
    ux = a * x / r2
    uy = a * y / r2
    uxx = a * (r2 - 2 * x ** 2) / (r2 ** 2)
    uxy = -2 * a * x * y / (r2 ** 2)
    uyy = a * (r2 - 2 * y ** 2) / (r2 ** 2)
    
    return u, ux, uy, uxx, uxy, uyy


def laplace_radial_3d_exact(x, y, z, a, b):
    """
    3D拉普拉斯方程的径向精确解。
    u(r) = a / r + b
    """
    r2 = x ** 2 + y ** 2 + z ** 2
    r2 = np.clip(r2, 1e-15, None)
    r = np.sqrt(r2)
    
    u = a / r + b
    ux = -a * x / (r2 ** 1.5)
    uy = -a * y / (r2 ** 1.5)
    uz = -a * z / (r2 ** 1.5)
    return u, ux, uy, uz


def sawtooth_wave(t, omega=2.0 * np.pi, amplitude=1.0):
    """
    周期锯齿波函数。
    saw(t) = amplitude * (2 * frac(ωt/(2π)) - 1)
    其中 frac 是小数部分。
    """
    phase = omega * t / (2.0 * np.pi)
    frac = phase - np.floor(phase)
    return amplitude * (2.0 * frac - 1.0)


def sawtooth_ode_rhs(t, y, omega=2.0 * np.pi):
    """
    锯齿波驱动ODE的右端项。
    y = [u, v]
    du/dt = v
    dv/dt = -u + saw(ω t)
    """
    u, v = y[0], y[1]
    dudt = v
    dvdt = -u + sawtooth_wave(t, omega)
    return np.array([dudt, dvdt])


def solve_sawtooth_ode(t_span, y0, omega=2.0 * np.pi, n_steps=1000):
    """
    使用显式Runge-Kutta 4阶方法求解锯齿波驱动ODE。
    """
    t0, tf = t_span
    dt = (tf - t0) / n_steps
    t = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros((n_steps + 1, 2))
    y[0] = y0
    
    for i in range(n_steps):
        k1 = sawtooth_ode_rhs(t[i], y[i], omega)
        k2 = sawtooth_ode_rhs(t[i] + 0.5 * dt, y[i] + 0.5 * dt * k1, omega)
        k3 = sawtooth_ode_rhs(t[i] + 0.5 * dt, y[i] + 0.5 * dt * k2, omega)
        k4 = sawtooth_ode_rhs(t[i] + dt, y[i] + dt * k3, omega)
        y[i + 1] = y[i] + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    
    return t, y


def compute_l2_error(u_num, u_exact, area):
    """
    在三角网格上计算L²误差: ||u_num - u_exact||₂ = √(∫ |e|² dx)
    近似: √(Σ_e |e_k|² * A_k)
    """
    diff = u_num - u_exact
    err_sq = np.sum(diff ** 2 * area)
    return np.sqrt(err_sq)
