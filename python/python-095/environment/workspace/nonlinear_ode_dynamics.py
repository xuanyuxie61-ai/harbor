"""
nonlinear_ode_dynamics.py
自适应ANC系统的非线性动力学与稳定性分析

融合原始项目:
  - 006_anishchenko_ode (非线性ODE系统)
  - 674_lindberg_exact (刚性ODE精确解)

科学背景:
  自适应滤波器系数在非平稳环境中的演化可建模为非线性ODE.
  对于简化双通道系统,权重动力学方程为:

      dw1/dt = mu [ e(t) x1(t) - gamma w1(t) ]
      dw2/dt = mu [ e(t) x2(t) - gamma w2(t) ]
      de/dt  = -lambda e(t) + eta w1(t) w2(t) + d(t)

  其中 e(t) 为残余误差, d(t) 为非平稳初级噪声.
  该耦合系统可能产生分岔、混沌等复杂动力学行为,
  类似于Anishchenko振荡器.

  本模块提供:
  1. Anishchenko-like自适应系统ODE
  2. 刚性ODE积分器验证 (Lindberg测试)
  3. 自适应系统稳定性边界分析
"""

import numpy as np
import math


def anishchenko_adaptive_deriv(t, state, mu=1.2, eta=0.5, gamma_leak=0.01):
    """
    基于Anishchenko模型的自适应系统ODE导数.

    状态变量:
        state[0] = w1: 第一个权重 (对应Anishchenko x)
        state[1] = w2: 第二个权重 (对应Anishchenko y)
        state[2] = e: 误差信号    (对应Anishchenko z)

    方程:
        dw1/dt = mu * w1 + w2 - w1 * e - gamma_leak * w1
        dw2/dt = -w1 - gamma_leak * w2
        de/dt  = -eta * e + eta * (w1 >= 0) * w1^2

    物理意义:
        前两个方程描述权重的耦合振荡,
        第三个方程描述误差能量的非线性耗散与再生.
    """
    w1, w2, e = state
    dw1 = mu * w1 + w2 - w1 * e - gamma_leak * w1
    dw2 = -w1 - gamma_leak * w2
    de = -eta * e + eta * (1.0 if w1 >= 0.0 else 0.0) * (w1 ** 2)
    return np.array([dw1, dw2, de], dtype=float)


def rk4_integrate(f, t0, y0, tstop, h=0.01):
    """
    四阶Runge-Kutta数值积分器.

    对于ODE: dy/dt = f(t, y)
    RK4格式:
        k1 = h f(t_n, y_n)
        k2 = h f(t_n + h/2, y_n + k1/2)
        k3 = h f(t_n + h/2, y_n + k2/2)
        k4 = h f(t_n + h, y_n + k3)
        y_{n+1} = y_n + (k1 + 2k2 + 2k3 + k4)/6

    稳定性区域: h*lambda 位于 RK4 稳定域内,
    对于刚性系统需采用隐式方法.本函数用于非刚性自适应动力学.
    """
    y = np.asarray(y0, dtype=float)
    t = t0
    trajectory = [(t, y.copy())]

    while t < tstop:
        if t + h > tstop:
            h = tstop - t
        k1 = h * f(t, y)
        k2 = h * f(t + h / 2.0, y + k1 / 2.0)
        k3 = h * f(t + h / 2.0, y + k2 / 2.0)
        k4 = h * f(t + h, y + k3)
        y = y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        t = t + h
        trajectory.append((t, y.copy()))

    return trajectory


def adaptive_lms_ode(t, W, R, P, mu, gamma_reg=0.0):
    """
    连续时间LMS算法的ODE近似 (均值分析).

    对于权重向量 W,参考信号自相关矩阵 R 和互相关向量 P:
        dW/dt = mu (P - R W) - gamma_reg W

    稳态解:
        W* = (R + (gamma_reg/mu) I)^{-1} P

    时间常数:
        tau_i = 1 / (mu lambda_i)
        lambda_i 为 R 的特征值.
    """
    W = np.asarray(W, dtype=float)
    R = np.asarray(R, dtype=float)
    P = np.asarray(P, dtype=float)
    dW = mu * (P - R @ W) - gamma_reg * W
    return dW


def stability_boundary_anishchenko(mu_range, gamma_range, n_grid=50):
    """
    分析Anishchenko-like自适应系统的稳定性边界.

    线性化Jacobian在(w1,w2,e)=(0,0,0)处:
        J = [[mu-gamma, 1, 0],
             [-1, -gamma, 0],
             [0, 0, -eta]]

    固定 eta=0.5, 分析 mu (增益) 与 gamma (泄漏) 的稳定性.
    稳定条件: Re(lambda_i) < 0.
    对于 2x2 子矩阵,特征值满足:
        lambda^2 + (2*gamma - mu)*lambda + (gamma^2 - mu*gamma + 1) = 0
    稳定条件:
        2*gamma - mu > 0  =>  gamma > mu/2
        gamma^2 - mu*gamma + 1 > 0

    参数:
        mu_range: (mu_min, mu_max)
        gamma_range: (gamma_min, gamma_max)
        n_grid: 网格点数

    返回:
        mu_grid, gamma_grid, stable_mask
    """
    mu_grid = np.linspace(mu_range[0], mu_range[1], n_grid)
    gamma_grid = np.linspace(gamma_range[0], gamma_range[1], n_grid)
    stable_mask = np.zeros((n_grid, n_grid), dtype=bool)

    eta = 0.5
    for i, mu in enumerate(mu_grid):
        for j, gamma in enumerate(gamma_grid):
            J = np.array([[mu - gamma, 1.0, 0.0],
                          [-1.0, -gamma, 0.0],
                          [0.0, 0.0, -eta]])
            eigs = np.linalg.eigvals(J)
            stable_mask[i, j] = np.all(np.real(eigs) < 0)

    return mu_grid, gamma_grid, stable_mask
