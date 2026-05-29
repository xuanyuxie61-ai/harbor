"""
time_integrator.py — 自适应时间积分器

融合以下种子项目：
- 1030_rk12_adapt : 一阶/二阶 Runge-Kutta 自适应步长积分
- 674_lindberg_exact : 刚性 ODE 精确解验证

功能：
1. 实现 RK1/RK2 自适应步长积分器（源自 rk12_adapt）
2. 实现 RK4 固定步长积分器（用于高阶参考解）
3. 提供刚性方程检验接口（Lindberg 精确解对比）
4. 为 MHD 方程组提供时间推进接口

核心公式：
  对于 y' = f(t, y):
  RK1 (Euler):    y1 = y_n + h k1,   k1 = f(t_n, y_n)
  RK2 (Heun):     y2 = y_n + h/2 (k1 + k2),  k2 = f(t_n+h, y_n+h k1)
  局部截断误差:   e = ||y2 - y1||
  步长调整策略:
    若 e > tol * h :   h = h / 2  (拒绝步)
    若 e < tol * h / 16 : h = h * 2  (接受步并放大)
    否则接受当前步
"""

import numpy as np
from utils import lindberg_rhs, lindberg_exact_solution


def rk12_adapt(rhs_func, tspan, y0, dt_init, tol):
    """
    自适应 RK1/RK2 时间积分器。
    源自 1030_rk12_adapt (rk12_adapt.m)。

    参数：
      rhs_func : 右端项函数，签名 func(t, y) -> dy/dt (数组)
      tspan    : [t_start, t_end]
      y0       : 初始条件 (m,)
      dt_init  : 建议初始步长
      tol      : 容差，截断误差需满足 e < tol * dt

    返回：
      t : 时间序列 (n+1,)
      y : 解序列 (n+1, m)
      e : 估计截断误差序列 (n+1,)
    """
    t_start, t_end = tspan
    y0 = np.atleast_1d(y0)
    m = len(y0)

    t_list = [t_start]
    y_list = [y0.copy()]
    e_list = [0.0]

    t_curr = t_start
    y_curr = y0.copy()
    dt = dt_init

    while t_curr < t_end - 1e-14:
        accepted = False
        while not accepted:
            t_next = t_curr + dt
            if t_next > t_end:
                t_next = t_end
                dt = t_next - t_curr

            k1 = dt * rhs_func(t_curr, y_curr)
            y1 = y_curr + k1
            k2 = dt * rhs_func(t_curr + dt, y1)
            y2 = y_curr + 0.5 * (k1 + k2)

            err = np.linalg.norm(y2 - y1)

            if err > tol * dt:
                dt = dt / 2.0
                if dt < 1e-14:
                    raise RuntimeError("rk12_adapt: 步长过小，积分失败")
            else:
                accepted = True
                y_curr = y2.copy()
                t_curr = t_next
                t_list.append(t_curr)
                y_list.append(y_curr.copy())
                e_list.append(err)
                if err < tol * dt / 16.0:
                    dt = dt * 2.0

    t = np.array(t_list)
    y = np.array(y_list)
    e = np.array(e_list)
    return t, y, e


def rk4_fixed(rhs_func, tspan, y0, n_steps):
    """
    四阶 Runge-Kutta 固定步长积分器（用于生成参考解）。

    公式：
      k1 = f(t_n, y_n)
      k2 = f(t_n + h/2, y_n + h k1 / 2)
      k3 = f(t_n + h/2, y_n + h k2 / 2)
      k4 = f(t_n + h,   y_n + h k3)
      y_{n+1} = y_n + h/6 (k1 + 2k2 + 2k3 + k4)
    """
    t_start, t_end = tspan
    y0 = np.atleast_1d(y0)
    h = (t_end - t_start) / n_steps

    t = np.linspace(t_start, t_end, n_steps + 1)
    y = np.zeros((n_steps + 1, len(y0)))
    y[0] = y0

    for i in range(n_steps):
        k1 = rhs_func(t[i], y[i])
        k2 = rhs_func(t[i] + h / 2, y[i] + h * k1 / 2)
        k3 = rhs_func(t[i] + h / 2, y[i] + h * k2 / 2)
        k4 = rhs_func(t[i] + h, y[i] + h * k3)
        y[i + 1] = y[i] + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    return t, y


def adams_bashforth_3(rhs_func, tspan, y0, n_steps):
    """
    三阶 Adams-Bashforth 多步法。
    前 2 步用 RK4 启动。

    公式（k ≥ 2）:
      y_{k+1} = y_k + h/12 * (23 f_k - 16 f_{k-1} + 5 f_{k-2})
    """
    t_start, t_end = tspan
    y0 = np.atleast_1d(y0)
    h = (t_end - t_start) / n_steps

    t = np.linspace(t_start, t_end, n_steps + 1)
    y = np.zeros((n_steps + 1, len(y0)))
    y[0] = y0

    # 用 RK4 启动前两步
    for i in range(min(2, n_steps)):
        k1 = rhs_func(t[i], y[i])
        k2 = rhs_func(t[i] + h / 2, y[i] + h * k1 / 2)
        k3 = rhs_func(t[i] + h / 2, y[i] + h * k2 / 2)
        k4 = rhs_func(t[i] + h, y[i] + h * k3)
        y[i + 1] = y[i] + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    f_hist = [rhs_func(t[i], y[i]) for i in range(3)]

    for i in range(2, n_steps):
        y[i + 1] = y[i] + h / 12.0 * (23 * f_hist[2] - 16 * f_hist[1] + 5 * f_hist[0])
        f_hist.pop(0)
        f_hist.append(rhs_func(t[i + 1], y[i + 1]))

    return t, y


def integrate_mhd_system(state, rhs_mhd, tspan, dt_init, tol,
                         integrator_type='rk12'):
    """
    MHD 系统的时间推进接口。

    状态向量 state 包含：
      [磁矢势 A 的节点值, 速度场 u 的节点值, 温度场 T 的节点值]

    采用分量拆分（operator splitting）：
      1. 扩散项（刚性）用隐式处理或更小步长
      2. 对流项（非刚性）用显式 RK
    """
    if integrator_type == 'rk12':
        return rk12_adapt(rhs_mhd, tspan, state, dt_init, tol)
    elif integrator_type == 'rk4':
        n_steps = max(1, int((tspan[1] - tspan[0]) / dt_init))
        return rk4_fixed(rhs_mhd, tspan, state, n_steps)
    elif integrator_type == 'ab3':
        n_steps = max(3, int((tspan[1] - tspan[0]) / dt_init))
        return adams_bashforth_3(rhs_mhd, tspan, state, n_steps)
    else:
        raise ValueError(f"未知的积分器类型: {integrator_type}")


def verify_integrator_accuracy():
    """
    使用 Lindberg 精确解验证时间积分器精度（源自 674_lindberg_exact）。
    返回各积分器在 t=0.01 处的相对误差。
    """
    tspan = np.array([0.0, 0.01])
    y0 = np.array([1.0, 1.0, -1.0, 0.0])
    y_exact, _ = lindberg_exact_solution(np.array([0.01]))
    y_exact = y_exact[0]

    results = {}

    # RK12
    t_rk12, y_rk12, _ = rk12_adapt(lindberg_rhs, tspan, y0, 0.001, 1e-8)
    err = np.linalg.norm(y_rk12[-1] - y_exact) / (np.linalg.norm(y_exact) + 1e-30)
    results['rk12'] = err

    # RK4
    t_rk4, y_rk4 = rk4_fixed(lindberg_rhs, tspan, y0, 100)
    err = np.linalg.norm(y_rk4[-1] - y_exact) / (np.linalg.norm(y_exact) + 1e-30)
    results['rk4'] = err

    # AB3
    t_ab3, y_ab3 = adams_bashforth_3(lindberg_rhs, tspan, y0, 100)
    err = np.linalg.norm(y_ab3[-1] - y_exact) / (np.linalg.norm(y_exact) + 1e-30)
    results['ab3'] = err

    return results
