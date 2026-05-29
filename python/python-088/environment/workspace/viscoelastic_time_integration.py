"""
viscoelastic_time_integration.py
粘弹性时间积分与演化方程模块

融入种子项目:
  - 316_doughnut_exact: 非线性 ODE 精确解思想

功能:
  - 老化粘弹性核的指数积分法
  - 隐式时间积分（Backward Euler）
  - 隐式-显式 (IMEX) 分裂格式
  - 变时间步长控制
  - 长期蠕变预测的外推方法
"""

import numpy as np
from typing import Callable, Tuple, Optional


def exponential_integrator_linear(
    A: np.ndarray, b: np.ndarray, y0: np.ndarray,
    t_span: Tuple[float, float], n_steps: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    线性系统 dy/dt = A y + b 的指数积分法。

    精确解:
        y(t) = e^{A t} y_0 + \\\\int_0^t e^{A(t-s)} b \, ds

    离散形式（常数 b 假设）:
        y_{n+1} = e^{A \Delta t} y_n + A^{-1}(e^{A \Delta t} - I) b

    参数:
        A: 系统矩阵
        b: 常数源项
        y0: 初始条件
        t_span: (t0, tf)
        n_steps: 时间步数

    返回:
        (t_array, y_array)
    """
    t0, tf = t_span
    dt = (tf - t0) / n_steps
    t_array = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros((n_steps + 1, len(y0)))
    y[0] = y0

    # 矩阵指数
    from scipy.linalg import expm
    exp_Adt = expm(A * dt)

    # 计算 A^{-1}(e^{A dt} - I)
    try:
        A_inv = np.linalg.inv(A)
        phi = A_inv @ (exp_Adt - np.eye(len(y0)))
    except np.linalg.LinAlgError:
        phi = dt * np.eye(len(y0))  # 退化情况

    for n in range(n_steps):
        y[n + 1] = exp_Adt @ y[n] + phi @ b

    return t_array, y


def backward_euler_viscoelastic(
    M: np.ndarray, K: np.ndarray, F: np.ndarray,
    u0: np.ndarray, v0: np.ndarray,
    t_span: Tuple[float, float], n_steps: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    粘弹性问题的 Backward Euler 隐式积分。

    二阶系统:
        M \ddot{u} + K u = F

    转化为一阶系统:
        [ \dot{u} ]   [ 0       I  ] [ u ]   [ 0 ]
        [ \ddot{u} ] = [ -M^{-1}K  0 ] [ \dot{u} ] + [ M^{-1}F ]

    简化为拟静态（忽略惯性）：
        K u_{n+1} = F_{n+1}

    参数:
        M: 质量矩阵（可选）
        K: 刚度矩阵
        F: 载荷历史，形状 (n_steps+1, n_dof) 或常数
        u0: 初始位移
        v0: 初始速度
        t_span: 时间区间
        n_steps: 步数

    返回:
        (t_array, u_array, v_array)
    """
    t0, tf = t_span
    dt = (tf - t0) / n_steps
    t_array = np.linspace(t0, tf, n_steps + 1)
    n_dof = len(u0)

    u = np.zeros((n_steps + 1, n_dof))
    v = np.zeros((n_steps + 1, n_dof))
    u[0] = u0
    v[0] = v0

    # 假设载荷为常数或数组
    if F.ndim == 1:
        F_hist = np.tile(F, (n_steps + 1, 1))
    else:
        F_hist = F

    # 隐式积分
    for n in range(n_steps):
        # 拟静态: K u_{n+1} = F_{n+1}
        # 若考虑蠕变，K 为有效刚度
        try:
            u[n + 1] = np.linalg.solve(K, F_hist[n + 1])
        except np.linalg.LinAlgError:
            u[n + 1] = np.linalg.lstsq(K, F_hist[n + 1], rcond=None)[0]

        v[n + 1] = (u[n + 1] - u[n]) / dt

    return t_array, u, v


def imex_time_integration(
    f_explicit: Callable, f_implicit: Callable,
    y0: np.ndarray, t_span: Tuple[float, float], n_steps: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    隐式-显式 (IMEX) 分裂时间积分。

    对于 ODE:
        dy/dt = f_{impl}(y) + f_{expl}(y)

    采用以下格式:
        y_{n+1} = y_n + \Delta t [f_{impl}(y_{n+1}) + f_{expl}(y_n)]

    若 f_{impl} 为线性: f_{impl}(y) = A y，则:
        (I - \Delta t A) y_{n+1} = y_n + \Delta t f_{expl}(y_n)

    参数:
        f_explicit: 显式部分函数
        f_implicit: 隐式部分矩阵（线性时）或函数
        y0: 初始条件
        t_span: 时间区间
        n_steps: 步数

    返回:
        (t_array, y_array)
    """
    t0, tf = t_span
    dt = (tf - t0) / n_steps
    t_array = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros((n_steps + 1, len(y0)))
    y[0] = y0

    for n in range(n_steps):
        f_expl_n = f_explicit(y[n])

        if callable(f_implicit):
            # 非线性隐式部分，使用不动点迭代
            y_next = y[n] + dt * f_expl_n
            for _ in range(10):
                y_next_new = y[n] + dt * (f_implicit(y_next) + f_expl_n)
                if np.linalg.norm(y_next_new - y_next) < 1e-12:
                    break
                y_next = y_next_new
            y[n + 1] = y_next
        else:
            # 线性隐式部分: f_implicit(y) = A y
            A = f_implicit
            lhs = np.eye(len(y0)) - dt * A
            rhs = y[n] + dt * f_expl_n
            try:
                y[n + 1] = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                y[n + 1] = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    return t_array, y


def adaptive_time_stepping(
    f: Callable, y0: np.ndarray, t_span: Tuple[float, float],
    dt_init: float, tol: float = 1e-6, dt_min: float = 1e-6, dt_max: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    自适应时间步长控制（基于局部截断误差估计）。

    使用 Euler 和 RK2 的差值作为误差估计:
        y_{n+1}^{(E)} = y_n + \Delta t f(y_n)
        y_{n+1}^{(RK2)} = y_n + \\frac{\Delta t}{2} [f(y_n) + f(y_n + \Delta t f(y_n))]
        e = \|y_{n+1}^{(RK2)} - y_{n+1}^{(E)}\|

    步长调整:
        \Delta t_{new} = \Delta t \\cdot \\min\\\left(2, \\max\\\left(0.5, 0.9 \\sqrt{\\frac{tol}{e}}\\right)\\right)

    参数:
        f: 右端项函数
        y0: 初始条件
        t_span: 时间区间
        dt_init: 初始步长
        tol: 误差容差
        dt_min: 最小步长
        dt_max: 最大步长

    返回:
        (t_array, y_array)
    """
    t0, tf = t_span
    t = t0
    y = y0.copy().astype(float)
    dt = dt_init

    t_list = [t]
    y_list = [y.copy()]

    while t < tf:
        dt = min(dt, tf - t)

        # Euler 步
        k1 = f(y)
        y_euler = y + dt * k1

        # RK2 步
        k2 = f(y + dt * k1)
        y_rk2 = y + 0.5 * dt * (k1 + k2)

        # 误差估计
        e = np.linalg.norm(y_rk2 - y_euler)

        if e < tol or dt <= dt_min:
            # 接受步
            y = y_rk2
            t += dt
            t_list.append(t)
            y_list.append(y.copy())

            # 增大步长
            if e > 0:
                dt = min(dt_max, dt * min(2.0, max(0.5, 0.9 * np.sqrt(tol / e))))
            else:
                dt = min(dt_max, dt * 2.0)
        else:
            # 拒绝步，减小步长
            dt = max(dt_min, dt * max(0.5, 0.9 * np.sqrt(tol / e)))

    return np.array(t_list), np.array(y_list)


def hereditary_integral_discrete(
    kernel: Callable, f_history: np.ndarray,
    t_history: np.ndarray
) -> np.ndarray:
    """
    离散遗传积分（卷积型）。

    对于遗传积分:
        y(t) = \\\\int_0^t K(t - s) f(s) \, ds

    离散形式（梯形法则）:
        y_n = \\\sum_{i=0}^{n-1} \\frac{1}{2} [K(t_n - t_i) f_i + K(t_n - t_{i+1}) f_{i+1}] \Delta t_i

    参数:
        kernel: 核函数 K(tau)
        f_history: f 的历史值
        t_history: 时间点

    返回:
        y 的历史值
    """
    n = len(t_history)
    y = np.zeros(n)

    for j in range(n):
        integral = 0.0
        for i in range(j):
            dt_i = t_history[i + 1] - t_history[i]
            k1 = kernel(t_history[j] - t_history[i])
            k2 = kernel(t_history[j] - t_history[i + 1])
            integral += 0.5 * dt_i * (k1 * f_history[i] + k2 * f_history[i + 1])
        y[j] = integral

    return y


def power_law_creep_kernel(
    tau: float, E0: float, n_power: float, A_c: float
) -> float:
    """
    幂律蠕变核函数（常用于混凝土长期蠕变）。

        K(\\tau) = \\frac{A_c}{E_0} \\tau^{-n}

    其中 n \\approx 0.1 \\sim 0.3 为蠕变指数。

    参数:
        tau: 时间差
        E0: 参考弹性模量
        n_power: 幂指数
        A_c: 幅值系数

    返回:
        核函数值
    """
    if tau <= 0:
        return 0.0
    return (A_c / E0) * (tau ** (-n_power))


def log_double_power_law(
    t: float, t_prime: float, q1: float, q2: float, m: float, n: float
) -> float:
    """
    对数双幂律蠕变函数（Bazant 提出）。

        J(t, t') = q_1 + q_2 \\ln\\\left[1 + \\\left(\\frac{t - t'}{\\lambda_0}\\right)^m\\right]
                    + q_3 \\ln\\\left(\\frac{t}{t'}\\right)^n

    参数:
        t: 观测时间
        t_prime: 加载龄期
        q1, q2: 幅值参数
        m, n: 幂指数

    返回:
        柔量
    """
    if t <= t_prime:
        return q1
    dt = t - t_prime
    lambda0 = 1.0
    J = q1 + q2 * np.log1p((dt / lambda0) ** m)
    if t / t_prime > 1.0:
        J += q2 * 0.5 * (np.log(t / t_prime) ** n)
    return J


def viscoelastic_relaxation_spectrum(
    E_t: Callable, times: np.ndarray, n_maxwell: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    通过 Prony 级数拟合松弛模量，提取 Maxwell 链参数。

    目标:
        E(t) \\approx E_\\\infty + \\\sum_{i=1}^N E_i e^{-t/\\tau_i}

    使用对数等距松弛时间和最小二乘拟合。

    参数:
        E_t: 松弛模量函数
        times: 时间数组
        n_maxwell: Maxwell 单元数

    返回:
        (E_inf, E_i_array, tau_i_array)
    """
    E_vals = np.array([E_t(t) for t in times])

    # 对数等距松弛时间
    t_min, t_max = times[0], times[-1]
    tau_i = np.logspace(np.log10(t_min + 1e-6), np.log10(t_max), n_maxwell)

    # 构造设计矩阵
    A = np.zeros((len(times), n_maxwell))
    for i, tau in enumerate(tau_i):
        A[:, i] = np.exp(-times / tau)

    # 最小二乘拟合
    E_inf = E_vals[-1]
    b = E_vals - E_inf

    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    E_i = np.maximum(coeffs, 0)  # 保证非负

    return E_inf, E_i, tau_i


def effective_time_for_aging_creep(
    t: float, t_prime: float, alpha_h: float = 1.0
) -> float:
    """
    老化混凝土的有效时间（基于水化度修正）。

    有效时间概念:
        t_{eff} = \\\\int_{t'}^t \\alpha_h(s) \, ds

    对于常数水化度:
        t_{eff} = \\alpha_h (t - t')

    参数:
        t: 观测时间
        t_prime: 加载龄期
        alpha_h: 水化度相关因子

    返回:
        有效时间
    """
    return alpha_h * (t - t_prime)
