"""
nonlinear_ode_solvers.py — 非线性 ODE 求解器与 Duffing 型拟合动力学
=================================================================
本模块将顶夸克质量拟合过程建模为非线性动力系统,
使用自适应步长 RK45 求解器追踪参数演化轨迹。

数学建模:
  将质量参数的梯度流视为连续时间 ODE:
    dm/dt = ∂lnL/∂m = f(m, t; α)

  对于含时滞的扩展系统:
    dm/dt = f(m, m(t-τ); α)

  简化为有限维:
    dm/dt = f(m, p; α)
    dp/dt = -γ p + β (m - m_target)

  类比 Duffing 振荡器:
    m'' + δ m' + α m + β m³ = γ cos(ωt)
  描述拟合参数在势能面中的运动:
    V(m) = α m²/2 + β m⁴/4 - γ m cos(ωt)

  Anishchenko 型开关动力学:
    dm/dt = μ m + p - m × z
    dp/dt = -m
    dz/dt = -η z + η H(m) m²
  用于拟合收敛的相空间分析。

  映射种子项目:
    - 322_duffing_ode: Duffing 振荡器 RK4 求解
    - 006_anishchenko_ode: Anishchenko 开关系统
"""

import numpy as np


def rk4_step(func, t, y, dt):
    """
    经典四阶 Runge-Kutta 单步。

    k₁ = f(t, y)
    k₂ = f(t + h/2, y + h k₁/2)
    k₃ = f(t + h/2, y + h k₂/2)
    k₄ = f(t + h, y + h k₃)
    y(t+h) = y(t) + (h/6)(k₁ + 2k₂ + 2k₃ + k₄)

    局部截断误差: O(h⁵)
    全局误差: O(h⁴)

    参数:
        func: 右端函数 f(t, y)
        t: 当前时间
        y: 当前状态
        dt: 步长

    返回:
        y_new: 更新后状态
    """
    k1 = func(t, y)
    k2 = func(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = func(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = func(t + dt, y + dt * k3)

    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk45_adaptive_step(func, t, y, dt, rtol=1e-6, atol=1e-9,
                       safety=0.9, max_factor=5.0, min_factor=0.2):
    """
    自适应步长 RK45 (Dormand-Prince) 单步。

    使用 4 阶和 5 阶方法的差作为误差估计:
      ε = ||y₅ - y₄|| / (atol + rtol × ||y||)

    步长调整:
      dt_new = dt × safety × (1/ε)^{1/5}
      限制: min_factor × dt ≤ dt_new ≤ max_factor × dt

    Dormand-Prince 系数 (Butcher tableau):
      c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
      a₂₁ = 1/5
      a₃₁ = 3/40, a₃₂ = 9/40
      ...

    参数:
        func: f(t, y)
        t: 当前时间
        y: 当前状态
        dt: 初始步长
        rtol, atol: 相对/绝对容差
        safety: 安全因子
        max_factor, min_factor: 步长调整限制

    返回:
        (y_new, dt_new, error, n_rejected)
    """
    y = np.asarray(y, dtype=float)
    rejected = 0

    while True:
        # Dormand-Prince stages
        k1 = func(t, y)
        k2 = func(t + dt / 5.0, y + dt * k1 / 5.0)
        k3 = func(t + 3.0 * dt / 10.0,
                  y + dt * (3.0 * k1 / 40.0 + 9.0 * k2 / 40.0))
        k4 = func(t + 4.0 * dt / 5.0,
                  y + dt * (44.0 * k1 / 45.0 - 56.0 * k2 / 15.0 + 32.0 * k3 / 9.0))
        k5 = func(t + 8.0 * dt / 9.0,
                  y + dt * (19372.0 * k1 / 6561.0 - 25360.0 * k2 / 2187.0 +
                            64448.0 * k3 / 6561.0 - 212.0 * k4 / 729.0))
        k6 = func(t + dt,
                  y + dt * (9017.0 * k1 / 3168.0 - 355.0 * k2 / 33.0 +
                            46732.0 * k3 / 5247.0 + 49.0 * k4 / 176.0 -
                            5103.0 * k5 / 18656.0))

        # 5 阶解
        y5 = y + dt * (35.0 * k1 / 384.0 + 500.0 * k3 / 1113.0 +
                       125.0 * k4 / 192.0 - 2187.0 * k5 / 6784.0 +
                       11.0 * k6 / 84.0)

        # 4 阶解 (误差估计)
        k7 = func(t + dt, y5)
        y4 = y + dt * (5179.0 * k1 / 57600.0 + 7571.0 * k3 / 16695.0 +
                       393.0 * k4 / 640.0 - 92097.0 * k5 / 339200.0 +
                       187.0 * k6 / 2100.0 + k7 / 40.0)

        # 误差估计
        error_vec = y5 - y4
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y5))
        error = np.sqrt(np.mean((error_vec / scale)**2))

        if error <= 1.0:
            # 接受步
            # 最优步长
            if error > 0:
                factor = safety * (1.0 / error)**0.2
            else:
                factor = max_factor
            factor = min(max_factor, max(min_factor, factor))
            dt_new = dt * factor
            return y5, dt_new, error, rejected
        else:
            # 拒绝步, 减小步长
            factor = safety * (1.0 / error)**0.25
            factor = max(min_factor, factor)
            dt = dt * factor
            rejected += 1
            if rejected > 20:
                return y5, dt, error, rejected


def duffing_fit_dynamics(t, y, params):
    """
    Duffing 型拟合动力学的右端函数。

    将拟合过程建模为阻尼受迫 Duffing 振荡器:
      y = [m, m'] (质量和质量变化率)
      m'' + δ m' + α(m - m_target) + β(m - m_target)³ = γ cos(ωt)

    这描述了拟合参数在有效势能面中的运动:
      V(m) = α(m-m₀)²/2 + β(m-m₀)⁴/4

    映射种子项目:
      - 322_duffing_ode: Duffing 系统

    参数:
        t: 时间 (迭代步)
        y: [m, m'] 状态
        params: 参数字典

    返回:
        dy/dt: [m', m'']
    """
    m, m_dot = y[0], y[1]

    delta = params.get('damping', 0.02)  # 阻尼
    alpha = params.get('linear_stiffness', 1.0)  # 线性刚度
    beta = params.get('nonlinear_stiffness', 0.1)  # 非线性刚度
    gamma = params.get('forcing_amplitude', 0.5)  # 强迫幅度
    omega = params.get('forcing_frequency', 0.3)  # 强迫频率
    m_target = params.get('target_mass', 172.5)  # 目标质量

    dm = m - m_target
    m_ddot = -delta * m_dot - alpha * dm - beta * dm**3 + gamma * np.cos(omega * t)

    return np.array([m_dot, m_ddot])


def anishchenko_fit_switch(t, y, params):
    """
    Anishchenko 型开关拟合动力学。

    dm/dt = μ m + p - m × z
    dp/dt = -m
    dz/dt = -η z + η H(m) m²

    其中 H(m) 是 Heaviside 函数,
    z 充当自适应"惯性"变量。

    映射种子项目:
      - 006_anishchenko_ode: Anishchenko 系统

    参数:
        t: 时间
        y: [m, p, z] 状态
        params: 参数字典

    返回:
        dy/dt
    """
    m, p, z = y[0], y[1], y[2]

    mu = params.get('mu', 1.2)
    eta = params.get('eta', 0.5)

    # Heaviside (平滑近似)
    H_m = 1.0 / (1.0 + np.exp(-20.0 * m))  # 平滑阶跃

    dm = mu * m + p - m * z
    dp = -m
    dz = -eta * z + eta * H_m * m**2

    return np.array([dm, dp, dz])


def integrate_fit_trajectory(ode_func, y0, t_span, params, rtol=1e-6):
    """
    积分拟合轨迹 (自适应 RK45)。

    参数:
        ode_func: 右端函数 (t, y) → dy/dt
        y0: 初始状态
        t_span: (t0, tf) 时间范围
        params: 参数
        rtol: 相对容差

    返回:
        result: 积分结果字典
    """
    t0, tf = t_span
    y = np.asarray(y0, dtype=float)
    t = t0

    dt = (tf - t0) / 100.0  # 初始步长
    dt_max = (tf - t0) / 10.0

    trajectory = [y.copy()]
    times = [t]

    def rhs(t_val, y_val):
        return ode_func(t_val, y_val, params)

    max_steps = 10000
    step_count = 0

    while t < tf and step_count < max_steps:
        dt = min(dt, dt_max, tf - t)

        y_new, dt_new, error, n_rejected = rk45_adaptive_step(
            rhs, t, y, dt, rtol=rtol
        )

        t += dt
        y = y_new
        dt = dt_new

        trajectory.append(y.copy())
        times.append(t)

        step_count += 1

    return {
        'times': np.array(times),
        'trajectory': np.array(trajectory),
        'n_steps': step_count,
        'final_state': y,
    }


def phase_space_analysis(ode_func, y_init, params, t_max=100.0):
    """
    拟合动力学的相空间分析。

    计算:
    1. 轨迹 (时间序列)
    2. 不动点
    3. 李雅普诺夫指数 (最大)

    参数:
        ode_func: 右端函数
        y_init: 初始状态
        params: 参数
        t_max: 最大时间

    返回:
        analysis: 分析结果字典
    """
    # 积分轨迹
    result = integrate_fit_trajectory(ode_func, y_init, (0, t_max), params)
    traj = result['trajectory']
    times = result['times']

    # 不动点检测 (最后 10% 的均值)
    n_tail = max(len(traj) // 10, 5)
    y_fixed = np.mean(traj[-n_tail:], axis=0)

    # 收敛检测
    if len(traj) > 10:
        diff = np.max(np.abs(traj[-1] - traj[-10]))
        converged = diff < 1e-4
    else:
        converged = False

    # 最大 Lyapunov 指数估计 ( Rosenstein 方法简化版)
    lyap = estimate_lyapunov(traj, times)

    return {
        'trajectory': traj,
        'times': times,
        'fixed_point': y_fixed,
        'converged': converged,
        'lyapunov_exponent': lyap,
        'n_steps': result['n_steps'],
    }


def estimate_lyapunov(trajectory, times):
    """
    估计最大 Lyapunov 指数 (简化方法)。

    对于嵌入空间中的邻近点对:
      d(t) ≈ d(0) e^{λt}
      λ ≈ (1/t) ln(d(t)/d(0))

    参数:
        trajectory: 状态时间序列
        times: 时间数组

    返回:
        最大 Lyapunov 指数估计
    """
    n = len(trajectory)
    if n < 20:
        return 0.0

    # 使用状态差的对数斜率
    dim = len(trajectory[0])
    d0 = np.linalg.norm(trajectory[n // 2] - trajectory[n // 2 + 1])

    if d0 < 1e-15:
        return 0.0

    # 取最后 1/3 的平均距离
    d_final = np.mean([np.linalg.norm(trajectory[-1] - trajectory[-i])
                       for i in range(2, min(10, n))])

    dt = times[-1] - times[n // 2]
    if dt > 0 and d_final > 0:
        lyap = np.log(max(d_final / d0, 1e-15)) / dt
    else:
        lyap = 0.0

    return lyap
