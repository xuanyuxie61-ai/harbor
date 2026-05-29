"""
ice_stream_dynamics.py
冰流非线性振荡动力学 — Duffing 型滞滑循环模型

将经典 Duffing 振子改造为冰流速度的非线性振荡方程，
描述冰流 (ice stream) 在基底水文耦合下的周期性 surging 行为。

物理模型:
  考虑冰流速度 u(t) 受基底有效压力 N(t) 控制的非线性方程:

      m \frac{du}{dt} = F_{drive} - F_{drag} - F_{damping}

  其中:
      F_{drive} = \tau_d = \rho g H \sin\alpha       (重力驱动力)
      F_{drag}  = \tau_b = \frac{C N^n u}{|u| + u_0}   (Weertman 型滑移律)
      F_{damping} = \beta u                             (线性阻尼)

  基底有效压力 N 受排水系统演化控制:

      \frac{dN}{dt} = a - b u - c N^3 + d \cos(\omega_{seas} t)

  其中 a, b, c, d 为排水参数，\omega_{seas} 为季节性强迫频率。

  这是一个广义的 Duffing 型系统:
      u' = v
      v' = -\delta v - \alpha u - \beta u^3 + \gamma \cos(\omega t) + f(N)
      N' = a - b u - c N^3 + d \cos(\omega_{seas} t)

数值方法:
  - 四阶 Runge-Kutta 时间积分
  - 事件检测: 检测速度反转点 (stick-slip 转变)
  - 长期统计: 平均流速、振荡周期、振幅分布
"""

import numpy as np
from typing import Tuple, Optional

from ice_constitutive_model import ICE_DENSITY, GRAVITY


def ice_stream_rhs(state: np.ndarray,
                   t: float,
                   params: dict) -> np.ndarray:
    """
    计算冰流-水文耦合系统的右端项。

    状态向量:  Y = [u, v, N]^T

    参数 (params):
        'H': 冰厚度 (m)
        'alpha': 表面坡度 (rad)
        'delta': 阻尼系数
        'alpha_u': 线性恢复系数
        'beta_u': 非线性 (Duffing) 系数
        'gamma': 周期性强迫振幅
        'omega': 强迫频率 (rad/s)
        'C_weertman': Weertman 滑移律系数
        'u0': 滑移律特征速度
        'a_drain': 排水源项
        'b_drain': 流速对排水的耦合
        'c_drain': 排水非线性耗散
        'd_season': 季节性排水强迫
        'omega_season': 季节频率
        'm_eff': 有效质量
    """
    u, v, N = state

    H = params.get('H', 1000.0)
    alpha = params.get('alpha', 0.001)
    delta = params.get('delta', 0.1)
    alpha_u = params.get('alpha_u', 0.01)
    beta_u = params.get('beta_u', 1e-8)
    gamma = params.get('gamma', 0.5)
    omega = params.get('omega', 2.0 * np.pi / (365.25 * 86400.0))
    C_weertman = params.get('C_weertman', 1e-4)
    u0 = params.get('u0', 1.0)
    a_drain = params.get('a_drain', 0.1)
    b_drain = params.get('b_drain', 1e-6)
    c_drain = params.get('c_drain', 1e-10)
    d_season = params.get('d_season', 0.05)
    omega_season = params.get('omega_season', 2.0 * np.pi / (365.25 * 86400.0))
    m_eff = params.get('m_eff', 1.0)

    # 驱动力
    tau_drive = ICE_DENSITY * GRAVITY * H * np.sin(alpha)

    # 滑移阻力 (Weertman 型, 正则化)
    denom = np.abs(u) + u0
    tau_drag = C_weertman * (N ** 3) * u / denom

    # 季节性强迫
    forcing = gamma * np.cos(omega * t)
    season_forcing = d_season * np.cos(omega_season * t)

    # Duffing 非线性项 (带溢出保护)
    u_clipped = np.clip(u, -1e4, 1e4)
    N_clipped = np.clip(N, 1.0, 1e7)
    duffing = -alpha_u * u_clipped - beta_u * (u_clipped ** 3)

    # 方程组
    du_dt = v
    tau_drag_safe = C_weertman * (N_clipped ** 3) * u_clipped / (np.abs(u_clipped) + u0)
    dv_dt = (tau_drive - tau_drag_safe - delta * v + duffing + forcing) / m_eff
    dN_dt = a_drain - b_drain * u_clipped - c_drain * (N_clipped ** 3) + season_forcing

    return np.array([du_dt, dv_dt, dN_dt], dtype=np.float64)


def rk4_step(y: np.ndarray, t: float, dt: float, rhs_func) -> np.ndarray:
    """经典四阶 Runge-Kutta 单步。"""
    k1 = rhs_func(y, t)
    k2 = rhs_func(y + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = rhs_func(y + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = rhs_func(y + dt * k3, t + dt)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def solve_ice_stream_oscillation(y0: np.ndarray,
                                 t_span: Tuple[float, float],
                                 dt: float,
                                 params: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    求解冰流非线性振荡。

    参数:
        y0: 初始状态 [u0, v0, N0]
        t_span: (t_start, t_end)
        dt: 时间步长 (s)
        params: 物理参数字典

    返回:
        t_array: 时间数组
        y_array: (nt, 3) 状态历史 [u, v, N]
    """
    y0 = np.asarray(y0, dtype=np.float64)
    t_start, t_end = t_span

    if dt <= 0:
        raise ValueError("dt must be positive.")

    nt = int(np.ceil((t_end - t_start) / dt)) + 1
    t_array = np.linspace(t_start, t_end, nt)
    y_array = np.zeros((nt, 3), dtype=np.float64)
    y_array[0] = y0

    y = y0.copy()
    for i in range(1, nt):
        y = rk4_step(y, t_array[i - 1], dt, lambda state, t: ice_stream_rhs(state, t, params))
        # 物理保护
        y[0] = np.clip(y[0], -5000.0, 5000.0)   # 速度范围 (m/a)
        y[1] = np.clip(y[1], -1000.0, 1000.0)   # 加速度
        y[2] = np.clip(y[2], 1e3, 1e7)          # 有效压力范围 (Pa)
        y_array[i] = y

    return t_array, y_array


def detect_stick_slip_events(t_array: np.ndarray,
                             y_array: np.ndarray,
                             velocity_threshold: float = 0.1) -> dict:
    """
    检测 stick-slip 转变事件。

    返回统计字典:
        'slip_events': 滑移事件索引列表
        'stick_events': 粘滞事件索引列表
        'mean_velocity': 平均流速
        'max_velocity': 最大流速
        'oscillation_period_estimate': 估计振荡周期
    """
    u = y_array[:, 0]

    # 过零点检测 (速度从负到正或从慢到快)
    above = u > velocity_threshold
    below = u <= velocity_threshold

    slip_events = []
    stick_events = []
    for i in range(1, len(u)):
        if below[i - 1] and above[i]:
            slip_events.append(i)
        if above[i - 1] and below[i]:
            stick_events.append(i)

    # 周期估计: 滑移事件之间的平均时间间隔
    period = None
    if len(slip_events) >= 2:
        intervals = np.diff(t_array[slip_events])
        period = float(np.mean(intervals))

    stats = {
        'slip_events': slip_events,
        'stick_events': stick_events,
        'mean_velocity': float(np.mean(np.abs(u))),
        'max_velocity': float(np.max(np.abs(u))),
        'oscillation_period_estimate': period,
    }
    return stats


def basal_shear_stress_from_state(state: np.ndarray,
                                   params: dict) -> float:
    """
    由状态计算基底剪应力。

        \tau_b = C N^n u / (|u| + u_0)
    """
    u, _, N = state
    C = params.get('C_weertman', 1e-4)
    u0 = params.get('u0', 1.0)
    n = params.get('n_weertman', 1.0)
    denom = np.abs(u) + u0
    tau_b = C * (N ** n) * u / denom
    return float(tau_b)


def driving_stress_from_params(params: dict) -> float:
    """
    计算重力驱动力。

        \tau_d = \rho g H \sin\alpha
    """
    H = params.get('H', 1000.0)
    alpha = params.get('alpha', 0.001)
    return ICE_DENSITY * GRAVITY * H * np.sin(alpha)
