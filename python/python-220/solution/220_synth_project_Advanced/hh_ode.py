"""
hh_ode.py — Hodgkin-Huxley 神经轴突模型
============================================
来源项目: 060_axon_ode

本模块实现 Hodgkin-Huxley (1952) 神经元膜电位动力学模型.
该模型描述神经轴突中动作电位的产生与传播, 是计算神经科学的基石.

状态方程:
  C_m dV/dt = I_ext - I_Na - I_K - I_L

离子电流:
  I_Na = g_Na * m³ * h * (V - E_Na)  (钠电流, 快速激活)
  I_K  = g_K * n⁴ * (V - E_K)        (钾电流, 慢速激活)
  I_L  = g_L * (V - E_L)             (漏电流)

门控动力学:
  dx/dt = α_x(V) * (1 - x) - β_x(V) * x,  x ∈ {n, m, h}

速率函数 (含 L'Hôpital 正则化):
  α_n(V) = 0.01(10-V) / (exp((10-V)/10) - 1)
  β_n(V) = 0.125 exp(-V/80)
  α_m(V) = 0.1(25-V) / (exp((25-V)/10) - 1)
  β_m(V) = 4.0 exp(-V/18)
  α_h(V) = 0.07 exp(-V/20)
  β_h(V) = 1 / (exp((30-V)/10) + 1)

守恒关系:
  n_∞(V) = α_n/(α_n+β_n),  τ_n(V) = 1/(α_n+β_n)
  m_∞(V) = α_m/(α_m+β_m),  τ_m(V) = 1/(α_m+β_m)
  h_∞(V) = α_h/(α_h+β_h),  τ_h(V) = 1/(α_h+β_h)

数值方法: 4 阶 Runge-Kutta (RK4)
  k1 = f(t_n, y_n)
  k2 = f(t_n + h/2, y_n + h*k1/2)
  k3 = f(t_n + h/2, y_n + h*k2/2)
  k4 = f(t_n + h, y_n + h*k3)
  y_{n+1} = y_n + (h/6)(k1 + 2k2 + 2k3 + k4)
"""

import numpy as np
from typing import Tuple, Optional, List
from config import HodgkinHuxleyConfig, EPS_NUM


# ============================================================
#  离子电流计算
# ============================================================
def compute_ionic_currents(V: float, n: float, m: float, h: float,
                           config: HodgkinHuxleyConfig) -> dict:
    """计算各离子电流分量

    I_Na = g_Na * m³ * h * (V - E_Na)  — 钠电流 (去极化)
    I_K  = g_K * n⁴ * (V - E_K)        — 钾电流 (复极化)
    I_L  = g_L * (V - E_L)             — 漏电流

    总离子电流: I_ion = I_Na + I_K + I_L

    Returns:
        dict with keys: I_Na, I_K, I_L, I_total
    """
    V_arr = np.array([V])
    I_Na = config.g_Na * (m ** 3) * h * (V - config.E_Na)
    I_K = config.g_K * (n ** 4) * (V - config.E_K)
    I_L = config.g_L * (V - config.E_L)

    return {
        'I_Na': I_Na,
        'I_K': I_K,
        'I_L': I_L,
        'I_total': I_Na + I_K + I_L,
    }


# ============================================================
#  ODE 右端项
# ============================================================
def hh_rhs(y: np.ndarray, config: HodgkinHuxleyConfig,
           I_ext: Optional[float] = None) -> np.ndarray:
    """Hodgkin-Huxley ODE 系统右端项

    状态向量 y = [V, n, m, h]

    dV/dt = (I_ext - I_Na - I_K - I_L) / C_m
    dn/dt = α_n(V)(1-n) - β_n(V)n
    dm/dt = α_m(V)(1-m) - β_m(V)m
    dh/dt = α_h(V)(1-h) - β_h(V)h

    Args:
        y: 状态向量 [V, n, m, h]
        config: HH 参数
        I_ext: 外部电流 (若 None, 使用 config.I_ext)

    Returns:
        dy/dt: 右端项向量
    """
    V, n, m, h = y[0], y[1], y[2], y[3]

    if I_ext is None:
        I_ext = config.I_ext

    # 速率函数
    a_n = config.alpha_n(np.array([V]))[0]
    b_n = config.beta_n(np.array([V]))[0]
    a_m = config.alpha_m(np.array([V]))[0]
    b_m = config.beta_m(np.array([V]))[0]
    a_h = config.alpha_h(np.array([V]))[0]
    b_h = config.beta_h(np.array([V]))[0]

    # 离子电流
    currents = compute_ionic_currents(V, n, m, h, config)

    # ODE 右端项
    dVdt = (I_ext - currents['I_total']) / config.C_m
    dndt = a_n * (1.0 - n) - b_n * n
    dmdt = a_m * (1.0 - m) - b_m * m
    dhdt = a_h * (1.0 - h) - b_h * h

    return np.array([dVdt, dndt, dmdt, dhdt])


# ============================================================
#  RK4 时间步进
# ============================================================
def hh_rk4_step(y: np.ndarray, dt: float, config: HodgkinHuxleyConfig,
                I_ext: Optional[float] = None) -> np.ndarray:
    """4 阶 Runge-Kutta 单步推进

    k1 = f(t_n, y_n)
    k2 = f(t_n + dt/2, y_n + dt*k1/2)
    k3 = f(t_n + dt/2, y_n + dt*k2/2)
    k4 = f(t_n + dt, y_n + dt*k3)
    y_{n+1} = y_n + (dt/6)(k1 + 2*k2 + 2*k3 + k4)

    边界保护:
      n, m, h ∈ [0, 1] (门控变量的物理约束)
    """
    k1 = hh_rhs(y, config, I_ext)
    k2 = hh_rhs(y + 0.5 * dt * k1, config, I_ext)
    k3 = hh_rhs(y + 0.5 * dt * k2, config, I_ext)
    k4 = hh_rhs(y + dt * k3, config, I_ext)

    y_new = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # 物理约束: 门控变量 ∈ [0, 1]
    y_new[1] = np.clip(y_new[1], 0.0, 1.0)  # n
    y_new[2] = np.clip(y_new[2], 0.0, 1.0)  # m
    y_new[3] = np.clip(y_new[3], 0.0, 1.0)  # h

    return y_new


# ============================================================
#  完整模拟
# ============================================================
def run_hh_simulation(config: HodgkinHuxleyConfig,
                      I_ext_profile: Optional[callable] = None) -> dict:
    """运行完整的 Hodgkin-Huxley 模拟

    Args:
        config: HH 参数
        I_ext_profile: 外部电流函数 I_ext(t), 默认恒定

    Returns:
        dict: 包含 'time', 'V', 'n', 'm', 'h', 'spike_times'
    """
    dt = config.dt
    n_steps = int(config.T_final / dt)

    # 初始化: 稳态值
    V0 = config.V0
    V_arr = np.array([V0])
    n0 = config.alpha_n(V_arr)[0] / (config.alpha_n(V_arr)[0] + config.beta_n(V_arr)[0])
    m0 = config.alpha_m(V_arr)[0] / (config.alpha_m(V_arr)[0] + config.beta_m(V_arr)[0])
    h0 = config.alpha_h(V_arr)[0] / (config.alpha_h(V_arr)[0] + config.beta_h(V_arr)[0])

    y = np.array([V0, n0, m0, h0])

    # 存储
    t_hist = np.zeros(n_steps + 1)
    V_hist = np.zeros(n_steps + 1)
    n_hist = np.zeros(n_steps + 1)
    m_hist = np.zeros(n_steps + 1)
    h_hist = np.zeros(n_steps + 1)

    t_hist[0] = 0.0
    V_hist[0] = y[0]
    n_hist[0] = y[1]
    m_hist[0] = y[2]
    h_hist[0] = y[3]

    # 动作电位检测
    spike_times = []
    spike_threshold = 0.0  # mV
    was_above = y[0] > spike_threshold

    for step in range(1, n_steps + 1):
        t = step * dt
        I_ext = I_ext_profile(t) if I_ext_profile else config.I_ext

        y = hh_rk4_step(y, dt, config, I_ext)

        t_hist[step] = t
        V_hist[step] = y[0]
        n_hist[step] = y[1]
        m_hist[step] = y[2]
        h_hist[step] = y[3]

        # 简单动作电位检测
        is_above = y[0] > spike_threshold
        if is_above and not was_above:
            spike_times.append(t)
        was_above = is_above

    return {
        'time': t_hist,
        'V': V_hist,
        'n': n_hist,
        'm': m_hist,
        'h': h_hist,
        'spike_times': spike_times,
        'firing_rate': len(spike_times) / config.T_final * 1000.0,  # Hz
    }


# ============================================================
#  稳态分析
# ============================================================
def compute_steady_state(V: float, config: HodgkinHuxleyConfig) -> dict:
    """计算给定电压下的稳态门控变量

    x_∞(V) = α_x(V) / (α_x(V) + β_x(V))
    τ_x(V) = 1 / (α_x(V) + β_x(V))

    Returns:
        dict: n_inf, m_inf, h_inf, tau_n, tau_m, tau_h
    """
    V_arr = np.array([V])
    a_n = config.alpha_n(V_arr)[0]
    b_n = config.beta_n(V_arr)[0]
    a_m = config.alpha_m(V_arr)[0]
    b_m = config.beta_m(V_arr)[0]
    a_h = config.alpha_h(V_arr)[0]
    b_h = config.beta_h(V_arr)[0]

    return {
        'n_inf': a_n / (a_n + b_n + EPS_NUM),
        'm_inf': a_m / (a_m + b_m + EPS_NUM),
        'h_inf': a_h / (a_h + b_h + EPS_NUM),
        'tau_n': 1.0 / (a_n + b_n + EPS_NUM),
        'tau_m': 1.0 / (a_m + b_m + EPS_NUM),
        'tau_h': 1.0 / (a_h + b_h + EPS_NUM),
    }


# ============================================================
#  I-V 曲线 (电流-电压关系)
# ============================================================
def compute_iv_curve(config: HodgkinHuxleyConfig,
                     V_range: Optional[np.ndarray] = None) -> dict:
    """计算稳态 I-V 曲线

    对于每个电压 V, 计算稳态电流:
      I_ss(V) = g_Na * m_∞³(V) * h_∞(V) * (V - E_Na)
              + g_K * n_∞⁴(V) * (V - E_K)
              + g_L * (V - E_L)

    Returns:
        dict: V_values, I_total, I_Na, I_K, I_L
    """
    if V_range is None:
        V_range = np.linspace(-100, 60, 200)

    I_total = np.zeros_like(V_range)
    I_Na_arr = np.zeros_like(V_range)
    I_K_arr = np.zeros_like(V_range)
    I_L_arr = np.zeros_like(V_range)

    for i, V in enumerate(V_range):
        ss = compute_steady_state(V, config)
        I_Na_arr[i] = config.g_Na * ss['m_inf']**3 * ss['h_inf'] * (V - config.E_Na)
        I_K_arr[i] = config.g_K * ss['n_inf']**4 * (V - config.E_K)
        I_L_arr[i] = config.g_L * (V - config.E_L)
        I_total[i] = I_Na_arr[i] + I_K_arr[i] + I_L_arr[i]

    return {
        'V_values': V_range,
        'I_total': I_total,
        'I_Na': I_Na_arr,
        'I_K': I_K_arr,
        'I_L': I_L_arr,
    }
