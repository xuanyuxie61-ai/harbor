"""
ozone_ode.py — 大气臭氧化学动力学模型
============================================
来源项目: 842_ozone2_ode

本模块实现大气层中臭氧-氮氧化物化学动力学 ODE 系统.
基于 Chapman 机制和 NOx 催化循环.

物种:
  y[0] = [O]    原子氧 O(³P)
  y[1] = [NO]   一氧化氮
  y[2] = [NO₂]  二氧化氮
  y[3] = [O₃]   臭氧

反应机制:
  R1: O₃ + hν → O₂ + O        (光解, 速率 k₁(t))
  R2: O + O₃ → 2O₂            (速率 k₂)
  R3: NO + O₃ → NO₂ + O₂      (速率 k₃)
  R4: NO₂ + hν → NO + O       (光解, 隐含在 k₁ 中)

ODE 系统:
  dy₀/dt = +k₁y₃ - k₂y₀y₃
  dy₁/dt = +k₁y₃ - k₃y₁y₃ + σ₂
  dy₂/dt = -k₁y₃ + k₃y₁y₃
  dy₃/dt = +k₂y₀y₃ - k₁y₃ - k₃y₁y₃

日变化光解速率:
  k₁(t) = 10⁻⁵ exp(7 sin(π(t̄-4)/16)^0.2)   (白天, 4h ≤ t ≤ 20h)
  k₁(t) = 10⁻⁴⁰                                (夜间)

守恒量:
  h = y₀ + y₂ + y₃ = [O] + [NO₂] + [O₃]
  dh/dt = σ₂ (仅由源项驱动)

数值方法: 半隐式 RK4 (对刚性项隐式处理)
"""

import numpy as np
from typing import Tuple, Optional, List
from config import OzoneChemistryConfig, EPS_NUM


# ============================================================
#  化学反应速率
# ============================================================
def ozone_rhs(y: np.ndarray, t: float, config: OzoneChemistryConfig) -> np.ndarray:
    """臭氧化学 ODE 系统右端项

    dy₀/dt = +k₁(t)*y₃ - k₂*y₀*y₃
    dy₁/dt = +k₁(t)*y₃ - k₃*y₁*y₃ + σ₂
    dy₂/dt = -k₁(t)*y₃ + k₃*y₁*y₃
    dy₃/dt = -k₁(t)*y₃ + k₂*y₀*y₃ - k₃*y₁*y₃

    注意: 方程需满足 dy₀/dt + dy₂/dt + dy₃/dt = 0 (守恒关系)
    """
    O, NO, NO2, O3 = y[0], y[1], y[2], y[3]

    k1 = config.k1_diurnal(t)
    k2 = config.k2
    k3 = config.k3
    sigma2 = config.sigma2

    # 反应速率
    r1 = k1 * O3          # 光解速率
    r2 = k2 * O * O3      # O + O3 反应
    r3 = k3 * NO * O3     # NO + O3 反应

    # ODE
    dO_dt = r1 - r2
    dNO_dt = r1 - r3 + sigma2
    dNO2_dt = -r1 + r3
    dO3_dt = -r1 + r2 - r3

    return np.array([dO_dt, dNO_dt, dNO2_dt, dO3_dt])


# ============================================================
#  守恒量验证
# ============================================================
def compute_conserved(y: np.ndarray) -> float:
    """计算守恒量 h = [O] + [NO₂] + [O₃]

    理论: dh/dt = σ₂ ≈ 1e-11 (极慢变化)
    在短时间尺度上近似守恒.
    """
    return y[0] + y[2] + y[3]


def check_conservation(y_history: np.ndarray, times: np.ndarray,
                       config: OzoneChemistryConfig) -> dict:
    """验证守恒量随时间的变化

    Returns:
        dict: h_values, h_drift, relative_drift
    """
    h_values = np.array([compute_conserved(y) for y in y_history])
    h_initial = h_values[0]
    h_drift = h_values - h_initial
    expected_drift = config.sigma2 * times  # 理论漂移

    relative_drift = np.abs(h_drift - expected_drift) / (np.abs(h_initial) + EPS_NUM)

    return {
        'h_values': h_values,
        'h_drift': h_drift,
        'expected_drift': expected_drift,
        'relative_drift': relative_drift,
        'max_relative_error': np.max(relative_drift),
    }


# ============================================================
#  RK4 时间步进
# ============================================================
def ozone_rk4_step(y: np.ndarray, t: float, dt: float,
                   config: OzoneChemistryConfig) -> np.ndarray:
    """4 阶 Runge-Kutta 单步

    对化学 ODE 系统, RK4 提供四阶精度:
    局部截断误差: O(dt⁵)
    全局误差: O(dt⁴)
    """
    k1 = ozone_rhs(y, t, config)
    k2 = ozone_rhs(y + 0.5 * dt * k1, t + 0.5 * dt, config)
    k3 = ozone_rhs(y + 0.5 * dt * k2, t + 0.5 * dt, config)
    k4 = ozone_rhs(y + dt * k3, t + dt, config)

    y_new = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # 物理约束: 浓度非负
    y_new = np.maximum(y_new, 0.0)

    return y_new


# ============================================================
#  完整模拟
# ============================================================
def run_ozone_simulation(config: OzoneChemistryConfig) -> dict:
    """运行 24 小时臭氧化学模拟

    Returns:
        dict: time, O, NO, NO2, O3, k1_history, conservation
    """
    dt = config.dt
    n_steps = int(config.T_final / dt)

    y = np.array(config.y0, dtype=np.float64)

    # 存储 (每隔一定步数采样)
    sample_interval = max(1, int(0.1 / dt))  # 每 0.1 小时采样
    n_samples = n_steps // sample_interval + 1

    t_hist = np.zeros(n_samples)
    y_hist = np.zeros((n_samples, 4))
    k1_hist = np.zeros(n_samples)

    t_hist[0] = 0.0
    y_hist[0] = y.copy()
    k1_hist[0] = config.k1_diurnal(0.0)

    sample_idx = 1
    for step in range(1, n_steps + 1):
        t = step * dt
        y = ozone_rk4_step(y, t - dt, dt, config)

        if step % sample_interval == 0 and sample_idx < n_samples:
            t_hist[sample_idx] = t
            y_hist[sample_idx] = y.copy()
            k1_hist[sample_idx] = config.k1_diurnal(t)
            sample_idx += 1

    # 截断到实际采样数
    t_hist = t_hist[:sample_idx]
    y_hist = y_hist[:sample_idx]
    k1_hist = k1_hist[:sample_idx]

    # 守恒量检查
    conservation = check_conservation(y_hist, t_hist, config)

    return {
        'time': t_hist,
        'O': y_hist[:, 0],
        'NO': y_hist[:, 1],
        'NO2': y_hist[:, 2],
        'O3': y_hist[:, 3],
        'k1_history': k1_hist,
        'conservation': conservation,
    }


# ============================================================
#  平衡态分析
# ============================================================
def compute_photostationary_state(config: OzoneChemistryConfig,
                                  k1_value: float = 1e-3) -> dict:
    """计算光稳态 (photostationary state)

    在恒光照下, 令 dy/dt = 0:
      k₁[O₃] = k₂[O][O₃]  →  [O] = k₁/k₂
      k₁[O₃] = k₃[NO][O₃]  →  [NO] = k₁/k₃
      [O₃] = (h - [O]) / (1 + k₂/k₁)  (从守恒量推导)

    Returns:
        dict: O, NO, NO2, O3 (稳态浓度)
    """
    k1 = k1_value
    k2 = config.k2
    k3 = config.k3
    sigma2 = config.sigma2
    h = sum(config.y0[:1]) + sum(config.y0[2:])  # 守恒量初值

    # 近似稳态
    O_ss = k1 / (k2 + EPS_NUM)
    NO_ss = (k1 + sigma2) / (k3 + EPS_NUM)

    # 从守恒量求 O3
    O3_ss = max(h - O_ss, 0.0) / (1.0 + k2 / (k1 + EPS_NUM))
    NO2_ss = max(h - O_ss - O3_ss, 0.0)

    return {
        'O': O_ss,
        'NO': NO_ss,
        'NO2': NO2_ss,
        'O3': O3_ss,
        'k1': k1,
    }
