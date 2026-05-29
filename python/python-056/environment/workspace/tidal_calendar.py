"""
tidal_calendar.py
================================================================================
潮汐日历与时间序列模块 (来源于 135_calpak 项目)
================================================================================
本模块提供儒略日计算与潮汐相位时间序列生成，用于确定潮汐能
提取系统在不同时刻的潮汐高度与流速。基于天文历法的日期转换
算法，计算月球和太阳对地球潮汐的相位影响。

核心公式:
    - 儒略日 (JED):
        JED = JD - 2400000.5
        用于统一时间基准

    - 朔望月周期:
        T_synodic = 29.53058868 天

    - 潮汐势相位角:
        θ(t) = ω_M · t + φ_M  (月球)
             + ω_S · t + φ_S  (太阳)

    - 平衡潮高:
        η_eq(t) = Σ_n A_n · cos(ω_n t + φ_n)
        其中主要分潮包括 M2, S2, K1, O1, N2 等
"""

import numpy as np
from typing import Tuple, List


def ymdf_to_jed_gregorian(y: int, m: int, d: int, f: float = 0.0) -> float:
    """
    将公历日期 (Y, M, D, F) 转换为儒略日 (JED)。

    公式 (基于 Fliegel & Van Flandern 1968):
        JD = D - 32075 + 1461*(Y + 4800 + (M-14)/12)/4
           + 367*(M - 2 - (M-14)/12 * 12)/12
           - 3*((Y + 4900 + (M-14)/12)/100)/4

    参数:
        y: 年份
        m: 月份 (1-12)
        d: 日
        f: 日的小数部分 [0, 1)

    返回:
        儒略日 JED
    """
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524
    jed = jd - 2400000.5 + f
    return float(jed)


def jed_to_ymdhms_common(jed: float) -> Tuple[int, int, int, int, int, float]:
    """
    将儒略日转换回公历日期时分秒。

    参数:
        jed: 儒略日

    返回:
        (year, month, day, hour, minute, second)
    """
    jd = jed + 2400000.5
    jd_int = int(np.floor(jd))
    f = jd - jd_int

    if jd_int < 2299161:
        a = jd_int
    else:
        alpha = int((jd_int - 1867216.25) / 36524.25)
        a = jd_int + 1 + alpha - int(alpha / 4)

    b = a + 1524
    c = int((b - 122.1) / 365.25)
    d = int(365.25 * c)
    e = int((b - d) / 30.6001)

    day = b - d - int(30.6001 * e)
    if e < 14:
        month = e - 1
    else:
        month = e - 13
    if month > 2:
        year = c - 4716
    else:
        year = c - 4715

    hours = f * 24.0
    hour = int(hours)
    minutes = (hours - hour) * 60.0
    minute = int(minutes)
    second = (minutes - minute) * 60.0
    return year, month, day, hour, minute, second


def compute_tidal_phases(jed: float) -> dict:
    """
    计算给定儒略日的主要潮汐分潮相位。

    分潮参数 (Doodson 参数):
        τ  = 月球时角
        s  = 月球平黄经
        h  = 太阳平黄经
        p  = 月球近地点平黄经
        N' = 月球升交点平黄经
        p_s = 太阳近地点平黄经

    主要分潮:
        M2: 2τ - 2s + 2h       周期 = 12.42 h
        S2: 2τ                 周期 = 12.00 h
        K1: τ + h              周期 = 23.93 h
        O1: τ - 2s + h         周期 = 25.82 h
        N2: 2τ - 3s + p + 2h   周期 = 12.66 h

    参数:
        jed: 儒略日

    返回:
        包含各分潮相位的字典
    """
    # 从 J2000.0 (JED = 2451545.0) 起算的天数
    d = jed - 2451545.0

    # Doodson 参数 (度)
    T = d / 36525.0  # 儒略世纪
    s = 218.316 + 13.176396 * d
    h = 280.460 + 0.985647 * d
    p = 83.353 + 0.111404 * d
    N_prime = 125.045 - 0.052954 * d

    # 转换为弧度
    deg2rad = np.pi / 180.0
    s_rad = (s % 360.0) * deg2rad
    h_rad = (h % 360.0) * deg2rad
    p_rad = (p % 360.0) * deg2rad
    N_rad = (N_prime % 360.0) * deg2rad
    tau = (15.0 * ((d % 1.0) * 24.0) - s) % 360.0
    tau_rad = tau * deg2rad

    phases = {
        'M2': 2.0 * tau_rad - 2.0 * s_rad + 2.0 * h_rad,
        'S2': 2.0 * tau_rad,
        'K1': tau_rad + h_rad,
        'O1': tau_rad - 2.0 * s_rad + h_rad,
        'N2': 2.0 * tau_rad - 3.0 * s_rad + p_rad + 2.0 * h_rad,
    }
    return phases


def generate_tidal_elevation(
    t_hours: np.ndarray,
    jed_base: float,
    amplitudes: dict = None,
) -> np.ndarray:
    """
    生成平衡潮高时间序列。

    公式:
        η(t) = A_M2 cos(ω_M2 t + φ_M2)
             + A_S2 cos(ω_S2 t + φ_S2)
             + A_K1 cos(ω_K1 t + φ_K1)
             + A_O1 cos(ω_O1 t + φ_O1)
             + A_N2 cos(ω_N2 t + φ_N2)

    参数:
        t_hours: 时间数组 (小时，相对 jed_base)
        jed_base: 基准儒略日
        amplitudes: 各分潮振幅字典 (m)，默认使用典型值

    返回:
        潮高数组 (m)
    """
    if amplitudes is None:
        amplitudes = {
            'M2': 0.80,
            'S2': 0.35,
            'K1': 0.25,
            'O1': 0.20,
            'N2': 0.15,
        }

    phases_base = compute_tidal_phases(jed_base)
    omega = {
        'M2': 2.0 * np.pi / 12.4206012,
        'S2': 2.0 * np.pi / 12.0,
        'K1': 2.0 * np.pi / 23.934472,
        'O1': 2.0 * np.pi / 25.819338,
        'N2': 2.0 * np.pi / 12.658348,
    }

    t = np.asarray(t_hours, dtype=float)
    eta = np.zeros_like(t)
    for comp, A in amplitudes.items():
        eta += A * np.cos(omega[comp] * t + phases_base[comp])
    return eta


def generate_tidal_velocity(
    t_hours: np.ndarray,
    jed_base: float,
    max_velocity: float = 2.5,
    phase_lag_deg: float = 45.0,
) -> np.ndarray:
    """
    生成潮流速度时间序列。

    物理模型:
        流速与潮高的梯度成正比，且存在相位滞后:
            u(t) = u_max · sin(ω t + φ - δ)
        其中 δ 为相位滞后角，典型值 45°。

    参数:
        t_hours: 时间数组 (小时)
        jed_base: 基准儒略日
        max_velocity: 最大流速 (m/s)
        phase_lag_deg: 相位滞后 (度)

    返回:
        流速数组 (m/s)
    """
    eta = generate_tidal_elevation(t_hours, jed_base)
    # 用潮高的一阶差分近似流速（正比于梯度）
    dt = np.diff(t_hours, prepend=t_hours[0])
    dt[0] = dt[1] if len(dt) > 1 else 1.0
    deta = np.gradient(eta, t_hours)
    # 归一化到最大流速
    u_raw = deta
    u_max = np.max(np.abs(u_raw))
    if u_max < 1e-12:
        return np.zeros_like(t_hours)
    u = max_velocity * u_raw / u_max
    return u
