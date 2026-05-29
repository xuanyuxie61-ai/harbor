"""
calendar_utils.py
=================
基于 weekday (1411_weekday) 与儒略日计算的核心算法，
为 ENSO 季节锁相 (phase-locking) 分析提供精密时间索引。

科学背景
--------
ENSO 事件具有显著的季节锁相特征：El Nino 发展峰值通常出现在北半球冬季
(11月–1月)。本模块将日历日期转换为儒略日数 (Julian Ephemeris Date, JED)，
进而计算事件相对于年周期的相位角，为海气耦合模式提供时间坐标。

核心公式
--------
1. 儒略日数 (JED) 计算（混合历：格里高利历 1582/10/15 之后，儒略历之前）：
   对于格里高利历日期 (Y, M, D, F)：
   
   A = floor(Y / 100)
   B = 2 - A + floor(A / 4)
   
   JED = floor(365.25 * (Y + 4716)) + floor(30.6001 * (M + 1))
         + D + F + B - 1524.5

   其中 M 的取值规则：1月、2月视为上一年的 13、14 月。

2. 星期计算（从 JED 推导）：
   jedmod = mod(JED, 7.0)
   j = mod(floor(jedmod), 7)
   f = (jedmod + 0.5) - j
   w = wrap(j + 2, 1, 7)

3. ENSO 季节相位角：
   以 1 月 1 日为参考零点，将日期映射到 [0, 2π) 的相位：
   
   θ = 2π * (day_of_year - 1) / days_in_year

   其中 day_of_year 通过累积月天数计算。

4. 季节锁相强度 (Phase-Locking Index, PLI)：
   对 N 个 ENSO 事件的发生月份 {m_k}，计算
   
   PLI = | (1/N) * Σ_{k=1}^{N} exp(i * 2π * m_k / 12) |

   PLI ∈ [0, 1]，越接近 1 表示锁相越强。
"""

import numpy as np
from typing import Tuple, List


def _ymd_to_jed_gregorian(y: int, m: int, d: int, f: float = 0.0) -> float:
    """
    将格里高利历日期转换为儒略日数 (JED)。

    公式 (Richards 1999, Mapping Time)：
    JED = 1721060.5 + 365.0 * (Y - 1)
          + floor((Y - 1) / 4) - floor((Y - 1) / 100)
          + floor((Y - 1) / 400)
          + floor((367 * M - 362) / 12)
          + D + F

    参数
    ----
    y, m, d : int
        年、月、日。
    f : float
        日的小数部分，默认 0.0。

    返回
    ----
    jed : float
        儒略日数。
    """
    if m <= 2:
        y_g = y - 1
        m_g = m + 12
    else:
        y_g = y
        m_g = m

    a = np.floor(y_g / 100.0)
    b = 2.0 - a + np.floor(a / 4.0)

    jed = (
        np.floor(365.25 * (y_g + 4716.0))
        + np.floor(30.6001 * (m_g + 1.0))
        + d
        + f
        + b
        - 1524.5
    )
    return float(jed)


def _ymd_to_jed_julian(y: int, m: int, d: int, f: float = 0.0) -> float:
    """
    将儒略历日期转换为儒略日数 (JED)。
    """
    if m <= 2:
        y_j = y - 1
        m_j = m + 12
    else:
        y_j = y
        m_j = m

    jed = (
        np.floor(365.25 * (y_j + 4716.0))
        + np.floor(30.6001 * (m_j + 1.0))
        + d
        + f
        - 1524.5
    )
    return float(jed)


def ymd_to_jed(y: int, m: int, d: int, f: float = 0.0) -> float:
    """
    混合历日期转 JED：1582/10/15 之前用儒略历，之后用格里高利历。
    本函数严格处理历法切换边界。
    """
    # 格里高利历起始日期的 JED
    jed_gregorian_start = 2299161.0

    # 先假设格里高利历计算
    jed_g = _ymd_to_jed_gregorian(y, m, d, f)

    if jed_g >= jed_gregorian_start:
        return jed_g

    # 对于更早的日期，使用儒略历
    jed_j = _ymd_to_jed_julian(y, m, d, f)

    # 边界检查：1582/10/05 至 1582/10/14 为历法空白期
    jed_boundary_low = 2299159.5  # 1582/10/04 Julian
    jed_boundary_high = 2299160.5  # 1582/10/15 Gregorian
    if jed_boundary_low < jed_j < jed_boundary_high:
        raise ValueError(
            "Date falls in the Gregorian calendar transition gap (1582/10/05–14)"
        )

    return jed_j


def jed_to_weekday(jed: float) -> Tuple[int, float]:
    """
    从 JED 计算星期几。

    公式：
    jedmod = mod(jed, 7.0)
    j = mod(floor(jedmod), 7)
    f = (jedmod + 0.5) - j
    w = wrap(j + 2, 1, 7)

    返回
    ----
    w : int
        星期编号，1=周日, 2=周一, ..., 7=周六。
    f : float
        日的小数部分。
    """
    jedmod = np.mod(jed, 7.0)
    j = int(np.mod(np.floor(jedmod), 7))
    f = (jedmod + 0.5) - j
    w = _i4_wrap(j + 2, 1, 7)
    return w, float(f)


def _i4_wrap(ival: int, ilo: int, ihi: int) -> int:
    """
    将整数包装到 [ilo, ihi] 区间。
    """
    if ilo > ihi:
        raise ValueError("ilo must not exceed ihi")
    wide = ihi - ilo + 1
    if wide == 0:
        return ival
    j = ilo + np.mod(ival - ilo, wide)
    return int(j)


def day_of_year(y: int, m: int, d: int) -> int:
    """
    计算一年中的第几天（1-based）。

    使用累积月天数表，自动处理闰年：
    days_in_month = [31, 28/29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    """
    is_leap = (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)
    month_days = [31, 29 if is_leap else 28, 31, 30, 31, 30,
                  31, 31, 30, 31, 30, 31]
    if m < 1 or m > 12:
        raise ValueError("Month must be in [1, 12]")
    if d < 1 or d > month_days[m - 1]:
        raise ValueError(f"Day out of range for month {m}")
    return sum(month_days[:m - 1]) + d


def days_in_year(y: int) -> int:
    """返回指定年份的总天数。"""
    return 366 if ((y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)) else 365


def seasonal_phase(y: int, m: int, d: int) -> float:
    """
    计算日期相对于年周期的相位角 θ ∈ [0, 2π)。

    公式：
    θ = 2π * (day_of_year - 1) / days_in_year

    物理意义：北半球冬季（12月–2月）对应 θ ≈ 0，
    夏季（6月–8月）对应 θ ≈ π。
    """
    doy = day_of_year(y, m, d)
    diy = days_in_year(y)
    theta = 2.0 * np.pi * (doy - 1) / diy
    return float(theta)


def phase_locking_index(event_months: List[int]) -> float:
    """
    计算 ENSO 事件的季节锁相强度 (Phase-Locking Index, PLI)。

    公式：
    PLI = | (1/N) * Σ_{k=1}^{N} exp(i * 2π * m_k / 12) |

    其中 m_k 为第 k 个事件的月份（1–12）。

    参数
    ----
    event_months : List[int]
        ENSO 事件发生月份列表。

    返回
    ----
    pli : float
        锁相强度，范围 [0, 1]。
    """
    if not event_months:
        return 0.0
    months = np.array(event_months, dtype=float)
    if np.any((months < 1) | (months > 12)):
        raise ValueError("Months must be in [1, 12]")
    complex_sum = np.sum(np.exp(2.0j * np.pi * months / 12.0))
    pli = np.abs(complex_sum / len(months))
    return float(pli)


def enso_event_timing(events: List[Tuple[int, int, int, float]]) -> dict:
    """
    分析一组 ENSO 事件的时间特征。

    参数
    ----
    events : List[Tuple[int, int, int, float]]
        每个元组为 (year, month, day, nino34_index)。

    返回
    ----
    dict : 包含 JED、相位角、锁相强度的分析结果。
    """
    if not events:
        return {}

    jeds = []
    phases = []
    months = []
    for y, m, d, nino in events:
        jed = ymd_to_jed(y, m, d)
        jeds.append(jed)
        phases.append(seasonal_phase(y, m, d))
        months.append(m)

    pli = phase_locking_index(months)

    # 计算事件间平均间隔（年）
    intervals = np.diff(jeds) / 365.25

    return {
        "jed_array": np.array(jeds),
        "phase_array": np.array(phases),
        "phase_locking_index": pli,
        "mean_interval_years": float(np.mean(intervals)) if len(intervals) > 0 else 0.0,
        "std_interval_years": float(np.std(intervals)) if len(intervals) > 0 else 0.0,
    }
