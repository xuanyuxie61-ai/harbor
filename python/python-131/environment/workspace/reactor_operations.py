"""
reactor_operations.py
=====================
基于 135_calpak 改造的反应器操作时间线模块。

在工业气泡柱反应器的运行中，时间记录、批次周期计算与 Julian 日期转换
是过程控制与数据记录的基础。本模块将 calpak 的日期算法迁移到反应器
工程领域，用于：
1. Julian Day Number (JDN) 与 Gregorian 日期的互相转换
2. 反应器连续运行天数的精确计算
3. 批次操作（batch cycle）时间线规划
4. 闰年判断与操作日历生成

核心公式
--------
1. Gregorian 日期到 Julian Day Number：
       a = floor((14 - month) / 12)
       y = year + 4800 - a
       m = month + 12a - 3
       JDN = day + floor((153m + 2)/5) + 365y + floor(y/4)
             - floor(y/100) + floor(y/400) - 32045

2. Julian Day Number 到 Gregorian 日期：
       l = JDN + 68569
       n = floor(4l / 146097)
       l = l - floor((146097n + 3)/4)
       i = floor(4000(l + 1) / 1461001)
       l = l - floor(1461i / 4) + 31
       j = floor(80l / 2447)
       day = l - floor(2447j / 80)
       l = floor(j / 11)
       month = j + 2 - 12l
       year = 100(n - 49) + i + l

3. 闰年判断（Gregorian）：
       year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

4. 反应器运行时间跨度：
       Δt_days = JDN_end - JDN_start
       Δt_hours = Δt_days × 24 + (hour_end - hour_start)

5. 批次循环时间线：
       t_cycle = t_fill + t_heat + t_reaction + t_cool + t_empty
       产能 = n_batches × V_reactor × X_conversion / t_total
"""

import math


# ---------------------------------------------------------------------------
# Date conversions (from 135_calpak)
# ---------------------------------------------------------------------------

def date_to_jdn(year, month, day):
    """
    Gregorian 日期转 Julian Day Number。
    """
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    jdn = (day + (153 * m + 2) // 5 + 365 * y
           + y // 4 - y // 100 + y // 400 - 32045)
    return jdn


def jdn_to_date(jdn):
    """
    Julian Day Number 转 Gregorian 日期 (year, month, day)。
    """
    l = jdn + 68569
    n = (4 * l) // 146097
    l = l - (146097 * n + 3) // 4
    i = (4000 * (l + 1)) // 1461001
    l = l - (1461 * i) // 4 + 31
    j = (80 * l) // 2447
    day = l - (2447 * j) // 80
    l = j // 11
    month = j + 2 - 12 * l
    year = 100 * (n - 49) + i + l
    return year, month, day


def is_leap_year_gregorian(year):
    """
    Gregorian 闰年判断。
    """
    return (year % 4 == 0) and ((year % 100 != 0) or (year % 400 == 0))


def days_in_month_gregorian(year, month):
    """
    返回某年某月的天数。
    """
    if month in (1, 3, 5, 7, 8, 10, 12):
        return 31
    elif month in (4, 6, 9, 11):
        return 30
    elif month == 2:
        return 29 if is_leap_year_gregorian(year) else 28
    else:
        raise ValueError("Invalid month")


# ---------------------------------------------------------------------------
# Reactor operational timeline
# ---------------------------------------------------------------------------

def reactor_operation_timeline(start_date, end_date,
                               cycle_phases=None):
    """
    计算反应器在指定日期范围内的操作时间线。

    Parameters
    ----------
    start_date : tuple (year, month, day)
    end_date : tuple (year, month, day)
    cycle_phases : dict or None
        各阶段小时数，如 {'fill':2, 'heat':4, 'reaction':48,
                          'cool':6, 'empty':2}。

    Returns
    -------
    timeline : dict
    """
    y1, m1, d1 = start_date
    y2, m2, d2 = end_date
    jdn1 = date_to_jdn(y1, m1, d1)
    jdn2 = date_to_jdn(y2, m2, d2)
    total_days = jdn2 - jdn1 + 1

    if cycle_phases is None:
        cycle_phases = {
            'fill': 2.0,
            'heat': 4.0,
            'reaction': 48.0,
            'cool': 6.0,
            'empty': 2.0,
        }

    cycle_hours = sum(cycle_phases.values())
    cycles_possible = int(total_days * 24.0 / cycle_hours)

    return {
        'start_jdn': jdn1,
        'end_jdn': jdn2,
        'total_days': total_days,
        'cycle_hours': cycle_hours,
        'max_cycles': cycles_possible,
        'phases': cycle_phases,
    }


def operating_calendar_year(year, scheduled_downtime_days=None):
    """
    生成某年的反应器操作日历，扣除计划停机日。

    Parameters
    ----------
    year : int
    scheduled_downtime_days : list of tuple or None
        计划停机日期列表，如 [(3,15), (6,20)] 表示 3月15日、6月20日。

    Returns
    -------
    calendar_info : dict
    """
    is_leap = is_leap_year_gregorian(year)
    total_days = 366 if is_leap else 365

    downtime = set()
    if scheduled_downtime_days:
        for month, day in scheduled_downtime_days:
            if 1 <= month <= 12 and 1 <= day <= days_in_month_gregorian(year, month):
                jdn = date_to_jdn(year, month, day)
                downtime.add(jdn)

    operating_days = total_days - len(downtime)
    availability = operating_days / total_days if total_days > 0 else 0.0

    return {
        'year': year,
        'is_leap': is_leap,
        'total_days': total_days,
        'downtime_days': len(downtime),
        'operating_days': operating_days,
        'availability': availability,
    }
