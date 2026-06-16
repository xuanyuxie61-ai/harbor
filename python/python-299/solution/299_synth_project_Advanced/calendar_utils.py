"""
calendar_utils.py — 多历法实验时间戳系统
========================================

种子项目映射: 135_calpak — 日历计算包.

原项目 calpak 实现了多种历法系统之间的日期转换:
  - Gregorian (公历)
  - Julian (儒略历)
  - Hebrew (希伯来历)
  - Islamic (伊斯兰历)
  - Republican (法国共和历)
  - Alexandrian, English, Roman 等

核心算法:
  - datenum_to_jed: 日期序号 → 儒略日
  - day_borrow/carry: 日期借位/进位运算
  - days_before_month: 月内天数累积

本项目映射:
  在等离子体模拟中, 使用多历法时间戳来:
  1. 为每次实验运行生成唯一标识符
  2. 提供跨时区的实验记录
  3. 在长时间模拟中追踪 "模拟日历"

  模拟日历: 将物理模拟时间映射到一种 "等离子体历法",
  其中 1 "等离子体年" = τ_c (碰撞时间), 以此类推.
"""

import math
import time
from datetime import datetime, timezone


# ===========================================================================
#  §1  儒略日计算  (源自 calpak datenum_to_jed)
# ===========================================================================
def gregorian_to_jdn(year, month, day):
    """Gregorian 日历 → 儒略日编号 (Julian Day Number).

    算法 (Meeus, Astronomical Algorithms):
      JDN = 367Y - ⌊7(Y + ⌊(M+9)/12⌋)/4⌋ + ⌊275M/9⌋ + D + 1721013.5

    源自 calpak 中的 datenum_values + datenum_to_jed.
    """
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    jdn = day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    return jdn


def jdn_to_gregorian(jdn):
    """儒略日编号 → Gregorian 日历.

    逆运算, 源自 calpak 中的逆向转换.
    """
    a = jdn + 32044
    b = (4 * a + 3) // 146097
    c = a - 146097 * b // 4
    d = (4 * c + 3) // 1461
    e = c - 1461 * d // 4
    m = (5 * e + 2) // 153

    day = e - (153 * m + 2) // 5 + 1
    month = m + 3 - 12 * (m // 10)
    year = 100 * b + d - 4800 + m // 10

    return year, month, day


# ===========================================================================
#  §2  日期借位/进位运算  (源自 calpak day_borrow/day_carry)
# ===========================================================================
def day_carry_common(year, month, day):
    """日期进位: 将日/月溢出进位到正确范围.

    源自 calpak day_carry_common.
    处理 month > 12 或 day > 当月天数 的情况.
    """
    days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    # 闰年修正
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        days_in_month[2] = 29

    while month > 12:
        month -= 12
        year += 1
    while month < 1:
        month += 12
        year -= 1

    while day > days_in_month[month]:
        day -= days_in_month[month]
        month += 1
        if month > 12:
            month = 1
            year += 1

    return year, month, day


# ===========================================================================
#  §3  等离子体模拟历法
# ===========================================================================
class PlasmaSimulationCalendar:
    """等离子体模拟专用历法.

    将物理模拟时间映射到可读的 "等离子体日期".

    定义:
      1 等离子体秒 (ps) = τ_c (碰撞时间)
      1 等离子体分 (pm) = 60 ps
      1 等离子体时 (ph) = 3600 ps
      1 等离子体日 (pd) = 86400 ps
      1 等离子体月 (pM) = 30 pd
      1 等离子体年 (pY) = 365 pd

    映射自 calpak:
      原项目: 多种历法系统的转换和运算
      本项目: 模拟时间 → 等离子体历法日期
    """

    def __init__(self, tau_c_seconds=1.0):
        """
        Parameters
        ----------
        tau_c_seconds : float  碰撞时间 [秒] (缩放因子)
        """
        self.tau_c = tau_c_seconds

    def sim_time_to_plasma_date(self, tau_dimensionless):
        """无量纲模拟时间 → 等离子体日期.

        Parameters
        ----------
        tau_dimensionless : float  无量纲时间 t/τ_c

        Returns
        -------
        date_str : str  等离子体日期字符串
        """
        # 无量纲时间 = 物理时间 / τ_c
        total_ps = tau_dimensionless  # 等离子体秒

        years = int(total_ps // 31536000)
        remainder = total_ps - years * 31536000
        months = int(remainder // 2592000)
        remainder -= months * 2592000
        days = int(remainder // 86400)
        remainder -= days * 86400
        hours = int(remainder // 3600)
        remainder -= hours * 3600
        minutes = int(remainder // 60)
        seconds = remainder - minutes * 60

        return (f"pY{years:04d}-pM{months+1:02d}-pD{days+1:02d} "
                f"{hours:02d}:{minutes:02d}:{seconds:06.3f}")

    def sim_time_to_wall_clock(self, tau_dimensionless):
        """无量纲模拟时间 → 实际物理时间 [秒]."""
        return tau_dimensionless * self.tau_c


# ===========================================================================
#  §4  实验时间戳生成器
# ===========================================================================
def generate_experiment_timestamp():
    """为实验运行生成唯一时间戳.

    结合:
    1. 真实世界日期 (Gregorian)
    2. 儒略日编号
    3. Unix 时间戳

    Returns
    -------
    timestamp : dict
    """
    now = datetime.now(timezone.utc)
    jdn = gregorian_to_jdn(now.year, now.month, now.day)
    unix_ts = time.time()

    return {
        "gregorian": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "jdn": jdn,
        "unix_timestamp": unix_ts,
        "iso_format": now.isoformat(),
    }


def format_experiment_header(experiment_name, params_dict):
    """格式化实验头信息.

    包含时间戳、参数、和历法信息.
    """
    ts = generate_experiment_timestamp()
    lines = [
        "=" * 70,
        f"实验: {experiment_name}",
        f"时间戳: {ts['gregorian']}",
        f"儒略日: {ts['jdn']}",
        f"Unix 时间: {ts['unix_timestamp']:.3f}",
        "-" * 70,
        "参数:",
    ]
    for key, val in params_dict.items():
        lines.append(f"  {key:20s} = {val}")
    lines.append("=" * 70)
    return "\n".join(lines)
