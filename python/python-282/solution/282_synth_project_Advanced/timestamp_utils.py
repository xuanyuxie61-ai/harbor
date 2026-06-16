"""
timestamp_utils.py
==================

实验时间戳、日历运算与循环调度工具模块。

融合种子项目：
    1412_weekday_zeller：Zeller 同余式计算星期几
    1412_weekday_zeller/i4_wrap：整数环绕运算

科学应用：
    1. 为 SEI 实验记录精确时间戳
    2. 基于日历的实验调度（循环充放电测试按工作日安排）
    3. 时间戳哈希用于结果文件命名

数学基础：
    Zeller 同余式（Gregorian 历法）：
        h = (q + ⌊13(m+1)/5⌋ + K + ⌊K/4⌋ + ⌊J/4⌋ - 2J) mod 7
    其中：
        q = 日, m = 月 (3=March, ..., 14=February)
        K = year mod 100, J = year // 100

作者: DA-Synthesis
"""

import math
import time
try:
    from . import sei_parameters as P
except ImportError:
    import sei_parameters as P


# ============================================================
#  整数环绕（参考 i4_wrap.m）
# ============================================================

def i4_wrap(value, lo, hi):
    """
    整数环绕到指定区间 [lo, hi]（参考 1412 i4_wrap.m）。

    数学形式：
        result = lo + mod(value - lo, hi - lo + 1)

    Parameters
    ----------
    value : int
        输入整数。
    lo : int
        下界。
    hi : int
        上界。

    Returns
    -------
    int
        环绕后的值。
    """
    if hi < lo:
        lo, hi = hi, lo
    width = hi - lo + 1
    if width <= 0:
        return lo
    return lo + (value - lo) % width


# ============================================================
#  Zeller 同余式（参考 weekday_gregorian.m）
# ============================================================

def weekday_gregorian(year, month, day):
    """
    使用 Zeller 同余式计算 Gregorian 历法的星期几（参考 weekday_gregorian.m）。

    算法：
        若 m < 3，则 m += 12, y -= 1
        h = (q + ⌊13(m+1)/5⌋ + K + ⌊K/4⌋ + ⌊J/4⌋ - 2J - 1) mod 7 + 1

    Parameters
    ----------
    year : int
        年份。
    month : int
        月份 (1-12)。
    day : int
        日期 (1-31)。

    Returns
    -------
    int
        星期几（1=Sunday, ..., 7=Saturday）。
    """
    m = month
    y = year
    if m < 3:
        m += 12
        y -= 1

    q = day
    K = y % 100
    J = y // 100

    h = (q
         + (13 * (m + 1)) // 5
         + K
         + K // 4
         - J // 4  # 修正：Gregorian 使用 -2J 或 +J//4 - J 取决于约定
         - 1) % 7
    # 调整到 1-7 范围
    w = h % 7 + 1
    return w


def weekday_julian(year, month, day):
    """
    Julian 历法的 Zeller 同余式（参考 weekday_julian.m）。

    Parameters
    ----------
    year, month, day : int
        日期。

    Returns
    -------
    int
        星期几。
    """
    m = month
    y = year
    if m < 3:
        m += 12
        y -= 1

    q = day
    K = y % 100
    J = y // 100

    h = (q
         + (13 * (m + 1)) // 5
         + K
         + K // 4
         - 2) % 7
    w = h % 7 + 1
    return w


def weekday_to_name(w):
    """
    将星期几的整数转换为名称。

    Parameters
    ----------
    w : int
        星期几 (1-7)。

    Returns
    -------
    str
        星期名称。
    """
    names = ["Sunday", "Monday", "Tuesday", "Wednesday",
             "Thursday", "Friday", "Saturday"]
    idx = i4_wrap(w - 1, 0, 6)
    return names[idx]


# ============================================================
#  日期有效性校验（参考 weekday_check_common.m）
# ============================================================

def is_valid_date(year, month, day):
    """
    检查日期是否有效。

    Parameters
    ----------
    year, month, day : int
        日期。

    Returns
    -------
    bool
        是否有效。
    """
    if month < 1 or month > 12:
        return False
    if day < 1:
        return False
    if year < 1:
        return False

    # 每月天数
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # 闰年判断
    is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    if is_leap:
        days_in_month[1] = 29

    return day <= days_in_month[month - 1]


# ============================================================
#  实验时间戳生成
# ============================================================

def generate_experiment_timestamp(year=None, month=None, day=None):
    """
    生成 SEI 实验的时间戳。

    Parameters
    ----------
    year, month, day : int or None
        日期，None 使用默认值。

    Returns
    -------
    dict
        时间戳信息。
    """
    if year is None:
        year = P.EXP_YEAR
    if month is None:
        month = P.EXP_MONTH
    if day is None:
        day = P.EXP_DAY

    valid = is_valid_date(year, month, day)
    w = weekday_gregorian(year, month, day)
    w_name = weekday_to_name(w)

    # 时间戳字符串（用于文件命名）
    ts_str = f"{year:04d}{month:02d}{day:02d}"

    # 时间戳哈希（简单哈希用于随机种子）
    ts_hash = (year * 10000 + month * 100 + day) % (2 ** 31)

    return {
        "year": year,
        "month": month,
        "day": day,
        "weekday": w,
        "weekday_name": w_name,
        "timestamp_string": ts_str,
        "timestamp_hash": ts_hash,
        "valid": valid,
    }


def schedule_cycling_experiments(n_cycles, start_date, rest_days=1):
    """
    安排循环充放电实验的日程（考虑休息日）。

    Parameters
    ----------
    n_cycles : int
        循环次数。
    start_date : tuple
        起始日期 (year, month, day)。
    rest_days : int
        每循环后的休息天数。

    Returns
    -------
    schedule : list[dict]
        每个循环的日期安排。
    """
    schedule = []
    y, m, d = start_date

    for cycle in range(n_cycles):
        # 检查是否为周末（不做实验）
        w = weekday_gregorian(y, m, d)
        # 周六 = 7, 周日 = 1 -> 顺延到周一
        while w == 1 or w == 7:
            d += 1
            if d > 30:  # 简化
                d = 1
                m += 1
                if m > 12:
                    m = 1
                    y += 1
            w = weekday_gregorian(y, m, d)

        schedule.append({
            "cycle": cycle + 1,
            "year": y,
            "month": m,
            "day": d,
            "weekday": weekday_to_name(w),
        })

        # 前进到下一个循环
        d += 1 + rest_days
        if d > 28:  # 简化月份处理
            d = d - 28
            m += 1
            if m > 12:
                m = 1
                y += 1

    return schedule


# ============================================================
#  Unix 时间戳工具
# ============================================================

def current_unix_timestamp():
    """
    获取当前 Unix 时间戳。

    Returns
    -------
    float
        Unix 时间戳。
    """
    return time.time()


def timestamp_to_iso(unix_ts):
    """
    将 Unix 时间戳转换为 ISO 格式字符串。

    Parameters
    ----------
    unix_ts : float
        Unix 时间戳。

    Returns
    -------
    str
        ISO 格式时间字符串。
    """
    t = time.gmtime(unix_ts)
    return f"{t.tm_year:04d}-{t.tm_mon:02d}-{t.tm_mday:02d}T{t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}Z"


# ============================================================
#  综合演示
# ============================================================

def run_timestamp_demo():
    """
    运行时间戳演示。

    Returns
    -------
    dict
        时间戳分析结果。
    """
    # 实验日期时间戳
    ts = generate_experiment_timestamp()

    # 循环实验调度
    schedule = schedule_cycling_experiments(
        n_cycles=5,
        start_date=(P.EXP_YEAR, P.EXP_MONTH, P.EXP_DAY),
        rest_days=1
    )

    # 当前时间
    now_unix = current_unix_timestamp()
    now_iso = timestamp_to_iso(now_unix)

    # Zeller 验证
    test_dates = [
        (2026, 6, 8),   # 项目日期
        (2000, 1, 1),   # 千年
        (1969, 7, 20),  # 登月
    ]
    zeller_results = []
    for y, m, d in test_dates:
        w = weekday_gregorian(y, m, d)
        name = weekday_to_name(w)
        zeller_results.append({
            "date": f"{y:04d}-{m:02d}-{d:02d}",
            "weekday_num": w,
            "weekday_name": name,
        })

    return {
        "experiment_timestamp": ts,
        "cycling_schedule": schedule,
        "current_time": {"unix": now_unix, "iso": now_iso},
        "zeller_verification": zeller_results,
    }


if __name__ == "__main__":
    result = run_timestamp_demo()
    ts = result['experiment_timestamp']
    print(f"[timestamp_utils] 实验时间戳")
    print(f"  日期: {ts['year']}-{ts['month']:02d}-{ts['day']:02d}")
    print(f"  星期: {ts['weekday_name']}")
    print(f"  时间戳字符串: {ts['timestamp_string']}")
    print(f"  时间戳哈希: {ts['timestamp_hash']}")

    print(f"\n  Zeller 验证:")
    for z in result['zeller_verification']:
        print(f"    {z['date']} -> {z['weekday_name']} ({z['weekday_num']})")

    print(f"\n  循环实验调度:")
    for s in result['cycling_schedule']:
        print(f"    Cycle {s['cycle']}: {s['year']}-{s['month']:02d}-{s['day']:02d} "
              f"({s['weekday']})")
