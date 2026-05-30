
import numpy as np
from typing import Tuple, List


def _ymd_to_jed_gregorian(y: int, m: int, d: int, f: float = 0.0) -> float:
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

    jed_gregorian_start = 2299161.0


    jed_g = _ymd_to_jed_gregorian(y, m, d, f)

    if jed_g >= jed_gregorian_start:
        return jed_g


    jed_j = _ymd_to_jed_julian(y, m, d, f)


    jed_boundary_low = 2299159.5
    jed_boundary_high = 2299160.5
    if jed_boundary_low < jed_j < jed_boundary_high:
        raise ValueError(
            "Date falls in the Gregorian calendar transition gap (1582/10/05–14)"
        )

    return jed_j


def jed_to_weekday(jed: float) -> Tuple[int, float]:
    jedmod = np.mod(jed, 7.0)
    j = int(np.mod(np.floor(jedmod), 7))
    f = (jedmod + 0.5) - j
    w = _i4_wrap(j + 2, 1, 7)
    return w, float(f)


def _i4_wrap(ival: int, ilo: int, ihi: int) -> int:
    if ilo > ihi:
        raise ValueError("ilo must not exceed ihi")
    wide = ihi - ilo + 1
    if wide == 0:
        return ival
    j = ilo + np.mod(ival - ilo, wide)
    return int(j)


def day_of_year(y: int, m: int, d: int) -> int:
    is_leap = (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)
    month_days = [31, 29 if is_leap else 28, 31, 30, 31, 30,
                  31, 31, 30, 31, 30, 31]
    if m < 1 or m > 12:
        raise ValueError("Month must be in [1, 12]")
    if d < 1 or d > month_days[m - 1]:
        raise ValueError(f"Day out of range for month {m}")
    return sum(month_days[:m - 1]) + d


def days_in_year(y: int) -> int:
    return 366 if ((y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)) else 365


def seasonal_phase(y: int, m: int, d: int) -> float:
    doy = day_of_year(y, m, d)
    diy = days_in_year(y)
    theta = 2.0 * np.pi * (doy - 1) / diy
    return float(theta)


def phase_locking_index(event_months: List[int]) -> float:
    if not event_months:
        return 0.0
    months = np.array(event_months, dtype=float)
    if np.any((months < 1) | (months > 12)):
        raise ValueError("Months must be in [1, 12]")
    complex_sum = np.sum(np.exp(2.0j * np.pi * months / 12.0))
    pli = np.abs(complex_sum / len(months))
    return float(pli)


def enso_event_timing(events: List[Tuple[int, int, int, float]]) -> dict:
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


    intervals = np.diff(jeds) / 365.25

    return {
        "jed_array": np.array(jeds),
        "phase_array": np.array(phases),
        "phase_locking_index": pli,
        "mean_interval_years": float(np.mean(intervals)) if len(intervals) > 0 else 0.0,
        "std_interval_years": float(np.std(intervals)) if len(intervals) > 0 else 0.0,
    }
