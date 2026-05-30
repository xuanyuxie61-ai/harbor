
import math






def date_to_jdn(year, month, day):
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    jdn = (day + (153 * m + 2) // 5 + 365 * y
           + y // 4 - y // 100 + y // 400 - 32045)
    return jdn


def jdn_to_date(jdn):
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
    return (year % 4 == 0) and ((year % 100 != 0) or (year % 400 == 0))


def days_in_month_gregorian(year, month):
    if month in (1, 3, 5, 7, 8, 10, 12):
        return 31
    elif month in (4, 6, 9, 11):
        return 30
    elif month == 2:
        return 29 if is_leap_year_gregorian(year) else 28
    else:
        raise ValueError("Invalid month")






def reactor_operation_timeline(start_date, end_date,
                               cycle_phases=None):
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
