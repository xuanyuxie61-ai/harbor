"""
temporal_scheduler.py
=====================

Modular-arithmetic temporal scheduling for the multi-fidelity UQ
pipeline.  Adapted from the weekday / Zeller's-congruence / Julian-
Ephemeris-Date code of projects 1411 and 1412.

Purpose
-------
The protoplanetary-disk simulation evolves over Myr timescales, and we
want to schedule fidelity evaluations according to a quasi-periodic
cadence that mimics the time-step hierarchy of a multi-scale integrator:
  - frequent cheap (level-0) evaluations,
  - occasional medium (level-1) evaluations,
  - rare expensive (level-2, level-3) evaluations.

We encode the cadence as modular arithmetic over a "simulation clock"
expressed in Julian Ephemeris Date (JED) format:
  - JED(T) = T [days] + 2440000.5  (standard offset)
  - weekday(T) = JED(T) mod 7      (from jed_to_weekday)
  - Zeller's congruence for calendar date <-> JED conversion.

A simulation time T [Myr] is assigned to a fidelity level l if:
  l = max { l' : T mod period[l'] == 0 (mod period[l']) }
with periods period = [1, 7, 30, 365] days.  This creates a hierarchical
time-step structure analogous to symplectic sub-stepping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple


# ----------------------------------------------------------------------
# Modular-arithmetic helpers (from i4_wrap, weekday_gregorian).
# ----------------------------------------------------------------------
def i4_wrap(i: int, ilo: int, ihi: int) -> int:
    """Wrap an integer into the range [ilo, ihi] (following i4_wrap.m)."""
    if ilo > ihi:
        ilo, ihi = ihi, ilo
    range_ = ihi - ilo + 1
    if range_ <= 0:
        raise ValueError("i4_wrap: empty range.")
    return ilo + ((i - ilo) % range_)


def weekday_gregorian(y: int, m: int, d: int) -> int:
    """Day of the week via Zeller's congruence (from weekday_gregorian.m).

    Returns 1 (Sunday) through 7 (Saturday).
    """
    if m < 3:
        m += 12
        y -= 1
    w = (
        d
        + ((m + 1) * 13) // 5
        + y
        + y // 4
        - y // 100
        + y // 400
        - 1
    ) % 7 + 1
    return w


def ymd_to_jed_gregorian(y: int, m: int, d: int) -> float:
    """Convert a YMD Gregorian date to Julian Ephemeris Date (JED)."""
    # Fliegel & Van Flandern (1968) formula.
    if m <= 2:
        y -= 1
        m += 12
    jed = (
        d
        + (153 * m - 457) // 5
        + 365 * y
        + y // 4
        - y // 100
        + y // 400
        + 1721119
    )
    return float(jed)


def jed_to_weekday(jed: float) -> int:
    """Return the weekday (1..7) from a Julian Ephemeris Date."""
    j = int(round(jed))
    w = i4_wrap(j + 2, 1, 7)
    return w


def weekday_to_name(w: int) -> str:
    """Map weekday index 1..7 to name."""
    names = ["Sunday", "Monday", "Tuesday", "Wednesday",
             "Thursday", "Friday", "Saturday"]
    if w < 1 or w > 7:
        return "Unknown"
    return names[w - 1]


# ----------------------------------------------------------------------
# Simulation clock in JED.
# ----------------------------------------------------------------------
def sim_time_to_jed(t_myr: float, t0_jed: float = 2440000.5) -> float:
    """Convert simulation time t [Myr] to JED-like clock value.

    We use t0_jed as the zero-point; 1 Myr = 365.25 * 1e6 days.
    """
    days = t_myr * 365.25e6
    return t0_jed + days


# ----------------------------------------------------------------------
# Fidelity cadence scheduler.
# ----------------------------------------------------------------------
@dataclass
class CadenceConfig:
    """Hierarchical fidelity cadence in simulation days."""
    period_days: List[float]      # period for each fidelity level
    name: str = "hierarchical"


DEFAULT_CADENCE = CadenceConfig(
    period_days=[1.0e5, 5.0e5, 2.0e6, 1.0e7],
    name="hierarchical",
)


def assign_fidelity_level(
    t_myr: float, cadence: CadenceConfig = DEFAULT_CADENCE
) -> int:
    """Assign a fidelity level to a simulation time t [Myr].

    The rule: we use the JED representation of t and pick the highest
    level l such that jed(t) mod period[l] is approximately zero.
    Fallback to level 0 if no period matches.
    """
    jed = sim_time_to_jed(t_myr)
    level = 0
    for l, period in enumerate(cadence.period_days):
        residue = jed % max(period, 1.0e-12)
        # Treat residue within 1% of the period as "zero".
        tol = 0.01 * period
        if residue < tol or (period - residue) < tol:
            level = l
    return level


def schedule_fidelity_batch(
    times_myr: List[float],
    cadence: CadenceConfig = DEFAULT_CADENCE,
) -> List[Tuple[float, int, float]]:
    """Schedule fidelity levels for a list of simulation times.

    Returns [(t_myr, level, jed), ...].
    """
    out: List[Tuple[float, int, float]] = []
    for t in times_myr:
        level = assign_fidelity_level(t, cadence)
        jed = sim_time_to_jed(t)
        out.append((t, level, jed))
    return out


# ----------------------------------------------------------------------
# Weekday diagnostic (for sanity-checking the JED conversion).
# ----------------------------------------------------------------------
def weekday_diagnostic() -> str:
    """Sanity check: known date 2026-06-07 should be a Sunday (weekday=1)."""
    y, m, d = 2026, 6, 7
    w = weekday_gregorian(y, m, d)
    name = weekday_to_name(w)
    jed = ymd_to_jed_gregorian(y, m, d)
    w2 = jed_to_weekday(jed)
    return (
        f"Date {y}-{m:02d}-{d:02d} -> weekday={w} ({name}), "
        f"JED={jed}, jed_to_weekday={w2}"
    )
