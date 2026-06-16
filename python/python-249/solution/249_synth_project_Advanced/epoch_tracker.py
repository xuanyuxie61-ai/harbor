# -*- coding: utf-8 -*-
"""
epoch_tracker.py
================
Cyclic phase / epoch bookkeeping for stellar burning stages.

Background
----------
A massive star progresses through a discrete sequence of nuclear burning
stages:

    H-burning  -> He-burning -> C-burning -> Ne-burning ->
    O-burning  -> Si-burning -> Fe-core (collapse)

During shell burning the same sequence appears *nested* radially
(onion-skin structure).  Each stage has a characteristic duration
spanning orders of magnitude (e.g. H-burning ~ 10^7 yr vs. Si-burning
~ 1 day for a 25 M_sun star).

This module adapts Zeller's-congruence-style modular arithmetic
(from the `weekday_zeller` seed project) to track

    - cyclic epoch index   e in {0, 1, ..., E-1}
    - nested epoch pairs   (outer, inner)  for onion-skin structures
    - phase within a stage phi in [0, 1)
    - wrapped stage counters that reset after passing through a cycle

The modular arithmetic is lifted from integers mod 7 (the weekdays)
to arbitrary cycle lengths, including *hierarchical* cycles where an
inner counter increments each time the outer counter completes a full
revolution (analogous to the way Julian Day -> calendar date
conversion handles leap-year cycles).

Key formulae
------------
Given cycle length C and an integer counter n,
    epoch      = n mod C
    full_cycle = n // C
For nested cycles with inner period c and outer period C:
    inner = n mod c
    outer = (n // c) mod C
These are exactly Zeller's modular reductions applied to
time-like counters in stellar evolution.
"""

from __future__ import annotations
from typing import List, Tuple, Dict
import math


# ---------------------------------------------------------------------
# Named stellar burning stages (order matters)
# ---------------------------------------------------------------------

BURNING_STAGES: List[Tuple[str, float]] = [
    # (name, typical duration in years for 25 M_sun)
    ("H_core_burning",   7.0e6),
    ("H_shell_burning",  1.0e6),
    ("He_core_burning",  7.0e5),
    ("He_shell_burning", 1.0e5),
    ("C_core_burning",   6.0e2),
    ("Ne_core_burning",  1.0),
    ("O_core_burning",   0.5),
    ("Si_core_burning",  3.0e-3),   # ~ 1 day
    ("Fe_core",          0.0),       # collapse
]

N_STAGES = len(BURNING_STAGES)
STAGE_NAMES = [name for name, _ in BURNING_STAGES]


# ---------------------------------------------------------------------
# Core modular-arithmetic helpers
# ---------------------------------------------------------------------

def i4_wrap(value: int, lo: int, hi: int) -> int:
    """Wrap integer value into the inclusive range [lo, hi].

    This is the same wrap-around used in the `weekday_check_common`
    seed project (1412_weekday_zeller), generalized from [1,7] to an
    arbitrary integer interval.

    The mapping is a modular reduction:
        i4_wrap(v, lo, hi) = lo + ((v - lo) mod (hi - lo + 1))
    which guarantees lo <= result <= hi for any integer input.
    """
    if hi < lo:
        raise ValueError("i4_wrap: hi must be >= lo")
    width = hi - lo + 1
    return lo + ((value - lo) % width)


def cyclic_phase(n: int, period: int) -> int:
    """Return the phase (index) of counter n within a cycle of length
    `period`.  Direct analogue of  w = mod(n, 7)  in Zeller's
    congruence but with user-specified period."""
    if period <= 0:
        raise ValueError("period must be positive")
    return n % period


def cyclic_full_cycles(n: int, period: int) -> int:
    """Return the number of complete cycles of length `period` in counter n.
    Direct analogue of  floor(y/4) - floor(y/100) + floor(y/400)
    in the Gregorian calendar."""
    if period <= 0:
        raise ValueError("period must be positive")
    return n // period


def hierarchical_counter(n: int, inner_period: int,
                          outer_period: int) -> Tuple[int, int]:
    """Hierarchical (outer, inner) counter.

    Like the Gregorian (century, year-within-century) or
    (year, day-of-year) decomposition.

    Given n ticks,
        inner = n mod inner_period
        outer = (n // inner_period) mod outer_period

    This is a two-level generalisation of Zeller's
        m' = mod(m + 12, 12)   (month-within-year)
        y' = y - floor(m'/12)  (year counter)
    """
    inner = n % inner_period
    outer = (n // inner_period) % outer_period
    return outer, inner


# ---------------------------------------------------------------------
# Stellar epoch tracker
# ---------------------------------------------------------------------

class StellarEpochTracker:
    """Track the burning-stage state of a stellar evolution run.

    Parameters
    ----------
    stage_index : int
        Initial burning stage index (0 = H-core, ...).
    subcycle : int
        Sub-index inside the stage (e.g. number of thermal pulses).

    Attributes
    ----------
    stage : int
        Current burning stage index.
    subcycle : int
        Sub-cycle counter.
    full_cycles : int
        Number of times the complete burning sequence has been cycled
        (useful for AGB stars that cycle H/He shell burning).
    """

    def __init__(self, stage_index: int = 0, subcycle: int = 0) -> None:
        self.stage = i4_wrap(stage_index, 0, N_STAGES - 1)
        self.subcycle = max(0, subcycle)
        self.full_cycles = 0
        self._n_ticks = 0

    @property
    def stage_name(self) -> str:
        return STAGE_NAMES[self.stage]

    @property
    def stage_duration_yr(self) -> float:
        return BURNING_STAGES[self.stage][1]

    # ------------------------------------------------------------------
    def advance(self, dt_yr: float) -> Dict[str, object]:
        """Advance the stage counter by dt_yr and return a record.

        The rule is:
            - accumulate dt_yr into a running clock
            - each time the clock crosses the stage duration, increment
              the subcycle and decrement the remaining clock
            - after a stage completes `max_subcycle` times, advance
              to the next stage (wrapping around)

        Returns a dict with
            stage, stage_name, subcycle, full_cycles, phase, ticks
        where phase in [0,1) is the fractional completion of the stage.
        """
        if dt_yr < 0.0:
            dt_yr = 0.0
        self._n_ticks += 1
        tau = self.stage_duration_yr
        if tau <= 0.0:
            # terminal Fe-core stage; no further advancement
            return self.snapshot()
        # Phase increment; if it exceeds 1 we wrap and advance stage
        phase_inc = dt_yr / tau
        phase = (self.subcycle + phase_inc)
        n_sub = int(phase)
        phase = phase - n_sub
        if n_sub > 0:
            new_stage = self.stage + n_sub
            if new_stage >= N_STAGES:
                self.full_cycles += new_stage // N_STAGES
                new_stage = new_stage % N_STAGES
            self.stage = new_stage
            self.subcycle = int(phase * 1000) % 1000   # keep internal counter
        else:
            self.subcycle = int(phase * 1000) % 1000
        return self.snapshot()

    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, object]:
        return {
            "stage":        self.stage,
            "stage_name":   self.stage_name,
            "subcycle":     self.subcycle,
            "full_cycles":  self.full_cycles,
            "phase":        self.subcycle / 1000.0,
            "ticks":        self._n_ticks,
            "stage_duration_yr": self.stage_duration_yr,
        }

    # ------------------------------------------------------------------
    def onion_skin(self, nshells: int) -> List[str]:
        """Build an onion-skin composition label for a star with
        `nshells` concentric burning shells.

        Each shell is assigned the stage that is `shell_index` steps
        behind the current core stage, wrapping modulo N_STAGES.
        This mirrors the hierarchical epoch decomposition
        hierarchical_counter(shell_idx, N_STAGES, N_STAGES).
        """
        if nshells < 1:
            return []
        skin = []
        for s in range(nshells):
            idx = i4_wrap(self.stage - s, 0, N_STAGES - 1)
            skin.append(STAGE_NAMES[idx])
        return skin


# ---------------------------------------------------------------------
# Diagnostic: "weekday name" for a stage (analogous to weekday_to_name)
# ---------------------------------------------------------------------

def stage_to_name(stage_index: int) -> str:
    """Return the name of burning stage `stage_index`, wrapping modulo
    the number of stages.  Direct analogue of weekday_to_name_common
    from seed 1412."""
    idx = i4_wrap(stage_index, 0, N_STAGES - 1)
    return STAGE_NAMES[idx]


def cycle_check(value: int, period: int) -> int:
    """Wrap integer value into [0, period-1].  Direct analogue of
    weekday_check_common."""
    return cyclic_phase(value, period)


# ---------------------------------------------------------------------
# Simple self-test
# ---------------------------------------------------------------------

def _self_test() -> None:
    """Exercise the basic modular arithmetic."""
    assert i4_wrap(10, 1, 7) == 3
    assert i4_wrap(-1, 1, 7) == 6    # -1 wraps to 6 (1 step before 1 = 1 step before 7)
    assert i4_wrap(0, 1, 7) == 7     # 0 wraps to 7 (boundary)
    assert cyclic_phase(23, 7) == 2
    outer, inner = hierarchical_counter(100, 12, 10)
    assert inner == 100 % 12 and outer == (100 // 12) % 10

    trk = StellarEpochTracker(stage_index=0, subcycle=0)
    for dt in [1.0e6, 5.0e6, 1.0e6, 2.0e5, 1.0e3, 0.5, 0.01]:
        rec = trk.advance(dt)
    assert 0 <= rec["stage"] < N_STAGES
    print("epoch_tracker self-test OK  (final stage =", rec["stage_name"], ")")


if __name__ == "__main__":
    _self_test()
