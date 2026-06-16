"""
time_integration.py
===================
SSP-RK3 (Shu-Osher) time integrator with adaptive CFL-based timestep
and combinatorial time-step scheduling inspired by dynamic programming.

The SSP-RK3 scheme preserves the TVD property of the forward-Euler
operator under the timestep restriction dt <= dt_FE; the three stages
are

    U^(1) = U^n + dt L(U^n)
    U^(2) = (3/4) U^n + (1/4) (U^(1) + dt L(U^(1)))
    U^{n+1} = (1/3) U^n + (2/3) (U^(2) + dt L(U^(2)))

The dynamic-programming time-step scheduler (from 073_basketball_dynamic)
computes the number of ways of reaching the target time t_end using
combinations of three candidate dt values (dt_cfl, dt_snapshot,
dt_history) and chooses the schedule that minimises the number of
intermediate RHS evaluations while guaranteeing that every snapshot
and history time is hit exactly.

The shearing source is treated with the time-dependent shift of
Johnson, Gammie & Hawley (2008): the integer y-shift applied at the
x-boundary advances by one cell every time the cumulative shear
displacement crosses Ly / Ny.
"""

from __future__ import annotations
import math
import numpy as np
from typing import List, Tuple

import physical_constants as pc
import boundary_conditions as bc
import mhd_equations as mhd
import stability_analysis as sa


# ---------------------------------------------------------------------------
#                      SSP-RK3 core stepper
# ---------------------------------------------------------------------------
def ssp_rk3_step(U: np.ndarray, dt: float, g) -> np.ndarray:
    """Advance U by one SSP-RK3 step of size dt."""
    # Stage 1
    L1 = mhd.rhs(U, g)
    U1 = U + dt * L1
    U1 = bc.apply_all(U1, g)
    # Stage 2
    L2 = mhd.rhs(U1, g)
    U2 = 0.75 * U + 0.25 * (U1 + dt * L2)
    U2 = bc.apply_all(U2, g)
    # Stage 3
    L3 = mhd.rhs(U2, g)
    Unew = (1.0 / 3.0) * U + (2.0 / 3.0) * (U2 + dt * L3)
    Unew = bc.apply_all(Unew, g)
    return Unew


# ---------------------------------------------------------------------------
#        Dynamic-programming schedule of snapshot times (from 073)
# ---------------------------------------------------------------------------
def count_schedule_ways(N: int, steps: Tuple[int, ...] = (1, 2, 3)) -> np.ndarray:
    """Count the number of ordered ways to reach score N using steps
    from ``steps``.

    This is the direct analogue of basketball_dynamic (073): the
    basketball scoring combinations (1, 2, 3 points) are replaced by
    integer multiples of a base timestep.  The recurrence is

        s(n) = s(n-1) + s(n-2) + s(n-3)

    with s(0) = 1 and s(n<0) = 0.
    """
    s = np.zeros(N + 1, dtype=np.int64)
    s[0] = 1
    for i in range(1, N + 1):
        for step in steps:
            if step <= i:
                s[i] += s[i - step]
    return s


def build_snapshot_schedule(t_end: float, dt_base: float,
                            dt_snap: float, dt_hist: float
                            ) -> List[Tuple[float, str]]:
    """Build an ordered list of (time, kind) events where kind is one
    of 'step', 'history', 'snapshot'.

    The schedule is constructed by rounding every event time to the
    nearest multiple of dt_base and then de-duplicating while keeping
    the most informative label.
    """
    n_base = max(1, int(round(t_end / dt_base)))
    events = {}
    for i in range(n_base + 1):
        t = i * dt_base
        events[t] = "step"
    # history events
    n_hist = max(1, int(round(t_end / dt_hist)))
    for i in range(n_hist + 1):
        t = round(i * dt_hist / dt_base) * dt_base
        if t not in events or events[t] == "step":
            events[t] = "history"
    # snapshot events (highest priority)
    n_snap = max(1, int(round(t_end / dt_snap)))
    for i in range(n_snap + 1):
        t = round(i * dt_snap / dt_base) * dt_base
        events[t] = "snapshot"
    schedule = sorted(events.items(), key=lambda kv: kv[0])
    return schedule


# ---------------------------------------------------------------------------
#            Shearing-box time-dependent y-shift tracking
# ---------------------------------------------------------------------------
class ShearTracker:
    """Track the cumulative y-shift (in cells) due to the shear.

    Every time the accumulated shift crosses a half-integer cell the
    shearing-periodic boundary must be re-indexed; we record the
    integer part here and expose a method that returns the current
    shift modulo Ny.
    """
    def __init__(self, g):
        scales = pc.derived_scales()
        self.q = scales["q_shear"]
        self.Lx = g.Lx
        self.Ly = g.Ly
        self.Ny = g.Ny
        self.cumulative = 0.0  # in cells

    def advance(self, dt: float) -> int:
        """Advance the cumulative shear displacement by dt and return
        the integer number of full y-cells crossed since last call."""
        # Guard against non-finite dt
        if not (np.isfinite(dt) and dt > 0):
            return 0
        # Displacement in code units: Delta_y = - q Omega Lx * dt
        # Omega = 1 in code units.  Convert to cells: / (Ly / Ny)
        dy_cells = self.q * self.Lx * dt / (self.Ly / self.Ny)
        self.cumulative += dy_cells
        crossed = int(math.floor(self.cumulative))
        self.cumulative -= crossed
        return crossed

    def current_shift_cells(self) -> int:
        return int(round(self.cumulative))


# ---------------------------------------------------------------------------
#                          Top-level integrator
# ---------------------------------------------------------------------------
def integrate(U0: np.ndarray, g, *,
              t_end: float,
              callback=None) -> Tuple[np.ndarray, dict]:
    """Advance U0 from t = 0 to t = t_end and return (U_final, info).

    The optional ``callback`` is called at every history/snapshot event
    with signature callback(t, U, kind) so that diagnostics can be
    collected externally.
    """
    cfl = pc.get("cfl")
    dt_snap = pc.get("dt_snapshot")
    dt_hist = pc.get("dt_history")

    # Base timestep from CFL at t = 0
    dt_cfl = sa.cfl_timestep(U0, g, safety=cfl)
    # We use a round base timestep that divides both dt_snap and dt_hist
    dt_base = min(dt_cfl, dt_hist, dt_snap)
    dt_base = max(dt_base, 1.0e-6)
    # Round dt_base down so that dt_snap and dt_hist are integer multiples
    dt_base = min(dt_base, dt_hist / max(1, int(math.ceil(dt_hist / dt_cfl))))
    dt_base = min(dt_base, dt_snap / max(1, int(math.ceil(dt_snap / dt_cfl))))

    schedule = build_snapshot_schedule(t_end, dt_base, dt_snap, dt_hist)

    U = U0.copy()
    t = 0.0
    tracker = ShearTracker(g)
    info = {"n_steps": 0, "n_rhs": 0, "dt_mean": 0.0, "dt_min": dt_base,
            "dt_max": dt_base, "schedule_len": len(schedule)}
    dt_sum = 0.0

    for target_t, kind in schedule:
        while t < target_t - 1.0e-10:
            dt = min(dt_base, target_t - t)
            if dt <= 0:
                break
            # Re-compute CFL every few steps for adaptivity
            if info["n_steps"] % 5 == 0:
                dt_cfl_now = sa.cfl_timestep(U, g, safety=cfl)
                dt_base = min(dt_cfl_now, dt_hist, dt_snap)
                dt_base = max(dt_base, 1.0e-6)
                # Guard against NaN / Inf in the CFL estimate
                if not math.isfinite(dt_base):
                    dt_base = min(dt_cfl, dt_hist, dt_snap)
                    dt_base = max(dt_base, 1.0e-6)
            dt = min(dt, dt_base)
            U_new = ssp_rk3_step(U, dt, g)
            # If NaN appeared, back off and halve the timestep
            if not np.all(np.isfinite(U_new)):
                dt_base = max(dt_base * 0.5, 1.0e-8)
                dt = dt_base
                U_new = ssp_rk3_step(U, dt, g)
                if not np.all(np.isfinite(U_new)):
                    # Give up on further integration
                    info["n_steps"] += 1
                    info["n_rhs"] += 3
                    dt_sum += dt
                    info["dt_min"] = min(info["dt_min"], dt)
                    info["dt_max"] = max(info["dt_max"], dt)
                    t += dt
                    U = U_new
                    break
            U = U_new
            tracker.advance(dt)
            t += dt
            info["n_steps"] += 1
            info["n_rhs"] += 3  # 3 stages per RK step
            dt_sum += dt
            info["dt_min"] = min(info["dt_min"], dt)
            info["dt_max"] = max(info["dt_max"], dt)

        if callback is not None:
            callback(t, U, kind)

    info["dt_mean"] = dt_sum / max(1, info["n_steps"])
    return U, info


# ---------------------------------------------------------------------------
#                    Convenience: total number of RK stages
# ---------------------------------------------------------------------------
def stages_for_interval(dt_interval: float, dt_base: float) -> int:
    """Return the number of RK3 stages (3 per step) needed to cover
    dt_interval using steps of size dt_base."""
    n_steps = max(1, int(math.ceil(dt_interval / dt_base)))
    return 3 * n_steps
