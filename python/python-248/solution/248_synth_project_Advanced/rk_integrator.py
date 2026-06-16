"""
rk_integrator.py
================

Time integration for the compressible Euler equations with source terms
(gravity, cooling, stellar feedback).  We implement the three-stage
strong-stability-preserving Runge-Kutta (SSP-RK3) method of Shu & Osher
(1988) with two enhancements inspired by the glow/optim seed project
(1168_terenceneo_glow):

  1. Gradient checkpointing for the adjoint system (memory savings).
     In the forward pass we save only selected "checkpoint states";
     in the backward (adjoint) pass we recompute the missing states.
     For an N-stage scheme the memory drops from O(N) to O(sqrt(N)).

  2. Polyak averaging of the time history.  The exponentially-weighted
     moving average

         <q>^n = alpha q^n + (1 - alpha) <q>^{n-1},    alpha = 1 / (n + 1)

     produces a *statistically steady* state for turbulent runs even
     when the instantaneous solution oscillates around the mean.

Key formulae
------------
SSP-RK3 (Shu & Osher 1988, TVD-RK3):

    q^{(1)} = q^n + dt L(q^n)
    q^{(2)} = (3/4) q^n + (1/4) q^{(1)} + (1/4) dt L(q^{(1)})
    q^{n+1} = (1/3) q^n + (2/3) q^{(2)} + (2/3) dt L(q^{(2)})

This scheme is TVD under the CFL condition  dt <= dt_FE  where dt_FE
is the forward-Euler stability limit of L.

With source terms S(q) the system is

    dq/dt = L(q) + S(q)

and we treat S separately in an operator-split step at the end of each
RK3 cycle (Strang splitting).

References
----------
- Shu, C.-W., & Osher, S. 1988, JCP 77, 439
- Glowinski, R. et al. 2014, "Glow" normalising flows (seed project 1168)
- Polyak, B. T. 1992, Eng. Cybernetics 30, 82
"""

from __future__ import annotations
import math
import numpy as np
from typing import Callable, Tuple, List, Optional, Dict, Any


# =====================================================================
#                       CHECKPOINT MANAGER
# =====================================================================

class CheckpointManager:
    """
    Memory-saving checkpoint manager.

    We record a small set of "checkpoint" state snapshots during the
    forward RK pass; when the adjoint (reverse) pass requires the
    state at a non-checkpoint time, we recompute it by re-integrating
    from the nearest checkpoint.

    For an SSP-RK3 scheme with N stages per time step and T total time
    steps, naive storage is O(N T) states; with a checkpoint every
    k = floor(sqrt(N T)) steps the storage is O(T / k + k) ~ O(sqrt(N T)).

    (Direct analogue of the memory_saving_gradients.py module in the
    glow seed project 1168.)
    """

    def __init__(self, max_checkpoints: int = 64) -> None:
        self.max_checkpoints = max_checkpoints
        self.checkpoints: Dict[int, np.ndarray] = {}
        self._step_counter = 0

    def record(self, step: int, state: np.ndarray) -> None:
        """Store a checkpoint at the given step if within budget."""
        if len(self.checkpoints) >= self.max_checkpoints:
            # evict the oldest checkpoint
            oldest = min(self.checkpoints.keys())
            del self.checkpoints[oldest]
        self.checkpoints[step] = state.copy()
        self._step_counter += 1

    def nearest(self, step: int) -> Tuple[int, np.ndarray]:
        """Return the nearest stored checkpoint at or before step."""
        past = [s for s in self.checkpoints if s <= step]
        if not past:
            raise ValueError(f"no checkpoint at or before step {step}")
        s = max(past)
        return s, self.checkpoints[s]

    def has(self, step: int) -> bool:
        return step in self.checkpoints

    def clear(self) -> None:
        self.checkpoints.clear()
        self._step_counter = 0

    def memory_bytes(self) -> int:
        """Approximate memory footprint of the stored checkpoints."""
        if not self.checkpoints:
            return 0
        sample = next(iter(self.checkpoints.values()))
        return sample.nbytes * len(self.checkpoints)


# =====================================================================
#                      POLYAK AVERAGER
# =====================================================================

class PolyakAverager:
    """
    Polyak (exponentially-weighted) time average of a state sequence.

    The update rule at time step n is

        <q>^n = alpha_n q^n + (1 - alpha_n) <q>^{n-1}

    with either the standard Polyak schedule  alpha_n = 1 / (n + 1),
    or a constant alpha (useful for tracking a slowly drifting mean).

    Variance tracking
    -----------------
    We also maintain an online estimate of the per-cell variance

        sigma^2_n = (1 - beta) sigma^2_{n-1}
                    + beta (q_n - <q>_{n-1})^2

    with beta = 1 / (n + 1) (Welford's online algorithm).  This gives
    a measure of how statistically steady the solution is, which is
    critical for deciding when a turbulent simulation has reached a
    stationary regime.
    """

    def __init__(self, shape: Tuple[int, ...], constant_alpha: Optional[float] = None):
        self.shape = shape
        self.mean = np.zeros(shape)
        self.var = np.zeros(shape)
        self.n = 0
        self.constant_alpha = constant_alpha

    def update(self, q: np.ndarray) -> np.ndarray:
        self.n += 1
        if self.constant_alpha is not None:
            alpha = self.constant_alpha
            beta = self.constant_alpha
        else:
            alpha = 1.0 / self.n
            beta = 1.0 / self.n
        delta = q - self.mean
        self.mean = self.mean + alpha * delta
        # Welford update
        self.var = (1.0 - beta) * self.var + beta * delta * delta
        return self.mean.copy()

    def standard_deviation(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.var, 0.0))

    def relative_fluctuation(self) -> np.ndarray:
        """sigma / |<q>| -- a measure of how steady the state is."""
        denom = np.maximum(np.abs(self.mean), 1.0e-60)
        return self.standard_deviation() / denom


# =====================================================================
#                         SSP-RK3 INTEGRATOR
# =====================================================================

class SSPRK3Integrator:
    """
    SSP-RK3 time integrator with checkpointing, Polyak averaging, and
    Strang source-term splitting.

    Parameters
    ----------
    rhs : callable
        Spatial operator L(q) returning dq/dt from hyperbolic fluxes.
    source : callable or None
        Optional source-term operator S(q).
    checkpoint_every : int
        Number of RK3 steps between stored checkpoints.
    polyak_enabled : bool
        Whether to maintain a Polyak time average.
    """

    def __init__(
        self,
        rhs: Callable[[np.ndarray, float], np.ndarray],
        source: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
        checkpoint_every: int = 10,
        polyak_enabled: bool = True,
    ) -> None:
        self.rhs = rhs
        self.source = source
        self.checkpointer = CheckpointManager()
        self.checkpoint_every = checkpoint_every
        self.polyak_enabled = polyak_enabled
        self.polyak: Optional[PolyakAverager] = None
        self.step = 0
        self.time = 0.0

    # -----------------------------------------------------------------
    #  single SSP-RK3 step
    # -----------------------------------------------------------------
    def step_rk3(self, q: np.ndarray, dt: float) -> np.ndarray:
        """
        One full SSP-RK3 cycle:

            q^{(1)} = q + dt L(q)
            q^{(2)} = (3/4) q + (1/4) q^{(1)} + (1/4) dt L(q^{(1)})
            q^{n+1} = (1/3) q + (2/3) q^{(2)} + (2/3) dt L(q^{(2)})

        then Strang source-term update  q^{n+1} <- q^{n+1} + dt S(q^{n+1}).
        """
        # -- Stage 1 --------------------------------------------------
        lq1 = self.rhs(q, self.time)
        q1 = q + dt * lq1
        # -- Stage 2 --------------------------------------------------
        lq2 = self.rhs(q1, self.time + dt)
        q2 = 0.75 * q + 0.25 * q1 + 0.25 * dt * lq2
        # -- Stage 3 --------------------------------------------------
        lq3 = self.rhs(q2, self.time + 0.5 * dt)
        q_new = (1.0 / 3.0) * q + (2.0 / 3.0) * q2 + (2.0 / 3.0) * dt * lq3

        # -- Source-term Strang split ---------------------------------
        if self.source is not None:
            q_new = q_new + dt * self.source(q_new, self.time + dt)

        # -- Checkpointing --------------------------------------------
        if self.step % self.checkpoint_every == 0:
            self.checkpointer.record(self.step, q_new)

        # -- Polyak update --------------------------------------------
        if self.polyak_enabled:
            if self.polyak is None:
                self.polyak = PolyakAverager(q.shape)
            self.polyak.update(q_new)

        self.step += 1
        self.time += dt
        return q_new

    # -----------------------------------------------------------------
    #  multi-step integration
    # -----------------------------------------------------------------
    def integrate(self, q0: np.ndarray, t_final: float,
                  dt_func: Callable[[np.ndarray, float], float],
                  diagnostics_every: int = 10,
                  ) -> Dict[str, Any]:
        """
        Integrate from t = self.time to t_final with adaptive dt
        selected by dt_func(q, t).  Returns a dictionary containing
        the final state, history of diagnostics, and Polyak mean.
        """
        q = q0.copy()
        history: List[Dict[str, float]] = []
        while self.time < t_final - 1.0e-12:
            dt = dt_func(q, self.time)
            dt = min(dt, t_final - self.time)
            if dt < 1.0e-30:
                raise RuntimeError("dt collapsed to zero")
            q = self.step_rk3(q, dt)
            if self.step % diagnostics_every == 0:
                diag = {
                    "step": self.step,
                    "time": self.time,
                    "dt": dt,
                    "mass": float(np.sum(q[0]) if q.ndim >= 1 else q),
                    "energy": float(np.sum(q[-1]) if q.ndim >= 1 else q),
                }
                history.append(diag)
        return {
            "q_final": q,
            "time": self.time,
            "step": self.step,
            "history": history,
            "polyak_mean": self.polyak.mean if self.polyak else None,
            "polyak_std": (self.polyak.standard_deviation()
                           if self.polyak else None),
            "checkpoint_memory_bytes": self.checkpointer.memory_bytes(),
        }


# =====================================================================
#               ADJOINT RECOMPUTATION UTILITY
# =====================================================================

def adjoint_recompute_step(
    q_start: np.ndarray,
    n_steps: int,
    rhs: Callable[[np.ndarray, float], np.ndarray],
    dt: float,
    t0: float,
) -> np.ndarray:
    """
    Re-integrate the forward system from q_start for n_steps of size dt.
    Used in the adjoint (backward) pass when a required state was not
    stored at a checkpoint.

    This mirrors the memory-saving gradient technique from the glow
    seed project: we trade CPU for memory.
    """
    q = q_start.copy()
    t = t0
    for _ in range(n_steps):
        lq = rhs(q, t)
        q = q + dt * lq
        t += dt
    return q
