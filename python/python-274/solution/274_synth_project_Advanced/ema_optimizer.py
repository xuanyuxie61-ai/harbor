# -*- coding: utf-8 -*-
"""
ema_optimizer.py
----------------
Exponential-moving-average (EMA) accelerated fixed-point iteration for the
self-consistent Eliashberg equations on the Matsubara axis.

Scientific origin of the fused algorithms
-----------------------------------------
* EMAOptimizer / long-horizon time-series benchmark
    (seed project 1174_MIMUW-RL_Unified-Long-Horizon-Time-Series-Benchmark)
    -> the PyTorch EMAOptimizer maintains a smoothed copy of parameters
          ema_w = decay * ema_w + (1 - decay) * w
       and uses it for evaluation.
    -> adapted: we keep a smoothed copy of the gap function
          Delta_ema = decay * Delta_ema + (1 - decay) * Delta_new
       and use it as the next iterate.  This is equivalent to *Anderson
       mixing* with a single memory and is dramatically more stable than
       plain Picard iteration for the Eliashberg equations.

Core physics / mathematics
--------------------------
* Eliashberg self-consistency at fixed T:
        Delta_n^{(t+1)} = F[Delta^{(t)}](omega_n)
  where F is the nonlinear RHS.  Plain iteration  Delta <- F(Delta)  often
  oscillates.  EMA damping replaces it with
        Delta_ema <- decay * Delta_ema + (1 - decay) * F(Delta_ema)
        Delta     <- Delta_ema
* Convergence criterion:
        ||Delta^{(t+1)} - Delta^{(t)}||_inf  <  tol
  combined with a maximum iteration count.

Stability / boundary notes
--------------------------
* Decay parameter close to 1 (e.g. 0.99) gives heavy damping (stable but
  slow); close to 0 (e.g. 0.1) is essentially plain Picard.
* If the iterates diverge (||Delta|| > 1e10) the loop aborts early.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# 1.  EMA fixed-point solver
# ---------------------------------------------------------------------------
@dataclass
class EMAResult:
    """Result of the EMA-accelerated self-consistency loop."""
    Delta: np.ndarray              # (M,) converged gap function
    residual_history: List[float] = field(default_factory=list)
    n_iter: int = 0
    converged: bool = False
    final_residual: float = 0.0


def ema_fixed_point(
    F: Callable[[np.ndarray], np.ndarray],
    Delta_init: np.ndarray,
    decay: float = 0.95,
    max_iter: int = 5000,
    tol: float = 1e-9,
    divergence_threshold: float = 1e10,
) -> EMAResult:
    """Solve  Delta = F(Delta)  with EMA damping.

    Parameters
    ----------
    F : callable
        The RHS of the Eliashberg equation; takes (M,) array, returns (M,).
    Delta_init : (M,) array
        Initial guess for the gap function.
    decay : float in (0, 1)
        EMA decay factor (closer to 1 = more damping).
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence threshold on the max-norm residual.
    divergence_threshold : float
        Abort if ||Delta||_inf exceeds this value.
    """
    if not (0.0 < decay < 1.0):
        raise ValueError("ema_fixed_point: decay must be in (0, 1)")

    Delta = np.array(Delta_init, dtype=float)
    Delta_ema = Delta.copy()
    history: List[float] = []

    for it in range(max_iter):
        # Evaluate RHS at the EMA point
        F_val = F(Delta_ema)

        # EMA update
        Delta_new = decay * Delta_ema + (1.0 - decay) * F_val

        # Residual
        res = float(np.max(np.abs(Delta_new - Delta_ema)))
        history.append(res)

        # Divergence guard
        if float(np.max(np.abs(Delta_new))) > divergence_threshold:
            return EMAResult(
                Delta=Delta_new,
                residual_history=history,
                n_iter=it + 1,
                converged=False,
                final_residual=res,
            )

        if res < tol:
            return EMAResult(
                Delta=Delta_new,
                residual_history=history,
                n_iter=it + 1,
                converged=True,
                final_residual=res,
            )
        Delta_ema = Delta_new

    return EMAResult(
        Delta=Delta_ema,
        residual_history=history,
        n_iter=max_iter,
        converged=False,
        final_residual=history[-1] if history else float("inf"),
    )


# ---------------------------------------------------------------------------
# 2.  Eliashberg RHS at fixed T
# ---------------------------------------------------------------------------
def build_eliashberg_rhs(
    temperature: float,
    lambda_kernel: np.ndarray,
    mu_star: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a callable F such that  F(Delta) is the RHS of the Eliashberg
    equation at fixed T, linearised in Delta near Tc:

        F(Delta)_n = (pi T) sum_m  [lambda_{n,m} - mu* delta_{n,m}]
                        Delta_m / |omega_m|
    where omega_m = pi T (2m+1).
    """
    import math
    M = lambda_kernel.shape[0]
    omega = math.pi * temperature * (2.0 * np.arange(M) + 1.0)
    prefactor = math.pi * temperature

    def rhs(Delta: np.ndarray) -> np.ndarray:
        out = np.zeros_like(Delta)
        for n in range(M):
            s = 0.0
            for m in range(M):
                kern_nm = (
                    lambda_kernel[n, m]
                    if lambda_kernel.ndim == 2
                    else lambda_kernel[abs(n - m)]
                )
                s += (kern_nm - mu_star * (1.0 if n == m else 0.0)) * Delta[m] / abs(
                    omega[m]
                )
            out[n] = prefactor * s
        return out

    return rhs


# ---------------------------------------------------------------------------
# 3.  Driver: self-consistent Tc search
# ---------------------------------------------------------------------------
@dataclass
class TcSearchResult:
    """Result of searching for Tc by bracketing."""
    Tc_estimate: float
    lambda_at_Tc: float
    search_temperatures: List[float]
    eigenvalues: List[float]


def search_tc_by_bracket(
    T_low: float,
    T_high: float,
    n_matsubara: int,
    build_kernel: Callable[[float], np.ndarray],
    mu_star: float,
    n_steps: int = 20,
    decay: float = 0.95,
) -> TcSearchResult:
    """Search for Tc by finding where the largest eigenvalue of the
    linearised Eliashberg kernel crosses 1.

    The kernel depends on T through both the Matsubara frequencies and
    the explicit pi T prefactor.
    """
    temps = np.linspace(T_low, T_high, n_steps)
    evals = []
    import math
    for T in temps:
        K = build_kernel(T)
        M = K.shape[0]
        omega = math.pi * T * (2.0 * np.arange(M) + 1.0)
        Mmat = np.zeros((M, M))
        for n in range(M):
            for m in range(M):
                kern_nm = K[n, m] if K.ndim == 2 else K[abs(n - m)]
                Mmat[n, m] = (
                    math.pi * T
                    * (kern_nm - mu_star * (1.0 if n == m else 0.0))
                    / abs(omega[m])
                )
        eigvals = np.linalg.eigvals(Mmat)
        evals.append(float(eigvals.real.max()))

    # Find where eval crosses 1
    tc_est = 0.5 * (T_low + T_high)
    lam_tc = 1.0
    for i in range(len(temps) - 1):
        if (evals[i] - 1.0) * (evals[i + 1] - 1.0) <= 0:
            # Linear interpolation
            slope = (evals[i + 1] - evals[i]) / (temps[i + 1] - temps[i])
            if abs(slope) > 1e-300:
                tc_est = float(temps[i] + (1.0 - evals[i]) / slope)
                lam_tc = 1.0
            break

    return TcSearchResult(
        Tc_estimate=tc_est,
        lambda_at_Tc=lam_tc,
        search_temperatures=temps.tolist(),
        eigenvalues=evals,
    )
