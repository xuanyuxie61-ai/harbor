# -*- coding: utf-8 -*-
"""
Echo-state-network adaptive estimator for the quantum critical point.

We reuse the ESN + RLS architecture of the online-learning RC-control
project (JonyeeShen, RoboSoft 2025) but replace the control-theoretic
target with the *detection* of the TFIM quantum critical point:

    input  u(t) = [lambda(t), L(t), E_0(L, lambda)]^T   (a trajectory
                   in parameter space)
    target y(t) = d^2 E_0 / d lambda^2   (the "specific heat" of the QCP)

As lambda crosses lambda_c, y(t) develops a sharp L-dependent peak;
the ESN is trained online via RLS to track this signal and the RLS
weights then give an *adaptive* estimator of lambda_c that improves
as more finite-L data is streamed in.

This approach complements the static FSS analysis in
``finite_size_scaling.py`` by producing a data-driven, sequential
estimate that can be updated in situ during an adaptive finite-L
sweep.
"""

from __future__ import annotations
from typing import Tuple, List
import numpy as np
try:
    from . import constants as C
except ImportError:
    import constants as C


# ---------------------------------------------------------------------------
# Echo State Network
# ---------------------------------------------------------------------------
class EchoStateNetwork:
    """Discrete-time ESN with tanh activation.

    Parameters
    ----------
    input_size : int
    reservoir_size : int
    spectral_radius : float
        Scales the recurrent weight matrix.  Values close to but below
        1 give the "edge of chaos" regime with long memory; values
        much smaller wash out temporal structure.
    leakage : float
        Leakage rate alpha in  (1-alpha) x + alpha tanh(...).
    """
    def __init__(self, input_size: int, reservoir_size: int = 200,
                  spectral_radius: float = 0.9, leakage: float = 0.3,
                  seed: int = 0):
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.leakage = leakage
        rng = np.random.default_rng(seed)

        # Input weights in [-1, 1]
        self.W_in = rng.uniform(-1.0, 1.0,
                                 size=(reservoir_size, input_size))
        # Sparse-ish recurrent matrix
        W = rng.uniform(-0.5, 0.5,
                         size=(reservoir_size, reservoir_size))
        # Scale to desired spectral radius (power iteration proxy)
        rho = float(np.max(np.abs(np.linalg.eigvals(W))))
        if rho > C.EPS_NUM:
            W *= spectral_radius / rho
        self.W = W
        self.state = np.zeros(reservoir_size)

    def reset(self) -> None:
        self.state[:] = 0.0

    def forward(self, x: np.ndarray) -> np.ndarray:
        """One ESN step.  x has shape (input_size,)."""
        u = self.W_in @ np.asarray(x, dtype=float).reshape(-1)
        self.state = ((1.0 - self.leakage) * self.state
                      + self.leakage * np.tanh(self.W @ self.state + u))
        return self.state


# ---------------------------------------------------------------------------
# Recursive Least Squares output layer
# ---------------------------------------------------------------------------
class RLSOutput:
    """RLS adaptive linear read-out mapping reservoir state -> output.

    The forgetting factor ``lam`` controls memory length.  Values
    close to 1 give an accumulating least-squares solution; smaller
    values track a time-varying target.
    """
    def __init__(self, input_size: int, output_size: int = 1,
                  delta: float = 1.0, lam: float = 0.995):
        self.delta = delta
        self.lam = lam
        self.P = np.eye(input_size) / delta
        self.w_out = np.zeros((output_size, input_size))

    def update(self, state: np.ndarray, target: np.ndarray) -> float:
        s = np.asarray(state, dtype=float).reshape(-1, 1)
        t = np.asarray(target, dtype=float).reshape(-1, 1)
        Ps = self.P @ s
        err = self.w_out @ s - t   # (output_size, 1)
        denom = self.lam + float(s.T @ Ps)
        if abs(denom) < C.EPS_NUM:
            return float(np.sum(np.abs(err)))
        K = Ps / denom   # (reservoir_size, 1)
        # w_out is (output_size, reservoir_size); update: w -= err @ K^T
        self.w_out = self.w_out - err @ K.T
        self.P = (self.P - K @ s.T @ self.P) / self.lam
        return float(np.sum(np.abs(err)))

    def predict(self, state: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=float).reshape(-1, 1)
        return (self.w_out @ s).flatten()


# ---------------------------------------------------------------------------
# Full adaptive estimator
# ---------------------------------------------------------------------------
class QCPAdaptiveEstimator:
    """Online estimator of the QCP location from a stream of
    (lambda, L, E_0) data."""
    def __init__(self, reservoir_size: int = 200,
                  spectral_radius: float = 0.9,
                  leakage: float = 0.3, lam: float = 0.995,
                  seed: int = 0):
        self.esn = EchoStateNetwork(input_size=3,
                                     reservoir_size=reservoir_size,
                                     spectral_radius=spectral_radius,
                                     leakage=leakage, seed=seed)
        self.rls = RLSOutput(input_size=reservoir_size,
                              output_size=1, delta=1.0, lam=lam)
        self.lam_history: List[float] = []
        self.pred_history: List[float] = []

    def ingest(self, lam: float, L: float,
                e0: float, target: float) -> Tuple[float, float]:
        """Feed one sample.  ``target`` is the known value of
        d^2 E_0 / d lambda^2 at (lam, L) (from FD, say).
        Returns (prediction, error)."""
        x = np.array([lam, L, e0])
        s = self.esn.forward(x)
        err = self.rls.update(s, np.array([target]))
        pred = float(self.rls.predict(s)[0])
        self.lam_history.append(lam)
        self.pred_history.append(pred)
        return pred, err

    def estimate_qcp(self) -> float:
        """Return the current best estimate of lambda_c as the value of
        lambda where the running prediction was maximal."""
        if not self.pred_history:
            return float("nan")
        arr = np.array(self.pred_history)
        idx = int(np.argmax(arr))
        return self.lam_history[idx]


# ---------------------------------------------------------------------------
# Offline training helper
# ---------------------------------------------------------------------------
def train_estimator(estimator: QCPAdaptiveEstimator,
                     lams: np.ndarray, Ls: np.ndarray,
                     e0s: np.ndarray, targets: np.ndarray,
                     warmup: int = 20) -> np.ndarray:
    """Train ``estimator`` on a batch and return the sequence of
    predictions.  The first ``warmup`` samples are used to warm up
    the reservoir without updating RLS (prevents corrupting P).
    """
    preds = np.zeros_like(lams, dtype=float)
    for k in range(len(lams)):
        x = np.array([lams[k], Ls[k], e0s[k]])
        s = estimator.esn.forward(x)
        if k < warmup:
            preds[k] = 0.0
            continue
        estimator.rls.update(s, np.array([targets[k]]))
        preds[k] = float(estimator.rls.predict(s)[0])
    return preds
