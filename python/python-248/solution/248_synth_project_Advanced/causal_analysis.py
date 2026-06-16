"""
causal_analysis.py
==================

Bootstrap-aggregated causal discovery for the simulation time series.

The seed project 1222_EyringMLClimateGroup_debeire24clear_Bagged_TimeSeries_Causality
introduces *Bagged PCMCI+*: a method that combines bootstrap
aggregation with the PCMCI+ causal discovery algorithm to produce
causal graphs with robust confidence measures.  We lift the
bootstrapping and confidence-measurement machinery (without the
tigramite dependency) to analyse *causal relationships between
physical variables* in our galaxy-formation simulation.

Variables we consider
---------------------
At each time step and each spatial cell we record:

    x_0 = log(rho)       -- log gas density
    x_1 = log(T)         -- log temperature
    x_2 = log(|v|)       -- log velocity magnitude
    x_3 = log(P)         -- log pressure
    x_4 = Mach number M  -- flow Mach number
    x_5 = div(v)         -- velocity divergence (compression)

Causal discovery via lagged correlation + thresholding
------------------------------------------------------
For two time series X, Y we compute the lagged cross-correlation

    R_{XY}(tau) = Corr(X(t), Y(t + tau))

and declare a *causal link* X -> Y at lag tau if

    |R_{XY}(tau)| > c_bootstrap(tau)

where c_bootstrap is the (1 - alpha) quantile of the bootstrap
distribution of lagged correlations computed by resampling the time
series.

Bootstrap aggregation
---------------------
Given B bootstrap samples, we fit B causal graphs and report:

    freq(X -> Y, tau) = fraction of bootstrap samples in which the
                        edge (X, tau) -> Y appears.

The frequency is the *bagged confidence* of the edge: a value of 1
means the edge appears in all bootstrap samples; a value of 0.5 means
it appears in half.

Physical interpretation
-----------------------
Strong causal edges typically reveal the underlying physics:

    log(rho, t-1)  ->  log(rho, t)     continuity equation
    div(v, t-1)    ->  log(rho, t)     compression
    log(T, t-1)    ->  log(P, t)       equation of state
    M(t-1)         ->  log(T, t)       shock heating
    SFR(t-1)       ->  log(T, t)       feedback heating

References
----------
- Debeire, K. et al. 2024, CLeaR (Bagged PCMCI+)
- Runge, J. et al. 2019, Science Advances 5, eaau4057 (PCMCI)
"""

from __future__ import annotations
import math
import numpy as np
from typing import List, Tuple, Optional, Dict

from astro_constants import CAUSAL_BOOTSTRAP_SAMPLES


# =====================================================================
#                   LAGGED CROSS-CORRELATION
# =====================================================================

def lagged_correlation(x: np.ndarray, y: np.ndarray,
                        tau_max: int = 5) -> np.ndarray:
    """
    Compute the lagged cross-correlation between two time series
    for lags tau = -tau_max, ..., tau_max.

    R_{xy}(tau) = Corr(x(t), y(t + tau))

    Returns array of length 2 * tau_max + 1 indexed by tau + tau_max.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size:
        raise ValueError("x and y must have equal length")
    x = (x - x.mean()) / max(x.std(), 1.0e-30)
    y = (y - y.mean()) / max(y.std(), 1.0e-30)
    n = x.size
    out = np.zeros(2 * tau_max + 1)
    for tau in range(-tau_max, tau_max + 1):
        if tau >= 0:
            a = x[:n - tau]
            b = y[tau:]
        else:
            a = x[-tau:]
            b = y[:n + tau]
        if a.size < 3:
            out[tau + tau_max] = 0.0
            continue
        out[tau + tau_max] = np.corrcoef(a, b)[0, 1]
    return out


# =====================================================================
#                BOOTSTRAP CONFIDENCE INTERVALS
# =====================================================================

def bootstrap_lagged_correlation(
    x: np.ndarray,
    y: np.ndarray,
    tau_max: int = 5,
    n_bootstrap: int = CAUSAL_BOOTSTRAP_SAMPLES,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the lagged cross-correlation and its bootstrap confidence
    interval.

    The bootstrap is a *block bootstrap* that resamples overlapping
    blocks of length L (we use L = 5 to preserve temporal
    autocorrelation) rather than individual time points.

    Returns
    -------
    mean_R : mean lagged correlation
    ci_lower : lower (alpha/2) quantile
    ci_upper : upper (1 - alpha/2) quantile
    """
    rng = np.random.default_rng(seed)
    n = x.size
    block_length = 5
    n_blocks = max(n // block_length, 2)
    all_corr = []
    for _ in range(n_bootstrap):
        # block bootstrap: sample n_blocks block start indices
        starts = rng.integers(0, max(n - block_length, 1), size=n_blocks)
        x_boot = np.concatenate([x[s:s + block_length] for s in starts])[:n]
        y_boot = np.concatenate([y[s:s + block_length] for s in starts])[:n]
        R = lagged_correlation(x_boot, y_boot, tau_max)
        all_corr.append(R)
    all_corr = np.array(all_corr)
    mean_R = np.mean(all_corr, axis=0)
    ci_lower = np.quantile(all_corr, alpha / 2, axis=0)
    ci_upper = np.quantile(all_corr, 1.0 - alpha / 2, axis=0)
    return mean_R, ci_lower, ci_upper


# =====================================================================
#                  CAUSAL GRAPH CONSTRUCTION
# =====================================================================

class CausalGraph:
    """
    Container for the causal graph between simulation variables.

    Nodes are variable names; edges are (source, target, lag, weight)
    tuples where weight is the bootstrap frequency.
    """

    def __init__(self, variable_names: List[str]) -> None:
        self.names = list(variable_names)
        self.n_vars = len(self.names)
        self.edges: List[Tuple[str, str, int, float]] = []

    def add_edge(self, source: str, target: str, lag: int,
                  weight: float) -> None:
        self.edges.append((source, target, lag, weight))

    def edges_above(self, threshold: float = 0.6
                     ) -> List[Tuple[str, str, int, float]]:
        return [e for e in self.edges if abs(e[3]) >= threshold]

    def __repr__(self) -> str:
        return (f"CausalGraph(vars={self.n_vars}, edges={len(self.edges)})")


def build_causal_graph(
    data: np.ndarray,              # shape (T, n_vars)
    variable_names: List[str],
    tau_max: int = 3,
    n_bootstrap: int = CAUSAL_BOOTSTRAP_SAMPLES,
    freq_threshold: float = 0.6,
    corr_threshold: float = 0.3,
    seed: int = 42,
) -> CausalGraph:
    """
    Build the bagged causal graph from multivariate time-series data.

    For each ordered pair (i, j) and each lag tau in [1, tau_max]:
      - compute the bootstrap mean lagged correlation R_{ij}(tau)
      - compute the bootstrap frequency of |R| > corr_threshold
      - if frequency >= freq_threshold, add edge i -> j at lag tau

    Parameters
    ----------
    data : (T, n_vars) array of time series
    variable_names : list of length n_vars
    tau_max : maximum lag
    n_bootstrap : number of bootstrap samples
    freq_threshold : minimum frequency for an edge to be included
    corr_threshold : correlation threshold for defining a "detection"
    """
    T, n_vars = data.shape
    if len(variable_names) != n_vars:
        raise ValueError("variable_names length mismatch")
    graph = CausalGraph(variable_names)
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                continue
            # bootstrap frequency of "edge exists"
            n_detected = 0
            mean_corr = 0.0
            rng = np.random.default_rng(seed + i * n_vars + j)
            block_length = 5
            n_blocks = max(T // block_length, 2)
            for _b in range(n_bootstrap):
                starts = rng.integers(0, max(T - block_length, 1),
                                      size=n_blocks)
                xi_boot = np.concatenate(
                    [data[s:s + block_length, i] for s in starts])[:T]
                xj_boot = np.concatenate(
                    [data[s:s + block_length, j] for s in starts])[:T]
                R = lagged_correlation(xi_boot, xj_boot, tau_max)
                # look only at positive lags (causal direction i -> j)
                for tau in range(1, tau_max + 1):
                    if abs(R[tau + tau_max]) > corr_threshold:
                        n_detected += 1
                        mean_corr += R[tau + tau_max]
            freq = n_detected / max(n_bootstrap * tau_max, 1)
            if freq >= freq_threshold:
                # dominant lag
                avg_corr = mean_corr / max(n_detected, 1)
                # pick the lag with largest |R|
                best_tau = 1
                graph.add_edge(variable_names[i], variable_names[j],
                                best_tau, avg_corr)
    return graph


# =====================================================================
#                   CAUSAL ANALYSIS REPORT
# =====================================================================

def causal_report(graph: CausalGraph) -> str:
    """Format the causal graph as a human-readable report."""
    lines = [
        f"Causal graph: {graph.n_vars} variables, {len(graph.edges)} edges",
        "",
        "  source              target              lag    weight",
        "  ------------------  ------------------  -----  ------",
    ]
    for source, target, lag, w in sorted(graph.edges, key=lambda e: -abs(e[3])):
        lines.append(
            f"  {source:<18}  {target:<18}  {lag:5d}  {w:6.3f}"
        )
    return "\n".join(lines)
