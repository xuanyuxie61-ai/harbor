# -*- coding: utf-8 -*-
"""
graph_optimization.py
---------------------
Combinatorial optimisation over the space of phonon-mode subsets that enter
the Eliashberg equations.  The "nodes" of the graph are the p Einstein
branches of the truncated phonon spectrum; an edge (i,j) means that the
two branches are coupled through the Migdal vertex.  The goal is to find
the *K-element subset* S of branches that maximises the effective coupling
    lambda_eff(S) = sum_{i,j in S}  V_{ij} / (1 + mu* N(0))

Scientific origin of the fused algorithms
-----------------------------------------
* GraphComBO / Blind  (seed project 1224_Conf-Blind-Sub_GraphComBO-Blind)
    -> graph-based Bayesian optimisation over combinatorial structures.
    -> adapted: we use a simpler *kernel-based greedy* variant where the
       "acquisition function" is the predicted lambda_eff of each candidate
       subset, computed from a precomputed kernel matrix on the graph.

Core physics / mathematics
--------------------------
* The coupling kernel on the mode graph is
        K_{ij} = int_BZ  g_i(k) g_j(k)  dk
  where g_i(k) is the electron-phonon matrix element for branch i at
  wavevector k.  After integrating out k we obtain a positive definite
  kernel matrix.
* The objective is a *quadratic pseudo-Boolean* function
        f(S) = sum_{i,j in S} K_{ij}
  which is submodular (since K is PSD) and can therefore be greedily
  maximised with a (1 - 1/e) approximation guarantee.

Stability / boundary notes
--------------------------
* K must be symmetric PSD; we project onto the PSD cone if small negative
  eigenvalues appear due to floating-point errors.
* The greedy search aborts early if the marginal gain drops below ``tol_gain``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1.  Graph construction
# ---------------------------------------------------------------------------
@dataclass
class ModeGraph:
    """Graph of phonon branches with coupling kernel."""
    n_modes: int
    K_matrix: np.ndarray            # (n_modes, n_modes) coupling kernel (PSD)
    eigenvalues: np.ndarray         # eigenvalues of K (for PSD check)
    branch_frequencies: np.ndarray  # (n_modes,) Einstein frequencies


def build_mode_graph(
    n_modes: int,
    frequencies: np.ndarray,
    coupling_strength: float = 1.0,
    decay_rate: float = 0.5,
) -> ModeGraph:
    """Construct the coupling kernel  K_{ij} = g * exp(-decay * |Omega_i - Omega_j|).

    The exponential kernel is PSD for any choice of frequencies, so the
    submodularity of the objective is guaranteed.
    """
    if n_modes < 1:
        raise ValueError("build_mode_graph: need at least 1 mode")
    freq = np.asarray(frequencies, dtype=float).reshape(-1)
    if freq.size < n_modes:
        freq = np.pad(freq, (0, n_modes - freq.size), constant_values=freq[-1] if freq.size > 0 else 1.0)
    freq = freq[:n_modes]

    diff = np.abs(freq[:, None] - freq[None, :])
    K = coupling_strength * np.exp(-decay_rate * diff)
    # Project onto PSD cone if needed
    eigvals, eigvecs = np.linalg.eigh(K)
    eigvals = np.maximum(eigvals, 0.0)
    K = eigvecs @ np.diag(eigvals) @ eigvecs.T
    K = 0.5 * (K + K.T)
    return ModeGraph(
        n_modes=n_modes,
        K_matrix=K,
        eigenvalues=eigvals,
        branch_frequencies=freq,
    )


# ---------------------------------------------------------------------------
# 2.  Greedy submodular maximisation
# ---------------------------------------------------------------------------
@dataclass
class GreedyResult:
    """Result of the greedy subset selection."""
    selected: List[int]            # indices of selected modes
    marginal_gains: List[float]    # marginal gain at each step
    total_gain: float              # f(selected)
    n_steps: int


def greedy_maximise(
    K: np.ndarray,
    budget: int,
    tol_gain: float = 1e-12,
) -> GreedyResult:
    """Greedy selection of a budget-sized subset S maximising  f(S) = 1^T K_{S,S} 1.

    At each step we add the element with the largest marginal gain
        Delta_i = f(S + {i}) - f(S) = 2 sum_{j in S} K_{ij} + K_{ii}.
    """
    n = K.shape[0]
    if budget < 1 or budget > n:
        raise ValueError("greedy_maximise: budget must be in [1, n]")
    selected: List[int] = []
    remaining = set(range(n))
    gains: List[float] = []
    total = 0.0

    for _ in range(budget):
        best_i = -1
        best_gain = -float("inf")
        for i in remaining:
            if not selected:
                gain = float(K[i, i])
            else:
                gain = 2.0 * sum(K[i, j] for j in selected) + float(K[i, i])
            if gain > best_gain:
                best_gain = gain
                best_i = i
        if best_i < 0 or best_gain < tol_gain:
            break
        selected.append(best_i)
        remaining.remove(best_i)
        gains.append(best_gain)
        total += best_gain

    return GreedyResult(
        selected=selected,
        marginal_gains=gains,
        total_gain=total,
        n_steps=len(selected),
    )


# ---------------------------------------------------------------------------
# 3.  Physics driver: optimal phonon subset for Tc
# ---------------------------------------------------------------------------
@dataclass
class OptimalPhononSubset:
    """Optimal K-mode subset with predicted effective coupling."""
    selected_indices: List[int]
    selected_frequencies: np.ndarray
    effective_lambda: float
    mu_star: float
    omega_log: float              # logarithmic average frequency of selected subset
    tc_mcmillan: float            # McMillan Tc estimate from the subset


def optimal_phonon_subset(
    mode_graph: ModeGraph,
    budget: int,
    mu_star: float = 0.10,
    V_ph: float = 0.30,
) -> OptimalPhononSubset:
    """Find the budget-sized phonon-mode subset that maximises lambda_eff.

    The effective coupling is
        lambda_eff(S) = V_ph * sum_{i,j in S} K_{ij}  -  mu_star * |S|
    and we use the greedy algorithm for submodular maximisation.
    """
    K = V_ph * mode_graph.K_matrix
    # Subtract mu* on the diagonal (the Coulomb pseudo-potential)
    K_shifted = K - mu_star * np.eye(mode_graph.n_modes)

    # Greedy on K_shifted (note: this is no longer PSD but the greedy
    # algorithm still gives a reasonable heuristic)
    res = greedy_maximise(K_shifted + mu_star * np.eye(mode_graph.n_modes), budget)
    # Compute lambda_eff on the selected subset
    S = res.selected
    K_sub = K[np.ix_(S, S)]
    lam_eff = float(K_sub.sum()) - mu_star * len(S)

    # Logarithmic average frequency
    freq_sub = mode_graph.branch_frequencies[S]
    if np.all(freq_sub > 0):
        omega_log = float(np.exp(np.mean(np.log(freq_sub))))
    else:
        omega_log = float(freq_sub.mean())

    # McMillan Tc
    try:
        from monte_carlo_fluctuations import mcmillan_tc
    except ImportError:
        from .monte_carlo_fluctuations import mcmillan_tc
    tc = mcmillan_tc(max(lam_eff, 1e-10), mu_star, omega_log)

    return OptimalPhononSubset(
        selected_indices=S,
        selected_frequencies=freq_sub,
        effective_lambda=lam_eff,
        mu_star=mu_star,
        omega_log=omega_log,
        tc_mcmillan=tc,
    )
