"""
protocol_optimizer.py
================================================================================
Optimal charging protocol design for lithium-ion batteries via combinatorial
optimization and trajectory search.

Injects core algorithms from:
  - 1363_tsp_brute  (exhaustive permutation enumeration via Trotter's algorithm)
  - 039_asa113      (transfer/swap iterative refinement)
  - 1291_treepack   (tree enumeration for protocol graph structure)

Scientific role:
  The charging protocol is a sequence of current steps I_1, I_2, ..., I_N
  with associated durations tau_1, ..., tau_N. Finding the optimal sequence
  that minimizes charging time while respecting thermal and degradation
  constraints is a combinatorial optimization problem. We:
    1. Enumerate current step permutations (TSP-inspired trajectory).
    2. Refine via transfer/swap to locally improve thermal objectives.
    3. Build tree-structured protocol graphs for hierarchical control.
================================================================================
"""

import numpy as np
from typing import List, Tuple, Callable
from particle_distribution import transfer_clustering, swap_clustering


# ==============================================================================
# Trotter permutation generation (from 1363_tsp_brute)
# ==============================================================================

def perm1_next3(p: np.ndarray) -> bool:
    """
    Generate next permutation in lexicographic order using Trotter's algorithm.
    Maps from 1363_tsp_brute/perm1_next3.m.
    Returns False if no more permutations.
    """
    n = len(p)
    # Find mobile element
    d = np.ones(n, dtype=int)
    # Simplified Steinhaus-Johnson-Trotter using adjacent transpositions
    # Find largest mobile integer
    mobile = -1
    mobile_idx = -1
    for i in range(n):
        if d[i] == 1 and i < n - 1 and p[i] > p[i + 1]:
            if p[i] > mobile:
                mobile = p[i]
                mobile_idx = i
        elif d[i] == -1 and i > 0 and p[i] > p[i - 1]:
            if p[i] > mobile:
                mobile = p[i]
                mobile_idx = i
    if mobile_idx == -1:
        return False
    # Swap mobile element
    if d[mobile_idx] == 1:
        p[mobile_idx], p[mobile_idx + 1] = p[mobile_idx + 1], p[mobile_idx]
        d[mobile_idx], d[mobile_idx + 1] = d[mobile_idx + 1], d[mobile_idx]
    else:
        p[mobile_idx], p[mobile_idx - 1] = p[mobile_idx - 1], p[mobile_idx]
        d[mobile_idx], d[mobile_idx - 1] = d[mobile_idx - 1], d[mobile_idx]
    # Reverse direction of all elements larger than mobile
    for i in range(n):
        if p[i] > mobile:
            d[i] = -d[i]
    return True


def all_permutations(n: int):
    """Yield all permutations of [0, 1, ..., n-1]."""
    p = np.arange(n, dtype=int)
    d = np.ones(n, dtype=int)
    yield p.copy()
    while True:
        mobile = -1
        mobile_idx = -1
        for i in range(n):
            if d[i] == 1 and i < n - 1 and p[i] > p[i + 1]:
                if p[i] > mobile:
                    mobile = p[i]
                    mobile_idx = i
            elif d[i] == -1 and i > 0 and p[i] > p[i - 1]:
                if p[i] > mobile:
                    mobile = p[i]
                    mobile_idx = i
        if mobile_idx == -1:
            break
        if d[mobile_idx] == 1:
            p[mobile_idx], p[mobile_idx + 1] = p[mobile_idx + 1], p[mobile_idx]
            d[mobile_idx], d[mobile_idx + 1] = d[mobile_idx + 1], d[mobile_idx]
        else:
            p[mobile_idx], p[mobile_idx - 1] = p[mobile_idx - 1], p[mobile_idx]
            d[mobile_idx], d[mobile_idx - 1] = d[mobile_idx - 1], d[mobile_idx]
        for i in range(n):
            if p[i] > mobile:
                d[i] = -d[i]
        yield p.copy()


def brute_force_protocol_search(
    current_levels: np.ndarray,
    durations: np.ndarray,
    objective: Callable[[np.ndarray, np.ndarray], float]
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Exhaustively search all permutations of current+duration pairs to minimize
    an objective function.  For small N only (N <= 6).
    Maps from 1363_tsp_brute/tsp_brute.m.
    """
    n = len(current_levels)
    if n > 6:
        raise ValueError("Brute force limited to N <= 6")
    best_perm = np.arange(n)
    best_cost = objective(current_levels[best_perm], durations[best_perm])
    for perm in all_permutations(n):
        cost = objective(current_levels[perm], durations[perm])
        if cost < best_cost:
            best_cost = cost
            best_perm = perm.copy()
    return current_levels[best_perm], durations[best_perm], best_cost


# ==============================================================================
# Thermal-aware protocol objective
# ==============================================================================

def thermal_charge_objective(currents: np.ndarray, durations: np.ndarray,
                             thermal_model_func: Callable,
                             max_temp_limit: float = 318.15) -> float:
    """
    Objective: minimize total charge time + penalty for temperature exceedance.
    """
    total_time = np.sum(durations)
    # Approximate max temperature rise proportional to I^2 * R * t
    # Simplified Joule heating estimate
    R_int = 0.01  # Ohm
    q_joule = np.sum(currents ** 2 * R_int * durations)
    T_rise_est = q_joule / 100.0  # simplified thermal mass
    penalty = max(0.0, T_rise_est - (max_temp_limit - 298.15))
    return total_time + 1e4 * penalty


# ==============================================================================
# Greedy + local refinement for larger protocols
# ==============================================================================

def greedy_thermal_protocol(
    target_capacity: float,
    current_options: np.ndarray,
    duration_step: float,
    thermal_model_func: Callable,
    max_temp: float = 318.15
) -> Tuple[List[float], List[float]]:
    """
    Build a charge protocol greedily: at each step, choose the largest
    current that does not violate the temperature constraint.
    """
    protocol_I = []
    protocol_t = []
    remaining = target_capacity
    t_accum = 0.0
    while remaining > 1e-6:
        best_I = current_options[0]
        best_dt = duration_step
        for I in sorted(current_options, reverse=True):
            dt = min(duration_step, remaining / abs(I + 1e-18))
            # Check thermal constraint (simplified)
            T_est = 298.15 + I ** 2 * 0.01 * (t_accum + dt) / 100.0
            if T_est <= max_temp:
                best_I = I
                best_dt = dt
                break
        protocol_I.append(best_I)
        protocol_t.append(best_dt)
        remaining -= abs(best_I) * best_dt
        t_accum += best_dt
        if len(protocol_I) > 1000:
            break
    return protocol_I, protocol_t


# ==============================================================================
# Clustered protocol segments (from 039_asa113)
# ==============================================================================

def cluster_protocol_segments(
    currents: np.ndarray,
    n_segments: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster current values into segments using transfer/swap optimization.
    Returns (segment_labels, segment_centers).
    """
    if len(currents) < n_segments:
        n_segments = max(1, len(currents))
    labels = np.zeros(len(currents), dtype=int)
    for k in range(n_segments):
        lo = int(k * len(currents) / n_segments)
        hi = int((k + 1) * len(currents) / n_segments)
        if hi <= lo and k > 0:
            hi = lo + 1
        labels[lo:hi] = k
    labels = transfer_clustering(currents, labels, n_segments)
    labels = swap_clustering(currents, labels, n_segments)
    centers = np.array([np.mean(currents[labels == k]) if np.any(labels == k) else 0.0
                        for k in range(n_segments)])
    return labels, centers
