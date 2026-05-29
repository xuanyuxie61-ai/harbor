"""
Multi-stage membrane cascade optimization and network flow analysis.

Adapted from:
  - pagerank (power-method eigenvector / network ranking)
  - subset_sum (dynamic programming for optimal module loading)
  - doomsday (modular arithmetic for cyclic scheduling)
"""

import numpy as np
from utils import cyclic_schedule_step


def build_cascade_adjacency(stages, recycle_ratio=0.2):
    """
    Build the adjacency matrix for a membrane cascade network.
    Nodes 0..stages-1 represent stages; edges represent flow.
    """
    n = stages
    A = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        A[i, i + 1] = 1.0 - recycle_ratio
        if i > 0:
            A[i, i - 1] = recycle_ratio
    # Final stage sends permeate to product
    A[n - 1, n - 1] = 1.0
    return A


def adjacency_to_google_matrix(A, damping=0.15):
    """
    Convert adjacency matrix to Google matrix (PageRank style).
    G = (1-d) * S + d * (1/n) * 1 1^T
    where S is the column-stochastic transition matrix.
    """
    n = A.shape[0]
    # Column normalization
    col_sums = A.sum(axis=0)
    S = np.zeros_like(A)
    for j in range(n):
        if col_sums[j] > 0:
            S[:, j] = A[:, j] / col_sums[j]
        else:
            S[:, j] = 1.0 / n
    G = (1.0 - damping) * S + damping / n * np.ones((n, n), dtype=float)
    return G


def power_method_rank(G, max_iter=100, tol=1e-12):
    """
    Compute the dominant eigenvector of G using the power method.
    This represents the steady-state flow distribution (PageRank analogy).
    """
    n = G.shape[0]
    x = np.ones(n, dtype=float) / n
    for _ in range(max_iter):
        x_new = G.dot(x)
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    x = x / np.sum(x)
    return x


def compute_stage_cuts_from_rank(rank, nominal_cut):
    """
    Map rank vector to stage-cut distribution.
    Higher rank -> higher stage cut.
    """
    rank = np.asarray(rank, dtype=float)
    rank = rank / np.sum(rank)
    cuts = nominal_cut + 0.2 * (rank - 1.0 / len(rank))
    cuts = np.clip(cuts, 0.05, 0.95)
    return cuts


def subset_sum_optimal_loading(weights, target):
    """
    Dynamic-programming subset-sum to find the subset of membrane modules
    (with given weight = footprint or cost) that exactly matches a target
    plant capacity.

    Returns the subset indices.
    """
    weights = np.asarray(weights, dtype=int)
    target = int(target)
    n = len(weights)
    table = np.zeros(target + 1, dtype=int)
    # table[j] = last weight used to form sum j, 0 if impossible
    for i in range(n):
        w = int(weights[i])
        for j in range(target - w, -1, -1):
            if j == 0:
                if table[w] == 0:
                    table[w] = w
            else:
                if table[j] != 0 and table[j + w] == 0:
                    table[j + w] = w

    if table[target] == 0:
        return []

    # Backtrack
    result = []
    remaining = target
    while remaining > 0:
        w = table[remaining]
        if w == 0:
            break
        result.append(w)
        remaining -= w
    return result


def cascade_mass_balance(feed_flow, stage_cuts, rank_distribution,
                          feed_composition):
    """
    Solve the multi-stage cascade mass balance assuming perfect mixing
    on the permeate side of each stage.
    """
    n = len(stage_cuts)
    comp = feed_composition.copy()
    flow = feed_flow
    permeate_flows = np.zeros(n, dtype=float)
    retentate_flows = np.zeros(n, dtype=float)
    permeate_comps = []

    for i in range(n):
        theta = stage_cuts[i] * rank_distribution[i]
        theta = np.clip(theta, 0.0, 1.0)
        perm_flow = flow * theta
        ret_flow = flow * (1.0 - theta)
        permeate_flows[i] = perm_flow
        retentate_flows[i] = ret_flow
        # Update composition (simplified: CO2 enriches in permeate)
        alpha = 20.0  # ideal selectivity
        x_co2 = comp["CO2"]
        x_ch4 = comp["CH4"]
        denom = theta + (1.0 - theta) * alpha
        if denom <= 0:
            denom = 1e-30
        y_co2 = alpha * x_co2 / denom
        y_ch4 = 1.0 - y_co2
        permeate_comps.append({"CO2": y_co2, "CH4": y_ch4})
        # Retentate composition update
        if ret_flow > 0:
            comp["CO2"] = (flow * x_co2 - perm_flow * y_co2) / ret_flow
            comp["CH4"] = 1.0 - comp["CO2"]
        flow = ret_flow

    return permeate_flows, retentate_flows, permeate_comps


def cyclic_regeneration_schedule(stages, cycle_days=30, start_day=1):
    """
    Generate a staggered cyclic regeneration schedule for a cascade.
    Uses modular arithmetic adapted from doomsday algorithm.
    """
    schedule = []
    for s in range(stages):
        next_day = cyclic_schedule_step(start_day + s * (cycle_days // stages), cycle_days)
        schedule.append({"stage": s, "next_regeneration": next_day, "cycle": cycle_days})
    return schedule
