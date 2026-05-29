# -*- coding: utf-8 -*-
"""
================================================================================
Metabolic Resource Allocation for Synaptic Plasticity
================================================================================

This module implements algorithms for distributing limited metabolic resources
(ATP, amino acids, protein synthesis capacity) among synapses undergoing
plasticity.

Biological Motivation:
----------------------
Protein synthesis for synaptic plasticity is metabolically expensive.
A neuron has a limited metabolic budget B (ATP equivalents per unit time).
Each synapse i requires a cost c_i per unit weight change Δw_i.

Optimization Problem:
---------------------
Given target weight changes Δw_i^{target} (determined by Hebbian rules),
find actual weight changes Δw_i that maximize total plasticity while
respecting the metabolic budget:

    maximize   Σ_i |Δw_i|
    subject to Σ_i c_i · |Δw_i| ≤ B
               |Δw_i| ≤ |Δw_i^{target}|

Greedy Algorithm:
-----------------
Sort synapses by benefit-to-cost ratio r_i = |Δw_i^{target}| / c_i in
descending order. Allocate to synapses in this order until budget is exhausted.

This is analogous to the change-making problem where:
    - "coins" are synapses with different costs
    - "target" is the metabolic budget B
    - "value" is the plasticity benefit

Continuous Relaxation:
----------------------
For continuous weight changes, the greedy algorithm is optimal. The
Lagrangian is:

    L = Σ_i Δw_i + λ·(B - Σ_i c_i·Δw_i) + Σ_i μ_i·(Δw_i^{target} - Δw_i)

KKT conditions yield the optimal allocation strategy.

================================================================================
"""

import numpy as np
from typing import Tuple, Optional


def greedy_resource_allocation(
    target_changes: np.ndarray,
    costs: np.ndarray,
    budget: float,
    maximize_total: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Allocate metabolic budget among synapses using a greedy algorithm.

    Algorithm:
    ----------
    1. Compute benefit-to-cost ratios: r_i = |target_i| / cost_i
    2. Sort synapses by r_i in descending order
    3. For each synapse in sorted order:
       - Allocate as much as possible (up to target)
       - Deduct cost from remaining budget
    4. Stop when budget is exhausted

    Parameters
    ----------
    target_changes : np.ndarray
        Target weight changes for each synapse.
    costs : np.ndarray
        Metabolic cost per unit weight change for each synapse.
    budget : float
        Total metabolic budget. Must be non-negative.
    maximize_total : bool
        If True, maximize total absolute weight change.

    Returns
    -------
    allocated : np.ndarray
        Actual allocated weight changes.
    ratios : np.ndarray
        Benefit-to-cost ratios.
    remaining_budget : float
        Unused budget (should be < min(costs) if budget exhausted).

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    n = target_changes.shape[0]
    if costs.shape[0] != n:
        raise ValueError("target_changes and costs must have same length.")
    if budget < 0.0:
        raise ValueError("Budget must be non-negative.")
    if np.any(costs <= 0.0):
        raise ValueError("All costs must be positive.")

    target_abs = np.abs(target_changes)

    # Benefit-to-cost ratios
    ratios = target_abs / costs

    # Sort in descending order of ratio
    sorted_indices = np.argsort(-ratios)

    allocated = np.zeros(n)
    remaining_budget = budget

    for idx in sorted_indices:
        if remaining_budget <= 0.0:
            break

        # Maximum we can allocate to this synapse
        max_alloc = target_abs[idx]
        if max_alloc <= 0.0:
            continue

        # Cost to fully satisfy this synapse
        full_cost = max_alloc * costs[idx]

        if full_cost <= remaining_budget:
            # Fully allocate
            allocated[idx] = max_alloc
            remaining_budget -= full_cost
        else:
            # Partial allocation
            partial = remaining_budget / costs[idx]
            allocated[idx] = partial
            remaining_budget = 0.0

    # Preserve signs from target changes
    signs = np.sign(target_changes)
    allocated = allocated * signs

    return allocated, ratios, remaining_budget


def knapsack_plasticity_allocation(
    values: np.ndarray,
    costs: np.ndarray,
    budget: float,
) -> Tuple[np.ndarray, float]:
    """
    Solve the 0/1 knapsack variant for discrete plasticity events.

    Each synapse can either undergo full plasticity (value = |Δw_i|)
    or none at all. This models all-or-none synaptic tagging.

    Dynamic Programming Formulation:
    --------------------------------
    Let dp[b][i] = maximum value achievable with budget b using first i synapses.

    Recurrence:
        dp[b][i] = max(dp[b][i-1],
                       dp[b-cost_i][i-1] + value_i)   if b >= cost_i

    Parameters
    ----------
    values : np.ndarray
        Plasticity values for each synapse.
    costs : np.ndarray
        Metabolic costs.
    budget : float
        Total budget.

    Returns
    -------
    selected : np.ndarray
        Boolean array indicating which synapses are selected.
    total_value : float
        Total achieved plasticity value.
    """
    n = values.shape[0]
    if costs.shape[0] != n:
        raise ValueError("values and costs must have same length.")
    if budget < 0.0:
        raise ValueError("Budget must be non-negative.")
    if np.any(costs < 0.0):
        raise ValueError("Costs must be non-negative.")

    # Discretize budget for DP
    scale = 100.0
    B = int(np.ceil(budget * scale))
    c_scaled = np.maximum(np.round(costs * scale).astype(int), 1)
    v_scaled = np.maximum(np.round(values * scale).astype(int), 0)

    # 1D DP to save memory
    dp = np.zeros(B + 1)
    choice = np.zeros((B + 1, n), dtype=bool)

    for i in range(n):
        ci = c_scaled[i]
        vi = v_scaled[i]
        if ci > B:
            continue
        for b in range(B, ci - 1, -1):
            if dp[b - ci] + vi > dp[b]:
                dp[b] = dp[b - ci] + vi
                choice[b, :] = choice[b - ci, :].copy()
                choice[b, i] = True

    best_b = np.argmax(dp)
    selected = choice[best_b, :]
    total_value = np.sum(values[selected])

    return selected, total_value


def proportional_allocation(
    target_changes: np.ndarray,
    costs: np.ndarray,
    budget: float,
) -> np.ndarray:
    """
    Allocate resources proportionally to target changes.

    This represents a homeostatic allocation strategy where all synapses
    receive resources in proportion to their needs.

    Formula:
        α = B / Σ_i (c_i · |target_i|)
        Δw_i = α · target_i

    Parameters
    ----------
    target_changes : np.ndarray
        Target weight changes.
    costs : np.ndarray
        Costs per unit change.
    budget : float
        Total budget.

    Returns
    -------
    allocated : np.ndarray
        Allocated weight changes.
    """
    n = target_changes.shape[0]
    if costs.shape[0] != n:
        raise ValueError("Arrays must have same length.")
    if budget < 0.0:
        raise ValueError("Budget must be non-negative.")

    total_cost = np.sum(costs * np.abs(target_changes))
    if total_cost <= 1e-15:
        return np.zeros(n)

    alpha = min(1.0, budget / total_cost)
    return alpha * target_changes


def evaluate_allocation_efficiency(
    allocated: np.ndarray,
    target_changes: np.ndarray,
    costs: np.ndarray,
    budget: float,
) -> dict:
    """
    Evaluate the efficiency of a resource allocation.

    Metrics:
    --------
    - Total plasticity: Σ |Δw_i|
    - Budget utilization: Σ c_i·|Δw_i| / B
    - Target achievement: Σ |Δw_i| / Σ |target_i|
    - Cost efficiency: Σ |Δw_i| / Σ c_i·|Δw_i|
    - Gini coefficient of allocation inequality

    Parameters
    ----------
    allocated : np.ndarray
        Actual allocations.
    target_changes : np.ndarray
        Target changes.
    costs : np.ndarray
        Costs.
    budget : float
        Total budget.

    Returns
    -------
    metrics : dict
        Dictionary of efficiency metrics.
    """
    n = allocated.shape[0]

    total_plasticity = np.sum(np.abs(allocated))
    total_target = np.sum(np.abs(target_changes))
    total_spent = np.sum(costs * np.abs(allocated))

    budget_util = total_spent / budget if budget > 0 else 0.0
    target_achievement = total_plasticity / total_target if total_target > 0 else 0.0
    cost_eff = total_plasticity / total_spent if total_spent > 0 else 0.0

    # Gini coefficient
    sorted_alloc = np.sort(np.abs(allocated))
    cumsum = np.cumsum(sorted_alloc)
    if cumsum[-1] > 1e-15:
        gini = (n + 1.0 - 2.0 * np.sum(cumsum) / cumsum[-1]) / n
    else:
        gini = 0.0

    return {
        "total_plasticity": total_plasticity,
        "budget_utilization": budget_util,
        "target_achievement": target_achievement,
        "cost_efficiency": cost_eff,
        "gini_coefficient": gini,
    }


def simulate_metabolic_allocation(
    n_synapses: int = 50,
    budget_factor: float = 0.6,
    seed: int = 42,
) -> dict:
    """
    Simulate resource allocation across a population of synapses.

    Parameters
    ----------
    n_synapses : int
        Number of synapses.
    budget_factor : float
        Budget as fraction of total required cost.
    seed : int
        Random seed.

    Returns
    -------
    results : dict
        Comparison of different allocation strategies.
    """
    if n_synapses < 1:
        raise ValueError("n_synapses must be >= 1.")
    if not (0.0 <= budget_factor <= 1.0):
        raise ValueError("budget_factor must be in [0,1].")

    rng = np.random.default_rng(seed)

    # Generate heterogeneous synapses
    target_changes = rng.normal(0.0, 0.1, n_synapses)
    costs = rng.uniform(0.5, 2.0, n_synapses)

    total_required = np.sum(costs * np.abs(target_changes))
    budget = budget_factor * total_required

    # Greedy allocation
    alloc_greedy, ratios, _ = greedy_resource_allocation(target_changes, costs, budget)
    metrics_greedy = evaluate_allocation_efficiency(alloc_greedy, target_changes, costs, budget)

    # Proportional allocation
    alloc_prop = proportional_allocation(target_changes, costs, budget)
    metrics_prop = evaluate_allocation_efficiency(alloc_prop, target_changes, costs, budget)

    # Knapsack allocation (all-or-none)
    values = np.abs(target_changes)
    selected, total_val = knapsack_plasticity_allocation(values, costs, budget)
    alloc_knapsack = target_changes * selected
    metrics_knapsack = evaluate_allocation_efficiency(alloc_knapsack, target_changes, costs, budget)

    return {
        "greedy": {"allocated": alloc_greedy, "metrics": metrics_greedy},
        "proportional": {"allocated": alloc_prop, "metrics": metrics_prop},
        "knapsack": {"allocated": alloc_knapsack, "metrics": metrics_knapsack},
        "target_changes": target_changes,
        "costs": costs,
        "budget": budget,
    }


if __name__ == "__main__":
    results = simulate_metabolic_allocation()
    print("Greedy allocation metrics:")
    for k, v in results["greedy"]["metrics"].items():
        print(f"  {k}: {v:.4f}")
