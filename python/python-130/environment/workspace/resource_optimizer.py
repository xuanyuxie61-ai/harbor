# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple, Optional


def greedy_resource_allocation(
    target_changes: np.ndarray,
    costs: np.ndarray,
    budget: float,
    maximize_total: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = target_changes.shape[0]
    if costs.shape[0] != n:
        raise ValueError("target_changes and costs must have same length.")
    if budget < 0.0:
        raise ValueError("Budget must be non-negative.")
    if np.any(costs <= 0.0):
        raise ValueError("All costs must be positive.")

    target_abs = np.abs(target_changes)


    ratios = target_abs / costs


    sorted_indices = np.argsort(-ratios)

    allocated = np.zeros(n)
    remaining_budget = budget

    for idx in sorted_indices:
        if remaining_budget <= 0.0:
            break


        max_alloc = target_abs[idx]
        if max_alloc <= 0.0:
            continue


        full_cost = max_alloc * costs[idx]

        if full_cost <= remaining_budget:

            allocated[idx] = max_alloc
            remaining_budget -= full_cost
        else:

            partial = remaining_budget / costs[idx]
            allocated[idx] = partial
            remaining_budget = 0.0


    signs = np.sign(target_changes)
    allocated = allocated * signs

    return allocated, ratios, remaining_budget


def knapsack_plasticity_allocation(
    values: np.ndarray,
    costs: np.ndarray,
    budget: float,
) -> Tuple[np.ndarray, float]:
    n = values.shape[0]
    if costs.shape[0] != n:
        raise ValueError("values and costs must have same length.")
    if budget < 0.0:
        raise ValueError("Budget must be non-negative.")
    if np.any(costs < 0.0):
        raise ValueError("Costs must be non-negative.")


    scale = 100.0
    B = int(np.ceil(budget * scale))
    c_scaled = np.maximum(np.round(costs * scale).astype(int), 1)
    v_scaled = np.maximum(np.round(values * scale).astype(int), 0)


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
    n = allocated.shape[0]

    total_plasticity = np.sum(np.abs(allocated))
    total_target = np.sum(np.abs(target_changes))
    total_spent = np.sum(costs * np.abs(allocated))

    budget_util = total_spent / budget if budget > 0 else 0.0
    target_achievement = total_plasticity / total_target if total_target > 0 else 0.0
    cost_eff = total_plasticity / total_spent if total_spent > 0 else 0.0


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
    if n_synapses < 1:
        raise ValueError("n_synapses must be >= 1.")
    if not (0.0 <= budget_factor <= 1.0):
        raise ValueError("budget_factor must be in [0,1].")

    rng = np.random.default_rng(seed)


    target_changes = rng.normal(0.0, 0.1, n_synapses)
    costs = rng.uniform(0.5, 2.0, n_synapses)

    total_required = np.sum(costs * np.abs(target_changes))
    budget = budget_factor * total_required


    alloc_greedy, ratios, _ = greedy_resource_allocation(target_changes, costs, budget)
    metrics_greedy = evaluate_allocation_efficiency(alloc_greedy, target_changes, costs, budget)


    alloc_prop = proportional_allocation(target_changes, costs, budget)
    metrics_prop = evaluate_allocation_efficiency(alloc_prop, target_changes, costs, budget)


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
