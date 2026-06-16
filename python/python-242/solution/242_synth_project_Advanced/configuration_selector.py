"""
configuration_selector.py  --  Shell-model configuration space selection
========================================================================
Fused seed:
    628_knapsack_values  -- 0/1 knapsack with all sub-knapsack values

The full shell-model many-body space grows combinatorially with the number
of valence nucleons and orbitals. For A ~ 20-40 nuclei in the sd-shell the
m-scheme dimension can exceed 10^8, far beyond the reach of exact Lanczos
diagonalisation.

We implement a knapsack-style pruning that selects the most important
configurations by a "physics value" (proximity to the Fermi surface) subject
to a maximum-dimension constraint.

Knapsack formulation:
    - Each candidate configuration c has
        weight(c)     = estimated cost of including c in the basis
        value(c)      = physical importance of c
    - We want to maximise  sum_{c in S} value(c)
      subject to        sum_{c in S} weight(c) <= MAX_DIM

The importance is estimated by:
    value(c) = exp(- (E_SPE(c) - E_Fermi)^2 / (2 sigma^2))
where E_SPE is the sum of single-particle energies in configuration c,
E_Fermi is the Fermi energy, and sigma is a width parameter.

Weight is taken as the computational cost ~ O(D^2) where D is the dimension
contribution of the configuration.

The knapsack_values seed provides the complete enumeration of sub-knapsack
values, which we reuse to efficiently evaluate the DP table for all possible
remaining capacities.

References:
    Whitehead et al., Rev. Mod. Phys. 49 (1977) 655
    Cottle et al., The Linear Complementarity Problem (1992)
"""

from __future__ import annotations
import math
import numpy as np
from nuclear_constants import R_EPSILON


def all_subknapsack_values(values: list, weights: list, capacity: int) -> list:
    """Compute the optimal value for every capacity 0, 1, ..., capacity.

    This is the "all sub-knapsack values" extension of the classical
    0/1 knapsack: instead of just the answer at capacity C, we return
    the full DP row dp[0..C].

    Adapted from 628_knapsack_values.

    Returns:
        list of length capacity + 1, where out[w] = best value achievable
        with total weight <= w.
    """
    n = len(values)
    dp = [0.0] * (capacity + 1)
    for i in range(n):
        v = values[i]
        w = weights[i]
        for j in range(capacity, w - 1, -1):
            cand = dp[j - w] + v
            if cand > dp[j]:
                dp[j] = cand
    return dp


def knapsack_selected_items(
    values: list, weights: list, capacity: int,
) -> tuple[float, list]:
    """0/1 knapsack returning (best_value, selected_indices)."""
    n = len(values)
    # DP with back-pointers
    dp = [[0.0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        v = values[i - 1]
        w = weights[i - 1]
        for j in range(capacity + 1):
            dp[i][j] = dp[i - 1][j]
            if j >= w:
                cand = dp[i - 1][j - w] + v
                if cand > dp[i][j]:
                    dp[i][j] = cand
    # Back-track
    selected = []
    j = capacity
    for i in range(n, 0, -1):
        if dp[i][j] != dp[i - 1][j]:
            selected.append(i - 1)
            j -= weights[i - 1]
    selected.reverse()
    return dp[n][capacity], selected


def configuration_value(
    config: tuple, spe_dict: dict, orbitals: list,
    E_fermi: float, sigma: float = 5.0,
) -> float:
    """Importance of a configuration (Gaussian weighting around the Fermi energy).

    config: tuple of occupation numbers (p_0, p_1, ..., p_{K-1})
    """
    E_SPE = 0.0
    for k, occ in enumerate(config):
        orb = orbitals[k]
        key = (orb[0], orb[1], orb[2], orb[3])
        E_SPE += occ * 2.0 * spe_dict.get(key, 0.0)
    return math.exp(-((E_SPE - E_fermi) ** 2) / (2.0 * sigma * sigma))


def configuration_weight(config: tuple, orbitals: list) -> int:
    """Computational weight of a configuration: product of (Omega_k choose p_k).

    This is the dimension of the subspace associated with the configuration.
    """
    w = 1
    for k, occ in enumerate(config):
        orb = orbitals[k]
        omega_k = max(1, int(round((2.0 * orb[2] + 1.0) / 2.0)))
        # Binomial coefficient C(Omega_k, p_k)
        w *= math.comb(omega_k, occ) if 0 <= occ <= omega_k else 0
    return max(w, 1)


def select_configurations(
    all_configs: list,
    spe_dict: dict,
    orbitals: list,
    max_dim: int,
    E_fermi: float = -10.0,
    sigma: float = 5.0,
) -> list:
    """Select a subset of configurations whose total dimension <= max_dim,
    maximising the total physics value.

    Returns the list of selected configurations.
    """
    if not all_configs:
        return []
    values = [configuration_value(c, spe_dict, orbitals, E_fermi, sigma) for c in all_configs]
    weights = [configuration_weight(c, orbitals) for c in all_configs]
    # Cap the knapsack capacity to max_dim
    capacity = max_dim
    # Use the selected-items version for small problems, greedy for large ones
    n = len(all_configs)
    if n * capacity < 500000:
        _, selected_idx = knapsack_selected_items(values, weights, capacity)
    else:
        # Greedy by value / weight ratio
        ratios = [(values[i] / max(weights[i], 1), i) for i in range(n)]
        ratios.sort(reverse=True)
        selected_idx = []
        running_weight = 0
        for _, i in ratios:
            if running_weight + weights[i] <= capacity:
                selected_idx.append(i)
                running_weight += weights[i]
    return [all_configs[i] for i in selected_idx]


def fermi_energy_estimate(spe_dict: dict, n_particles: int, n_kind: str = "neutron") -> float:
    """Estimate the Fermi energy from the sorted SPE spectrum.

    Fill the lowest SPE states with n_particles particles (two per orbital).
    """
    pairs = [(key, val) for key, val in spe_dict.items() if key[3] == n_kind]
    pairs.sort(key=lambda x: x[1])
    filled = 0
    E_fermi = 0.0
    for key, val in pairs:
        if filled + 2 > n_particles:
            break
        E_fermi = val
        filled += 2
    return E_fermi


def dimension_breakdown(selected_configs: list, orbitals: list) -> dict:
    """Diagnostic: breakdown of the selected basis by configuration."""
    breakdown = []
    total_dim = 0
    for config in selected_configs:
        w = configuration_weight(config, orbitals)
        breakdown.append({"config": config, "dim": w})
        total_dim += w
    return {"total_dim": total_dim, "n_configs": len(selected_configs), "breakdown": breakdown}
