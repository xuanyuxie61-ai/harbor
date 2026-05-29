# -*- coding: utf-8 -*-
"""
Adaptive Spectral Truncation and Combinatorial Optimization
============================================================
Selects optimal spectral mode subsets under computational budget constraints
using combinatorial optimization, and validates random number quality.

Inspired by:
- knapsack_brute: brute-force subset enumeration for optimization
- prime_fermat: probabilistic primality testing for RNG seed validation

Mathematical formulation:
- Given N spectral modes with importance weights w_i (e.g., coefficient
  magnitudes |a_i|) and computational costs c_i (e.g., operation counts),
  find subset S maximizing sum_{i in S} w_i subject to sum_{i in S} c_i <= B.
- This is the classic 0/1 knapsack problem, NP-hard but solvable exactly
  for small N by brute-force enumeration over 2^N subsets.
- For spectral methods, we use a greedy approximation for large N combined
  with exact optimization for the most significant modes.
- Fermat primality test: for prime p and random a in [2,p-2], if a^{p-1} ≠ 1 (mod p),
  then p is composite. Used to validate the quality of pseudo-random seeds.
"""

import numpy as np


def fermat_is_prime(n, k=5):
    """
    Fermat probabilistic primality test.
    Adapted from is_prime_fermat.m.

    Parameters
    ----------
    n : int
        Number to test.
    k : int
        Number of test rounds.

    Returns
    -------
    is_probable_prime : bool
    """
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    for _ in range(k):
        a = np.random.randint(2, n - 1)
        if np.gcd(a, n) != 1:
            return False
        # Compute a^{n-1} mod n via binary exponentiation
        result = pow(int(a), int(n - 1), int(n))
        if result != 1:
            return False
    return True


def validate_random_seed(seed, min_bits=16):
    """
    Validate that a random seed corresponds to a prime number (probabilistic).
    This ensures the seed has good number-theoretic properties.

    Parameters
    ----------
    seed : int
        Proposed seed.
    min_bits : int
        Minimum bit length.

    Returns
    -------
    valid : bool
    """
    if seed < 2 ** min_bits:
        return False
    return fermat_is_prime(seed, k=5)


def generate_robust_seed():
    """
    Generate a random seed that passes the Fermat primality test.

    Returns
    -------
    seed : int
    """
    while True:
        seed = np.random.randint(2 ** 16, 2 ** 31)
        if fermat_is_prime(seed, k=3):
            return seed


def knapsack_brute(weights, costs, budget):
    """
    Solve 0/1 knapsack by brute-force enumeration.
    Adapted from knapsack_brute.m.

    Parameters
    ----------
    weights : ndarray
        Values/weights of items.
    costs : ndarray
        Costs of items.
    budget : float
        Total budget.

    Returns
    -------
    best_value : float
        Maximum achievable value.
    best_subset : ndarray
        Boolean mask of selected items.
    """
    n = len(weights)
    best_value = -1.0
    best_subset = np.zeros(n, dtype=bool)

    # Enumerate all 2^n subsets using binary representation
    for s in range(1 << n):
        subset = np.zeros(n, dtype=bool)
        total_cost = 0.0
        total_value = 0.0
        for i in range(n):
            if (s >> i) & 1:
                subset[i] = True
                total_cost += costs[i]
                total_value += weights[i]
        if total_cost <= budget and total_value > best_value:
            best_value = total_value
            best_subset = subset.copy()

    return best_value, best_subset


def greedy_spectral_truncation(coefficients, budget_ratio=0.5):
    """
    Greedy selection of spectral modes based on coefficient magnitude per mode index.
    Combines with exact knapsack for the top modes.

    Parameters
    ----------
    coefficients : ndarray
        Spectral coefficients.
    budget_ratio : float
        Fraction of total modes to keep.

    Returns
    -------
    mask : ndarray
        Boolean mask of retained modes.
    truncated_coef : ndarray
        Coefficients with truncated modes zeroed.
    """
    n = len(coefficients)
    weights = np.abs(coefficients)
    costs = np.ones(n)
    budget = int(budget_ratio * n)

    if n <= 20:
        # Exact brute-force for small n
        _, mask = knapsack_brute(weights, costs, budget)
    else:
        # Greedy: sort by weight/cost ratio (here cost=1, so sort by |coef|)
        indices = np.argsort(weights)[::-1]
        mask = np.zeros(n, dtype=bool)
        mask[indices[:budget]] = True
        # Always retain the mean mode (index 0)
        mask[0] = True

    truncated_coef = coefficients.copy()
    truncated_coef[~mask] = 0.0
    return mask, truncated_coef


def adaptive_truncation_error_analysis(coefficients, threshold=1e-12):
    """
    Analyze truncation error as a function of number of retained modes.

    Parameters
    ----------
    coefficients : ndarray
        Spectral coefficients.
    threshold : float
        Error tolerance.

    Returns
    -------
    n_required : int
        Minimum number of modes to achieve error < threshold.
    errors : ndarray
        Truncation error vs number of modes.
    """
    n = len(coefficients)
    errors = np.zeros(n)
    sorted_coef = np.sort(np.abs(coefficients))[::-1]
    cumulative = np.cumsum(sorted_coef ** 2)
    total = np.sum(coefficients ** 2)
    errors = np.sqrt(total - cumulative)
    n_required = np.searchsorted(errors, threshold, side='left') + 1
    n_required = min(n_required, n)
    return n_required, errors
