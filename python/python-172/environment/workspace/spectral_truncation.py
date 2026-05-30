# -*- coding: utf-8 -*-

import numpy as np


def fermat_is_prime(n, k=5):
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

        result = pow(int(a), int(n - 1), int(n))
        if result != 1:
            return False
    return True


def validate_random_seed(seed, min_bits=16):
    if seed < 2 ** min_bits:
        return False
    return fermat_is_prime(seed, k=5)


def generate_robust_seed():
    while True:
        seed = np.random.randint(2 ** 16, 2 ** 31)
        if fermat_is_prime(seed, k=3):
            return seed


def knapsack_brute(weights, costs, budget):
    n = len(weights)
    best_value = -1.0
    best_subset = np.zeros(n, dtype=bool)


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
    n = len(coefficients)
    weights = np.abs(coefficients)
    costs = np.ones(n)
    budget = int(budget_ratio * n)

    if n <= 20:

        _, mask = knapsack_brute(weights, costs, budget)
    else:

        indices = np.argsort(weights)[::-1]
        mask = np.zeros(n, dtype=bool)
        mask[indices[:budget]] = True

        mask[0] = True

    truncated_coef = coefficients.copy()
    truncated_coef[~mask] = 0.0
    return mask, truncated_coef


def adaptive_truncation_error_analysis(coefficients, threshold=1e-12):
    n = len(coefficients)
    errors = np.zeros(n)
    sorted_coef = np.sort(np.abs(coefficients))[::-1]
    cumulative = np.cumsum(sorted_coef ** 2)
    total = np.sum(coefficients ** 2)
    errors = np.sqrt(total - cumulative)
    n_required = np.searchsorted(errors, threshold, side='left') + 1
    n_required = min(n_required, n)
    return n_required, errors
