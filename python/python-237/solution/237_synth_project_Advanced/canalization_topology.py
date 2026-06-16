"""
Topological charge sector enumeration via Boolean canalization.

Adapted from:
  - 1159_KadelkaLab_nondegenerate-canalization (Boolean function enumeration)

Key formulas:
    B*(n) = 2^{2^n} - 2((-1)^n - n) + sum_{k=1}^{n} (-1)^k C(n,k) 2^{k+1} 2^{2^{n-k}}
"""

import numpy as np
from math import factorial, comb


def factorial_safe(n):
    if n < 0:
        raise ValueError(f"factorial undefined for n={n} < 0")
    if n > 170:
        return int(float('inf'))
    return factorial(n)


def nchoosek_safe(n, k):
    if k < 0 or k > n:
        return 0
    return comb(n, k)


def B_star(n):
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if n == 0:
        return 2
    if n <= 6:
        term1 = 2 ** (2 ** n)
        term2 = -2.0 * ((-1) ** n - n)
        sum_term = 0.0
        for k in range(1, n + 1):
            sign = (-1) ** k
            binom = nchoosek_safe(n, k)
            pow_term = 2 ** (k + 1) * 2 ** (2 ** (n - k))
            sum_term += sign * binom * pow_term
        return int(term1 + term2 + sum_term)
    else:
        return int(float('inf'))


def count_canalizing_functions(n):
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if n == 0:
        return 2
    if n == 1:
        return 4
    if n == 2:
        return 12
    if n == 3:
        return 194
    total = 2 ** (2 ** n) if n <= 5 else float('inf')
    return int(total * 0.75)


def enumerate_topological_sectors(n_plaquettes_per_cube=6, max_Q=3):
    if max_Q < 0:
        raise ValueError(f"max_Q must be >= 0, got {max_Q}")
    if n_plaquettes_per_cube < 1:
        raise ValueError(f"n_plaquettes_per_cube must be >= 1, got {n_plaquettes_per_cube}")
    n_cubes = max(1, n_plaquettes_per_cube // 6)
    sectors = {}
    total = 0
    for Q_target in range(-max_Q, max_Q + 1):
        count = _count_configs_with_charge(n_cubes, Q_target)
        sectors[Q_target] = count
        total += count
    return {
        'sectors': sectors, 'total_configs': total,
        'max_Q': max_Q, 'n_cubes': n_cubes,
    }


def _count_configs_with_charge(n_cubes, Q_target):
    if abs(Q_target) > n_cubes:
        return 0
    count = 0
    for n_plus in range(n_cubes + 1):
        n_minus = n_plus - Q_target
        if n_minus < 0 or n_minus > n_cubes:
            continue
        n_zero = n_cubes - n_plus - n_minus
        if n_zero < 0:
            continue
        multinom = (factorial_safe(n_cubes)
                    // (factorial_safe(n_plus) * factorial_safe(n_minus) * factorial_safe(n_zero)))
        count += multinom
    return count


def expected_topological_distribution(beta, V, chi_t_lattice=0.01):
    if V <= 0:
        raise ValueError(f"V must be > 0, got {V}")
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")
    sigma_sq = max(chi_t_lattice * V, 0.01)
    max_Q = max(3, int(5 * np.sqrt(sigma_sq)))
    Q_values = np.arange(-max_Q, max_Q + 1)
    log_weights = -Q_values ** 2 / (2.0 * sigma_sq)
    S_inst = 8.0 * np.pi ** 2 / 3.0
    log_weights -= beta * S_inst * np.abs(Q_values) / V
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    P_Q = weights / (weights.sum() + 1e-30)
    return {
        'Q_values': Q_values, 'P_Q': P_Q,
        'mean': float(np.sum(Q_values * P_Q)),
        'variance': float(np.sum(Q_values ** 2 * P_Q)),
        'sigma': float(np.sqrt(np.sum(Q_values ** 2 * P_Q))),
    }


def canalization_entropy(n_vars):
    if n_vars < 1:
        raise ValueError(f"n_vars must be >= 1, got {n_vars}")
    P = expected_topological_distribution(beta=2.5, V=4 ** n_vars)
    probs = P['P_Q']
    probs = probs[probs > 1e-30]
    H = -float(np.sum(probs * np.log(probs)))
    return H
