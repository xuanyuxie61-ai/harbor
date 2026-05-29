"""
FFT Dimension Optimization via Prime Factorization
==================================================
Derived from seed project 911_prime_factors (trial-division prime
factorization).

The Fast Fourier Transform (FFT) is most efficient when the array size
has only small prime factors (2, 3, 5, 7). We optimize grid dimensions
by finding the nearest integer N' ≥ N whose largest prime factor is
below a threshold (typically 7).

Prime factorization of n:
    n = p₁^{e₁} · p₂^{e₂} · … · p_k^{e_k}

where p₁ < p₂ < … < p_k are prime and e_i ≥ 1.

FFT optimization algorithm:
    For candidate n' in [N, N+max_search]:
        factors = prime_factors(n')
        if max(factors) ≤ 7:
            return n'

This ensures the FFT engine can use mixed-radix algorithms with
O(N log N) complexity dominated by highly composite numbers.
"""

import numpy as np

def prime_factors(n):
    """
    Compute all prime factors of n (with multiplicity) using trial division.

    Parameters
    ----------
    n : int
        Positive integer.

    Returns
    -------
    factors : list of int
    """
    if n < 1:
        raise ValueError("n must be positive.")
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

def largest_prime_factor(n):
    """Return the largest prime factor of n."""
    factors = prime_factors(n)
    return max(factors) if factors else 1

def next_fft_friendly_size(N, max_prime=7, max_search=128):
    """
    Find the smallest n ≥ N such that all prime factors of n ≤ max_prime.

    Parameters
    ----------
    N : int
        Minimum desired size.
    max_prime : int
        Maximum acceptable prime factor.
    max_search : int
        Maximum search range above N.

    Returns
    -------
    n : int
        FFT-friendly size.
    """
    for n in range(N, N + max_search):
        if n < 2:
            continue
        factors = prime_factors(n)
        if all(p <= max_prime for p in factors):
            return n
    # Fallback: next power of 2
    return 2**int(np.ceil(np.log2(N)))

def optimize_qg_grid(Nx, Ny, max_prime=7):
    """
    Optimize grid dimensions for spectral QG solver.

    Returns
    -------
    Nx_opt, Ny_opt : int
    """
    return next_fft_friendly_size(Nx, max_prime), next_fft_friendly_size(Ny, max_prime)
