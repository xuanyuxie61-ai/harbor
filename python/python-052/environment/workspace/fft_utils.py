
import numpy as np

def prime_factors(n):
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
    factors = prime_factors(n)
    return max(factors) if factors else 1

def next_fft_friendly_size(N, max_prime=7, max_search=128):
    for n in range(N, N + max_search):
        if n < 2:
            continue
        factors = prime_factors(n)
        if all(p <= max_prime for p in factors):
            return n

    return 2**int(np.ceil(np.log2(N)))

def optimize_qg_grid(Nx, Ny, max_prime=7):
    return next_fft_friendly_size(Nx, max_prime), next_fft_friendly_size(Ny, max_prime)
