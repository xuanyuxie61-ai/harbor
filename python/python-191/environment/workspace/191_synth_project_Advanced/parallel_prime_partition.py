"""
parallel_prime_partition.py

Prime Number Distribution for Parallel Load Balancing in Matrix Multiplication.

Scientific Background:
----------------------
1. Prime Number Theorem:
   The number of primes <= n is asymptotically:
   
       pi(n) ~ n / ln(n)
   
   More precisely (Hadamard-de la Vallee Poussin, 1896):
       pi(n) = Li(n) + O(n * exp(-c * sqrt(ln n)))
   where Li(n) = integral_2^n dt / ln(t) is the logarithmic integral.

2. Load Balancing via Prime Distribution:
   For a matrix of size N divided among P processes,
   the block size is N/P. To avoid pathological divisions,
   we prefer process counts P such that P is close to a square
   (for Cannon's algorithm) or has favorable factorizations.

3. Prime Sieving:
   The Sieve of Eratosthenes finds all primes up to N in O(N log log N):
       is_prime[2..N] = true
       for p = 2 to sqrt(N):
           if is_prime[p]:
               for multiple = p^2, p^2+p, ... <= N:
                   is_prime[multiple] = false

4. Application to Parallel Matrix Multiply:
   When partitioning an N x N matrix across P processes,
   if P is prime or has large prime factors, the grid dimensions
   may be non-optimal. This module selects process counts with
   small prime factors (smooth numbers) for better partitioning.
"""

import numpy as np
import math
from typing import List, Tuple


def prime_sieve(n: int) -> np.ndarray:
    """
    Sieve of Eratosthenes to find all primes <= n.
    
    Complexity: O(n log log n) time, O(n) space.
    
    Args:
        n: upper bound
    
    Returns:
        primes: array of prime numbers <= n
    """
    if n < 2:
        return np.array([], dtype=np.int64)
    
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    
    for p in range(2, int(math.sqrt(n)) + 1):
        if is_prime[p]:
            is_prime[p * p:n + 1:p] = False
    
    return np.nonzero(is_prime)[0]


def prime_pi(n: int) -> int:
    """
    Count primes <= n using sieve.
    
    Args:
        n: upper bound
    
    Returns:
        pi(n): prime counting function value
    """
    if n < 2:
        return 0
    return len(prime_sieve(n))


def prime_factorization(n: int) -> List[Tuple[int, int]]:
    """
    Compute prime factorization of n.
    
    Returns list of (prime, exponent) pairs.
    
    Args:
        n: positive integer
    
    Returns:
        factors: list of (p, e)
    """
    if n < 1:
        raise ValueError("n must be positive")
    
    factors = []
    d = 2
    while d * d <= n:
        count = 0
        while n % d == 0:
            count += 1
            n //= d
        if count > 0:
            factors.append((d, count))
        d += 1
    
    if n > 1:
        factors.append((n, 1))
    
    return factors


def is_smooth(n: int, max_prime: int = 7) -> bool:
    """
    Check if n is B-smooth (all prime factors <= max_prime).
    
    Args:
        n: positive integer
        max_prime: smoothness bound
    
    Returns:
        True if n is B-smooth
    """
    if n < 1:
        return False
    factors = prime_factorization(n)
    return all(p <= max_prime for p, _ in factors)


def find_optimal_process_count(
    matrix_size: int,
    max_procs: int,
    algorithm: str = "cannon"
) -> int:
    """
    Find an optimal process count for parallel matrix multiplication.
    
    For Cannon's algorithm: prefer perfect squares.
    For SUMMA: prefer smooth numbers for rectangular grids.
    
    Mathematical criterion:
        minimize |matrix_size / P - round(matrix_size / P)|
        subject to P <= max_procs and P has favorable factors.
    
    Args:
        matrix_size: dimension n of n x n matrix
        max_procs: maximum available processes
        algorithm: 'cannon' or 'summa'
    
    Returns:
        optimal process count
    """
    if max_procs < 1:
        return 1
    if matrix_size < 1:
        return 1
    
    best_p = 1
    best_score = float('inf')
    
    for p in range(1, max_procs + 1):
        block = matrix_size / p
        remainder = abs(block - round(block))
        
        if algorithm == "cannon":
            q = int(math.sqrt(p))
            is_square = (q * q == p)
            # Prefer larger p with same score for better parallelism
            score = remainder + (0.0 if is_square else 100.0)
        elif algorithm == "summa":
            factors = prime_factorization(p)
            n_factors = len(factors)
            score = remainder + 0.5 * n_factors
        else:
            score = remainder
        
        # Use <= to prefer larger process counts when scores tie
        if score <= best_score:
            best_score = score
            best_p = p
    
    return best_p


def balanced_block_sizes(
    n: int,
    p: int
) -> List[int]:
    """
    Compute balanced block sizes for dividing n items among p processes.
    
    Some processes get ceil(n/p) items, others get floor(n/p).
    
        q = n // p, r = n % p
        first r processes get q+1 items
        remaining p-r processes get q items
    
    Args:
        n: total items
        p: number of processes
    
    Returns:
        block_sizes: list of length p
    """
    if p < 1:
        raise ValueError("p must be >= 1")
    if n < 0:
        raise ValueError("n must be non-negative")
    
    q = n // p
    r = n % p
    
    sizes = []
    for i in range(p):
        if i < r:
            sizes.append(q + 1)
        else:
            sizes.append(q)
    
    return sizes


def compute_load_imbalance(block_sizes: List[int]) -> float:
    """
    Compute load imbalance metric.
    
        imbalance = (max_size - min_size) / avg_size
    
    Args:
        block_sizes: list of block sizes
    
    Returns:
        imbalance ratio
    """
    if not block_sizes:
        return 0.0
    avg = sum(block_sizes) / len(block_sizes)
    if avg == 0:
        return 0.0
    return (max(block_sizes) - min(block_sizes)) / avg


def prime_balanced_partition(
    n: int,
    max_procs: int
) -> Tuple[int, List[int], float]:
    """
    Use prime distribution theory to create a balanced partition.
    
    1. Find optimal process count
    2. Compute block sizes
    3. Evaluate load imbalance
    
    Args:
        n: matrix dimension
        max_procs: maximum processes
    
    Returns:
        p: chosen process count
        blocks: block sizes
        imbalance: load imbalance metric
    """
    p = find_optimal_process_count(n, max_procs, algorithm="cannon")
    blocks = balanced_block_sizes(n, p)
    imbalance = compute_load_imbalance(blocks)
    return p, blocks, imbalance


if __name__ == "__main__":
    primes = prime_sieve(100)
    print("Primes <= 100:", primes)
    print("pi(100) =", prime_pi(100))
    
    p, blocks, imb = prime_balanced_partition(256, 16)
    print("Partition 256x256 matrix:")
    print("  Processes:", p)
    print("  Block sizes:", blocks)
    print("  Imbalance:", imb)
