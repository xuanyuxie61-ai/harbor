
import numpy as np
import math
from typing import List, Tuple


def prime_sieve(n: int) -> np.ndarray:
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
    if n < 2:
        return 0
    return len(prime_sieve(n))


def prime_factorization(n: int) -> List[Tuple[int, int]]:
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
    if n < 1:
        return False
    factors = prime_factorization(n)
    return all(p <= max_prime for p, _ in factors)


def find_optimal_process_count(
    matrix_size: int,
    max_procs: int,
    algorithm: str = "cannon"
) -> int:
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

            score = remainder + (0.0 if is_square else 100.0)
        elif algorithm == "summa":
            factors = prime_factorization(p)
            n_factors = len(factors)
            score = remainder + 0.5 * n_factors
        else:
            score = remainder
        

        if score <= best_score:
            best_score = score
            best_p = p
    
    return best_p


def balanced_block_sizes(
    n: int,
    p: int
) -> List[int]:
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
