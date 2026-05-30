
import math
import random
from typing import List, Tuple, Optional


def miller_rabin_prime(n: int, k: int = 10) -> bool:
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False


    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def generate_primes(start: int, count: int) -> List[int]:
    primes = []
    candidate = max(start, 2)
    while len(primes) < count:
        if miller_rabin_prime(candidate):
            primes.append(candidate)
        candidate += 1
    return primes


def hash_family_seed(prime_idx: int, num_hashes: int = 4) -> List[Tuple[int, int, int]]:
    base_primes = generate_primes(1000 + prime_idx * 50, num_hashes + 1)
    p = base_primes[-1]
    seeds = []
    for i in range(num_hashes):
        a = base_primes[i] % (p - 1) + 1
        b = (base_primes[i] * 7 + 13) % (p - 1)
        seeds.append((p, a, b))
    return seeds


def robust_division(a: float, b: float, fallback: float = 0.0) -> float:
    if abs(b) < 1e-300:
        return fallback
    result = a / b
    if math.isinf(result) or math.isnan(result):
        return fallback
    return result


def safe_sqrt(x: float, fallback: float = 0.0) -> float:
    if x < 0:
        if x > -1e-12:
            return 0.0
        return fallback
    return math.sqrt(x)


def check_boundary(value: float, lower: float, upper: float, name: str = "value") -> float:
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"{name} is NaN or Inf: {value}")
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


def compute_gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def entropy_of_distribution(probs: List[float]) -> float:
    h = 0.0
    for p in probs:
        if p > 1e-15:
            h -= p * math.log2(p)
    return h


def kldivergence(p: List[float], q: List[float]) -> float:
    d = 0.0
    for pi, qi in zip(p, q):
        if pi > 1e-15 and qi > 1e-15:
            d += pi * math.log(pi / qi)
    return d
