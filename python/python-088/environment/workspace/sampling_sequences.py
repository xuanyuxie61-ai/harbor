
import numpy as np
from typing import Tuple



_PRIMES = np.array([
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
    179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
    233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
    283, 293, 307, 311, 313, 317, 331, 337, 347, 349,
    353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
    419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
    467, 479, 487, 491, 499, 503, 509, 521, 523, 541
], dtype=int)


def _radical_inverse(n: int, base: int) -> float:
    n = int(abs(n))
    base = int(base)
    if base < 2:
        raise ValueError("Base must be >= 2")
    result = 0.0
    inv_base = 1.0 / base
    factor = inv_base
    while n > 0:
        result += (n % base) * factor
        n //= base
        factor *= inv_base
    return result


def hammersley_sequence(i1: int, i2: int, m: int, n_base: int = 1) -> np.ndarray:
    if m < 1 or m > 100:
        raise ValueError("Dimension m must be in [1, 100]")
    if i1 < 0 or i2 < 0:
        raise ValueError("Indices must be non-negative")
    if n_base < 1:
        raise ValueError("n_base must be >= 1")

    step = 1 if i1 <= i2 else -1
    length = abs(i2 - i1) + 1
    r = np.zeros((m, length))

    k = 0
    for i in range(i1, i2 + step, step):
        r[0, k] = (i % (n_base + 1)) / n_base if n_base > 0 else 0.0
        for dim in range(1, m):
            r[dim, k] = _radical_inverse(i, int(_PRIMES[dim - 1]))
        k += 1

    return r


def halton_sequence(n_samples: int, m: int, skip: int = 0) -> np.ndarray:
    if m < 1 or m > 100:
        raise ValueError("Dimension m must be in [1, 100]")
    r = np.zeros((n_samples, m))
    for i in range(n_samples):
        idx = i + skip
        for dim in range(m):
            r[i, dim] = _radical_inverse(idx, int(_PRIMES[dim]))
    return r


def latin_hypercube_sampling(n_samples: int, m: int) -> np.ndarray:
    samples = np.zeros((n_samples, m))
    for dim in range(m):

        perm = np.random.permutation(n_samples)
        samples[:, dim] = (perm + np.random.rand(n_samples)) / n_samples
    return samples


def quasi_monte_carlo_integral(
    f, dim: int, n_samples: int, domain: Tuple[float, float] = (0.0, 1.0),
    method: str = "hammersley"
) -> Tuple[float, float]:
    a, b = domain

    if method == "hammersley":
        points = hammersley_sequence(0, n_samples - 1, dim, n_base=n_samples)
        points = points.T
    elif method == "halton":
        points = halton_sequence(n_samples, dim, skip=100)
    elif method == "lhs":
        points = latin_hypercube_sampling(n_samples, dim)
    else:
        points = np.random.rand(n_samples, dim)


    points = a + (b - a) * points

    vals = f(points)
    mean = np.mean(vals)
    std_err = np.std(vals, ddof=1) / np.sqrt(n_samples) if n_samples > 1 else 0.0


    volume = (b - a) ** dim
    return float(mean * volume), float(std_err * volume)


def transform_to_gaussian(uniform_samples: np.ndarray) -> np.ndarray:
    n, m = uniform_samples.shape
    if m % 2 != 0:

        uniform_samples = np.column_stack([uniform_samples, uniform_samples[:, -1:]])
        m += 1

    result = np.zeros((n, m))
    for i in range(0, m, 2):
        u1 = np.clip(uniform_samples[:, i], 1e-15, 1.0 - 1e-15)
        u2 = np.clip(uniform_samples[:, i + 1], 1e-15, 1.0 - 1e-15)
        r = np.sqrt(-2.0 * np.log(u1))
        theta = 2.0 * np.pi * u2
        result[:, i] = r * np.cos(theta)
        result[:, i + 1] = r * np.sin(theta)

    return result[:, :m] if m % 2 == 0 else result[:, :-1]


def stratified_sampling(n_strata: int, dim: int) -> np.ndarray:
    total = n_strata ** dim
    if total > 1000000:
        raise ValueError("Too many strata for high dimensions")

    if dim == 1:
        samples = np.zeros((total, 1))
        for i in range(n_strata):
            samples[i, 0] = (i + np.random.rand()) / n_strata
        return samples


    grids = [np.linspace(0, 1, n_strata + 1) for _ in range(dim)]
    samples = []
    for idx in np.ndindex(*([n_strata] * dim)):
        point = np.zeros(dim)
        for d in range(dim):
            low = grids[d][idx[d]]
            high = grids[d][idx[d] + 1]
            point[d] = low + np.random.rand() * (high - low)
        samples.append(point)
    return np.array(samples)
