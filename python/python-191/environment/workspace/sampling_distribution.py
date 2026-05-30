
import numpy as np
from typing import Tuple
import math


def normal_01_cdf(x: float) -> float:
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x) / math.sqrt(2.0)
    
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    
    return 0.5 * (1.0 + sign * y)


def normal_01_cdf_inv(p: float) -> float:
    if p <= 0.0:
        return -10.0
    if p >= 1.0:
        return 10.0
    

    a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00
    ]
    b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00
    ]
    
    p_low = 0.02425
    p_high = 1.0 - p_low
    
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    
    return x


def log_normal_truncated_ab_sample(
    mu: float,
    sigma: float,
    a: float,
    b: float
) -> float:
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if a < 0.0:
        raise ValueError("a must be non-negative")
    if a >= b:
        raise ValueError("require a < b")
    
    ln_a = math.log(a) if a > 0 else -1e300
    ln_b = math.log(b)
    
    cdf_a = normal_01_cdf((ln_a - mu) / sigma) if a > 0 else 0.0
    cdf_b = normal_01_cdf((ln_b - mu) / sigma)
    
    if cdf_b <= cdf_a:
        return math.exp(mu)
    
    u = np.random.uniform(cdf_a, cdf_b)
    z = normal_01_cdf_inv(u)
    x = math.exp(mu + sigma * z)
    
    return max(a, min(b, x))


def walker_build(prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    prob = np.asarray(prob, dtype=np.float64)
    if np.any(prob < -1e-12):
        raise ValueError("probabilities must be non-negative")
    n = len(prob)
    if n == 0:
        return np.array([]), np.array([])
    
    prob = prob / np.sum(prob)
    
    y = np.zeros(n)
    a = np.zeros(n, dtype=np.int64)
    

    over = []
    under = []
    
    for i in range(n):
        y[i] = n * prob[i]
        a[i] = i
        if y[i] > 1.0:
            over.append(i)
        elif y[i] < 1.0:
            under.append(i)
    
    while over and under:
        i = over.pop()
        j = under.pop()
        a[j] = i
        y[i] -= (1.0 - y[j])
        if y[i] > 1.0:
            over.append(i)
        elif y[i] < 1.0:
            under.append(i)
    

    return y, a


def walker_sampler(y: np.ndarray, a: np.ndarray) -> int:
    n = len(y)
    if n == 0:
        raise ValueError("Empty alias table")
    i = np.random.randint(0, n)
    u = np.random.uniform(0.0, 1.0)
    if u < y[i]:
        return int(i)
    else:
        return int(a[i])


def cvt_energy_2d(
    generators: np.ndarray,
    samples: np.ndarray
) -> float:
    if generators.shape[0] == 0 or samples.shape[0] == 0:
        return 0.0
    

    diffs = samples[:, np.newaxis, :] - generators[np.newaxis, :, :]
    dists_sq = np.sum(diffs ** 2, axis=2)
    min_dists = np.min(dists_sq, axis=1)
    return np.mean(min_dists)


def lloyd_step_2d(
    generators: np.ndarray,
    samples: np.ndarray
) -> np.ndarray:
    n = generators.shape[0]
    if n == 0:
        return generators.copy()
    
    sums = np.zeros((n, 2))
    counts = np.zeros(n)
    
    for s in samples:
        dists = np.sum((generators - s) ** 2, axis=1)
        nearest = np.argmin(dists)
        sums[nearest] += s
        counts[nearest] += 1
    
    new_gens = generators.copy()
    for i in range(n):
        if counts[i] > 0:
            new_gens[i] = sums[i] / counts[i]
        else:

            new_gens[i] = generators[i]
    
    return new_gens


def disk01_positive_sample(n: int) -> np.ndarray:
    if n < 0:
        raise ValueError("n must be non-negative")








    raise NotImplementedError("Hole_2: disk01_positive_sample uniform disk sampling not implemented")
    return np.column_stack([x, y])


def generate_random_matrix_lognormal(
    n: int,
    mu: float = 0.0,
    sigma: float = 1.0,
    a: float = 1e-6,
    b: float = 10.0
) -> np.ndarray:
    if n < 0:
        raise ValueError("n must be non-negative")
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i, j] = log_normal_truncated_ab_sample(mu, sigma, a, b)
    return mat


if __name__ == "__main__":

    samples = [log_normal_truncated_ab_sample(0.0, 1.0, 0.1, 5.0) for _ in range(1000)]
    print("Log-normal mean:", np.mean(samples), "std:", np.std(samples))
    

    prob = np.array([0.1, 0.3, 0.4, 0.2])
    y, a = walker_build(prob)
    counts = np.zeros(4)
    for _ in range(10000):
        counts[walker_sampler(y, a)] += 1
    print("Walker empirical frequencies:", counts / 10000)
    

    gens = np.random.rand(5, 2)
    samps = np.random.rand(1000, 2)
    E0 = cvt_energy_2d(gens, samps)
    for _ in range(10):
        gens = lloyd_step_2d(gens, samps)
    E1 = cvt_energy_2d(gens, samps)
    print("CVT energy before/after Lloyd:", E0, E1)
