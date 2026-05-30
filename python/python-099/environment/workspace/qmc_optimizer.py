
import numpy as np
import math
from utils import clamp, ensure_positive





_DIRECTION_NUMBERS = {
    1: {"m": [1], "a": 0, "s": 1},
    2: {"m": [1, 3], "a": 1, "s": 2},
    3: {"m": [1, 3, 1], "a": 1, "s": 3},
    4: {"m": [1, 1, 1], "a": 1, "s": 3},
    5: {"m": [1, 3, 3], "a": 2, "s": 3},
    6: {"m": [1, 3, 5], "a": 1, "s": 3},
    7: {"m": [1, 1, 5], "a": 4, "s": 3},
    8: {"m": [1, 3, 7], "a": 2, "s": 3},
    9: {"m": [1, 3, 7], "a": 4, "s": 3},
    10: {"m": [1, 1, 5], "a": 7, "s": 3},
}


def sobol_sequence(dim_num: int, n: int, skip: int = 0) -> np.ndarray:
    dim_num = int(clamp(dim_num, 1, 10))
    n = max(int(n), 1)
    skip = max(int(skip), 0)

    L = max(int(math.ceil(math.log2(n + skip + 1))), 1)
    points = np.zeros((n, dim_num), dtype=float)

    for d in range(dim_num):
        dn = _DIRECTION_NUMBERS.get(d + 1, _DIRECTION_NUMBERS[1])
        s = dn["s"]
        a = dn["a"]
        m = dn["m"][:s]
        while len(m) < L:
            m.append(0)


        for j in range(s, L):
            m[j] = m[j - s] ^ (m[j - s] << s)
            for k in range(1, s):
                m[j] ^= ((a >> (k - 1)) & 1) * (m[j - k] << k)

            m[j] &= (1 << (j + 1)) - 1

        v = np.zeros(L, dtype=int)
        for j in range(L):
            v[j] = m[j] << (L - 1 - j)

        x = 0
        g = skip

        for j in range(L):
            if ((g >> j) & 1):
                x ^= v[j]

        for i in range(n):
            points[i, d] = x / (2.0 ** L)

            g = i + skip
            c = 0
            while ((g >> c) & 1) == 1:
                c += 1
            if c < L:
                x ^= v[c]

    return points





def lattice_rule_integrate(
    func,
    dim_num: int,
    m: int,
    z: np.ndarray = None,
) -> float:
    dim_num = int(dim_num)
    m = max(int(m), 1)
    if z is None:
        z = np.ones(dim_num, dtype=int)
        for d in range(1, dim_num):
            z[d] = (z[d - 1] * 177) % m
    else:
        z = np.asarray(z, dtype=int)

    total = 0.0
    for j in range(m):
        x = np.mod(j * z / m, 1.0)
        total += func(x)
    return total / m


def fibonacci_lattice_integrate_2d(func, m: int) -> float:
    m = max(int(m), 1)

    a, b = 1, 1
    while b < m:
        a, b = b, a + b
    z = np.array([1, a], dtype=int)
    return lattice_rule_integrate(func, 2, m, z=z)





def normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    sigma = max(float(sigma), 1e-12)
    return (1.0 / (sigma * math.sqrt(2.0 * math.pi))) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    sigma = max(float(sigma), 1e-12)
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


def normal_sample(n: int, mu: float = 0.0, sigma: float = 1.0, seed: int = None) -> np.ndarray:
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    return rng.normal(loc=mu, scale=max(sigma, 1e-12), size=n)


def gamma_sample(n: int, shape: float = 1.0, scale: float = 1.0, seed: int = None) -> np.ndarray:
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    return rng.gamma(shape=max(shape, 1e-12), scale=max(scale, 1e-12), size=n)


def uniform_sample(n: int, low: float = 0.0, high: float = 1.0, seed: int = None) -> np.ndarray:
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    return rng.uniform(low=low, high=high, size=n)





def optimize_coating_parameters_qmc(
    objective_func,
    dim: int,
    n_samples: int = 512,
    param_bounds: np.ndarray = None,
    seed: int = 42,
) -> tuple:
    dim = int(dim)
    n_samples = max(int(n_samples), 4)

    points = sobol_sequence(dim, n_samples, skip=seed % 64)

    if param_bounds is None:
        param_bounds = np.array([[0.0, 1.0]] * dim, dtype=float)
    else:
        param_bounds = np.asarray(param_bounds, dtype=float)
        if param_bounds.shape != (dim, 2):
            raise ValueError("param_bounds must have shape (dim, 2).")

    best_value = float("inf")
    best_params = None

    for i in range(n_samples):
        p = points[i]

        params = param_bounds[:, 0] + p * (param_bounds[:, 1] - param_bounds[:, 0])
        val = objective_func(params)
        if val < best_value:
            best_value = val
            best_params = params.copy()

    return best_params, best_value


def monte_carlo_uncertainty_propagation(
    func,
    param_means: np.ndarray,
    param_stds: np.ndarray,
    n_mc: int = 1000,
    seed: int = 42,
) -> tuple:
    param_means = np.asarray(param_means, dtype=float)
    param_stds = np.asarray(param_stds, dtype=float)
    dim = param_means.size
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    samples = np.zeros(n_mc, dtype=float)
    for i in range(n_mc):
        params = param_means + rng.normal(size=dim) * param_stds
        samples[i] = func(params)

    return float(np.mean(samples)), float(np.std(samples)), samples
