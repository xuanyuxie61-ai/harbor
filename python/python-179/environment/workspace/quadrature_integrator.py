
import numpy as np
from typing import Tuple
from system_utils import EPS, TOL_RANK






def van_der_corput_sequence(n: int, base: int = 2) -> np.ndarray:
    seq = np.zeros(n, dtype=float)
    for i in range(n):
        idx = i
        f = 1.0
        r = 0.0
        while idx > 0:
            f /= base
            r += f * (idx % base)
            idx //= base
        seq[i] = r
    return seq


def hammersley_sequence(n: int, d: int) -> np.ndarray:
    if d < 1:
        raise ValueError("d must be >= 1")

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    points = np.zeros((n, d), dtype=float)
    points[:, 0] = np.arange(n, dtype=float) / max(n, 1)
    for k in range(1, d):
        base = primes[k - 1] if k - 1 < len(primes) else primes[-1]
        points[:, k] = van_der_corput_sequence(n, base)
    return points






def grid_integrate_1d(f, a: float, b: float, n: int) -> float:
    if n < 2:
        raise ValueError("n must be >= 2")
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    val = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
    return float(val)


def monte_carlo_integrate(f, dim: int, n: int,
                          bounds: np.ndarray = None,
                          seed: int = None) -> Tuple[float, float]:
    if seed is not None:
        np.random.seed(seed)
    if bounds is None:
        bounds = np.tile([0.0, 1.0], (dim, 1))
    bounds = np.asarray(bounds, dtype=float)
    x = np.random.rand(n, dim)

    for k in range(dim):
        x[:, k] = bounds[k, 0] + x[:, k] * (bounds[k, 1] - bounds[k, 0])
    y = np.array([f(xi) for xi in x], dtype=float)
    volume = np.prod(bounds[:, 1] - bounds[:, 0])
    mean = np.mean(y)
    var = np.var(y, ddof=1) if n > 1 else 0.0
    estimate = volume * mean
    stderr = volume * np.sqrt(var / max(n, 1))
    return float(estimate), float(stderr)


def qmc_integrate(f, dim: int, n: int,
                  bounds: np.ndarray = None) -> float:
    if bounds is None:
        bounds = np.tile([0.0, 1.0], (dim, 1))
    bounds = np.asarray(bounds, dtype=float)
    x = hammersley_sequence(n, dim)
    for k in range(dim):
        x[:, k] = bounds[k, 0] + x[:, k] * (bounds[k, 1] - bounds[k, 0])
    y = np.array([f(xi) for xi in x], dtype=float)
    volume = np.prod(bounds[:, 1] - bounds[:, 0])
    return float(volume * np.mean(y))






def estimate_tensor_frobenius_norm_mc(tensor: np.ndarray, n_samples: int = 10000,
                                      seed: int = None) -> float:
    tensor = np.asarray(tensor)
    shape = tensor.shape
    d = len(shape)
    N = int(np.prod(shape))
    if seed is not None:
        np.random.seed(seed)

    flat_idx = np.random.randint(0, N, size=n_samples)

    strides = [int(np.prod(shape[k+1:], dtype=np.int64)) for k in range(d)]
    samples = np.zeros(n_samples, dtype=float)
    for s in range(n_samples):
        idx = flat_idx[s]
        multi = []
        rem = idx
        for stride in strides:
            multi.append(rem // stride)
            rem = rem % stride
        samples[s] = tensor[tuple(multi)]
    norm_sq_est = (N / n_samples) * np.sum(samples * samples)
    return float(np.sqrt(norm_sq_est))


from typing import Tuple
