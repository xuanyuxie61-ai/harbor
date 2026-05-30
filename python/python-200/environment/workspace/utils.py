
import time
import numpy as np
from typing import List, Tuple, Optional






class Timer:

    def __init__(self):
        self._start = None
        self._elapsed = 0.0
        self._running = False

    def start(self):
        if not self._running:
            self._start = time.perf_counter()
            self._running = True

    def stop(self):
        if self._running:
            self._elapsed += time.perf_counter() - self._start
            self._running = False
        return self._elapsed

    def reset(self):
        self._elapsed = 0.0
        self._start = None
        self._running = False

    @property
    def elapsed(self) -> float:
        if self._running:
            return self._elapsed + (time.perf_counter() - self._start)
        return self._elapsed

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def benchmark_function(func, *args, n_runs: int = 3, **kwargs) -> dict:
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times = np.array(times)
    return {
        'mean_time': float(np.mean(times)),
        'min_time': float(np.min(times)),
        'max_time': float(np.max(times)),
        'std_time': float(np.std(times, ddof=1)),
        'result': result,
    }






class HistogramStats:

    def __init__(self, data: np.ndarray, n_bins: int = 20,
                 range_limits: Tuple[float, float] = None):
        self.data = np.asarray(data)
        self.n_bins = n_bins
        if range_limits is None:
            self.range = (float(np.min(data)), float(np.max(data)))
        else:
            self.range = range_limits

        self.counts, self.bin_edges = np.histogram(
            self.data, bins=n_bins, range=self.range
        )
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        self.bin_width = self.bin_edges[1] - self.bin_edges[0]

    @property
    def probabilities(self) -> np.ndarray:
        total = np.sum(self.counts)
        if total < 1:
            return np.zeros_like(self.counts, dtype=float)
        return self.counts.astype(float) / (total * self.bin_width)

    @property
    def entropy(self) -> float:
        p = self.probabilities * self.bin_width
        p = p[p > 1e-30]
        return -np.sum(p * np.log(p))

    @property
    def mean(self) -> float:
        return float(np.mean(self.data))

    @property
    def variance(self) -> float:
        return float(np.var(self.data, ddof=1))

    @property
    def skewness(self) -> float:
        mu = self.mean
        sigma = np.sqrt(self.variance)
        if sigma < 1e-30:
            return 0.0
        return float(np.mean(((self.data - mu) / sigma) ** 3))

    @property
    def kurtosis(self) -> float:
        mu = self.mean
        sigma = np.sqrt(self.variance)
        if sigma < 1e-30:
            return 0.0
        return float(np.mean(((self.data - mu) / sigma) ** 4) - 3.0)

    def summary(self) -> dict:
        return {
            'n_samples': len(self.data),
            'mean': self.mean,
            'variance': self.variance,
            'std': np.sqrt(self.variance),
            'min': float(np.min(self.data)),
            'max': float(np.max(self.data)),
            'median': float(np.median(self.data)),
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'entropy': self.entropy,
            'n_bins': self.n_bins,
        }


def compute_cumulative_distribution(data: np.ndarray,
                                    n_bins: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    sorted_data = np.sort(data)
    n = len(sorted_data)
    cdf = np.arange(1, n + 1) / n
    return sorted_data, cdf






def safe_divide(a: np.ndarray, b: np.ndarray,
                default: float = 0.0) -> np.ndarray:
    result = np.full_like(a, default, dtype=float)
    mask = np.abs(b) > 1e-30
    result[mask] = a[mask] / b[mask]
    return result


def clip_to_range(x: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip(x, low, high)


def relative_error(approx: float, exact: float) -> float:
    if abs(exact) < 1e-30:
        return abs(approx)
    return abs((approx - exact) / exact)


def convergence_rate(errors: List[float]) -> List[float]:
    rates = []
    for i in range(2, len(errors)):
        e0, e1, e2 = errors[i - 2], errors[i - 1], errors[i]
        if e0 < 1e-30 or e1 < 1e-30:
            rates.append(0.0)
        else:
            r = np.log(e2 / e1) / np.log(e1 / e0)
            rates.append(float(r))
    return rates






def simulation_time_to_seconds(step: int, dt: float,
                                time_unit: str = "reduced") -> float:
    t_reduced = step * dt
    if time_unit == "reduced":
        return t_reduced
    elif time_unit == "picoseconds":

        return t_reduced * 2.16
    elif time_unit == "femtoseconds":
        return t_reduced * 2160.0
    else:
        return t_reduced


def format_time_interval(seconds: float) -> str:
    if seconds < 1e-6:
        return f"{seconds*1e9:.3f} ns"
    elif seconds < 1e-3:
        return f"{seconds*1e6:.3f} μs"
    elif seconds < 1.0:
        return f"{seconds*1e3:.3f} ms"
    elif seconds < 60:
        return f"{seconds:.3f} s"
    elif seconds < 3600:
        return f"{seconds/60:.2f} min"
    else:
        return f"{seconds/3600:.2f} h"






def condition_number_estimate(A: np.ndarray) -> float:
    s = np.linalg.svd(A, compute_uv=False)
    if len(s) == 0 or s[-1] < 1e-30:
        return 1e30
    return float(s[0] / s[-1])


def is_symmetric_positive_definite(A: np.ndarray, tol: float = 1e-10) -> bool:
    if A.shape[0] != A.shape[1]:
        return False
    if not np.allclose(A, A.T, atol=tol):
        return False
    try:
        eigvals = np.linalg.eigvalsh(A)
        return np.all(eigvals > -tol)
    except np.linalg.LinAlgError:
        return False


def safe_cholesky(A: np.ndarray) -> Optional[np.ndarray]:
    try:
        return np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        return None
