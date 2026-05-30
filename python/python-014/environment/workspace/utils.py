
import numpy as np
from typing import Tuple, List


EPS_MACHINE = np.finfo(float).eps
EPS_SQRT = np.sqrt(EPS_MACHINE)


def safe_divide(a: float, b: float, fallback: float = 0.0) -> float:
    if np.abs(b) < EPS_MACHINE:
        return fallback
    return a / b


def safe_sqrt(x: float) -> float:
    if x < 0.0:
        if x > -EPS_MACHINE:
            return 0.0
        raise ValueError(f"safe_sqrt: negative argument {x}")
    return np.sqrt(x)


def clip_spin_norm(s: np.ndarray, target_norm: float = 1.0) -> np.ndarray:
    norm = np.linalg.norm(s)
    if norm < EPS_MACHINE:

        out = np.zeros_like(s)
        out[2] = target_norm
        return out
    return s * (target_norm / norm)


def rms_norm(v: np.ndarray) -> float:
    n = v.size
    if n == 0:
        return 0.0
    return np.sqrt(np.sum(v ** 2) / n)


def skyline_mv(n: int, diag: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = np.zeros(n, dtype=float)
    for j in range(n):
        dj = diag[j]

        if j == 0:
            height = 1
        else:
            height = dj - diag[j - 1]
        i_start = j - height + 1
        for idx in range(height):
            i = i_start + idx
            val = a[dj - height + 1 + idx]
            y[i] += val * x[j]
            if i != j:
                y[j] += val * x[i]
    return y


def build_skyline_from_tridiagonal(
    lower: np.ndarray, diag: np.ndarray, upper: np.ndarray
) -> Tuple[int, np.ndarray, np.ndarray]:
    n = diag.size
    na = 2 * n - 1
    a = np.zeros(na, dtype=float)
    diag_idx = np.zeros(n, dtype=int)
    pos = 0
    for j in range(n):
        if j > 0:
            a[pos] = lower[j - 1]
            pos += 1
        a[pos] = diag[j]
        diag_idx[j] = pos
        pos += 1
    return na, diag_idx, a


def histogram_stats_1d(data: np.ndarray, bins: int = 20) -> Tuple[np.ndarray, np.ndarray, dict]:
    if data.size == 0:
        counts = np.zeros(bins, dtype=int)
        edges = np.linspace(0.0, 1.0, bins + 1)
        stats = {"min": 0.0, "max": 0.0, "mean": 0.0, "variance": 0.0, "skewness": 0.0, "kurtosis": 0.0}
        return counts, edges, stats

    dmin = float(np.min(data))
    dmax = float(np.max(data))
    if dmax - dmin < EPS_MACHINE:
        dmax = dmin + 1.0

    edges = np.linspace(dmin, dmax, bins + 1)
    counts, _ = np.histogram(data, bins=edges)

    mean = float(np.mean(data))
    variance = float(np.var(data))
    std = safe_sqrt(variance)
    if std < EPS_MACHINE:
        skewness = 0.0
        kurtosis = 0.0
    else:
        skewness = float(np.mean(((data - mean) / std) ** 3))
        kurtosis = float(np.mean(((data - mean) / std) ** 4)) - 3.0

    stats = {
        "min": dmin,
        "max": dmax,
        "mean": mean,
        "variance": variance,
        "skewness": skewness,
        "kurtosis": kurtosis,
    }
    return counts, edges, stats


def triangle_area_histogram_2d(
    points: np.ndarray, n_sub: int = 5
) -> Tuple[np.ndarray, dict]:
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")


    valid = (
        (points[:, 0] >= -EPS_MACHINE)
        & (points[:, 1] >= -EPS_MACHINE)
        & (points[:, 0] + points[:, 1] <= 1.0 + EPS_MACHINE)
    )
    pts = points[valid]
    n_points = pts.shape[0]

    sub_num = n_sub * n_sub
    histo = np.zeros(sub_num + 1, dtype=int)

    for p in pts:
        x, y = p

        i = int(np.floor(x * n_sub))
        j = int(np.floor(y * n_sub))
        k = int(np.floor((1.0 - x - y) * n_sub))
        i = max(0, min(n_sub - 1, i))
        j = max(0, min(n_sub - 1, j))
        k = max(0, min(n_sub - 1, k))


        t = i + j * n_sub
        if t < 0 or t >= sub_num:
            t = sub_num
        histo[t] += 1

    histo_ave = np.sum(histo[:sub_num]) / max(sub_num, 1)
    histo_max = int(np.max(histo[:sub_num])) if sub_num > 0 else 0
    histo_min = int(np.min(histo[:sub_num])) if sub_num > 0 else 0
    histo_var = float(np.var(histo[:sub_num])) if sub_num > 0 else 0.0

    info = {
        "n_points": n_points,
        "n_sub": n_sub,
        "sub_num": sub_num,
        "min": histo_min,
        "max": histo_max,
        "average": histo_ave,
        "variance": histo_var,
        "out_of_range": int(histo[sub_num]),
    }
    return histo[:sub_num], info


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    r = int(np.sqrt(n))
    for d in range(3, r + 1, 2):
        if n % d == 0:
            return False
    return True


def primes_up_to(n_max: int) -> List[int]:
    if n_max < 2:
        return []
    sieve = np.ones(n_max + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(np.sqrt(n_max)) + 1):
        if sieve[p]:
            sieve[p * p :: p] = False
    return np.nonzero(sieve)[0].tolist()
