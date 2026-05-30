
import numpy as np


def r8_next(x: float) -> float:
    if np.isnan(x):
        return np.nan
    if x == np.inf:
        return np.inf
    if x == -np.inf:
        return -np.finfo(float).max
    eps = np.finfo(float).eps
    if x >= 0.0:
        if x == 0.0:
            return eps * 0.5 ** 52
        return x / (1.0 - eps * 0.5)
    else:
        return x * (1.0 - eps * 0.5)


def r8_previous(x: float) -> float:
    if np.isnan(x):
        return np.nan
    if x == -np.inf:
        return -np.inf
    if x == np.inf:
        return np.finfo(float).max
    eps = np.finfo(float).eps
    if x > 0.0:
        return x * (1.0 - eps * 0.5)
    elif x == 0.0:
        return -eps * 0.5 ** 52
    else:
        return x / (1.0 - eps * 0.5)


def safe_population_density(u: np.ndarray, u_min: float = 0.0, u_max: float = 1e6) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    u_min_bound = r8_next(u_min) if u_min == 0.0 else u_min
    u_max_bound = r8_previous(u_max) if np.isfinite(u_max) else u_max
    u = np.clip(u, u_min_bound, u_max_bound)

    u = np.where(np.isnan(u), u_min_bound, u)
    u = np.where(np.isinf(u), u_max_bound, u)
    return u


def critical_threshold_check(value: float, threshold: float = 1.0, tol: float = None) -> int:
    if tol is None:
        tol = max(np.finfo(float).eps * max(abs(value), abs(threshold)), 1e-12)
    if abs(value - threshold) < tol:
        return 0
    return 1 if value > threshold else -1


def thin_index_2d(nx: int, ny: int, factor: int = 2) -> np.ndarray:
    x_keep = np.arange(0, nx, factor)
    y_keep = np.arange(0, ny, factor)
    ix, iy = np.meshgrid(x_keep, y_keep, indexing='ij')
    indices = iy * nx + ix
    return indices.ravel()


def filename_inc(filename: str) -> str:
    import re
    match = re.search(r'(\d+)', filename)
    if not match:
        return filename
    num_str = match.group(1)
    new_num = str(int(num_str) + 1).zfill(len(num_str))
    return filename[:match.start()] + new_num + filename[match.end():]
