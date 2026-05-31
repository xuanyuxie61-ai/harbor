"""
numerical_robustness.py
Machine-precision boundary utilities and spatial data thinning.

Adapted from:
  - 961_r8_scale: IEEE-754 neighbor computation for floating-point robustness
  - 142_cavity_flow_movie: Data thinning/subsampling for spatial grids

Role in synthesis:
  Provides floating-point-aware numerical safeguards and spatial subsampling
  for population density fields and contact network data.
"""

import numpy as np


def r8_next(x: float) -> float:
    """
    Return the nearest representable double-precision float strictly greater than x.
    Based on IEEE-754 neighbor computation.
    """
    if np.isnan(x):
        return np.nan
    if x == np.inf:
        return np.inf
    if x == -np.inf:
        return -np.finfo(float).max
    eps = np.finfo(float).eps
    if x >= 0.0:
        if x == 0.0:
            return eps * 0.5 ** 52  # smallest positive subnormal approx
        return x / (1.0 - eps * 0.5)
    else:
        return x * (1.0 - eps * 0.5)


def r8_previous(x: float) -> float:
    """
    Return the nearest representable double-precision float strictly less than x.
    """
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
    """
    Clamp population density to physically valid range with machine-precision awareness.
    Ensures non-negativity and prevents overflow.
    """
    u = np.asarray(u, dtype=float)
    u_min_bound = r8_next(u_min) if u_min == 0.0 else u_min
    u_max_bound = r8_previous(u_max) if np.isfinite(u_max) else u_max
    u = np.clip(u, u_min_bound, u_max_bound)
    # Replace any NaN/Inf with safe defaults
    u = np.where(np.isnan(u), u_min_bound, u)
    u = np.where(np.isinf(u), u_max_bound, u)
    return u


def critical_threshold_check(value: float, threshold: float = 1.0, tol: float = None) -> int:
    """
    Check if a critical epidemiological threshold (e.g., R0) is near a bifurcation point.
    Returns:
        -1: below threshold
         0: near threshold (within machine-precision-aware tolerance)
         1: above threshold
    """
    if tol is None:
        tol = max(np.finfo(float).eps * max(abs(value), abs(threshold)), 1e-12)
    if abs(value - threshold) < tol:
        return 0
    return 1 if value > threshold else -1


def thin_index_2d(nx: int, ny: int, factor: int = 2) -> np.ndarray:
    """
    Generate indices to subsample a structured 2D grid by a checkerboard thinning factor.
    Adapted from cavity_flow_movie thinning for spatial contact network reduction.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions.
    factor : int
        Thinning factor (keep every factor-th point in each direction).

    Returns
    -------
    indices : 1D array of flat indices to keep.
    """
    x_keep = np.arange(0, nx, factor)
    y_keep = np.arange(0, ny, factor)
    ix, iy = np.meshgrid(x_keep, y_keep, indexing='ij')
    indices = iy * nx + ix
    return indices.ravel()


def filename_inc(filename: str) -> str:
    """
    Increment the numeric portion of a filename (e.g., 'data001.txt' -> 'data002.txt').
    """
    import re
    match = re.search(r'(\d+)', filename)
    if not match:
        return filename
    num_str = match.group(1)
    new_num = str(int(num_str) + 1).zfill(len(num_str))
    return filename[:match.start()] + new_num + filename[match.end():]
