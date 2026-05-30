
import numpy as np


def safe_divide(a, b, fill_value=0.0):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    result = np.empty_like(a, dtype=float)
    mask = np.abs(b) > 1e-15
    result[mask] = a[mask] / b[mask]
    result[~mask] = fill_value
    return result


def clip_with_warning(x, xmin, xmax, name="variable"):
    x = np.asarray(x, dtype=float)
    if np.any(x < xmin - 1e-12) or np.any(x > xmax + 1e-12):
        print(f"[WARN] {name} out of bounds [{xmin}, {xmax}], clipping applied.")
    return np.clip(x, xmin, xmax)


def ensure_positive(x, eps=1e-12, name="variable"):
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0):
        print(f"[WARN] {name} contains non-positive values, raised to {eps}.")
    return np.where(x > eps, x, eps)


def relative_change(new, old):
    new = np.asarray(new, dtype=float)
    old = np.asarray(old, dtype=float)
    denom = np.abs(old) + 1e-15
    return np.max(np.abs(new - old) / denom)


def thermo_factor_check(T, Tmin=200.0, Tmax=800.0, Pmin=1e3, Pmax=5e6):
    return float(clip_with_warning(T, Tmin, Tmax, "Temperature"))
