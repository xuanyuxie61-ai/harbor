"""
utils.py

Utility functions for the OCT simulation framework.
Includes parameter management, file I/O helpers, numerical safeguards,
and statistical analysis tools.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Parameter management (inspired by sensitive_parameters, *_parameters)
# ---------------------------------------------------------------------------

class ParameterManager:
    """
    Persistent parameter storage with defaults and user overrides.
    """

    def __init__(self, defaults=None):
        self._params = {}
        if defaults is not None:
            for key, val in defaults.items():
                self._params[key] = val

    def set(self, **kwargs):
        """Update parameters."""
        for key, val in kwargs.items():
            self._params[key] = val

    def get(self, key, default=None):
        """Get parameter with optional default."""
        return self._params.get(key, default)

    def get_all(self):
        """Return copy of all parameters."""
        return self._params.copy()

    def validate_ranges(self, ranges):
        """
        Validate that all parameters are within specified ranges.

        Parameters
        ----------
        ranges : dict
            {key: (min, max)}

        Returns
        -------
        ok : bool
        violations : list
        """
        violations = []
        for key, (vmin, vmax) in ranges.items():
            val = self._params.get(key)
            if val is not None and (val < vmin or val > vmax):
                violations.append((key, val, vmin, vmax))
        return len(violations) == 0, violations


# ---------------------------------------------------------------------------
# Numerical safeguards
# ---------------------------------------------------------------------------

def safe_divide(a, b, fill_value=0.0):
    """
    Element-wise division with zero handling.

    Parameters
    ----------
    a, b : array_like
    fill_value : float
        Value to use where b == 0.

    Returns
    -------
    result : ndarray
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    result = np.empty_like(a, dtype=float)
    mask = np.abs(b) > 1e-14
    result[mask] = a[mask] / b[mask]
    result[~mask] = fill_value
    return result


def clip_to_finite(arr, bounds=(-1e6, 1e6)):
    """
    Replace non-finite values and clip to bounds.

    Parameters
    ----------
    arr : array_like
    bounds : tuple

    Returns
    -------
    clipped : ndarray
    """
    arr = np.asarray(arr, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return np.clip(arr, bounds[0], bounds[1])


def robust_mean_std(arr, outlier_z=3.0):
    """
    Compute mean and standard deviation excluding outliers.

    Parameters
    ----------
    arr : array_like
    outlier_z : float
        Z-score threshold.

    Returns
    -------
    mean : float
    std : float
    """
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0, 0.0
    m = np.median(arr)
    mad = np.median(np.abs(arr - m))
    if mad < 1e-14:
        return np.mean(arr), np.std(arr, ddof=1)
    z_scores = 0.6745 * (arr - m) / mad
    mask = np.abs(z_scores) < outlier_z
    clean = arr[mask]
    if len(clean) == 0:
        clean = arr
    return np.mean(clean), np.std(clean, ddof=1)


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def save_array_with_header(filename, array, header_lines=None):
    """
    Save a numpy array to text file with optional comment header.

    Parameters
    ----------
    filename : str
    array : ndarray
    header_lines : list of str, optional
    """
    array = np.asarray(array, dtype=float)
    with open(filename, 'w') as f:
        if header_lines is not None:
            for line in header_lines:
                f.write(f"# {line}\n")
        if array.ndim == 1:
            for val in array:
                f.write(f"{val:.16g}\n")
        else:
            for row in array:
                f.write("  ".join(f"{v:.16g}" for v in row) + "\n")


def load_array_skip_header(filename, skip_comments=True):
    """
    Load numeric array from text file, skipping comment/blank lines.

    Parameters
    ----------
    filename : str
    skip_comments : bool

    Returns
    -------
    array : ndarray
    """
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if skip_comments and line.startswith('#'):
                continue
            parts = line.split()
            row = [float(p) for p in parts]
            data.append(row)
    return np.array(data, dtype=float)


# ---------------------------------------------------------------------------
# Convergence analysis
# ---------------------------------------------------------------------------

def convergence_rate(errors):
    """
    Estimate asymptotic convergence rate from error sequence.

    Assumes errors ~ C * r^k, estimates r from successive ratios.

    Parameters
    ----------
    errors : array_like
        Sequence of error norms.

    Returns
    -------
    rate : float
        Estimated convergence rate.
    """
    errors = np.asarray(errors, dtype=float)
    errors = errors[errors > 1e-14]
    if len(errors) < 2:
        return 0.0
    ratios = errors[1:] / errors[:-1]
    # Use geometric mean of last half
    n_use = max(1, len(ratios) // 2)
    rate = np.exp(np.mean(np.log(ratios[-n_use:])))
    return rate


def relative_error(approx, exact):
    """
    Compute relative L2 error.

    Parameters
    ----------
    approx, exact : array_like

    Returns
    -------
    err : float
    """
    approx = np.asarray(approx, dtype=float)
    exact = np.asarray(exact, dtype=float)
    denom = np.linalg.norm(exact)
    if denom < 1e-14:
        return np.linalg.norm(approx - exact)
    return np.linalg.norm(approx - exact) / denom
