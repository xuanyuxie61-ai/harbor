
import numpy as np






class ParameterManager:

    def __init__(self, defaults=None):
        self._params = {}
        if defaults is not None:
            for key, val in defaults.items():
                self._params[key] = val

    def set(self, **kwargs):
        for key, val in kwargs.items():
            self._params[key] = val

    def get(self, key, default=None):
        return self._params.get(key, default)

    def get_all(self):
        return self._params.copy()

    def validate_ranges(self, ranges):
        violations = []
        for key, (vmin, vmax) in ranges.items():
            val = self._params.get(key)
            if val is not None and (val < vmin or val > vmax):
                violations.append((key, val, vmin, vmax))
        return len(violations) == 0, violations






def safe_divide(a, b, fill_value=0.0):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    result = np.empty_like(a, dtype=float)
    mask = np.abs(b) > 1e-14
    result[mask] = a[mask] / b[mask]
    result[~mask] = fill_value
    return result


def clip_to_finite(arr, bounds=(-1e6, 1e6)):
    arr = np.asarray(arr, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return np.clip(arr, bounds[0], bounds[1])


def robust_mean_std(arr, outlier_z=3.0):
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






def save_array_with_header(filename, array, header_lines=None):
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






def convergence_rate(errors):
    errors = np.asarray(errors, dtype=float)
    errors = errors[errors > 1e-14]
    if len(errors) < 2:
        return 0.0
    ratios = errors[1:] / errors[:-1]

    n_use = max(1, len(ratios) // 2)
    rate = np.exp(np.mean(np.log(ratios[-n_use:])))
    return rate


def relative_error(approx, exact):
    approx = np.asarray(approx, dtype=float)
    exact = np.asarray(exact, dtype=float)
    denom = np.linalg.norm(exact)
    if denom < 1e-14:
        return np.linalg.norm(approx - exact)
    return np.linalg.norm(approx - exact) / denom
