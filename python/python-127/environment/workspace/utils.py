
import numpy as np
import sys


def safe_divide(a, b, fill_value=0.0):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    result = np.where(np.abs(b) < 1e-15, fill_value, a / b)
    return result


def clip_and_warn(arr, vmin, vmax, name="array"):
    arr = np.asarray(arr, dtype=float)
    clipped = np.clip(arr, vmin, vmax)
    n_clipped = np.sum((arr < vmin) | (arr > vmax))
    if n_clipped > 0:
        print(f"[WARN] {name}: {n_clipped} 个值被裁剪到 [{vmin}, {vmax}]",
              file=sys.stderr)
    return clipped


def check_finite(arr, name="array"):
    arr = np.asarray(arr)
    if not np.all(np.isfinite(arr)):
        n_bad = np.sum(~np.isfinite(arr))
        raise ValueError(f"{name} 包含 {n_bad} 个非有限值 (nan/inf)")
    return True


def print_section(title, width=70):
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_subsection(title, width=70):
    print("\n" + "-" * width)
    print(f"  {title}")
    print("-" * width)


def save_results_to_text(filename, results_dict):
    with open(filename, 'w', encoding='utf-8') as f:
        for key, value in results_dict.items():
            f.write(f"{key}:\n")
            if isinstance(value, np.ndarray):
                f.write(str(value))
            else:
                f.write(str(value))
            f.write("\n\n")


def compute_rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("a 和 b 形状必须相同")
    return np.sqrt(np.mean((a - b)**2))


def compute_relative_error(exact, approx):
    exact = np.asarray(exact, dtype=float)
    approx = np.asarray(approx, dtype=float)
    denom = np.maximum(np.abs(exact), 1e-15)
    return np.mean(np.abs(exact - approx) / denom)


def gaussian_2d(X, Y, cx, cy, sigma_x, sigma_y, amplitude=1.0):
    return amplitude * np.exp(
        -((X - cx)**2) / (2.0 * sigma_x**2)
        - ((Y - cy)**2) / (2.0 * sigma_y**2)
    )


def finite_difference_gradient_2d(F, dx, dy):
    F = np.asarray(F, dtype=float)
    dFdx = np.zeros_like(F)
    dFdy = np.zeros_like(F)
    dFdx[1:-1, :] = (F[2:, :] - F[:-2, :]) / (2.0 * dx)
    dFdy[:, 1:-1] = (F[:, 2:] - F[:, :-2]) / (2.0 * dy)
    return dFdx, dFdy
