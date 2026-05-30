
import sys
import numpy as np


def is_running_in_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


def safe_exp(x, max_val=700.0):
    x = np.asarray(x, dtype=np.float64)
    x_clipped = np.clip(x, -max_val, max_val)
    return np.exp(x_clipped)


def safe_divide(a, b, fill_value=0.0):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    result = np.empty_like(a, dtype=np.float64)
    mask = np.abs(b) > 1e-15
    result[mask] = a[mask] / b[mask]
    result[~mask] = fill_value
    return result


def check_bounds(values, lower, upper, name="variable"):
    values = np.asarray(values, dtype=np.float64)
    if np.any(values < lower) or np.any(values > upper):
        print(f"[警告] {name} 越界: 范围应为 [{lower}, {upper}], "
              f"实际范围 [{np.min(values):.4e}, {np.max(values):.4e}]")
    return np.clip(values, lower, upper)


def timestamp():
    from datetime import datetime
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def finite_diff_jacobian(func, x, h=1e-8):
    n = len(x)
    x = np.asarray(x, dtype=np.float64)
    f0 = func(x)
    m = len(f0)
    J = np.zeros((m, n), dtype=np.float64)
    for j in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[j] += h
        x_minus[j] -= h
        J[:, j] = (func(x_plus) - func(x_minus)) / (2.0 * h)
    return J


def print_matrix(A, name="Matrix", max_rows=6, max_cols=6):
    A = np.asarray(A)
    print(f"\n{name} (shape={A.shape}):")
    rows_to_print = min(A.shape[0], max_rows)
    cols_to_print = min(A.shape[1], max_cols) if A.ndim > 1 else 1
    for i in range(rows_to_print):
        if A.ndim > 1:
            line = "  ".join(f"{A[i, j]:12.6e}" for j in range(cols_to_print))
            if A.shape[1] > max_cols:
                line += "  ..."
        else:
            line = f"{A[i]:12.6e}"
        print(f"  {line}")
    if A.shape[0] > max_rows:
        print("  ...")


def cond_number_estimate(A):
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return np.inf

    s_max = np.linalg.norm(A, 2)
    try:
        s_min = 1.0 / np.linalg.norm(np.linalg.inv(A), 2)
    except np.linalg.LinAlgError:
        return np.inf
    if s_min < 1e-15:
        return np.inf
    return s_max / s_min
