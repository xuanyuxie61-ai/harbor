
import numpy as np
import sys


def r8mat_print_some(m, n, a, ilo, jlo, ihi, jhi, title):
    if 0 < len(title.strip()):
        print(title)
    incx = 5
    for i2lo in range(max(ilo, 1), min(ihi, m) + 1, incx):
        i2hi = min(i2lo + incx - 1, m, ihi)
        print("  Row: ", end="")
        for i in range(i2lo, i2hi + 1):
            print(f"{i:7d}       ", end="")
        print()
        print("  Col")
        for j in range(max(jlo, 1), min(jhi, n) + 1):
            print(f"{j:5d} ", end="")
            for i in range(i2lo, i2hi + 1):
                print(f"{a[i - 1, j - 1]:12.6f}", end="")
            print()


def r8vec_print(n, a, title):
    if 0 < len(title.strip()):
        print(title)
    for i in range(n):
        print(f"  a[{i}] = {a[i]:14.6f}")


def validate_positive(val, name, strict=True):
    if strict:
        if val <= 0.0:
            raise ValueError(f"{name} must be strictly positive, got {val}")
    else:
        if val < 0.0:
            raise ValueError(f"{name} must be non-negative, got {val}")


def validate_matrix_nonsingular(a, tol=1e-12):
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Matrix must be square.")
    det = np.linalg.det(a)
    if abs(det) < tol:
        raise ValueError(f"Matrix is numerically singular (det={det:.3e}).")


def safe_inverse(a, rcond=1e-15):
    u, s, vh = np.linalg.svd(a, full_matrices=False)
    s_inv = np.where(s > rcond * s[0], 1.0 / s, 0.0)
    return vh.T @ np.diag(s_inv) @ u.T


def file_row_count(filepath):
    try:
        with open(filepath, 'r') as f:
            row_num = 0
            for line in f:
                line = line.strip()
                if len(line) == 0 or line.startswith('#'):
                    continue
                row_num += 1
        return row_num
    except FileNotFoundError:
        return 0


def file_column_count(filepath):
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0 or line.startswith('#'):
                    continue
                return len(line.split())
        return 0
    except FileNotFoundError:
        return 0


def compute_condition_number(a, norm_type=2):
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        return np.inf
    return np.linalg.cond(a, norm_type)


def compute_residual_norm(A, x, b, norm_type=np.inf):
    r = b - A @ x
    return np.linalg.norm(r, ord=norm_type)


def compute_normalized_residual(A, x, b):
    eps = np.finfo(float).eps
    r_norm = compute_residual_norm(A, x, b, np.inf)
    A_norm = np.linalg.norm(A, ord=np.inf)
    x_norm = np.linalg.norm(x, ord=np.inf)
    if A_norm == 0.0 or x_norm == 0.0:
        return np.inf
    return r_norm / (A_norm * x_norm * eps)
