
import numpy as np
from typing import Tuple, Optional


def validate_array_1d(arr: np.ndarray, name: str = "array") -> np.ndarray:
    if arr is None:
        raise ValueError(f"{name} cannot be None")
    arr = np.asarray(arr)
    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty")
    return arr.ravel()


def validate_array_2d(arr: np.ndarray, name: str = "array") -> np.ndarray:
    if arr is None:
        raise ValueError(f"{name} cannot be None")
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {arr.ndim}D")
    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty")
    return arr


def safe_inverse(x: np.ndarray, eps: float = 1e-14) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.where(np.abs(x) < eps, np.sign(x + eps) * eps, x)
    return 1.0 / x


def build_sparse_hamiltonian_indices(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n < 2:
        raise ValueError("Matrix dimension must be >= 2")
    rows = []
    cols = []
    data = []
    for i in range(n):
        rows.append(i)
        cols.append(i)
        data.append(2.0)
        if i > 0:
            rows.append(i)
            cols.append(i - 1)
            data.append(-1.0)
        if i < n - 1:
            rows.append(i)
            cols.append(i + 1)
            data.append(-1.0)
    return np.array(rows, dtype=int), np.array(cols, dtype=int), np.array(data, dtype=float)


def spmatvec(rows: np.ndarray, cols: np.ndarray, data: np.ndarray, vec: np.ndarray) -> np.ndarray:
    vec = validate_array_1d(vec, "vec")
    n = vec.size
    out = np.zeros(n, dtype=float)
    for r, c, d in zip(rows, cols, data):
        if 0 <= r < n and 0 <= c < n:
            out[r] += d * vec[c]
    return out


def estimate_condition_number_dense(A: np.ndarray) -> float:
    A = validate_array_2d(A, "A")
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")

    eigvals = np.linalg.eigvalsh(A)
    abs_eig = np.abs(eigvals)
    min_eig = np.min(abs_eig)
    max_eig = np.max(abs_eig)
    if min_eig < 1e-15:
        min_eig = 1e-15
    return max_eig / min_eig


def fio_write_matrix(filename: str, A: np.ndarray, fmt: str = "%.16e") -> None:
    A = validate_array_2d(A, "A")
    np.savetxt(filename, A, fmt=fmt)


def fio_read_matrix(filename: str) -> Optional[np.ndarray]:
    try:
        return np.loadtxt(filename)
    except Exception as e:
        print(f"[utils] Warning: failed to read {filename}: {e}")
        return None


def tridiagonal_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    n = d.size
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    d = np.asarray(d, dtype=float)
    if not (a.size == n - 1 and b.size == n and c.size == n - 1):
        raise ValueError("Diagonal lengths mismatch with RHS")
    cp = np.zeros(n - 1, dtype=float)
    dp = np.zeros(n, dtype=float)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n - 1):
        denom = b[i] - a[i - 1] * cp[i - 1]
        if abs(denom) < 1e-15:
            denom = 1e-15
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / denom
    dp[n - 1] = (d[n - 1] - a[n - 2] * dp[n - 2]) / (b[n - 1] - a[n - 2] * cp[n - 2])
    x = np.zeros(n, dtype=float)
    x[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x
