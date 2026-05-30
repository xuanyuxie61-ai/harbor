import numpy as np
from typing import Tuple, Optional


def safe_divide(a: float, b: float, fallback: float = 0.0) -> float:
    if abs(b) < 1e-15:
        return fallback
    return a / b


def sign_with_zero(x: float, tol: float = 1e-12) -> int:
    if abs(x) < tol:
        return 0
    return 1 if x > 0 else -1


def clip_to_range(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(x, lo, hi)


def macaulay_bracket(x: float) -> float:
    return max(x, 0.0)


def heaviside_step(x: float, tol: float = 1e-12) -> float:
    if abs(x) < tol:
        return 0.5 * (1.0 + x / tol)
    return 1.0 if x > 0 else 0.0


def c8_norm_l2(z: np.ndarray) -> float:
    return float(np.sqrt(np.sum(np.abs(z) ** 2)))


def c8mat_norm_fro(A: np.ndarray) -> float:
    return float(np.sqrt(np.sum(np.abs(A) ** 2)))


def r8mat_print_some(A: np.ndarray, title: str = "", max_rows: int = 5, max_cols: int = 5):
    m, n = A.shape
    print(f"\n{title}")
    print(f"  Shape = ({m}, {n}), showing top-left ({min(m, max_rows)}, {min(n, max_cols)})")
    for i in range(min(m, max_rows)):
        row = "  ".join(f"{A[i, j]:12.6e}" for j in range(min(n, max_cols)))
        print(f"  [{i}] {row}")


def r8vec_indicator1(n: int) -> np.ndarray:
    return np.arange(1, n + 1, dtype=float)


def i4_log_10(n: int) -> int:
    if n <= 0:
        return 0
    return int(np.floor(np.log10(float(n))))


def check_symmetry(A: np.ndarray, tol: float = 1e-10) -> bool:
    return bool(np.allclose(A, A.T, atol=tol))


def condition_number_estimate(A: np.ndarray) -> float:
    s = np.linalg.svd(A, compute_uv=False)
    return float(s.max() / max(s.min(), 1e-20))


def solve_2x2_symmetric(a11: float, a12: float, a22: float, b1: float, b2: float) -> Tuple[float, float]:
    det = a11 * a22 - a12 * a12
    if abs(det) < 1e-20:
        raise ValueError("2x2 system nearly singular in solve_2x2_symmetric")
    x1 = (a22 * b1 - a12 * b2) / det
    x2 = (-a12 * b1 + a11 * b2) / det
    return x1, x2
