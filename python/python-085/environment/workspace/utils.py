"""
utils.py
通用数学工具与辅助函数
融合种子项目：c8lib（复数运算基础）、r8cb/r8gb（数值精度控制）
"""
import numpy as np
from typing import Tuple, Optional


def safe_divide(a: float, b: float, fallback: float = 0.0) -> float:
    """安全除法，避免除零。"""
    if abs(b) < 1e-15:
        return fallback
    return a / b


def sign_with_zero(x: float, tol: float = 1e-12) -> int:
    """带容差的符号函数，零附近返回0。"""
    if abs(x) < tol:
        return 0
    return 1 if x > 0 else -1


def clip_to_range(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """将数组裁剪到闭区间 [lo, hi]。"""
    return np.clip(x, lo, hi)


def macaulay_bracket(x: float) -> float:
    r"""
    Macaulay 括号：\langle x \rangle_+ = max(x, 0)
    在接触力学中广泛用于正部投影。
    """
    return max(x, 0.0)


def heaviside_step(x: float, tol: float = 1e-12) -> float:
    r"""
    Heaviside 阶跃函数（光滑近似）：
    H(x) = 1 / (1 + exp(-2x / tol))
    用于摩擦定律的光滑正则化。
    """
    if abs(x) < tol:
        return 0.5 * (1.0 + x / tol)
    return 1.0 if x > 0 else 0.0


def c8_norm_l2(z: np.ndarray) -> float:
    r"""
    复向量 L2 范数：
    \|z\|_2 = \sqrt{\sum_i |z_i|^2}
    融合自 c8lib 的复数范数思想。
    """
    return float(np.sqrt(np.sum(np.abs(z) ** 2)))


def c8mat_norm_fro(A: np.ndarray) -> float:
    r"""
    复矩阵 Frobenius 范数：
    \|A\|_F = \sqrt{\sum_{i,j} |A_{ij}|^2}
    """
    return float(np.sqrt(np.sum(np.abs(A) ** 2)))


def r8mat_print_some(A: np.ndarray, title: str = "", max_rows: int = 5, max_cols: int = 5):
    """打印矩阵的前若干行列。"""
    m, n = A.shape
    print(f"\n{title}")
    print(f"  Shape = ({m}, {n}), showing top-left ({min(m, max_rows)}, {min(n, max_cols)})")
    for i in range(min(m, max_rows)):
        row = "  ".join(f"{A[i, j]:12.6e}" for j in range(min(n, max_cols)))
        print(f"  [{i}] {row}")


def r8vec_indicator1(n: int) -> np.ndarray:
    """返回 1-based 指示向量 [1, 2, ..., n]。"""
    return np.arange(1, n + 1, dtype=float)


def i4_log_10(n: int) -> int:
    """计算以10为底的整数对数。"""
    if n <= 0:
        return 0
    return int(np.floor(np.log10(float(n))))


def check_symmetry(A: np.ndarray, tol: float = 1e-10) -> bool:
    """检查矩阵对称性。"""
    return bool(np.allclose(A, A.T, atol=tol))


def condition_number_estimate(A: np.ndarray) -> float:
    """基于 SVD 的矩阵条件数估计。"""
    s = np.linalg.svd(A, compute_uv=False)
    return float(s.max() / max(s.min(), 1e-20))


def solve_2x2_symmetric(a11: float, a12: float, a22: float, b1: float, b2: float) -> Tuple[float, float]:
    r"""
    解析求解 2x2 对称正定系统：
    \begin{bmatrix} a11 & a12 \\ a12 & a22 \end{bmatrix}
    \begin{bmatrix} x1 \\ x2 \end{bmatrix} =
    \begin{bmatrix} b1 \\ b2 \end{bmatrix}
    用于局部接触状态的解析更新。
    """
    det = a11 * a22 - a12 * a12
    if abs(det) < 1e-20:
        raise ValueError("2x2 system nearly singular in solve_2x2_symmetric")
    x1 = (a22 * b1 - a12 * b2) / det
    x2 = (-a12 * b1 + a11 * b2) / det
    return x1, x2
