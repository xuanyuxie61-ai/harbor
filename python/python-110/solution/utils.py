"""
utils.py - 通用工具模块

本模块提供稀疏矩阵操作、数据校验与文件 I/O 工具，
源自原项目 380_fem_to_tec 与 508_hb_to_mm 的矩阵/文件处理思想。
"""

import numpy as np
from typing import Tuple, Optional


def validate_array_1d(arr: np.ndarray, name: str = "array") -> np.ndarray:
    """校验一维数组，处理边界情况并返回展平后的数组。"""
    if arr is None:
        raise ValueError(f"{name} cannot be None")
    arr = np.asarray(arr)
    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty")
    return arr.ravel()


def validate_array_2d(arr: np.ndarray, name: str = "array") -> np.ndarray:
    """校验二维数组。"""
    if arr is None:
        raise ValueError(f"{name} cannot be None")
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {arr.ndim}D")
    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty")
    return arr


def safe_inverse(x: np.ndarray, eps: float = 1e-14) -> np.ndarray:
    """安全求逆，避免除零。"""
    x = np.asarray(x, dtype=float)
    x = np.where(np.abs(x) < eps, np.sign(x + eps) * eps, x)
    return 1.0 / x


def build_sparse_hamiltonian_indices(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    构造一维有限差分稀疏 Hamiltonian 的三元组 (row, col, data)。
    
    用于模拟量子点中电子/空穴的离散化动能项：
        H_{i,i}   =  2 / dx^2
        H_{i,i+1} = -1 / dx^2
        H_{i,i-1} = -1 / dx^2
    """
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
    """稀疏矩阵-向量乘法 (COO 格式)。"""
    vec = validate_array_1d(vec, "vec")
    n = vec.size
    out = np.zeros(n, dtype=float)
    for r, c, d in zip(rows, cols, data):
        if 0 <= r < n and 0 <= c < n:
            out[r] += d * vec[c]
    return out


def estimate_condition_number_dense(A: np.ndarray) -> float:
    """估算稠密矩阵的条件数（简化版，基于特征值）。"""
    A = validate_array_2d(A, "A")
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")
    # 使用幂法估算最大与最小特征值绝对值之比
    eigvals = np.linalg.eigvalsh(A)
    abs_eig = np.abs(eigvals)
    min_eig = np.min(abs_eig)
    max_eig = np.max(abs_eig)
    if min_eig < 1e-15:
        min_eig = 1e-15
    return max_eig / min_eig


def fio_write_matrix(filename: str, A: np.ndarray, fmt: str = "%.16e") -> None:
    """将矩阵写入文本文件（源自 fem_to_tec 的数据导出思想）。"""
    A = validate_array_2d(A, "A")
    np.savetxt(filename, A, fmt=fmt)


def fio_read_matrix(filename: str) -> Optional[np.ndarray]:
    """从文本文件读取矩阵。"""
    try:
        return np.loadtxt(filename)
    except Exception as e:
        print(f"[utils] Warning: failed to read {filename}: {e}")
        return None


def tridiagonal_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    求解三对角线性方程组 T x = d，
    其中 T 的下对角为 a，主对角为 b，上对角为 c。
    
    这是有限差分离散化后求波函数的关键步骤。
    """
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
