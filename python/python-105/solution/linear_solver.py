r"""
linear_solver.py
================
高阶线性代数求解引擎 —— 融合原项目 337_eros (Gauss 消元 / PLU 分解)。

在量子光源纠缠态数值模拟中，大量耦合模方程经过线性化后形成稠密或稀疏
线性系统。本模块提供：
1. 带部分主元的高斯消元（Gauss elimination with partial pivoting）。
2. PLU 分解与三角替换求解。
3. 边界条件检查与数值稳定性诊断。

核心公式
--------
对于线性系统 :math:`A x = b`，PLU 分解将 :math:`A` 写为

.. math::
    P A = L U

其中 :math:`P` 为置换矩阵，:math:`L` 为单位下三角矩阵，:math:`U` 为上三角矩阵。
求解分两步：

.. math::
    L y = P b, \quad U x = y

前向替换 :math:`y_i = \left( (Pb)_i - \sum_{j=1}^{i-1} L_{ij} y_j \right)`
后向替换 :math:`x_i = \left( y_i - \sum_{j=i+1}^{n} U_{ij} x_j \right) / U_{ii}`

数值稳定性通过部分主元保证：

.. math::
    |U_{ii}| \ge \max_{j \ge i} |A_{ji}^{(i)}|

若发现 :math:`|U_{ii}| < \varepsilon_{\text{mach}} \cdot \|A\|_{\infty}`，则判定矩阵奇异或病态。
"""

import numpy as np
from typing import Tuple, Optional


def gauss_elimination_partial_pivot(A: np.ndarray, b: np.ndarray,
                                    tol: Optional[float] = None) -> np.ndarray:
    """
    带部分主元的高斯消元求解 A x = b。

    参数
    ----
    A : np.ndarray, shape (n, n)
        系数矩阵。
    b : np.ndarray, shape (n,) 或 (n, k)
        右端项。
    tol : float, optional
        奇异判定阈值，默认为 eps * n * max(|A_ij|)。

    返回
    ----
    x : np.ndarray
        解向量或解矩阵。

    异常
    ----
    ValueError
        当矩阵奇异、非方阵、或维度不匹配时抛出。
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A 必须为方阵。")
    n = A.shape[0]
    b = np.atleast_2d(b).T if b.ndim == 1 else np.array(b)
    if b.shape[0] != n:
        raise ValueError("b 的行数必须与 A 的维数一致。")

    # 构造增广矩阵
    Ab = np.hstack([A.astype(np.float64), b.astype(np.float64)])
    if tol is None:
        tol = np.finfo(float).eps * n * np.max(np.abs(A))

    # 消元过程
    for col in range(n):
        # 部分主元：在当前列下方寻找最大元
        pivot_row = col + np.argmax(np.abs(Ab[col:, col]))
        max_val = np.abs(Ab[pivot_row, col])
        if max_val < tol:
            raise ValueError(f"矩阵在第 {col} 列无主元，可能奇异。")
        if pivot_row != col:
            Ab[[col, pivot_row], :] = Ab[[pivot_row, col], :]

        # 归一化主元行
        pivot = Ab[col, col]
        Ab[col, :] /= pivot

        # 消去下方各行
        for row in range(col + 1, n):
            factor = Ab[row, col]
            if factor != 0.0:
                Ab[row, :] -= factor * Ab[col, :]

    # 回代
    x = np.zeros((n, b.shape[1]), dtype=np.float64)
    for i in range(n - 1, -1, -1):
        x[i, :] = Ab[i, n:] - Ab[i, i + 1:n] @ x[i + 1:, :]

    return x.squeeze() if x.shape[1] == 1 else x


def plu_decomposition(A: np.ndarray,
                      tol: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Doolittle 型 PLU 分解：P A = L U。

    参数
    ----
    A : np.ndarray, shape (n, n)
    tol : float, optional

    返回
    ----
    P, L, U : np.ndarray
        置换矩阵、单位下三角、上三角。
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A 必须为方阵。")
    n = A.shape[0]
    if tol is None:
        tol = np.finfo(float).eps * n * np.max(np.abs(A) + 1.0)

    L = np.eye(n, dtype=np.float64)
    U = A.astype(np.float64).copy()
    P = np.eye(n, dtype=np.float64)

    for k in range(n - 1):
        pivot = np.argmax(np.abs(U[k:, k])) + k
        if abs(U[pivot, k]) < tol:
            raise ValueError(f"U_{k},{k} 接近零，矩阵奇异。")
        if pivot != k:
            U[[k, pivot], :] = U[[pivot, k], :]
            P[[k, pivot], :] = P[[pivot, k], :]
            if k > 0:
                L[[k, pivot], :k] = L[[pivot, k], :k]

        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
            U[i, k] = 0.0

    return P, L, U


def solve_plu(P: np.ndarray, L: np.ndarray, U: np.ndarray,
              b: np.ndarray) -> np.ndarray:
    """
    利用已分解的 PLU 求解 P A x = L U x = P b。

    前向替换解 L y = P b，后向替换解 U x = y。
    """
    n = L.shape[0]
    b = np.atleast_1d(b).astype(np.float64)
    if b.ndim == 1:
        b = b.reshape(-1, 1)

    pb = P @ b
    y = np.zeros_like(pb)
    for i in range(n):
        y[i, :] = pb[i, :] - L[i, :i] @ y[:i, :]

    x = np.zeros_like(pb)
    for i in range(n - 1, -1, -1):
        denom = U[i, i]
        if abs(denom) < 1e-15:
            raise ValueError("U 对角元接近零，无法回代。")
        x[i, :] = (y[i, :] - U[i, i + 1:] @ x[i + 1:, :]) / denom

    return x.squeeze() if x.shape[1] == 1 else x


def condition_number_estimate(A: np.ndarray) -> float:
    r"""
    使用无穷范数估计条件数 :math:`\kappa_{\infty}(A)`。

    .. math::
        \kappa_{\infty}(A) = \|A\|_{\infty} \cdot \|A^{-1}\|_{\infty}
    """
    norm_A = np.linalg.norm(A, ord=np.inf)
    try:
        norm_Ainv = np.linalg.norm(np.linalg.inv(A), ord=np.inf)
    except np.linalg.LinAlgError:
        return np.inf
    return norm_A * norm_Ainv
