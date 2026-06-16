# -*- coding: utf-8 -*-
"""
matrix_analysis.py
==================

矩阵分析与整行简化阶梯形 (RREF) 模块

对应种子项目 569_i4mat_rref2: 整数矩阵的行简化阶梯形算法,
用于分析有限差分算子矩阵的秩与零空间结构。

科学应用
--------
1. 有限差分矩阵的秩分析: 验证离散算子的满秩性
2. 约束系统求解: DM 相空间守恒律的约束矩阵
3. 反卷积问题: 探测器响应矩阵的条件数
4. 整数 RREF: 避免浮点对消误差 (小矩阵关键)

整数 RREF 算法 (源自 569):
  1. 选择主元 (最大元素)
  2. 用整数运算消元 (LCM 避免分数)
  3. 每行除以行 GCD 保持整数
  4. 判断秩 = 非零行数
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Optional
from math import gcd


# ---------------------------------------------------------------------------
# 第一部分: 整数 GCD 与 LCM
# ---------------------------------------------------------------------------

def igcd(a: int, b: int) -> int:
    """整数最大公约数 (扩展 Euclid)。"""
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def ilcm(a: int, b: int) -> int:
    """整数最小公倍数。"""
    return abs(a * b) // igcd(a, b) if a != 0 and b != 0 else 0


def row_gcd(row: np.ndarray) -> int:
    """计算整数行的 GCD。"""
    g = 0
    for x in row:
        g = igcd(g, int(abs(x)))
    return max(g, 1)


# ---------------------------------------------------------------------------
# 第二部分: 整数 RREF (源自 569_i4mat_rref2)
# ---------------------------------------------------------------------------

def i4mat_rref2(
    M: np.ndarray,
    n1: int,
    n2: int = 0,
) -> Tuple[np.ndarray, int]:
    """
    计算整数矩阵的整行简化阶梯形 (IRREF)。

    源自 569_i4mat_rref2 核心算法。

    输入:
      M   : (m, n1+n2) 整数矩阵
      n1  : 主矩阵列数 (进行主元消元)
      n2  : 增广列数 (随主矩阵变换但不做主元)

    输出:
      A     : IRREF 形式矩阵
      rank  : 矩阵秩

    算法:
      1. 对每一行, 找主元列 (第一个非零元素)
      2. 用整数运算消元: A[j] = (pivot * A[j] - A[i] * A[j,pivot_col]) / scale
      3. 每行约去 GCD
      4. 确保主元为正
    """
    A = M.copy().astype(np.int64)
    m, n_total = A.shape
    if n1 + n2 != n_total:
        raise ValueError(f"n1 + n2 = {n1+n2} != n_total = {n_total}")

    rank = 0
    pivot_col = 0

    for pivot_col in range(n1):
        # 在当前行 (rank) 及以下找非零主元
        pivot_row = -1
        for i in range(rank, m):
            if A[i, pivot_col] != 0:
                pivot_row = i
                break
        if pivot_row == -1:
            continue  # 此列全零, 跳过

        # 交换行
        if pivot_row != rank:
            A[[rank, pivot_row]] = A[[pivot_row, rank]]

        # 如果主元为负, 整行取反
        if A[rank, pivot_col] < 0:
            A[rank] = -A[rank]

        pivot_val = A[rank, pivot_col]

        # 对其他行消元
        for i in range(m):
            if i == rank:
                continue
            if A[i, pivot_col] == 0:
                continue
            # A[i] = pivot_val * A[i] - A[i, pivot_col] * A[rank]
            # 为避免溢出, 先计算 GCD
            factor_i = A[i, pivot_col]
            g = igcd(abs(factor_i), abs(pivot_val))
            scale_i = factor_i // g
            scale_p = pivot_val // g

            A[i] = scale_p * A[i] - scale_i * A[rank]
            # 约去行 GCD
            g_row = row_gcd(A[i])
            if g_row > 1:
                A[i] //= g_row
            # 确保首非零元素为正
            for k in range(n_total):
                if A[i, k] != 0:
                    if A[i, k] < 0:
                        A[i] = -A[i]
                    break

        rank += 1

    return A, rank


def i4mat_rref2_solve(
    A_aug: np.ndarray,
    n1: int,
) -> Tuple[Optional[np.ndarray], str]:
    """
    使用 IRREF 求解整数线性方程组 A x = b。

    A_aug 是增广矩阵 [A | b], 列数 = n1 + 1。

    返回: (x, status_message)
      status: 'unique', 'infinite', 'inconsistent'
    """
    m = A_aug.shape[0]
    n2 = A_aug.shape[1] - n1
    A_rref, rank = i4mat_rref2(A_aug, n1, n2)

    # 检查一致性: 如果有 [0, ..., 0 | c] 行 (c != 0), 无解
    for i in range(rank, m):
        if np.any(A_rref[i, :n1] != 0):
            continue
        if A_rref[i, n1] != 0:
            return None, 'inconsistent'

    if rank < n1:
        return None, 'infinite'

    # 唯一解: x_i = A_rref[i, n1] / A_rref[i, pivot_col]
    x = np.zeros(n1)
    pivot_cols = []
    for i in range(rank):
        for j in range(n1):
            if A_rref[i, j] != 0:
                pivot_cols.append(j)
                x[j] = A_rref[i, n1] / A_rref[i, j]
                break
    return x, 'unique'


# ---------------------------------------------------------------------------
# 第三部分: 浮点 RREF (用于实际数值分析)
# ---------------------------------------------------------------------------

def rref_float(
    A: np.ndarray,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, int]:
    """
    浮点 RREF (带阈值判断)。

    与整数版本互补: 用于实际数值矩阵,
    但需要注意小矩阵中浮点误差导致秩判断错误。
    """
    A = A.astype(np.float64).copy()
    m, n = A.shape
    rank = 0

    for j in range(n):
        # 找主元
        pivot_row = -1
        max_val = tol
        for i in range(rank, m):
            if abs(A[i, j]) > max_val:
                max_val = abs(A[i, j])
                pivot_row = i
        if pivot_row == -1:
            continue

        # 交换
        if pivot_row != rank:
            A[[rank, pivot_row]] = A[[pivot_row, rank]]

        # 归一化
        A[rank] /= A[rank, j]

        # 消元
        for i in range(m):
            if i != rank:
                A[i] -= A[i, j] * A[rank]

        rank += 1

    return A, rank


# ---------------------------------------------------------------------------
# 第四部分: FD 矩阵分析
# ---------------------------------------------------------------------------

def fd_matrix_rank_analysis(
    N: int,
    h: float,
    order: int = 4,
) -> Dict:
    """
    分析有限差分矩阵的秩和条件数。

    返回:
      {
        'rank': 矩阵秩,
        'nullity': 零度,
        'condition_number': 条件数,
        'min_singular_value': 最小奇异值,
        'max_singular_value': 最大奇异值,
        'is_full_rank': 是否满秩,
      }
    """
    from stability_analysis import build_diffusion_matrix_1d
    A = build_diffusion_matrix_1d(N, 1.0, h, order)

    # SVD 分析
    singular_values = np.linalg.svd(A, compute_uv=False)
    cond = singular_values[0] / max(singular_values[-1], 1e-15)

    # 整数 RREF 分析 (转换为整数矩阵)
    scale = 12 * h * h  # 4 阶 Laplacian 分母
    A_int = np.round(A * scale).astype(np.int64)
    _, rank_int = i4mat_rref2(A_int, N)

    return {
        'N': N,
        'order': order,
        'rank_integer_rref': rank_int,
        'nullity': N - rank_int,
        'condition_number': float(cond),
        'min_singular_value': float(singular_values[-1]),
        'max_singular_value': float(singular_values[0]),
        'is_full_rank': rank_int == N,
    }


# ---------------------------------------------------------------------------
# 第五部分: 响应矩阵反卷积
# ---------------------------------------------------------------------------

def unfold_response(
    observed: np.ndarray,
    response_matrix: np.ndarray,
    method: str = 'svd',
    reg_param: float = 1e-6,
) -> np.ndarray:
    """
    探测器响应矩阵反卷积 (unfolding)。

    观测: y = R x + noise
    求解: x = R^{-1} y (正则化)

    方法:
      'svd': 截断 SVD (Tikhonov 正则化)
      'rref': 整数 RREF (精确但仅适用于小整数问题)

    SVD 方法:
      R = U Σ V^T
      x_reg = V Σ_reg^{-1} U^T y
    其中 Σ_reg^{-1}_i = σ_i / (σ_i² + λ²)
    """
    if method == 'svd':
        U, s, Vt = np.linalg.svd(response_matrix, full_matrices=False)
        # Tikhonov 正则化
        s_reg = s / (s**2 + reg_param**2)
        x = Vt.T @ np.diag(s_reg) @ U.T @ observed
        return x
    elif method == 'rref':
        # 构建增广矩阵
        A_aug = np.hstack([response_matrix, observed.reshape(-1, 1)])
        A_int = np.round(A_aug * 1e6).astype(np.int64)
        x, status = i4mat_rref2_solve(A_int, response_matrix.shape[1])
        if x is None:
            # 退回到 SVD
            return unfold_response(observed, response_matrix, 'svd', reg_param)
        return x.astype(float)
    else:
        raise ValueError(f"未知反卷积方法: {method}")


# ---------------------------------------------------------------------------
# 第六部分: Wilson 矩阵 (源自 569)
# ---------------------------------------------------------------------------

def wilson_matrix() -> np.ndarray:
    """
    Wilson 矩阵 (源自 569_i4mat_rref2 的测试用例):

    4×4 整数矩阵, 用于测试 RREF 算法。
      [ 5  7  6  5 ]
      [ 7 10  8  7 ]
      [ 6  8 10  9 ]
      [ 5  7  9 10 ]
    """
    return np.array([
        [5, 7, 6, 5],
        [7, 10, 8, 7],
        [6, 8, 10, 9],
        [5, 7, 9, 10],
    ], dtype=np.int64)


def test_rref_wilson() -> Dict:
    """测试 RREF 算法在 Wilson 矩阵上的行为。"""
    W = wilson_matrix()
    W_rref, rank = i4mat_rref2(W, 4)
    return {
        'original': W.tolist(),
        'rref': W_rref.tolist(),
        'rank': rank,
    }
