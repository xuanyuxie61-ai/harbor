"""
rank_analysis.py
秩分析与代数结构模块
==================
对应原项目 1048_rref2（RREF 秩判定与列空间提取）与 198_collatz_polynomial（GF(2) 多项式序列），
实现张量展开的数值秩分析、Hankel 张量构造及代数张量秩的符号验证。

核心概念：
- 张量 A∈ℝ^{n1×...×nd} 的 mode-k 展开（matricization）矩阵的秩称为 n-秩或 multilinear rank。
- Hankel 张量由多项式系数构造，其秩与多项式线性递推长度密切相关（Prony 方法）。
"""

import numpy as np
from typing import Tuple, List
from system_utils import EPS, TOL_RANK, check_finite


# ---------------------------------------------------------------------------
# RREF 核心算法（原 rref_compute 迁移）
# ---------------------------------------------------------------------------

def rref_compute(A: np.ndarray, tol: float = None) -> Tuple[np.ndarray, List[int]]:
    """
    计算矩阵的简化行阶梯形（Reduced Row Echelon Form）。

    算法步骤（带部分主元的高斯-约当消元）
    --------------------------------------
    对列 j = 0,...,n-1：
      1. 在行 pivot_row..m-1 中寻找列 j 中绝对值最大元作为主元。
      2. 若 |主元| ≤ tol，跳过该列（自由变量）。
      3. 交换主元至 pivot_row 行。
      4. 将该行归一化，使主元为 1。
      5. 用该行消去所有其他行在列 j 上的分量。

    返回 (RREF, pivot_columns)。
    """
    A = np.asarray(A, dtype=float).copy()
    m, n = A.shape
    if tol is None:
        tol = TOL_RANK * max(m, n)
    pivot_cols = []
    r = 0
    for j in range(n):
        # 部分主元
        pivot_val = 0.0
        pivot_row = -1
        for i in range(r, m):
            if abs(A[i, j]) > pivot_val:
                pivot_val = abs(A[i, j])
                pivot_row = i
        if pivot_val <= tol:
            continue
        # 交换
        if pivot_row != r:
            A[[r, pivot_row], :] = A[[pivot_row, r], :]
        # 归一化
        A[r, :] /= A[r, j]
        # 消去其他行
        for i in range(m):
            if i != r and abs(A[i, j]) > tol:
                A[i, :] -= A[i, j] * A[r, :]
        pivot_cols.append(j)
        r += 1
        if r >= m:
            break
    return A, pivot_cols


def rref_rank(A: np.ndarray, tol: float = None) -> int:
    """
    通过 RREF 计算数值秩。
    """
    _, pivots = rref_compute(A, tol=tol)
    return len(pivots)


def rref_columns(A: np.ndarray, tol: float = None) -> np.ndarray:
    """
    提取线性无关列（列空间基）：
        C = A[:, pivot_columns]
    对应原 rref_columns。
    """
    _, pivots = rref_compute(A, tol=tol)
    return A[:, pivots]


# ---------------------------------------------------------------------------
# 张量展开秩分析
# ---------------------------------------------------------------------------

def unfold_tensor(tensor: np.ndarray, mode: int) -> np.ndarray:
    """
    张量的 mode-k 展开（matricization）：
        A_{(k)} ∈ ℝ^{n_k × (n1...n_{k-1} n_{k+1}...n_d)}
    将 mode-k 索引作为行，其余所有模式展平为列。
    """
    tensor = np.asarray(tensor)
    d = tensor.ndim
    if not (0 <= mode < d):
        raise ValueError(f"mode must be in [0, {d-1}]")
    # 将 mode 移到最前，再 reshape
    perm = [mode] + [i for i in range(d) if i != mode]
    tensor_perm = np.transpose(tensor, perm)
    n_mode = tensor.shape[mode]
    rest = int(np.prod([tensor.shape[i] for i in range(d) if i != mode]))
    return tensor_perm.reshape(n_mode, rest)


def tensor_multilinear_ranks(tensor: np.ndarray, tol: float = None) -> List[int]:
    """
    计算张量的 multilinear ranks（各 mode 展开矩阵的秩）。

    数学定义
    --------
    对 d 阶张量 A，其 multilinear rank 为向量
        r = (rank(A_{(1)}), rank(A_{(2)}), ..., rank(A_{(d)}))
    这是 Tucker 分解的核心秩参数，满足 r_k ≤ n_k。
    """
    tensor = np.asarray(tensor)
    d = tensor.ndim
    ranks = []
    for k in range(d):
        A_k = unfold_tensor(tensor, k)
        ranks.append(rref_rank(A_k, tol=tol))
    return ranks


def estimate_tensor_train_ranks(tensor: np.ndarray, tol: float = None) -> List[int]:
    """
    估计张量列车（Tensor Train）格式的 TT-ranks：
        r_k = rank( A^{(k)} ),  k=1,...,d-1
    其中 A^{(k)} 为左 unfolding：将前 k 个 mode 展平为行，其余展平为列。

    TT-ranks 满足 r_0 = r_d = 1，且 r_k ≤ min(∏_{i=1}^k n_i, ∏_{i=k+1}^d n_i)。
    """
    tensor = np.asarray(tensor)
    d = tensor.ndim
    shape = tensor.shape
    tt_ranks = [1]
    for k in range(1, d):
        left_size = int(np.prod(shape[:k]))
        right_size = int(np.prod(shape[k:]))
        unfolding = tensor.reshape(left_size, right_size)
        r = rref_rank(unfolding, tol=tol)
        tt_ranks.append(r)
    tt_ranks.append(1)
    return tt_ranks


# ---------------------------------------------------------------------------
# Hankel 张量与 Collatz 多项式（原 collatz_polynomial 扩展）
# ---------------------------------------------------------------------------

def collatz_polynomial_next(p: np.ndarray) -> np.ndarray:
    """
    GF(2) 上的 Collatz 多项式递推一步（原 collatz_polynomial_next）：
        若 p(0)=0（即常数项为 0），则 p ← p / x
        否则 p ← p*(x+1) + 1   (mod 2)
    系数向量 p 中 p[k] 对应 x^k 的系数，取值为 0 或 1。
    """
    p = np.asarray(p, dtype=int)
    p = p % 2
    if p.size == 0:
        return np.array([1], dtype=int)
    if p[0] == 0:
        # 除以 x
        if len(p) == 1:
            return np.array([0], dtype=int)
        return p[1:]
    else:
        # 乘以 (x+1) 再加 1
        q = np.zeros(len(p) + 1, dtype=int)
        q[1:] = p
        q[:len(p)] = (q[:len(p)] + p) % 2
        q[0] = (q[0] + 1) % 2
        return q


def collatz_polynomial_sequence(p0: np.ndarray, max_steps: int = 100) -> List[np.ndarray]:
    """
    生成 Collatz 多项式序列，直到达到常数多项式或达到 max_steps。
    """
    seq = [np.asarray(p0, dtype=int).copy()]
    p = seq[0].copy()
    for _ in range(max_steps):
        if len(p) == 1:
            break
        p = collatz_polynomial_next(p)
        seq.append(p.copy())
    return seq


def build_hankel_tensor_from_sequence(seq: List[np.ndarray],
                                       dimensions: Tuple[int, ...]) -> np.ndarray:
    """
    从一维序列构造高阶 Hankel 张量。

    数学定义
    --------
    给定序列 s[0], s[1], ..., s[N-1]，d 阶 Hankel 张量 H∈ℝ^{n1×...×nd} 满足
        H_{i1,i2,...,id} = s[i1 + i2 + ... + id]
    其中 n_k 为各 mode 维度，且 max_index = Σ (n_k - 1) < N。

    若序列满足线性递推关系（如 Collatz 序列的截断），则 Hankel 张量的
    多线性秩受递推阶数限制，可通过张量分解压缩存储。
    """
    d = len(dimensions)
    tensor = np.zeros(dimensions, dtype=float)
    it = np.nditer(tensor, flags=['multi_index'], op_flags=[['writeonly']])
    while not it.finished:
        idx = it.multi_index
        flat_idx = sum(idx)
        if flat_idx < len(seq):
            # 取序列向量值（若 seq 元素为向量，取第一个分量）
            val = float(seq[flat_idx][0]) if seq[flat_idx].size > 0 else 0.0
        else:
            val = 0.0
        it[0] = val
        it.iternext()
    return tensor


def hankel_matrix_from_sequence(s: np.ndarray, m: int, n: int) -> np.ndarray:
    """
    构造经典 Hankel 矩阵 H_{ij} = s_{i+j}，i=0..m-1, j=0..n-1。
    对应 Prony 方法中的核心矩阵，若序列满足 p 阶线性递推，则 rank(H) ≤ p。
    """
    s = np.asarray(s, dtype=float)
    H = np.zeros((m, n), dtype=float)
    for i in range(m):
        for j in range(n):
            idx = i + j
            H[i, j] = s[idx] if idx < len(s) else 0.0
    return H
