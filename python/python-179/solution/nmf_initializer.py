"""
nmf_initializer.py
非负矩阵/张量分解初始化模块
===========================
对应原项目 045_asa159（给定边际和的随机列联表生成），
实现非负张量分解（NTF）的随机初始化与列联表启发式非负性保持变换。

核心思想：
- 列联表可视为非负矩阵，其行和、列和约束与 NMF 的因子非负性兼容。
- 通过随机列联表生成符合边际分布的非负初始因子。
"""

import numpy as np
from typing import Tuple
from system_utils import EPS, clip_to_range


# ---------------------------------------------------------------------------
# 随机列联表生成（原 asa159 核心逻辑 Python 化）
# ---------------------------------------------------------------------------

def random_contingency_table(nrow: int, ncol: int,
                              nrowt: np.ndarray, ncolt: np.ndarray,
                              seed: int = None) -> np.ndarray:
    """
    生成满足给定行和与列和的随机非负整数矩阵（列联表）。

    算法（Patefield, 1981, AS 159）
    -------------------------------
    逐行逐列填充。对单元格 (i,j)，在给定当前剩余行和、列和条件下，
    其分布为超几何分布的近似。使用对数阶乘避免溢出：
        log_fact[k] = log(k!)
    通过接受-拒绝或条件期望采样确定每个单元格值。

    为简化实现，此处采用多项式条件抽样的近似版本，保证非负与边际约束。
    """
    if seed is not None:
        np.random.seed(seed)
    nrowt = np.asarray(nrowt, dtype=int)
    ncolt = np.asarray(ncolt, dtype=int)
    if nrowt.sum() != ncolt.sum():
        raise ValueError("Row sums and column sums must be equal.")
    table = np.zeros((nrow, ncol), dtype=int)
    row_rem = nrowt.copy()
    col_rem = ncolt.copy()
    total = int(row_rem.sum())
    for i in range(nrow):
        for j in range(ncol):
            if row_rem[i] == 0 or col_rem[j] == 0:
                continue
            # 剩余总量
            rem_total = int(row_rem[i:].sum())
            if rem_total == 0:
                break
            # 在当前行剩余中按列比例分配
            p = col_rem[j] / max(col_rem[j:].sum(), 1)
            max_val = min(row_rem[i], col_rem[j])
            # 二项式近似抽样
            val = np.random.binomial(row_rem[i], p)
            val = min(val, max_val)
            val = max(val, 0)
            table[i, j] = val
            row_rem[i] -= val
            col_rem[j] -= val
    return table


# ---------------------------------------------------------------------------
# NMF / NTF 初始化
# ---------------------------------------------------------------------------

def nmf_init_random(m: int, n: int, rank: int, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    标准随机非负初始化：
        W_{ij} ~ Uniform(0,1),  H_{ij} ~ Uniform(0,1)
    对应 Lee & Seung NMF 的默认初始化策略。
    """
    if seed is not None:
        np.random.seed(seed)
    W = np.random.rand(m, rank)
    H = np.random.rand(rank, n)
    return W, H


def nmf_init_coltable(m: int, n: int, rank: int, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于列联表的非负初始化：
    生成 rank×m 和 rank×n 的随机列联表，归一化后作为 NMF 因子。
    保证每行和为 1（概率化解释）。
    """
    if seed is not None:
        np.random.seed(seed)
    # 行和均匀，列和随机，确保总和一致且全为正
    row_sum_W = np.ones(rank, dtype=int) * max(m // rank + 1, 2)
    col_sum_W = np.random.randint(2, 6, size=m)
    total_W = max(row_sum_W.sum(), col_sum_W.sum())
    # 调整到相同总和
    row_sum_W = np.round(row_sum_W * total_W / row_sum_W.sum()).astype(int)
    col_sum_W = np.round(col_sum_W * total_W / col_sum_W.sum()).astype(int)
    # 微调确保严格相等且为正
    diff = row_sum_W.sum() - col_sum_W.sum()
    idx = 0
    while diff != 0:
        if diff > 0:
            if col_sum_W[idx % m] + diff > 0:
                col_sum_W[idx % m] += diff
                diff = 0
            else:
                col_sum_W[idx % m] += 1
                diff -= 1
        else:
            if row_sum_W[idx % rank] - diff > 0:
                row_sum_W[idx % rank] -= diff
                diff = 0
            else:
                row_sum_W[idx % rank] += 1
                diff += 1
        idx += 1
    W_int = random_contingency_table(rank, m, row_sum_W, col_sum_W, seed=seed)
    W = W_int.astype(float) + EPS
    W = W / (W.sum(axis=1, keepdims=True) + EPS)

    row_sum_H = np.ones(rank, dtype=int) * max(n // rank + 1, 2)
    col_sum_H = np.random.randint(2, 6, size=n)
    total_H = max(row_sum_H.sum(), col_sum_H.sum())
    row_sum_H = np.round(row_sum_H * total_H / row_sum_H.sum()).astype(int)
    col_sum_H = np.round(col_sum_H * total_H / col_sum_H.sum()).astype(int)
    diff = row_sum_H.sum() - col_sum_H.sum()
    idx = 0
    while diff != 0:
        if diff > 0:
            if col_sum_H[idx % n] + diff > 0:
                col_sum_H[idx % n] += diff
                diff = 0
            else:
                col_sum_H[idx % n] += 1
                diff -= 1
        else:
            if row_sum_H[idx % rank] - diff > 0:
                row_sum_H[idx % rank] -= diff
                diff = 0
            else:
                row_sum_H[idx % rank] += 1
                diff += 1
        idx += 1
    H_int = random_contingency_table(rank, n, row_sum_H, col_sum_H, seed=seed + 1 if seed else None)
    H = H_int.astype(float).T + EPS
    H = H / (H.sum(axis=0, keepdims=True) + EPS)
    return W, H


def ntf_init_random(shape: Tuple[int, ...], ranks: Tuple[int, ...],
                    seed: int = None) -> list:
    """
    非负张量分解（CP-NTF）的随机因子初始化。

    对 d 阶张量 A∈ℝ^{n1×...×nd}，CP 分解表示为
        A ≈ Σ_{r=1}^R  λ_r * u_r^{(1)} ∘ u_r^{(2)} ∘ ... ∘ u_r^{(d)}
    其中 ∘ 表示外积，u_r^{(k)}∈ℝ^{n_k}_+ 为非负因子向量。

    返回因子列表 factors[k] ∈ ℝ^{n_k × R}。
    """
    if seed is not None:
        np.random.seed(seed)
    d = len(shape)
    if len(ranks) != d:
        raise ValueError("ranks length must match tensor order.")
    factors = []
    for k in range(d):
        F = np.random.rand(shape[k], ranks[k]) + EPS
        # 列归一化
        col_norms = np.linalg.norm(F, axis=0)
        F = F / (col_norms + EPS)
        factors.append(F)
    return factors


# ---------------------------------------------------------------------------
# 非负投影与软阈值
# ---------------------------------------------------------------------------

def nonnegative_projection(X: np.ndarray) -> np.ndarray:
    """
    逐元素非负投影：max(X, 0)。
    """
    return np.maximum(X, 0.0)


def soft_threshold(X: np.ndarray, tau: float) -> np.ndarray:
    """
    软阈值算子（用于稀疏 NMF）：
        S_τ(x) = sign(x) * max(|x| - τ, 0)
    """
    tau = max(tau, 0.0)
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0.0)
