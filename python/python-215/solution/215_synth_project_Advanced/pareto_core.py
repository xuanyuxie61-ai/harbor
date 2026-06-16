#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pareto_core.py — Pareto 支配关系、非支配排序、拥挤距离、前沿提取

对应种子项目:
  - 909_predator_prey_ode_period: 捕食-被捕食的竞争平衡思想映射为
    目标函数间的竞争 (Pareto 支配等价于 Lotka-Volterra 平衡)
  - 906_pram_view: PRAM 分块思想用于种群的非支配层分块

核心数学公式:
  Pareto 支配:  x ≺ y  ⟺  ∀i f_i(x) ≤ f_i(y) ∧ ∃j f_j(x) < f_j(y)
  非支配排序:   将种群按支配关系分层 F_1, F_2, ...
  拥挤距离:
      CD(i) = Σ_k [ f_k(i+1) - f_k(i-1) ] / [ f_k^max - f_k^min ]
  Hypervolume 指标 (2D):
      HV = Σ_{i=1}^{|PF|} (f_1^{(i)} - f_1^{(i+1)}) · (f_2^{ref} - f_2^{(i)})
"""

import numpy as np
from typing import List, Tuple, Optional, Set


# ---------------------------------------------------------------------------
# Pareto 支配关系
# ---------------------------------------------------------------------------
def dominates(f_a: np.ndarray, f_b: np.ndarray,
              tol: float = 1e-12) -> bool:
    """
    判断个体 a 是否 Pareto 支配个体 b (最小化问题).

    数学定义:
        a ≺ b  ⟺  ∀i: f_i(a) ≤ f_i(b)  ∧  ∃j: f_j(a) < f_j(b)

    Parameters
    ----------
    f_a : ndarray, shape (M,)
        个体 a 的 M 个目标值 (均已转为最小化)
    f_b : ndarray, shape (M,)
        个体 b 的 M 个目标值
    tol : float
        数值容差, 防止浮点抖动

    Returns
    -------
    bool
    """
    f_a = np.asarray(f_a, dtype=np.float64)
    f_b = np.asarray(f_b, dtype=np.float64)
    if f_a.shape != f_b.shape:
        raise ValueError("目标维度不匹配")
    # 数值鲁棒: 用容差比较
    le_all = np.all(f_a <= f_b + tol)
    lt_some = np.any(f_a < f_b - tol)
    return bool(le_all and lt_some)


def fast_non_dominated_sort(F: np.ndarray,
                            tol: float = 1e-12) -> List[List[int]]:
    """
    Deb et al. (2002) 快速非支配排序.

    对种群 F (N×M 目标矩阵) 分层, 返回 fronts 列表.

    算法:
        对每个个体 p:
            S_p = {q : p ≺ q}    (p 支配的集合)
            n_p = |{q : q ≺ p}|  (支配 p 的个体数)
        第一前沿 F_1 = {p : n_p = 0}
        后续前沿通过递减 n_q 获得.

    复杂度: O(M N²)

    Parameters
    ----------
    F : ndarray, shape (N, M)
    tol : float

    Returns
    -------
    List[List[int]]  — 每个子列表是一个非支配层的个体索引
    """
    N, M = F.shape
    S: List[Set[int]] = [set() for _ in range(N)]
    n = np.zeros(N, dtype=int)
    rank = np.full(N, -1, dtype=int)

    for p in range(N):
        for q in range(p + 1, N):
            if dominates(F[p], F[q], tol):
                S[p].add(q)
                n[q] += 1
            elif dominates(F[q], F[p], tol):
                S[q].add(p)
                n[p] += 1

    fronts: List[List[int]] = []
    current_front = [i for i in range(N) if n[i] == 0]
    if not current_front:
        # 所有个体互不支配 (极端情况)
        current_front = list(range(N))

    front_idx = 0
    while current_front:
        fronts.append(current_front)
        for p in current_front:
            rank[p] = front_idx
        next_front = []
        for p in current_front:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        current_front = next_front
        front_idx += 1

    return fronts


def crowding_distance(F: np.ndarray,
                      front: List[int]) -> np.ndarray:
    """
    计算一个非支配层中每个个体的拥挤距离.

    公式:
        CD(i) = Σ_{k=1}^{M} [ f_k^{(i+1)} - f_k^{(i-1)} ]
                            / [ f_k^max - f_k^min + ε ]

    边界个体 CD = ∞ (保证保留极端解).

    Parameters
    ----------
    F : ndarray, shape (N, M)  — 全局目标矩阵
    front : List[int]          — 该层个体在 F 中的索引

    Returns
    -------
    ndarray, shape (len(front),)
    """
    n = len(front)
    if n == 0:
        return np.array([])
    M = F.shape[1]
    cd = np.zeros(n, dtype=np.float64)
    eps = 1e-14

    if n <= 2:
        cd[:] = np.inf
        return cd

    F_front = F[front]  # (n, M)
    for k in range(M):
        col = F_front[:, k]
        sorted_idx = np.argsort(col)
        f_range = col[sorted_idx[-1]] - col[sorted_idx[0]]
        if f_range < eps:
            continue
        cd[sorted_idx[0]] = np.inf
        cd[sorted_idx[-1]] = np.inf
        for i in range(1, n - 1):
            cd[sorted_idx[i]] += (col[sorted_idx[i + 1]]
                                  - col[sorted_idx[i - 1]]) / f_range
    return cd


# ---------------------------------------------------------------------------
# Hypervolume 指标
# ---------------------------------------------------------------------------
def hypervolume_2d(pf: np.ndarray,
                   ref: np.ndarray) -> float:
    """
    2D 前沿的超体积指标 (精确计算).

    HV = Σ_{i=0}^{|PF|-1} (f_1^{ref} - f_1^{(i)}) · (f_2^{(i)} - f_2^{(i+1)})

    其中 PF 按 f_1 升序排列, f_2^{(|PF|)} = f_2^{ref}.

    Parameters
    ----------
    pf  : ndarray, shape (K, 2) — Pareto 前沿 (已去支配)
    ref : ndarray, shape (2,)   — 参考点

    Returns
    -------
    float — 超体积 (越大越好)
    """
    if pf.shape[1] != 2:
        raise ValueError("仅支持 2D 超体积")
    pf = pf[np.argsort(pf[:, 0])]
    hv = 0.0
    n = pf.shape[0]
    for i in range(n):
        f2_next = pf[i + 1, 1] if i + 1 < n else ref[1]
        width = ref[0] - pf[i, 0]
        height = f2_next - pf[i, 1]
        if width > 0 and height > 0:
            hv += width * height
    return hv


def hypervolume_nd_approx(pf: np.ndarray,
                          ref: np.ndarray,
                          n_samples: int = 50000,
                          seed: int = 42) -> float:
    """
    N 维超体积的蒙特卡洛近似.

    HV ≈ (1/N_s) Σ_{s=1}^{N_s} 𝟙[ ∃ p ∈ PF : p ≺ s ]
            · Π_k (f_k^ref - f_k^min)

    Parameters
    ----------
    pf       : ndarray, shape (K, M)
    ref      : ndarray, shape (M,)
    n_samples: int
    seed     : int

    Returns
    -------
    float
    """
    rng = np.random.default_rng(seed)
    M = pf.shape[1]
    lb = pf.min(axis=0)
    box_vol = np.prod(ref - lb)
    if box_vol <= 0:
        return 0.0

    samples = rng.uniform(lb, ref, size=(n_samples, M))
    # 对每个样本, 检查是否被至少一个 Pareto 点支配
    dominated_count = 0
    for s in samples:
        for p in pf:
            if np.all(p <= s):
                dominated_count += 1
                break
    return box_vol * dominated_count / n_samples


# ---------------------------------------------------------------------------
# Pareto 前沿质量指标
# ---------------------------------------------------------------------------
def spacing_metric(pf: np.ndarray) -> float:
    """
    Schott 间距指标 — 衡量 Pareto 前沿分布均匀性.

    SM = sqrt( (1/|PF|) Σ_{i=1}^{|PF|} (d_i - d̄)² )

    其中 d_i = min_{j≠i} Σ_k |f_k^{(i)} - f_k^{(j)}|

    SM 越小, 分布越均匀.

    Parameters
    ----------
    pf : ndarray, shape (K, M)

    Returns
    -------
    float
    """
    K = pf.shape[0]
    if K <= 1:
        return 0.0
    d = np.zeros(K)
    for i in range(K):
        dists = []
        for j in range(K):
            if i == j:
                continue
            dists.append(np.sum(np.abs(pf[i] - pf[j])))
        d[i] = min(dists)
    d_bar = np.mean(d)
    return float(np.sqrt(np.mean((d - d_bar) ** 2)))


def generational_distance(pf: np.ndarray,
                          pf_true: np.ndarray,
                          p: float = 2.0) -> float:
    """
    世代距离 GD — 收敛性指标.

    GD = (1/|PF|)^(1/p) · [ Σ_{i=1}^{|PF|} d_i^p ]^(1/p)

    其中 d_i = min_{j} || f^{(i)} - f_true^{(j)} ||

    Parameters
    ----------
    pf      : ndarray, shape (K, M)  — 近似前沿
    pf_true : ndarray, shape (L, M)  — 真实前沿 (或参考前沿)
    p       : float

    Returns
    -------
    float
    """
    K = pf.shape[0]
    if K == 0 or pf_true.shape[0] == 0:
        return np.inf
    d = np.zeros(K)
    for i in range(K):
        dists = np.linalg.norm(pf_true - pf[i], axis=1)
        d[i] = dists.min()
    return float((1.0 / K) ** (1.0 / p) * np.linalg.norm(d, ord=p))


def extract_pareto_front(F: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """
    提取目标矩阵 F 中的第一非支配前沿.

    Returns: F[front_1] — shape (K, M)
    """
    fronts = fast_non_dominated_sort(F, tol)
    return F[fronts[0]]
