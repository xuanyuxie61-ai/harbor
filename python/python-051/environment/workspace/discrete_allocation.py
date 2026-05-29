"""
discrete_allocation.py
======================
海洋生态离散资源分配与数据压缩模块。

融合算法
--------
1. 多维丢番图方程（源自 743_mcnuggets_diophantine）：
   将海洋氮库的有限整数资源（如 mmol N）分配给不同生态组分，
   求解：
       a₁N₁ + a₂N₂ + ... + a_k N_k = b
   其中 a_i 为各生态功能组的氮需求系数，b 为总可用氮预算，
   N_i ≥ 0 为整数分配量。

2. 字典编码（源自 278_dictionary_code）：
   对高维模拟状态向量进行无损压缩，通过构建状态字典减少存储。
   适用于长期海洋模拟的状态快照归档。

数学描述
--------
给定预算 B 与需求向量 a = (a₁, ..., a_k)，非负整数解的枚举
等价于 k 维单形 lattice 点的计数。解的个数为组合数：
    #solutions ≤ C(⌊B/min(a)⌋ + k - 1, k - 1)

字典编码压缩率：
    CR = (原始字符数) / (编码后索引数 × 索引位宽 + 字典大小)
"""

import numpy as np


# ---------------------------------------------------------------------------
# 多维丢番图方程非负整数解（源自 743_mcnuggets_diophantine）
# ---------------------------------------------------------------------------

def diophantine_nd_nonnegative(a, b):
    """
    求解 a₁x₁ + a₂x₂ + ... + a_n x_n = b 的所有非负整数解。

    参数
    ----
    a : ndarray (n,)
        正整数系数
    b : int
        右端项（非负整数）

    返回
    ----
    solutions : ndarray (k, n)
        k 个解，每行一个
    """
    a = np.asarray(a).flatten()
    n = len(a)
    if b < 0:
        return np.array([]).reshape(0, n)
    if np.any(a <= 0):
        raise ValueError("系数 a 必须全为正整数")

    solutions = []
    y = np.zeros(n, dtype=int)
    j = 0

    while True:
        r = b - np.dot(a[:j], y[:j])
        if j < n:
            y[j] = r // a[j]
            j += 1
        else:
            if r == 0:
                solutions.append(y.copy())
            # 回溯
            while j > 0:
                j -= 1
                if y[j] > 0:
                    y[j] -= 1
                    j += 1
                    break
            else:
                break

    if len(solutions) == 0:
        return np.array([]).reshape(0, n)
    return np.array(solutions)


def allocate_nutrient_budget(budget, demand_coeffs, objective="min_variance"):
    """
    在离散非负整数约束下分配营养盐预算。

    参数
    ----
    budget : int
        总预算
    demand_coeffs : ndarray
        各功能组单位生物量氮需求
    objective : str
        "min_variance" 或 "max_even"

    返回
    ----
    best_alloc : ndarray
        最优分配方案
    """
    sols = diophantine_nd_nonnegative(demand_coeffs, budget)
    if sols.shape[0] == 0:
        return np.zeros(len(demand_coeffs), dtype=int)

    if objective == "min_variance":
        # 最小化分配量的方差（最均匀分配）
        variances = np.var(sols, axis=1)
        idx = np.argmin(variances)
    elif objective == "max_even":
        # 最大化最小分配量
        mins = np.min(sols, axis=1)
        idx = np.argmax(mins)
    else:
        idx = 0

    return sols[idx]


# ---------------------------------------------------------------------------
# 字典编码（源自 278_dictionary_code）
# ---------------------------------------------------------------------------

def dictionary_encode(state_vectors, tol=1e-8):
    """
    对状态向量集合进行字典编码压缩。

    参数
    ----
    state_vectors : ndarray (n_snapshots, n_features)
        模拟状态快照矩阵
    tol : float
        判断两个状态是否相等的容差

    返回
    ----
    dictionary : ndarray (n_unique, n_features)
        唯一状态字典
    indices : ndarray (n_snapshots,)
        每个快照对应的字典索引
    compression_ratio : float
        压缩比
    """
    n_snapshots, n_features = state_vectors.shape

    # 聚类：以 tol 为阈值合并近似相等的状态
    dictionary = []
    indices = np.zeros(n_snapshots, dtype=int)

    for i in range(n_snapshots):
        vec = state_vectors[i]
        found = False
        for j, ref in enumerate(dictionary):
            if np.linalg.norm(vec - ref) <= tol * max(1.0, np.linalg.norm(ref)):
                indices[i] = j
                found = True
                break
        if not found:
            dictionary.append(vec.copy())
            indices[i] = len(dictionary) - 1

    dictionary = np.array(dictionary)
    n_unique = dictionary.shape[0]

    # 压缩比 = 原始大小 / 编码后大小
    original_size = n_snapshots * n_features
    encoded_size = n_unique * n_features + n_snapshots * np.log2(max(n_unique, 2)) / 8.0
    compression_ratio = original_size / max(encoded_size, 1.0)

    return dictionary, indices, compression_ratio


def dictionary_decode(dictionary, indices):
    """
    由字典与索引重建原始状态序列。
    """
    return dictionary[indices]


# ---------------------------------------------------------------------------
# 生态状态快照管理
# ---------------------------------------------------------------------------

def snapshot_matrix(ocean_fields):
    """
    将海洋场字典展平为状态向量矩阵。

    参数
    ----
    ocean_fields : dict
        {'T': ndarray, 'S': ndarray, 'N': ndarray, 'P': ndarray, ...}

    返回
    ----
    flat : ndarray (n_fields * nx * nz,)
    """
    parts = []
    for key in sorted(ocean_fields.keys()):
        parts.append(ocean_fields[key].ravel())
    return np.concatenate(parts)


def pack_snapshots(snapshots_list):
    """
    将多个时间步的状态向量打包为矩阵 (n_times, n_features)。
    """
    return np.vstack(snapshots_list)
