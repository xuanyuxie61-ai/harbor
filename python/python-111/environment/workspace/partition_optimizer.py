"""
贪心划分优化模块
基于 partition_greedy 核心算法：整数贪心划分。

在蛋白质折叠中的应用：
- 并行 MD 负载均衡：将残基按计算负载分配到不同进程
- 蛋白质结构域划分：基于残基间接触强度二分蛋白质
- 接触图二分：用于粗粒化多尺度模拟
- 自由能景观的盆地划分

数学基础:
    贪心算法:
        1. 将数组 W 按降序排序
        2. 初始化 S_0 = 0, S_1 = 0
        3. 依次将当前最大元素放入当前和较小的子集:
            x_j = 0 if S_0 < S_1 else 1
        
    时间复杂度: O(N log N)（主要由排序决定）
"""

import numpy as np
from typing import Tuple, List


def partition_greedy(w: np.ndarray) -> np.ndarray:
    """
    用贪心算法将整数/实数数组划分为两个子集，使两子集之和的绝对差最小。
    
    Parameters
    ----------
    w : np.ndarray
        待划分数组。
    
    Returns
    -------
    x : np.ndarray
        划分标记 (0 或 1)。
    """
    w = np.array(w, dtype=float)
    n = len(w)
    if n == 0:
        return np.array([])
    
    # 按降序排序，同时记录原始索引
    idx_sorted = np.argsort(-w)
    x = np.zeros(n, dtype=int)
    
    s0 = 0.0
    s1 = 0.0
    for idx in idx_sorted:
        if s0 < s1:
            x[idx] = 0
            s0 += w[idx]
        else:
            x[idx] = 1
            s1 += w[idx]
    
    return x


def partition_residues_by_contact(contacts: np.ndarray, n_partitions: int = 2) -> List[np.ndarray]:
    """
    基于接触强度将残基划分为多个子集（结构域划分）。
    
    策略:
        1. 计算每个残基的总接触权重
        2. 递归使用贪心二分将残基分组
    
    Parameters
    ----------
    contacts : np.ndarray, shape (N, N)
        残基接触权重矩阵（对称）。
    n_partitions : int
        目标划分组数（2的幂次）。
    
    Returns
    -------
    groups : list of np.ndarray
        每组包含的残基索引。
    """
    N = contacts.shape[0]
    total_weights = np.sum(contacts, axis=1)
    
    groups = [np.arange(N)]
    while len(groups) < n_partitions:
        new_groups = []
        for g in groups:
            if len(g) <= 1:
                new_groups.append(g)
                continue
            sub_weights = total_weights[g]
            partition = partition_greedy(sub_weights)
            g0 = g[partition == 0]
            g1 = g[partition == 1]
            new_groups.append(g0)
            new_groups.append(g1)
        groups = new_groups
    
    return groups


def partition_free_energy_landscape(energies: np.ndarray, n_bins: int = 4) -> List[Tuple[float, float]]:
    """
    将自由能景观按能量值划分为近似等体积的盆地。
    
    策略:
        1. 对能量值排序
        2. 贪心划分为 n_bins 组，使每组总概率近似相等
        3. 返回每组的能量范围
    
    Parameters
    ----------
    energies : np.ndarray
        能量值数组。
    n_bins : int
        划分数。
    
    Returns
    -------
    ranges : list of tuples
        每组的 (E_min, E_max)。
    """
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    
    # 将能量转换为概率权重: w = exp(-E) (设 kT=1)
    w = np.exp(-energies)
    sorted_idx = np.argsort(energies)
    sorted_w = w[sorted_idx]
    
    total_weight = np.sum(sorted_w)
    target = total_weight / n_bins
    
    ranges = []
    start_idx = 0
    current_weight = 0.0
    
    for i in range(len(sorted_w)):
        current_weight += sorted_w[i]
        if current_weight >= target or i == len(sorted_w) - 1:
            e_min = energies[sorted_idx[start_idx]]
            e_max = energies[sorted_idx[i]]
            ranges.append((float(e_min), float(e_max)))
            start_idx = i + 1
            current_weight = 0.0
            if len(ranges) >= n_bins:
                break
    
    # 如果最后一组没加入
    if start_idx < len(sorted_w) and len(ranges) < n_bins:
        e_min = energies[sorted_idx[start_idx]]
        e_max = energies[sorted_idx[-1]]
        ranges.append((float(e_min), float(e_max)))
    
    return ranges
