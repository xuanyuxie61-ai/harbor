"""
state_clustering.py - 量子发射态聚类与优化模块

融合原项目 039_asa113（非层次聚类的 transfer/swap 优化算法）的核心思想，
用于将量子点在不同制备条件下的发射光谱/能态进行分类，
以识别最优发光态簇。

核心算法：
    - Transfer 操作：将对象从当前类转移到另一类，若准则函数降低则执行
    - Swap 操作：交换两个不同类中的对象，若准则函数降低则执行
    - 准则函数：类内方差和（等价于 k-means 的目标函数）

物理背景：
    量子点由于尺寸涨落，其发射波长呈现统计分布。
    通过聚类可将相似尺寸的量子点分组，评估各组的单色性指标。
"""

import numpy as np
from typing import Tuple, List, Dict
from utils import validate_array_1d, validate_array_2d


def criterion_variance(
    data: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
) -> float:
    """
    计算类内方差和（准则函数，越小越好）：
    
        J = sum_{k=1}^K sum_{i in C_k} ||x_i - mu_k||^2
    
    其中 mu_k 为第 k 类的均值向量。
    """
    data = validate_array_2d(data, "data")
    labels = validate_array_1d(labels, "labels")
    n_samples = data.shape[1]
    if labels.size != n_samples:
        raise ValueError("labels size must match number of samples")
    total = 0.0
    for k in range(n_clusters):
        mask = labels == k
        if not np.any(mask):
            continue
        cluster_data = data[:, mask]
        mu_k = np.mean(cluster_data, axis=1, keepdims=True)
        diff = cluster_data - mu_k
        total += float(np.sum(diff ** 2))
    return total


def _compute_cluster_means(data: np.ndarray, labels: np.ndarray, n_clusters: int):
    """辅助函数：计算各类均值。"""
    dim, n = data.shape
    means = np.zeros((dim, n_clusters), dtype=float)
    for k in range(n_clusters):
        mask = labels == k
        if np.any(mask):
            means[:, k] = np.mean(data[:, mask], axis=1)
    return means


def transfer_step(
    data: np.ndarray,
    labels: np.ndarray,
    cluster_sizes: np.ndarray,
    n_clusters: int,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    执行一次 Transfer 优化（源自 trnsfr 函数）：
    
    遍历每个对象 i，尝试将其从当前类 m 转移到其他各类 l，
    若类内方差降低则执行转移。
    准则变化通过重新计算精确方差得到（对小规模数据足够高效）。
    
    返回：
        (new_labels, new_cluster_sizes, new_criterion, n_transfers)
    """
    data = validate_array_2d(data, "data")
    labels = np.array(labels, dtype=int).copy()
    cluster_sizes = np.array(cluster_sizes, dtype=int).copy()
    n_samples = data.shape[1]
    if labels.size != n_samples or cluster_sizes.size != n_clusters:
        raise ValueError("Dimension mismatch")

    eps = 1e-12
    ntrans = 0
    current_crit = criterion_variance(data, labels, n_clusters)

    for i in range(n_samples):
        m = labels[i]
        if cluster_sizes[m] <= 1:
            continue
        best_l = m
        best_crit = current_crit
        for l in range(n_clusters):
            if l == m:
                continue
            # 模拟转移
            labels[i] = l
            new_crit = criterion_variance(data, labels, n_clusters)
            if new_crit < best_crit - eps:
                best_crit = new_crit
                best_l = l
            labels[i] = m  # 恢复
        if best_l != m:
            labels[i] = best_l
            cluster_sizes[m] -= 1
            cluster_sizes[best_l] += 1
            current_crit = best_crit
            ntrans += 1

    return labels, cluster_sizes, current_crit, ntrans


def swap_step(
    data: np.ndarray,
    labels: np.ndarray,
    cluster_sizes: np.ndarray,
    n_clusters: int,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    执行一次 Swap 优化（源自 swap 函数）：
    
    遍历每对对象 (i, j)，若交换它们所属的类能降低准则函数，则执行交换。
    """
    data = validate_array_2d(data, "data")
    labels = np.array(labels, dtype=int).copy()
    cluster_sizes = np.array(cluster_sizes, dtype=int).copy()
    n_samples = data.shape[1]
    eps = 1e-12
    ntrans = 0
    current_crit = criterion_variance(data, labels, n_clusters)

    for i in range(n_samples):
        l = labels[i]
        for j in range(i):
            m = labels[j]
            if l == m:
                continue
            if cluster_sizes[l] <= 1 or cluster_sizes[m] <= 1:
                continue
            # 模拟交换
            labels[i] = m
            labels[j] = l
            new_crit = criterion_variance(data, labels, n_clusters)
            if new_crit < current_crit - eps:
                current_crit = new_crit
                ntrans += 1
            else:
                # 撤销
                labels[i] = l
                labels[j] = m

    return labels, cluster_sizes, current_crit, ntrans


def optimize_clustering(
    data: np.ndarray,
    n_clusters: int,
    max_iter: int = 50,
) -> Dict[str, np.ndarray]:
    """
    交替执行 Transfer 与 Swap 优化，直到收敛。
    
    返回：
        labels: 最终聚类标签
        cluster_centers: 类中心
        criterion_history: 准则函数演化历史
    """
    data = validate_array_2d(data, "data")
    n_samples = data.shape[1]
    if n_clusters < 2 or n_clusters > n_samples:
        raise ValueError("Invalid number of clusters")

    # k-means++ 风格初始化
    labels = np.zeros(n_samples, dtype=int)
    centers_init = np.zeros((data.shape[0], n_clusters), dtype=float)
    # 第一个中心随机选
    first_idx = np.random.randint(0, n_samples)
    centers_init[:, 0] = data[:, first_idx]
    for k in range(1, n_clusters):
        dists = np.full(n_samples, float('inf'))
        for i in range(n_samples):
            d_min = float('inf')
            for j in range(k):
                d = np.sum((data[:, i] - centers_init[:, j]) ** 2)
                if d < d_min:
                    d_min = d
            dists[i] = d_min
        probs = dists / (np.sum(dists) + 1e-15)
        next_idx = np.random.choice(n_samples, p=probs)
        centers_init[:, k] = data[:, next_idx]
    # 根据最近中心分配标签
    for i in range(n_samples):
        best_k = 0
        best_d = float('inf')
        for k in range(n_clusters):
            d = np.sum((data[:, i] - centers_init[:, k]) ** 2)
            if d < best_d:
                best_d = d
                best_k = k
        labels[i] = best_k
    cluster_sizes = np.array([np.sum(labels == k) for k in range(n_clusters)], dtype=int)

    crit_history = []
    for it in range(max_iter):
        labels, cluster_sizes, crit, nt = transfer_step(data, labels, cluster_sizes, n_clusters)
        crit_history.append(crit)
        if nt == 0:
            labels, cluster_sizes, crit, ns = swap_step(data, labels, cluster_sizes, n_clusters)
            crit_history.append(crit)
            if ns == 0:
                break

    # 计算类中心
    centers = np.zeros((data.shape[0], n_clusters), dtype=float)
    for k in range(n_clusters):
        mask = labels == k
        if np.any(mask):
            centers[:, k] = np.mean(data[:, mask], axis=1)

    return {
        "labels": labels,
        "cluster_centers": centers,
        "cluster_sizes": cluster_sizes,
        "criterion_history": np.array(crit_history, dtype=float),
    }


def classify_quantum_dot_ensemble(
    energies_eV: np.ndarray,
    linewidths_meV: np.ndarray,
    n_clusters: int = 3,
) -> Dict[str, np.ndarray]:
    """
    对量子点系综按发射能量与线宽进行聚类分类。
    
    输入：
        energies_eV: 各量子点的发射能量 (eV)
        linewidths_meV: 各量子点的发射线宽 (meV)
    输出：
        聚类结果，包含优质单光子源候选组（线宽最窄的类）
    """
    energies_eV = validate_array_1d(energies_eV, "energies_eV")
    linewidths_meV = validate_array_1d(linewidths_meV, "linewidths_meV")
    if energies_eV.size != linewidths_meV.size:
        raise ValueError("energies and linewidths must have same size")
    n = energies_eV.size
    if n < n_clusters:
        raise ValueError("Not enough samples for clustering")
    # 构造二维特征：能量、线宽
    data = np.vstack([energies_eV, linewidths_meV])
    result = optimize_clustering(data, n_clusters)
    return result


def spectral_purity_index(
    cluster_center_energy: float,
    cluster_mean_linewidth: float,
    temperature_K: float = 4.0,
) -> float:
    """
    计算光谱纯度指标（Purity Index）：
    
        PI = (k_B T) / (hbar * Delta_omega) * exp(-Delta_E / k_B T)
    
    值越小表示纯度越高（线宽窄、能量分离大）。
    """
    kB = 1.380649e-23
    if cluster_mean_linewidth <= 0 or temperature_K <= 0:
        raise ValueError("Linewidth and temperature must be positive")
    # 线宽转换为角频率
    Delta_omega = cluster_mean_linewidth * 1e-3 * 1.602176634e-19 / 1.054571817e-34
    PI = (kB * temperature_K) / (1.054571817e-34 * Delta_omega)
    return float(PI)
