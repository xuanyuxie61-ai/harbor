"""
neighbor_graph.py
基于Gray码优化的近邻图构建与拓扑保持
融合原项目: 485_gray_code_display, 668_levenshtein_distance

在高维流形学习中，近邻图的构建是核心步骤。
本项目利用Gray码的Hamming距离性质优化邻域搜索，
并引入Levenshtein编辑距离处理混合类型数据。

数学模型:
给定数据集 X = {x_1, ..., x_N} ⊂ R^D，构建加权图 G = (V, E, W)。
边权重由度量 d(x_i, x_j) 定义。

Gray码性质: 相邻整数的Gray码Hamming距离恒为1，
这一性质可用于构建保序的量化索引，加速近似最近邻搜索。

Levenshtein距离: 对于序列型特征，定义编辑距离
    d_L(s, t) = min_{编辑序列} (插入数 + 删除数 + 替换数)
"""

import numpy as np
from typing import List, Tuple, Dict


def binary_to_gray(n: int, m: int) -> np.ndarray:
    """
    将整数 n 转换为 m 位Gray码
    Gray(n) = n XOR (n >> 1)
    """
    gray = n ^ (n >> 1)
    bits = np.zeros(m, dtype=int)
    for i in range(m):
        bits[i] = (gray >> i) & 1
    return bits


def gray_to_binary(g: np.ndarray) -> int:
    """Gray码转二进制整数"""
    n = 0
    mask = 0
    for i in range(len(g) - 1, -1, -1):
        mask ^= g[i]
        n = (n << 1) | mask
    return n


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Hamming距离: 两个二进制向量不同位的数量"""
    return int(np.sum(np.abs(a - b)))


def quantize_to_gray_code(x: np.ndarray, bounds: np.ndarray, m_bits: int = 8) -> np.ndarray:
    """
    将实数向量量化为Gray码索引
    x: (D,) 实数向量
    bounds: (D, 2) 每维的 [min, max]
    返回: (D,) Gray码整数索引
    """
    D = len(x)
    indices = np.zeros(D, dtype=int)
    for d in range(D):
        xmin, xmax = bounds[d]
        if xmax <= xmin:
            indices[d] = 0
            continue
        # 归一化到 [0, 2^m - 1]
        norm = (x[d] - xmin) / (xmax - xmin)
        norm = np.clip(norm, 0.0, 1.0)
        idx = int(norm * ((1 << m_bits) - 1))
        indices[d] = idx
    return indices


def gray_code_neighborhood_search(data: np.ndarray, query: np.ndarray,
                                   bounds: np.ndarray, m_bits: int = 8,
                                   max_hamming: int = 2) -> np.ndarray:
    """
    基于Gray码的近似最近邻搜索
    原理: 在Gray码空间中，Hamming距离小的点对应原始空间中相近的点
    """
    N, D = data.shape
    query_gray = quantize_to_gray_code(query, bounds, m_bits)
    candidates = []
    for i in range(N):
        pt_gray = quantize_to_gray_code(data[i], bounds, m_bits)
        hd = hamming_distance(query_gray, pt_gray)
        if hd <= max_hamming:
            candidates.append(i)
    if len(candidates) == 0:
        # 回退到暴力搜索
        dists = np.linalg.norm(data - query, axis=1)
        candidates = [int(np.argmin(dists))]
    return np.array(candidates, dtype=int)


def levenshtein_distance(s: List, t: List) -> int:
    """
    Levenshtein编辑距离
    s, t: 序列 (列表或字符串)
    动态规划: d[i,j] = min(d[i-1,j]+1, d[i,j-1]+1, d[i-1,j-1]+cost)
    """
    m, n = len(s), len(t)
    d = np.zeros((m + 1, n + 1), dtype=int)
    for i in range(m + 1):
        d[i, 0] = i
    for j in range(n + 1):
        d[0, j] = j
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1,
                          min(d[i, j - 1] + 1,
                              d[i - 1, j - 1] + cost))
    return int(d[m, n])


def mixed_distance(x: np.ndarray, y: np.ndarray,
                   categorical_dims: List[int] = None,
                   sequence_dims: List[int] = None) -> float:
    """
    混合数据类型距离
    数值维度: 欧氏距离
    类别维度: Hamming距离 (0/1)
    序列维度: Levenshtein距离 (归一化)
    """
    D = len(x)
    if categorical_dims is None:
        categorical_dims = []
    if sequence_dims is None:
        sequence_dims = []
    numerical_dims = [d for d in range(D)
                      if d not in categorical_dims and d not in sequence_dims]
    dist = 0.0
    # 数值维度欧氏距离
    if len(numerical_dims) > 0:
        diff = x[numerical_dims] - y[numerical_dims]
        dist += np.sum(diff ** 2)
    # 类别维度Hamming距离
    for d in categorical_dims:
        if x[d] != y[d]:
            dist += 1.0
    # 序列维度归一化Levenshtein距离
    for d in sequence_dims:
        s = list(str(x[d]))
        t = list(str(y[d]))
        max_len = max(len(s), len(t))
        if max_len > 0:
            dist += levenshtein_distance(s, t) / max_len
    return np.sqrt(dist)


def build_knn_graph(data: np.ndarray, k: int = 10,
                    method: str = "exact") -> Tuple[np.ndarray, np.ndarray]:
    """
    构建k近邻图
    返回: (edges, weights)
        edges: (M, 2) 边列表
        weights: (M,) 边权重 (高斯核)
    """
    N = len(data)
    edges = []
    weights = []
    # 计算中位数距离用于带宽选择
    all_dists = []
    for i in range(min(N, 100)):
        dists = np.linalg.norm(data - data[i], axis=1)
        all_dists.extend(dists[dists > 0])
    median_dist = np.median(all_dists) if len(all_dists) > 0 else 1.0
    bandwidth = median_dist
    for i in range(N):
        dists = np.linalg.norm(data - data[i], axis=1)
        idx = np.argsort(dists)[1:k + 1]  # 排除自身
        for j in idx:
            w = np.exp(-dists[j] ** 2 / (2.0 * bandwidth ** 2))
            edges.append([i, j])
            weights.append(w)
    return np.array(edges, dtype=int), np.array(weights, dtype=np.float64)


def graph_laplacian(edges: np.ndarray, weights: np.ndarray,
                    n_vertices: int, normalize: bool = True) -> np.ndarray:
    """
    计算图Laplacian矩阵
    L = D - W (非归一化) 或 L = I - D^{-1/2} W D^{-1/2} (归一化)
    """
    W = np.zeros((n_vertices, n_vertices), dtype=np.float64)
    for (i, j), w in zip(edges, weights):
        W[i, j] = w
        W[j, i] = w
    D = np.diag(np.sum(W, axis=1))
    if normalize:
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-15))
        L = np.eye(n_vertices) - D_inv_sqrt @ W @ D_inv_sqrt
    else:
        L = D - W
    return L
