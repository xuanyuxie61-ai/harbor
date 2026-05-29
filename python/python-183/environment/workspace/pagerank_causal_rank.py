r"""
pagerank_causal_rank.py
================================================================================
基于 PageRank 随机游走的因果网络节点重要性排序与结构因果模型排序

原项目映射: 844_pagerank — Google 矩阵的幂迭代特征向量计算

科学背景
--------
在复杂因果网络中，并非所有变量具有同等的因果重要性。
PageRank 的思想可迁移为 **CausalRank**：将因果骨架视为有向图，
沿因果边进行随机游走，稳态分布 $\pi$ 越高的节点，其因果影响力越大。
该排名可用于：
- 识别关键混淆变量（Confounders）
- 优先选择干预靶点
- 评估因果发现算法的可靠性

核心公式
--------
1. 因果邻接矩阵 $A$：若 $i\to j$ 存在因果边，则 $A_{ji}=1$（注意列→行方向）。

2. 随机冲浪转移矩阵（处理悬挂节点/dangling nodes）：
   $$ S_{ij} = \frac{A_{ij}}{\sum_k A_{kj}} \quad \text{(列归一化)} $$
   若第 $j$ 列全零（无出边），则 $S_{ij}=1/n$。

3. Google 矩阵（带阻尼因子 $\alpha$ 的随机跳转）：
   $$ G = \alpha S + \frac{1-\alpha}{n}\mathbf{1}\mathbf{1}^T $$
   其中 $\alpha\in(0,1)$ 通常取 0.85。

4. 幂迭代求解主特征向量（CausalRank）：
   $$ \pi_{k+1} = G \pi_k, \qquad \pi_0 = \frac{1}{n}\mathbf{1} $$
   收敛判据：$\|\pi_{k+1}-\pi_k\|_1 < \varepsilon$。

5. 与结构因果模型 (SCM) 结合：
   若节点 $i$ 的 CausalRank 高，且其出边指向多个高排名节点，
   则 $i$ 为潜在的高阶混淆因子，需在因果效应估计中优先控制。
r"""

import numpy as np
from typing import List, Tuple, Optional


def adjacency_from_edges(edges: List[Tuple[int, int, float]], n: int,
                         use_weights: bool = True) -> np.ndarray:
    r"""
    从因果骨架边构造邻接矩阵（有向）。

    Parameters
    ----------
    edges : list of (i, j, w)
        有向边 $i \to j$（注意方向：i 导致 j）。
    n : int
        节点总数。
    use_weights : bool
        是否使用权重的绝对值作为邻接强度。

    Returns
    -------
    A : ndarray, shape (n, n)
        邻接矩阵，$A_{ji}$ 表示从 $i$ 到 $j$ 的边。
    r"""
    A = np.zeros((n, n), dtype=float)
    for i, j, w in edges:
        if 0 <= i < n and 0 <= j < n:
            A[j, i] = abs(w) if use_weights else 1.0
    return A


def build_google_matrix(A: np.ndarray, alpha: float = 0.85) -> np.ndarray:
    r"""
    由邻接矩阵构建 Google 矩阵 $G$。

    Parameters
    ----------
    A : ndarray, shape (n, n)
        邻接矩阵。
    alpha : float
        阻尼因子，必须在 $(0,1)$ 内。

    Returns
    -------
    G : ndarray, shape (n, n)
        Google 矩阵（列随机）。
    r"""
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha 必须在 (0,1) 区间内。")
    n = A.shape[0]
    # 列归一化得到 S
    col_sums = A.sum(axis=0)
    S = np.zeros_like(A)
    for j in range(n):
        if col_sums[j] > 0.0:
            S[:, j] = A[:, j] / col_sums[j]
        else:
            # 悬挂节点：均匀分布
            S[:, j] = 1.0 / n

    # Google 矩阵
    G = alpha * S + (1.0 - alpha) / n * np.ones((n, n))
    return G


def power_method_rank(G: np.ndarray,
                      max_iter: int = 200,
                      tol: float = 1e-10) -> np.ndarray:
    r"""
    幂迭代求解 Google 矩阵的主特征向量（CausalRank）。

    迭代格式：$\pi^{(k+1)} = G \pi^{(k)}$。
    由于 $G$ 为列随机矩阵，其最大特征值为 1，对应的左特征向量即为稳态分布。
    r"""
    n = G.shape[0]
    pi = np.ones(n) / n
    for it in range(max_iter):
        pi_new = G @ pi
        # 归一化（理论上已归一，数值上再确保）
        s = np.sum(pi_new)
        if s > 0.0:
            pi_new = pi_new / s
        diff = np.linalg.norm(pi_new - pi, 1)
        pi = pi_new
        if diff < tol:
            break
    return pi


def surf_rank(pi_history: np.ndarray) -> np.ndarray:
    r"""
    计算稳态分布的曲率/变化率（Surf Rank），用于检测收敛过程中的振荡节点。

    公式：
    $$ \text{surf}_i = \sum_{k} |\pi_i^{(k+1)} - \pi_i^{(k)}| $$
    r"""
    if pi_history.ndim != 2:
        raise ValueError("pi_history 必须是二维数组 (iterations, n_nodes)。")
    return np.sum(np.abs(np.diff(pi_history, axis=0)), axis=0)


def identify_confounders_by_rank(edges: List[Tuple[int, int, float]],
                                  n: int,
                                  top_k: int = 3) -> List[Tuple[int, float]]:
    r"""
    综合 CausalRank 与出度/入度比值识别高阶混淆变量。

    策略：高 CausalRank 且 $outdegree \gg indegree$ 的节点更可能是
    上游混淆因子（同时影响多个下游变量）。

    Returns
    -------
    confounders : list of (node_id, score)
        按 score 降序排列的潜在混淆变量列表。
    r"""
    A = adjacency_from_edges(edges, n, use_weights=False)
    G = build_google_matrix(A, alpha=0.85)
    pi = power_method_rank(G)

    indeg = A.sum(axis=0)  # 入度（列和）
    outdeg = A.sum(axis=1)  # 出度（行和）

    scores = []
    for i in range(n):
        ratio = (outdeg[i] + 1.0) / (indeg[i] + 1.0)
        score = pi[i] * np.log1p(ratio)
        scores.append((i, float(score)))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def demo():
    r"""模块自测试。"""
    np.random.seed(3)
    n = 10
    # 构造有向无环因果图（DAG）
    edges = [(0, 1, 0.5), (0, 2, 0.4), (1, 3, 0.3), (2, 3, 0.3),
             (3, 4, 0.6), (4, 5, 0.5), (0, 5, 0.2), (6, 7, 0.4),
             (7, 8, 0.5), (8, 9, 0.3), (6, 9, 0.2)]
    A = adjacency_from_edges(edges, n)
    G = build_google_matrix(A, alpha=0.85)
    pi = power_method_rank(G, max_iter=300)
    confounders = identify_confounders_by_rank(edges, n, top_k=3)
    print(f"[pagerank_causal_rank] CausalRank (前5): {pi[:5].round(4)}")
    print(f"[pagerank_causal_rank] 潜在混淆变量: {confounders}")
    return pi, confounders


if __name__ == "__main__":
    demo()
