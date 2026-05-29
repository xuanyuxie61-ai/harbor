"""
proxy_network_graph.py
古气候代理空间关联网络模块

将有向图邻接矩阵理论（融合种子项目 1204_test_digraph_adj）应用于代理数据的空间依赖结构，
构建代理关联网络，计算 PageRank 重要性，并进行谱聚类分组。
"""

import numpy as np


def build_proxy_network(n_proxies, proxy_locations, vertices, correlation_threshold=0.5):
    """
    构建代理数据空间关联网络。

    代理 i -> j 的边权重基于:
      (1) 高斯核空间距离衰减
      (2) 大气遥相关（teleconnection）距离尺度 ~3000 km（球面上约 0.47 弧度）
    """
    adjacency = np.zeros((n_proxies, n_proxies), dtype=np.float64)

    for i in range(n_proxies):
        for j in range(n_proxies):
            if i == j:
                continue
            loc_i = proxy_locations[i]
            loc_j = proxy_locations[j]
            dot = np.clip(np.dot(vertices[loc_i], vertices[loc_j]), -1.0, 1.0)
            angular_dist = np.arccos(dot)

            correlation = np.exp(-(angular_dist / 0.5)**2)
            teleconnection = np.exp(-((angular_dist - 0.47) / 0.3)**2)
            combined = max(correlation, teleconnection)

            if combined > correlation_threshold:
                adjacency[i, j] = combined

    # 行归一化 + 自环保证遍历性
    row_sums = np.sum(adjacency, axis=1)
    row_sums = np.where(row_sums < 1e-10, 1.0, row_sums)
    adjacency = adjacency / row_sums[:, np.newaxis]
    adjacency = 0.85 * adjacency + 0.15 * np.eye(n_proxies)

    pagerank = compute_pagerank(adjacency)
    return adjacency, pagerank


def compute_pagerank(adjacency, damping=0.85, tol=1e-10, max_iter=100):
    """
    PageRank 幂迭代:
        PR = (1-d)/N + d * A^T * PR
    """
    n = adjacency.shape[0]
    pr = np.ones(n) / n
    for _ in range(max_iter):
        pr_new = (1.0 - damping) / n + damping * adjacency.T @ pr
        if np.max(np.abs(pr_new - pr)) < tol:
            break
        pr = pr_new
    return pr


def detect_proxy_clusters(adjacency, n_clusters=3):
    """
    基于 Fiedler 向量的递归谱二分聚类。
    """
    W = 0.5 * (adjacency + adjacency.T)
    D = np.diag(np.sum(W, axis=1))
    L = D - W

    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        fiedler = eigvecs[:, 1]
        clusters = np.where(fiedler > 0, 0, 1)
        current = 2

        while current < n_clusters:
            sizes = [np.sum(clusters == k) for k in range(current)]
            largest = int(np.argmax(sizes))
            mask = clusters == largest
            sub_W = W[np.ix_(mask, mask)]
            sub_D = np.diag(np.sum(sub_W, axis=1))
            sub_L = sub_D - sub_W
            if sub_L.shape[0] > 2:
                sub_eigvals, sub_eigvecs = np.linalg.eigh(sub_L)
                sub_fiedler = sub_eigvecs[:, 1]
                sub_idx = np.where(mask)[0]
                for idx, val in zip(sub_idx, sub_fiedler):
                    if val > 0:
                        clusters[idx] = current
                    else:
                        clusters[idx] = largest
                current += 1
            else:
                break
        return clusters
    except np.linalg.LinAlgError:
        return np.zeros(len(adjacency), dtype=int)


def network_centrality_metrics(adjacency):
    """计算度中心性与特征向量中心性。"""
    n = adjacency.shape[0]
    degree = np.sum(adjacency, axis=1)
    try:
        eigvals, eigvecs = np.linalg.eig(adjacency)
        max_idx = int(np.argmax(np.real(eigvals)))
        eigenvector = np.abs(np.real(eigvecs[:, max_idx]))
        eigenvector = eigenvector / np.sum(eigenvector)
    except Exception:
        eigenvector = np.ones(n) / n
    return {'degree': degree, 'eigenvector': eigenvector}
