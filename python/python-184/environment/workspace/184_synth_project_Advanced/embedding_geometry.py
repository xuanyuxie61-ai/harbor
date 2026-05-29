"""
Time-Delay Embedding and Geometric Quality Assessment
======================================================
源自种子项目 1348_triangulation_quality (Triangulation quality metrics)。

Takens 嵌入定理：对于 d 维动力系统，观测序列 {x_t} 的 m 维延迟嵌入
    X_i = [x_i, x_{i+τ}, x_{i+2τ}, ..., x_{i+(m-1)τ}]^T
在 m >= 2d+1 时，拓扑等价于原系统的吸引子。

本模块实现：
1. 延迟嵌入构造
2. 嵌入空间中的 Delaunay 三角化质量评估
3. 几何异常检测：低质量三角形对应状态空间中的异常区域

三角化质量指标：
- alpha_measure: 最小内角 / 60° (等边三角形为 1)
- q_measure: 2r_in / r_out，内切圆与外接圆半径之比 (等边为 1)
- area_measure: 面积均匀性

数学公式：
给定三角形顶点 A, B, C，边长 a=|BC|, b=|CA|, c=|AB|：
    s = (a+b+c)/2                         (半周长)
    面积  Δ = sqrt(s(s-a)(s-b)(s-c))     (Heron)
    r_in = Δ / s                          (内切圆半径)
    r_out = abc / (4Δ)                    (外接圆半径)
    alpha = min(角A, 角B, 角C) / 60°
    q = 2 r_in / r_out
"""

import numpy as np
from typing import List, Tuple


class EmbeddingGeometry:
    """
    时间延迟嵌入与几何质量分析。
    """

    def __init__(self, embedding_dim: int = 3, delay: int = 1):
        self.m = embedding_dim
        self.tau = delay

    def delay_embed(self, series: np.ndarray) -> np.ndarray:
        """
        构造延迟嵌入矩阵 X ∈ R^{N x m}，其中 N = len(series) - (m-1)*tau。
        """
        n = len(series)
        N = n - (self.m - 1) * self.tau
        if N <= 0:
            raise ValueError("Series too short for given embedding parameters.")
        X = np.zeros((N, self.m))
        for i in range(self.m):
            X[:, i] = series[i * self.tau: i * self.tau + N]
        return X

    def _triangle_quality(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> dict:
        """
        计算单个三角形的几何质量指标。
        """
        a = np.linalg.norm(p2 - p3)
        b = np.linalg.norm(p3 - p1)
        c = np.linalg.norm(p1 - p2)

        # 退化检查
        min_side = min(a, b, c)
        max_side = max(a, b, c)
        if min_side < 1e-12 or max_side > 1e6:
            return {"alpha": 0.0, "q": 0.0, "area": 0.0, "degenerate": True}

        s = 0.5 * (a + b + c)
        area_sq = s * (s - a) * (s - b) * (s - c)
        if area_sq <= 1e-24:
            return {"alpha": 0.0, "q": 0.0, "area": 0.0, "degenerate": True}
        area = np.sqrt(area_sq)

        # 内切圆与外接圆半径
        r_in = area / s
        r_out = a * b * c / (4.0 * area)

        # 最小内角（用余弦定理）
        cos_A = max(-1.0, min(1.0, (b ** 2 + c ** 2 - a ** 2) / (2.0 * b * c)))
        cos_B = max(-1.0, min(1.0, (c ** 2 + a ** 2 - b ** 2) / (2.0 * c * a)))
        cos_C = max(-1.0, min(1.0, (a ** 2 + b ** 2 - c ** 2) / (2.0 * a * b)))
        angle_A = np.arccos(cos_A)
        angle_B = np.arccos(cos_B)
        angle_C = np.arccos(cos_C)
        min_angle = min(angle_A, angle_B, angle_C)

        alpha = min_angle / (np.pi / 3.0)  # 归一化到 60°
        q = 2.0 * r_in / r_out if r_out > 1e-12 else 0.0

        return {
            "alpha": alpha,
            "q": q,
            "area": area,
            "degenerate": False,
            "r_in": r_in,
            "r_out": r_out
        }

    def local_triangulation_quality(self, X: np.ndarray, k_neighbors: int = 6) -> np.ndarray:
        """
        对每个嵌入点，取其 k 近邻构造局部三角化，计算平均质量得分。
        得分低的点对应吸引子中的异常区域（折叠、稀疏或噪声）。
        """
        n = X.shape[0]
        scores = np.zeros(n)

        for i in range(n):
            # k 近邻
            dists = np.linalg.norm(X - X[i], axis=1)
            dists[i] = np.inf
            if k_neighbors + 1 >= n:
                neighbors = np.argsort(dists)[:min(k_neighbors, n - 1)]
            else:
                neighbors = np.argpartition(dists, k_neighbors)[:k_neighbors]

            if len(neighbors) < 2:
                scores[i] = 0.0
                continue

            # 构造所有可能的三角形 (i, j, k)
            qual_sum = 0.0
            count = 0
            for idx_j in range(len(neighbors)):
                for idx_k in range(idx_j + 1, len(neighbors)):
                    j = neighbors[idx_j]
                    k = neighbors[idx_k]
                    q = self._triangle_quality(X[i], X[j], X[k])
                    if not q["degenerate"]:
                        # 综合质量：alpha 和 q 的调和平均
                        qual = 2.0 * q["alpha"] * q["q"] / (q["alpha"] + q["q"] + 1e-12)
                        qual_sum += qual
                        count += 1
            scores[i] = qual_sum / (count + 1e-12)

        # 反转：低质量 = 高异常
        scores = 1.0 - scores
        return scores

    def embedding_dimension_estimate(self, series: np.ndarray, max_dim: int = 10,
                                     threshold: float = 0.05) -> int:
        """
        使用假近邻法 (False Nearest Neighbors, FNN) 估计最小嵌入维度。
        原理：在 d 维嵌入中，若两个点在 d+1 维中距离急剧增大，则为假近邻。
        """
        N = len(series)
        fnn_ratio = []
        for d in range(1, max_dim + 1):
            X_d = np.zeros((N - (d - 1) * self.tau, d))
            for i in range(d):
                X_d[:, i] = series[i * self.tau: i * self.tau + X_d.shape[0]]

            if d == 1:
                fnn_ratio.append(1.0)
                continue

            X_prev = np.zeros((N - (d - 2) * self.tau, d - 1))
            for i in range(d - 1):
                X_prev[:, i] = series[i * self.tau: i * self.tau + X_prev.shape[0]]

            n_points = min(X_d.shape[0], X_prev.shape[0])
            fnn_count = 0
            valid = 0
            for i in range(n_points):
                dists = np.linalg.norm(X_prev[:n_points] - X_prev[i], axis=1)
                dists[i] = np.inf
                j = np.argmin(dists)
                if dists[j] < 1e-12:
                    continue
                # 检查 d 维和 d-1 维距离比
                dist_d = np.linalg.norm(X_d[i] - X_d[j])
                ratio = np.sqrt(abs(dist_d ** 2 - dists[j] ** 2)) / dists[j]
                if ratio > threshold:
                    fnn_count += 1
                valid += 1

            fnn_ratio.append(fnn_count / (valid + 1e-12))

        # 选择 FNN 首次低于阈值的维度
        for d, ratio in enumerate(fnn_ratio, start=1):
            if ratio < threshold:
                return d
        return max_dim
