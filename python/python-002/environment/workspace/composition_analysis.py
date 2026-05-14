# -*- coding: utf-8 -*-
"""
composition_analysis.py
基于 602_jaccard_distance 合成
恒星化学丰度模式相似度分析与核合成产物分类。
"""

import numpy as np
from typing import Tuple, List


class CompositionAnalysis:
    """
    恒星化学组成分析工具。
    
    核心方法：
      1) Jaccard 指数/距离 — 比较两个恒星的核素集合重叠度
      2) 余弦相似度 — 比较丰度向量方向
      3) 欧氏距离 — 比较丰度向量幅度
      4) 金属丰度 [Fe/H] 计算
      5) CNO 循环产物比 C/N, O/N
    """

    @staticmethod
    def jaccard_index(set_a: np.ndarray, set_b: np.ndarray, threshold: float = 1e-6) -> float:
        """
        Jaccard 相似度指数：
          J(A,B) = |A ∩ B| / |A ∪ B|
        
        在丰度分析中，将核素按阈值二值化为"存在"集合：
          A = {i | X_i > threshold}
        """
        a = np.asarray(set_a, dtype=np.float64)
        b = np.asarray(set_b, dtype=np.float64)
        mask_a = a > threshold
        mask_b = b > threshold
        intersection = np.sum(mask_a & mask_b)
        union = np.sum(mask_a | mask_b)
        if union == 0:
            return 0.0
        return float(intersection / union)

    @staticmethod
    def jaccard_distance(set_a: np.ndarray, set_b: np.ndarray, threshold: float = 1e-6) -> float:
        """Jaccard 距离 = 1 - J(A,B)，满足度量空间性质。"""
        return 1.0 - CompositionAnalysis.jaccard_index(set_a, set_b, threshold)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        余弦相似度：
          cos(θ) = (a·b) / (||a|| ||b||)
        """
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        """欧氏距离。"""
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))

    @staticmethod
    def metallicity_feh(X_metal: float, X_metal_sun: float = 0.014) -> float:
        """
        金属丰度 [Fe/H]：
          [Fe/H] = log10(Z / Z_sun)
        简化处理：使用总金属质量分数替代 Fe。
        """
        if X_metal <= 0 or X_metal_sun <= 0:
            return -99.0
        return float(np.log10(X_metal / X_metal_sun))

    @staticmethod
    def cno_ratios(X: np.ndarray, species_names: List[str]) -> Tuple[float, float, float]:
        """
        计算 CNO 循环相关比值。
        输入 X 为质量分数向量。
        返回 (C/N, O/N, C/O)。
        """
        idx = {s: i for i, s in enumerate(species_names)}
        c12 = X[idx.get('C12', 3)]
        n14 = X[idx.get('N14', 4)]
        o16 = X[idx.get('O16', 5)]
        cn = c12 / n14 if n14 > 1e-15 else 1e10
        on = o16 / n14 if n14 > 1e-15 else 1e10
        co = c12 / o16 if o16 > 1e-15 else 1e10
        return float(cn), float(on), float(co)

    @staticmethod
    def nucleosynthetic_yield(X_final: np.ndarray, X_initial: np.ndarray,
                              dm: np.ndarray) -> np.ndarray:
        """
        计算核合成产额 [M_sun]：
          Yield_i = ∫ (X_i,final - X_i,initial) dm
        """
        Xf = np.asarray(X_final, dtype=np.float64)
        Xi = np.asarray(X_initial, dtype=np.float64)
        dm_arr = np.asarray(dm, dtype=np.float64)
        delta = Xf - Xi
        return np.sum(delta[:, np.newaxis] * dm_arr[np.newaxis, :], axis=1)

    @staticmethod
    def entropy_abundance(X: np.ndarray) -> float:
        """
        Shannon 熵（化学多样性度量）：
          S = -Σ_i X_i log(X_i)
        """
        X = np.asarray(X, dtype=np.float64)
        X = np.clip(X, 1e-30, 1.0)
        X = X / np.sum(X)
        return float(-np.sum(X * np.log(X)))

    @staticmethod
    def cluster_compositions(compositions: np.ndarray, n_clusters: int = 3) -> np.ndarray:
        """
        简单的 k-means 聚类，按丰度模式将恒星分组。
        返回聚类标签。
        """
        X = np.asarray(compositions, dtype=np.float64)
        n_samples, n_features = X.shape
        # 随机初始化中心
        rng = np.random.default_rng(42)
        centers = X[rng.choice(n_samples, n_clusters, replace=False)]
        labels = np.zeros(n_samples, dtype=int)
        for _ in range(100):
            # 分配
            for i in range(n_samples):
                dists = [np.linalg.norm(X[i] - c) for c in centers]
                labels[i] = int(np.argmin(dists))
            # 更新
            new_centers = np.zeros_like(centers)
            for k in range(n_clusters):
                mask = labels == k
                if np.any(mask):
                    new_centers[k] = np.mean(X[mask], axis=0)
                else:
                    new_centers[k] = centers[k]
            if np.allclose(centers, new_centers, atol=1e-8):
                break
            centers = new_centers
        return labels
