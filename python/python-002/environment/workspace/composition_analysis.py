# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple, List


class CompositionAnalysis:

    @staticmethod
    def jaccard_index(set_a: np.ndarray, set_b: np.ndarray, threshold: float = 1e-6) -> float:
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
        return 1.0 - CompositionAnalysis.jaccard_index(set_a, set_b, threshold)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))

    @staticmethod
    def metallicity_feh(X_metal: float, X_metal_sun: float = 0.014) -> float:
        if X_metal <= 0 or X_metal_sun <= 0:
            return -99.0
        return float(np.log10(X_metal / X_metal_sun))

    @staticmethod
    def cno_ratios(X: np.ndarray, species_names: List[str]) -> Tuple[float, float, float]:
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
        Xf = np.asarray(X_final, dtype=np.float64)
        Xi = np.asarray(X_initial, dtype=np.float64)
        dm_arr = np.asarray(dm, dtype=np.float64)
        delta = Xf - Xi
        return np.sum(delta[:, np.newaxis] * dm_arr[np.newaxis, :], axis=1)

    @staticmethod
    def entropy_abundance(X: np.ndarray) -> float:
        X = np.asarray(X, dtype=np.float64)
        X = np.clip(X, 1e-30, 1.0)
        X = X / np.sum(X)
        return float(-np.sum(X * np.log(X)))

    @staticmethod
    def cluster_compositions(compositions: np.ndarray, n_clusters: int = 3) -> np.ndarray:
        X = np.asarray(compositions, dtype=np.float64)
        n_samples, n_features = X.shape

        rng = np.random.default_rng(42)
        centers = X[rng.choice(n_samples, n_clusters, replace=False)]
        labels = np.zeros(n_samples, dtype=int)
        for _ in range(100):

            for i in range(n_samples):
                dists = [np.linalg.norm(X[i] - c) for c in centers]
                labels[i] = int(np.argmin(dists))

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
