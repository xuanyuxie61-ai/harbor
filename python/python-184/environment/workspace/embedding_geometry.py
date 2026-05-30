
import numpy as np
from typing import List, Tuple


class EmbeddingGeometry:

    def __init__(self, embedding_dim: int = 3, delay: int = 1):
        self.m = embedding_dim
        self.tau = delay

    def delay_embed(self, series: np.ndarray) -> np.ndarray:
        n = len(series)
        N = n - (self.m - 1) * self.tau
        if N <= 0:
            raise ValueError("Series too short for given embedding parameters.")
        X = np.zeros((N, self.m))
        for i in range(self.m):
            X[:, i] = series[i * self.tau: i * self.tau + N]
        return X

    def _triangle_quality(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> dict:
        a = np.linalg.norm(p2 - p3)
        b = np.linalg.norm(p3 - p1)
        c = np.linalg.norm(p1 - p2)


        min_side = min(a, b, c)
        max_side = max(a, b, c)
        if min_side < 1e-12 or max_side > 1e6:
            return {"alpha": 0.0, "q": 0.0, "area": 0.0, "degenerate": True}

        s = 0.5 * (a + b + c)
        area_sq = s * (s - a) * (s - b) * (s - c)
        if area_sq <= 1e-24:
            return {"alpha": 0.0, "q": 0.0, "area": 0.0, "degenerate": True}
        area = np.sqrt(area_sq)


        r_in = area / s
        r_out = a * b * c / (4.0 * area)


        cos_A = max(-1.0, min(1.0, (b ** 2 + c ** 2 - a ** 2) / (2.0 * b * c)))
        cos_B = max(-1.0, min(1.0, (c ** 2 + a ** 2 - b ** 2) / (2.0 * c * a)))
        cos_C = max(-1.0, min(1.0, (a ** 2 + b ** 2 - c ** 2) / (2.0 * a * b)))
        angle_A = np.arccos(cos_A)
        angle_B = np.arccos(cos_B)
        angle_C = np.arccos(cos_C)
        min_angle = min(angle_A, angle_B, angle_C)

        alpha = min_angle / (np.pi / 3.0)
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
        n = X.shape[0]
        scores = np.zeros(n)

        for i in range(n):

            dists = np.linalg.norm(X - X[i], axis=1)
            dists[i] = np.inf
            if k_neighbors + 1 >= n:
                neighbors = np.argsort(dists)[:min(k_neighbors, n - 1)]
            else:
                neighbors = np.argpartition(dists, k_neighbors)[:k_neighbors]

            if len(neighbors) < 2:
                scores[i] = 0.0
                continue


            qual_sum = 0.0
            count = 0
            for idx_j in range(len(neighbors)):
                for idx_k in range(idx_j + 1, len(neighbors)):
                    j = neighbors[idx_j]
                    k = neighbors[idx_k]
                    q = self._triangle_quality(X[i], X[j], X[k])
                    if not q["degenerate"]:

                        qual = 2.0 * q["alpha"] * q["q"] / (q["alpha"] + q["q"] + 1e-12)
                        qual_sum += qual
                        count += 1
            scores[i] = qual_sum / (count + 1e-12)


        scores = 1.0 - scores
        return scores

    def embedding_dimension_estimate(self, series: np.ndarray, max_dim: int = 10,
                                     threshold: float = 0.05) -> int:
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

                dist_d = np.linalg.norm(X_d[i] - X_d[j])
                ratio = np.sqrt(abs(dist_d ** 2 - dists[j] ** 2)) / dists[j]
                if ratio > threshold:
                    fnn_count += 1
                valid += 1

            fnn_ratio.append(fnn_count / (valid + 1e-12))


        for d, ratio in enumerate(fnn_ratio, start=1):
            if ratio < threshold:
                return d
        return max_dim
