
import numpy as np
from typing import Literal


class RBFReconstructor:

    def __init__(self, kernel: Literal["mq", "imq", "tps", "gaussian"] = "gaussian",
                 shape_param: float = 1.0, regularization: float = 1e-10):
        self.kernel = kernel
        self.c = shape_param
        self.reg = regularization
        self.centers: np.ndarray | None = None
        self.weights: np.ndarray | None = None

    def _phi(self, r: np.ndarray) -> np.ndarray:
        if self.kernel == "mq":
            return np.sqrt(r ** 2 + self.c ** 2)
        elif self.kernel == "imq":
            return 1.0 / np.sqrt(r ** 2 + self.c ** 2)
        elif self.kernel == "tps":

            out = np.zeros_like(r)
            mask = r > 1e-15
            out[mask] = r[mask] ** 2 * np.log(r[mask])
            return out
        elif self.kernel == "gaussian":
            return np.exp(-(self.c * r) ** 2)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def fit(self, centers: np.ndarray, values: np.ndarray) -> "RBFReconstructor":
        if centers.ndim == 1:
            centers = centers.reshape(-1, 1)
        n = centers.shape[0]
        if values.shape != (n,):
            raise ValueError("values length must match centers count.")


        A = np.zeros((n, n))
        for i in range(n):
            diff = centers - centers[i]
            r = np.linalg.norm(diff, axis=1)
            A[i, :] = self._phi(r)


        A += self.reg * np.eye(n)


        self.weights = np.linalg.solve(A, values)
        self.centers = centers.copy()
        return self

    def predict(self, points: np.ndarray) -> np.ndarray:
        if self.weights is None or self.centers is None:
            raise RuntimeError("Model not fitted yet.")
        if points.ndim == 1:
            points = points.reshape(-1, 1)
        m = points.shape[0]
        n = self.centers.shape[0]
        result = np.zeros(m)
        for j in range(n):
            r = np.linalg.norm(points - self.centers[j], axis=1)
            result += self.weights[j] * self._phi(r)
        return result

    def reconstruct_series(self, timestamps: np.ndarray, observed_values: np.ndarray,
                           all_timestamps: np.ndarray) -> np.ndarray:
        centers = timestamps.reshape(-1, 1)
        query = all_timestamps.reshape(-1, 1)
        self.fit(centers, observed_values)
        return self.predict(query)

    def anomaly_score(self, timestamps: np.ndarray, values: np.ndarray) -> np.ndarray:
        n = len(timestamps)
        scores = np.zeros(n)
        centers = timestamps.reshape(-1, 1)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            temp = RBFReconstructor(kernel=self.kernel, shape_param=self.c, regularization=self.reg)
            temp.fit(centers[mask], values[mask])
            pred_i = temp.predict(centers[i:i + 1])[0]
            scores[i] = abs(values[i] - pred_i)

        s_max = scores.max()
        if s_max > 1e-12:
            scores /= s_max
        return scores
