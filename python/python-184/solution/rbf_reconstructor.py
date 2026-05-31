"""
Radial Basis Function Reconstruction and Anomaly Scoring
=========================================================
源自种子项目 1015_rbf_interp_nd (RBF interpolation in N dimensions)。

给定散乱数据点 {x_i}_{i=1}^N ⊂ R^d 及函数值 {f_i}，RBF 插值形如：
    s(x) = sum_{j=1}^N w_j φ(||x - x_j||_2)

权向量 w 通过求解线性系统 A w = f 得到，其中：
    A_{ij} = φ(||x_i - x_j||_2)

常用核函数 φ(r)：
1. Multiquadric (MQ):        φ(r) = sqrt(r^2 + c^2)
2. Inverse MQ:               φ(r) = 1 / sqrt(r^2 + c^2)
3. Thin-plate spline (TPS):  φ(r) = r^2 log(r)    (d=2 时自然推广)
4. Gaussian:                 φ(r) = exp(-(ε r)^2)

在 time series 中的应用：
- 缺失值插补：将已知时间点作为中心，重建完整序列
- 异常检测：RBF 重构误差 ||f_i - s(x_i)|| 大的点为异常
- 非均匀采样信号的恢复

数学性质：
- MQ、IMQ、Gaussian 为正定径向函数，A 正定，解唯一
- TPS 为条件正定，需附加多项式项保证唯一性
"""

import numpy as np
from typing import Literal


class RBFReconstructor:
    """
    N 维 RBF 插值器，用于 time series 缺失值重建与异常检测。
    """

    def __init__(self, kernel: Literal["mq", "imq", "tps", "gaussian"] = "gaussian",
                 shape_param: float = 1.0, regularization: float = 1e-10):
        self.kernel = kernel
        self.c = shape_param
        self.reg = regularization
        self.centers: np.ndarray | None = None
        self.weights: np.ndarray | None = None

    def _phi(self, r: np.ndarray) -> np.ndarray:
        """径向基函数求值（向量化）。"""
        if self.kernel == "mq":
            return np.sqrt(r ** 2 + self.c ** 2)
        elif self.kernel == "imq":
            return 1.0 / np.sqrt(r ** 2 + self.c ** 2)
        elif self.kernel == "tps":
            # r^2 log(r)，处理 r=0
            out = np.zeros_like(r)
            mask = r > 1e-15
            out[mask] = r[mask] ** 2 * np.log(r[mask])
            return out
        elif self.kernel == "gaussian":
            return np.exp(-(self.c * r) ** 2)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def fit(self, centers: np.ndarray, values: np.ndarray) -> "RBFReconstructor":
        """
        求解 RBF 插值权值。

        Parameters
        ----------
        centers : np.ndarray, shape (N, d)
            中心点（已知数据点）。
        values : np.ndarray, shape (N,)
            函数值。
        """
        if centers.ndim == 1:
            centers = centers.reshape(-1, 1)
        n = centers.shape[0]
        if values.shape != (n,):
            raise ValueError("values length must match centers count.")

        # 构造插值矩阵
        A = np.zeros((n, n))
        for i in range(n):
            diff = centers - centers[i]
            r = np.linalg.norm(diff, axis=1)
            A[i, :] = self._phi(r)

        # Tikhonov 正则化保证数值稳定性
        A += self.reg * np.eye(n)

        # 求解
        self.weights = np.linalg.solve(A, values)
        self.centers = centers.copy()
        return self

    def predict(self, points: np.ndarray) -> np.ndarray:
        """
        在新的点集上评估 RBF 插值。
        """
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
        """
        对时间序列进行缺失值重建。
        已知点 -> fit -> 在所有时间点 predict。
        """
        centers = timestamps.reshape(-1, 1)
        query = all_timestamps.reshape(-1, 1)
        self.fit(centers, observed_values)
        return self.predict(query)

    def anomaly_score(self, timestamps: np.ndarray, values: np.ndarray) -> np.ndarray:
        """
        留一法 RBF 重构误差作为异常得分。
        对每个点 i，用其余 N-1 个点训练 RBF，然后计算在 x_i 处的重构误差。
        """
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
        # 归一化
        s_max = scores.max()
        if s_max > 1e-12:
            scores /= s_max
        return scores
