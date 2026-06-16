"""
分类代理模型模块 (Classification Surrogate for Solution Regimes)
=================================================================
基于分类方法识别随机参数空间中的解行为区域 (regime)。

核心思想:
  对于参数化 PDE, 不同参数区域可能导致不同的解行为:
    - 光滑解 vs 激波形成
    - 稳定 vs 不稳定
    - 唯一解 vs 多解 (分岔)

  使用分类方法预先标记参数空间区域,
  然后在每个区域内使用不同的数值方法或配置策略。

分类器:
  使用多项式分类边界:
    g(ξ) = Σ_{|α|≤p} c_α Ψ_α(ξ) = 0

  样本 ξ 被分为:
    class 1 if g(ξ) > 0
    class 0 if g(ξ) ≤ 0

训练:
  给定标记样本 {(ξ_k, y_k)}, y_k ∈ {0, 1}:
    min Σ_k L(y_k, sign(g(ξ_k))) + λ ||c||²

  其中 L 是铰链损失或交叉熵损失。

弱监督:
  当精确标记成本高时, 使用弱标签:
    - 仅知部分样本的确切类别
    - 利用未标记样本的空间结构

应用:
  - 自适应配置: 在分岔边界附近加密网格
  - 多模型策略: 不同区域使用不同数值格式
  - 奇异性检测: 识别解可能失去正则性的区域
"""

import numpy as np
from typing import Tuple, List, Optional, Dict


class SolutionRegimeClassifier:
    """
    解行为区域分类器。

    将随机参数空间划分为不同解行为区域,
    用于指导自适应配置策略。
    """

    def __init__(
        self,
        dimension: int,
        max_order: int = 3,
        regularization: float = 1e-3
    ):
        """
        参数:
            dimension: 参数空间维度
            max_order: 分类边界多项式阶数
            regularization: 正则化参数 λ
        """
        self.D = dimension
        self.p = max_order
        self.lam = regularization

        # 生成多指标集
        self.multi_indices = []
        self._enumerate(dimension, max_order, [], self.multi_indices)
        self.n_features = len(self.multi_indices)

        # 模型参数
        self.weights = np.zeros(self.n_features)
        self.bias = 0.0
        self.is_fitted = False

    def _enumerate(
        self, remaining: int, remaining_sum: int,
        current: List[int], result: List[Tuple[int, ...]]
    ):
        if remaining == 0:
            result.append(tuple(current))
            return
        for v in range(remaining_sum + 1):
            current.append(v)
            self._enumerate(remaining - 1, remaining_sum - v, current, result)
            current.pop()

    def _feature_map(self, xi: np.ndarray) -> np.ndarray:
        """
        多项式特征映射:
          Φ(ξ) = [Ψ_α(ξ)]_{|α|≤p}

        使用 Legendre 多项式基。
        """
        from polynomial_utils import legendre_evaluate

        xi = np.atleast_2d(xi)
        N = xi.shape[0]

        # 各维度的 Legendre 多项式
        poly_vals = {}
        for d in range(self.D):
            vals = legendre_evaluate(xi[:, d], self.p)
            for n in range(self.p + 1):
                poly_vals[(d, n)] = vals[n, :]

        # 多元特征
        Phi = np.ones((N, self.n_features))
        for j, alpha in enumerate(self.multi_indices):
            for d in range(self.D):
                Phi[:, j] *= poly_vals[(d, alpha[d])]

        return Phi

    def fit(
        self,
        xi_train: np.ndarray,
        y_train: np.ndarray,
        n_epochs: int = 100,
        learning_rate: float = 0.01
    ) -> Dict[str, float]:
        """
        训练分类器 (梯度下降 + 铰链损失):

          min_{w,b} (1/N) Σ_k max(0, 1 - y_k(w^T Φ(ξ_k) + b)) + λ ||w||²

          梯度:
            ∂L/∂w = (1/N) Σ_k [-y_k Φ(ξ_k) 1_{margin<0}] + 2λ w
            ∂L/∂b = (1/N) Σ_k [-y_k 1_{margin<0}]

        参数:
            xi_train: 训练样本, shape (N_train, D)
            y_train: 标签 ∈ {-1, +1}, shape (N_train,)
            n_epochs: 训练轮数
            learning_rate: 学习率

        返回:
            训练历史
        """
        Phi = self._feature_map(xi_train)  # (N_train, n_features)
        N = Phi.shape[0]
        y = np.sign(y_train).astype(np.float64)
        y[y == 0] = 1.0  # 确保 ±1

        history = {'loss': [], 'accuracy': []}

        for epoch in range(n_epochs):
            # 前向
            scores = Phi @ self.weights + self.bias
            margins = y * scores

            # 铰链损失
            hinge = np.maximum(0.0, 1.0 - margins)
            loss = np.mean(hinge) + self.lam * np.sum(self.weights ** 2)
            history['loss'].append(float(loss))

            # 梯度
            mask = (margins < 1.0).astype(float)
            grad_w = -Phi.T @ (y * mask) / N + 2.0 * self.lam * self.weights
            grad_b = -np.sum(y * mask) / N

            # 更新
            self.weights -= learning_rate * grad_w
            self.bias -= learning_rate * grad_b

            # 准确率
            preds = np.sign(scores)
            acc = np.mean(preds == y)
            history['accuracy'].append(float(acc))

        self.is_fitted = True
        return history

    def predict(self, xi: np.ndarray) -> np.ndarray:
        """
        预测类别标签。

        参数:
            xi: shape (N, D)

        返回:
            labels: shape (N,), ∈ {-1, +1}
        """
        Phi = self._feature_map(xi)
        scores = Phi @ self.weights + self.bias
        return np.sign(scores)

    def decision_function(self, xi: np.ndarray) -> np.ndarray:
        """
        计算决策函数值 (到边界的有符号距离)。

        g(ξ) > 0: 正类区域
        g(ξ) < 0: 负类区域
        |g(ξ)| 小: 靠近边界 (需要加密配置)
        """
        Phi = self._feature_map(xi)
        return Phi @ self.weights + self.bias

    def find_boundary_points(
        self,
        candidate_points: np.ndarray,
        n_select: int = 10
    ) -> np.ndarray:
        """
        找到最靠近分类边界的候选点。

        这些点位于解行为变化的过渡区域,
        是自适应细化最有价值的区域。

        参数:
            candidate_points: shape (N_cand, D)
            n_select: 选择的点数

        返回:
            boundary_points: shape (n_select, D)
        """
        scores = self.decision_function(candidate_points)
        # 按 |score| 升序 (最靠近边界)
        idx = np.argsort(np.abs(scores))[:n_select]
        return candidate_points[idx]

    def classify_solution_behavior(
        self,
        xi: np.ndarray,
        criterion: str = 'solution_norm',
        threshold: float = 1.0
    ) -> np.ndarray:
        """
        自动标记解行为:

        根据给定的准则函数计算每个参数点处的解特征,
        然后根据阈值确定类别。

        参数:
            xi: 参数点, shape (N, D)
            criterion: 分类准则名称
            threshold: 分类阈值

        返回:
            labels: shape (N,)
        """
        # 默认: 基于参数范数分类
        norms = np.linalg.norm(xi, axis=1)
        labels = np.where(norms > threshold, 1.0, -1.0)
        return labels


class WeakSupervisionLabeler:
    """
    弱监督标记器: 利用部分标记数据传播标签。

    思想 (源自半监督学习):
      1. 有少量精确标记样本 {(ξ_k, y_k)} (高成本)
      2. 有大量未标记样本 {ξ_j}
      3. 利用参数空间的连续性, 从未标记样本推断标签

    方法: 标签传播
      - 构建参数空间的 k-NN 图
      - 迭代传播: y_j = Σ_k W_{jk} y_k / Σ_k W_{jk}
      - 其中 W_{jk} = exp(-||ξ_j - ξ_k||² / (2σ²))
    """

    def __init__(self, sigma: float = 0.5, n_neighbors: int = 5):
        self.sigma = sigma
        self.n_neighbors = n_neighbors

    def propagate_labels(
        self,
        xi_all: np.ndarray,
        labeled_idx: np.ndarray,
        labeled_values: np.ndarray,
        n_iterations: int = 20
    ) -> np.ndarray:
        """
        标签传播算法。

        参数:
            xi_all: 所有参数点, shape (N, D)
            labeled_idx: 已标记点的索引
            labeled_values: 已标记值, shape (n_labeled,)
            n_iterations: 传播迭代次数

        返回:
            propagated_labels: shape (N,)
        """
        N = xi_all.shape[0]
        labels = np.zeros(N)
        labels[labeled_idx] = labeled_values

        # 构建权重矩阵 (稀疏)
        from scipy.spatial.distance import cdist
        dists = cdist(xi_all, xi_all)
        W = np.exp(-dists ** 2 / (2.0 * self.sigma ** 2))
        np.fill_diagonal(W, 0.0)

        # 归一化
        row_sums = np.sum(W, axis=1)
        row_sums = np.maximum(row_sums, 1e-30)
        T = W / row_sums[:, np.newaxis]

        # 迭代传播
        for _ in range(n_iterations):
            labels_new = T @ labels
            # 固定已标记点
            labels_new[labeled_idx] = labeled_values
            labels = labels_new

        return labels
