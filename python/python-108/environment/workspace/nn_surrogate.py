# -*- coding: utf-8 -*-
"""
nn_surrogate.py
神经网络代理模型模块

核心公式与物理背景
------------------
1. 前馈神经网络（Fully-connected Feedforward NN）
   对输入 x，第 l 层的激活：
       a^{[l]} = σ( W^{[l]} · a^{[l-1]} + b^{[l]} )
   其中 σ(z) = 1 / (1 + exp(-z)) 为 sigmoid 激活函数。

2. 代价函数（均方误差）
   对训练集 { (x_i, y_i) }_{i=1}^{N}：
       J = (1/2N) Σ_i || a^{[L]}(x_i) - y_i ||²

3. 反向传播（Backpropagation）
   输出层误差：
       δ^{[L]} = (a^{[L]} - y) ⊙ σ'(z^{[L]})
   隐层反向传播：
       δ^{[l]} = ( (W^{[l+1]})^T · δ^{[l+1]} ) ⊙ σ'(z^{[l]})
   其中 ⊙ 为 Hadamard 积，σ'(z) = σ(z)·(1-σ(z))。

   权重与偏置梯度：
       ∂J/∂W^{[l]} = δ^{[l]} · (a^{[l-1]})^T
       ∂J/∂b^{[l]} = δ^{[l]}

4. 随机梯度下降（SGD）
   每次迭代随机选取一个样本 (x_i, y_i)，更新：
       W ← W - η · ∂J_i/∂W
       b ← b - η · ∂J_i/∂b

融合来源
--------
- 799_neural_network : 前馈神经网络结构与反向传播训练
"""

import numpy as np
from typing import List, Tuple, Optional


class SigmoidActivation:
    """Sigmoid 激活函数及其导数。"""

    @staticmethod
    def f(z: np.ndarray) -> np.ndarray:
        # 截断避免溢出
        z = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def df_dz(a: np.ndarray) -> np.ndarray:
        """已知激活值 a = σ(z)，导数为 a·(1-a)"""
        return a * (1.0 - a)


class NeuralNetworkSurrogate:
    """
    全连接前馈神经网络代理模型。
    用于快速预测微腔传感器的谐振波长漂移。
    """

    def __init__(self, layer_sizes: List[int], seed: Optional[int] = None):
        """
        layer_sizes : List[int]
            各层神经元数量，例如 [2, 16, 8, 1] 表示：
            2 输入 → 16 隐层 → 8 隐层 → 1 输出
        """
        if len(layer_sizes) < 2:
            raise ValueError("至少需要输入层和输出层")
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        self.rng = np.random.default_rng(seed)
        self._init_weights()
        self.activation = SigmoidActivation()

    def _init_weights(self):
        """
        Xavier 初始化：
            W_{ij} ~ U[-√(6/(n_in+n_out)), √(6/(n_in+n_out))]
        """
        self.W = []
        self.b = []
        for i in range(1, self.n_layers):
            n_in = self.layer_sizes[i - 1]
            n_out = self.layer_sizes[i]
            limit = np.sqrt(6.0 / (n_in + n_out))
            W = self.rng.uniform(-limit, limit, size=(n_out, n_in))
            b = np.zeros((n_out, 1))
            self.W.append(W)
            self.b.append(b)

    def forward(self, x: np.ndarray) -> List[np.ndarray]:
        """
        前向传播，返回每层的激活值列表 a[0..L-1]。
        x 形状为 (n_features, n_samples)。
        """
        a = [x]
        for l in range(self.n_layers - 1):
            z = self.W[l] @ a[-1] + self.b[l]
            a_next = self.activation.f(z)
            a.append(a_next)
        return a

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        对输入 x (n_features, n_samples) 做预测。
        """
        a = self.forward(x)
        return a[-1]

    def cost(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        计算均方误差代价：
            J = (1/2N) Σ ||ŷ - y||²
        """
        y_pred = self.predict(X)
        m = Y.shape[1]
        return float(np.sum((y_pred - Y) ** 2) / (2.0 * m))

    def train_sgd(self, X: np.ndarray, Y: np.ndarray,
                  eta: float = 0.5,
                  n_iterations: int = 5000,
                  print_every: int = 1000) -> List[float]:
        """
        随机梯度下降训练。

        参数
        ----
        X : np.ndarray, shape (n_features, n_samples)
        Y : np.ndarray, shape (n_outputs, n_samples)
        eta : float
            学习率
        n_iterations : int
            迭代次数（每次随机选一个样本）
        """
        m = X.shape[1]
        costs = []
        for it in range(1, n_iterations + 1):
            # 随机选一个样本
            idx = self.rng.integers(0, m)
            x_i = X[:, idx:idx + 1]
            y_i = Y[:, idx:idx + 1]

            # 前向
            a = self.forward(x_i)

            # 反向传播
            deltas = [None] * (self.n_layers - 1)
            # 输出层
            deltas[-1] = (a[-1] - y_i) * self.activation.df_dz(a[-1])
            # 隐层反向
            for l in range(self.n_layers - 3, -1, -1):
                deltas[l] = (self.W[l + 1].T @ deltas[l + 1]) * self.activation.df_dz(a[l + 1])

            # 梯度更新
            for l in range(self.n_layers - 1):
                dW = deltas[l] @ a[l].T
                db = deltas[l]
                self.W[l] -= eta * dW
                self.b[l] -= eta * db

            if it % print_every == 0:
                c = self.cost(X, Y)
                costs.append(c)
        return costs

    def train_batch(self, X: np.ndarray, Y: np.ndarray,
                    eta: float = 0.1,
                    n_epochs: int = 2000,
                    batch_size: Optional[int] = None,
                    print_every: int = 500) -> List[float]:
        """
        Mini-batch 梯度下降训练。
        """
        m = X.shape[1]
        if batch_size is None or batch_size > m:
            batch_size = m
        costs = []
        for epoch in range(1, n_epochs + 1):
            # 随机打乱
            perm = self.rng.permutation(m)
            X_shuffled = X[:, perm]
            Y_shuffled = Y[:, perm]

            for start in range(0, m, batch_size):
                end = min(start + batch_size, m)
                X_batch = X_shuffled[:, start:end]
                Y_batch = Y_shuffled[:, start:end]

                a = self.forward(X_batch)
                deltas = [None] * (self.n_layers - 1)
                deltas[-1] = (a[-1] - Y_batch) * self.activation.df_dz(a[-1])
                for l in range(self.n_layers - 3, -1, -1):
                    deltas[l] = (self.W[l + 1].T @ deltas[l + 1]) * self.activation.df_dz(a[l + 1])

                batch_m = end - start
                for l in range(self.n_layers - 1):
                    dW = (deltas[l] @ a[l].T) / batch_m
                    db = np.mean(deltas[l], axis=1, keepdims=True)
                    self.W[l] -= eta * dW
                    self.b[l] -= eta * db

            if epoch % print_every == 0:
                c = self.cost(X, Y)
                costs.append(c)
        return costs

    def evaluate_metrics(self, X: np.ndarray, Y: np.ndarray) -> dict:
        """
        评估预测指标：MSE、RMSE、MAE、R²。
        """
        y_pred = self.predict(X)
        mse = float(np.mean((y_pred - Y) ** 2))
        rmse = np.sqrt(mse)
        mae = float(np.mean(np.abs(y_pred - Y)))
        ss_res = np.sum((Y - y_pred) ** 2)
        ss_tot = np.sum((Y - np.mean(Y)) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-30)
        return {"mse": mse, "rmse": float(rmse), "mae": mae, "r2": float(r2)}


class SensorResponseSurrogate:
    """
    封装 NN 代理模型，用于微腔传感器响应预测。
    输入：环境温度变化 ΔT_env、环境折射率 n_env
    输出：谐振波长漂移 Δλ [pm]
    """

    def __init__(self):
        self.nn = NeuralNetworkSurrogate(layer_sizes=[2, 12, 8, 1], seed=42)
        self._trained = False

    def generate_training_data(self, n_points: int = 200,
                                delta_T_range: Tuple[float, float] = (-10.0, 10.0),
                                n_env_range: Tuple[float, float] = (1.00, 1.05)) -> Tuple[np.ndarray, np.ndarray]:
        """
        基于物理模型生成训练数据。
        使用简化解析模型：
            Δλ/λ₀ = (1/n_g)·(dn/dT·ΔT + ∂n/∂n_env·Δn_env)
        """
        rng = np.random.default_rng(123)
        delta_T = rng.uniform(delta_T_range[0], delta_T_range[1], size=n_points)
        n_env = rng.uniform(n_env_range[0], n_env_range[1], size=n_points)

        # 物理参数
        lambda0 = 1550.0  # nm
        ng = 3.5
        dn_dT = 1.86e-4   # /K
        S = 100.0         # pm/RIU 灵敏度
        delta_n_env = n_env - 1.00

        delta_lambda = (lambda0 / ng) * dn_dT * delta_T + S * delta_n_env
        # 添加微小噪声
        delta_lambda += rng.normal(0.0, 0.5, size=n_points)

        X = np.vstack([delta_T, n_env])  # shape (2, n)
        Y = delta_lambda.reshape(1, -1)   # shape (1, n)
        return X, Y

    def train(self, n_points: int = 200):
        """生成数据并训练代理模型"""
        X, Y = self.generate_training_data(n_points)
        # 对输出做 min-max 归一化到 [0,1] 以适应 sigmoid
        self._y_min = float(np.min(Y))
        self._y_max = float(np.max(Y))
        self._y_range = self._y_max - self._y_min
        if self._y_range < 1e-12:
            self._y_range = 1.0
        Y_norm = (Y - self._y_min) / self._y_range
        self.nn.train_batch(X, Y_norm, eta=0.5, n_epochs=5000, batch_size=32)
        self._trained = True

    def predict(self, delta_T: float, n_env: float) -> float:
        """预测谐振波长漂移 [pm]"""
        if not self._trained:
            self.train()
        X = np.array([[delta_T], [n_env]])
        y_norm = float(self.nn.predict(X)[0, 0])
        # 反归一化
        y_raw = y_norm * self._y_range + self._y_min
        return y_raw
