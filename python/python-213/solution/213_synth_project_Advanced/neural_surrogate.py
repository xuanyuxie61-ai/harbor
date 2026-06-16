"""
neural_surrogate.py - 神经网络代理模型模块
==========================================

融合种子项目:
  - 1171_jaggbow_magnet: PyTorch Lightning 神经网络模型

核心数学: 神经网络作为优化子问题的代理模型

在内点法中, 每次 Newton 步需要:
  1. 计算 PDE 残差 (需要求解 PDE)
  2. 计算 Jacobian (需要 PDE 导数)

这对大规模问题计算代价极高.
神经网络代理模型可以加速:
  1. 学习 PDE 的解映射: q -> u(q)
  2. 预测最优控制: 直接输出 q*
  3. 估计 Hessian: 学习 Newton 步的方向

代理模型架构 (简化版, 融合 1171 的模型工厂模式):

  class SurrogateNet:
    input: q in R^n_control
    hidden: 3 层全连接 + ReLU 激活
    output: u_pred in R^n_spatial

  训练目标:
    min_theta sum_i ||u_pred(q_i; theta) - u_true(q_i)||^2

  其中 u_true(q_i) 是通过有限元求解器计算的.

  训练后, 代理模型提供:
    - 快速预测 (O(1) vs O(n^3) 的 PDE 求解)
    - 可用于 warm-start 内点法
    - 可提供初始 Hessian 近似

  数学保证:
    Universal Approximation Theorem:
      对任意连续函数 f 和 epsilon > 0,
      存在单隐层网络 g 使得 ||f - g||_inf < epsilon.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict


class SurrogateNet:
    """
    简化版神经网络代理 (纯 NumPy, 无 PyTorch 依赖).

    全连接前馈网络:
      y = W_3 * ReLU(W_2 * ReLU(W_1 * x + b_1) + b_2) + b_3

    激活函数: ReLU(z) = max(0, z)
    导数: ReLU'(z) = 1 if z > 0 else 0

    训练使用 mini-batch SGD + momentum:
      v_t = beta * v_{t-1} + grad
      theta_{t+1} = theta_t - lr * v_t

    Parameters
    ----------
    input_dim : int
        输入维度
    hidden_dims : list of int
        隐藏层维度
    output_dim : int
        输出维度
    """

    def __init__(self, input_dim: int, hidden_dims: List[int],
                 output_dim: int):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.layer_dims = [input_dim] + hidden_dims + [output_dim]

        # Xavier 初始化
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self._initialize_weights()

        # 训练状态
        self.velocities_w: List[np.ndarray] = [
            np.zeros_like(w) for w in self.weights]
        self.velocities_b: List[np.ndarray] = [
            np.zeros_like(b) for b in self.biases]

    def _initialize_weights(self):
        """Xavier/Glorot 权重初始化.

        W ~ Uniform(-sqrt(6/(n_in + n_out)), sqrt(6/(n_in + n_out)))

        这确保了前向传播时激活值的方差在层间保持一致.
        """
        for i in range(len(self.layer_dims) - 1):
            n_in = self.layer_dims[i]
            n_out = self.layer_dims[i + 1]
            limit = np.sqrt(6.0 / (n_in + n_out))
            W = np.random.uniform(-limit, limit, (n_in, n_out))
            b = np.zeros(n_out)
            self.weights.append(W)
            self.biases.append(b)

    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        """ReLU 激活函数."""
        return np.maximum(0.0, z)

    @staticmethod
    def relu_derivative(z: np.ndarray) -> np.ndarray:
        """ReLU 导数."""
        return (z > 0).astype(float)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播.

        h_0 = x
        h_{l+1} = ReLU(W_l h_l + b_l)  (隐藏层)
        y = W_L h_L + b_L               (输出层, 无激活)

        Parameters
        ----------
        x : ndarray, shape (batch, input_dim)
            输入

        Returns
        -------
        ndarray
            输出
        """
        x = np.atleast_2d(x)
        self._activations = [x]
        self._pre_activations = []

        h = x
        n_layers = len(self.weights)

        for i in range(n_layers):
            z = h @ self.weights[i] + self.biases[i]
            self._pre_activations.append(z)

            if i < n_layers - 1:
                h = self.relu(z)
            else:
                h = z  # 输出层无激活

            self._activations.append(h)

        return h

    def backward(self, grad_output: np.ndarray
                  ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        反向传播.

        链式法则:
          delta_L = grad_output * output'(z_L)
          delta_l = (W_{l+1}^T delta_{l+1}) * ReLU'(z_l)

          dL/dW_l = h_l^T delta_l
          dL/db_l = sum(delta_l)

        Parameters
        ----------
        grad_output : ndarray
            输出梯度

        Returns
        -------
        grad_W, grad_b : lists of ndarray
            权重和偏置的梯度
        """
        n_layers = len(self.weights)
        grad_W = [None] * n_layers
        grad_b = [None] * n_layers

        delta = grad_output

        for l in range(n_layers - 1, -1, -1):
            h_l = self._activations[l]
            grad_W[l] = h_l.T @ delta / h_l.shape[0]
            grad_b[l] = np.mean(delta, axis=0)

            if l > 0:
                delta = (delta @ self.weights[l].T) * \
                    self.relu_derivative(self._pre_activations[l - 1])

        return grad_W, grad_b

    def train_step(self, x_batch: np.ndarray, y_batch: np.ndarray,
                    learning_rate: float = 0.001,
                    momentum: float = 0.9) -> float:
        """
        一步 SGD 训练.

        损失: L = 0.5 * ||y_pred - y_true||^2 / batch_size
        梯度: dL/dW_l, dL/db_l (通过反向传播)
        更新: v = beta*v + grad, theta -= lr * v

        Parameters
        ----------
        x_batch, y_batch : ndarray
            训练批次
        learning_rate : float
            学习率
        momentum : float
            动量系数

        Returns
        -------
        float
            损失值
        """
        # 前向
        y_pred = self.forward(x_batch)

        # MSE 损失
        loss = 0.5 * np.mean((y_pred - y_batch) ** 2)

        # 反向
        grad_output = (y_pred - y_batch) / y_batch.shape[0]
        grad_W, grad_b = self.backward(grad_output)

        # SGD + momentum 更新
        for l in range(len(self.weights)):
            self.velocities_w[l] = (momentum * self.velocities_w[l] +
                                     grad_W[l])
            self.velocities_b[l] = (momentum * self.velocities_b[l] +
                                     grad_b[l])
            self.weights[l] -= learning_rate * self.velocities_w[l]
            self.biases[l] -= learning_rate * self.velocities_b[l]

        return float(loss)

    def train(self, x_train: np.ndarray, y_train: np.ndarray,
              epochs: int = 100, batch_size: int = 32,
              learning_rate: float = 0.001) -> List[float]:
        """
        完整训练循环.

        Parameters
        ----------
        x_train, y_train : ndarray
            训练数据
        epochs : int
            训练轮数
        batch_size : int
            批次大小
        learning_rate : float
            学习率

        Returns
        -------
        losses : list of float
            每 epoch 的损失
        """
        n_samples = len(x_train)
        losses = []

        for epoch in range(epochs):
            # 随机打乱
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = indices[start:end]
                loss = self.train_step(
                    x_train[batch_idx], y_train[batch_idx],
                    learning_rate)
                epoch_loss += loss
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)

        return losses

    def predict(self, x: np.ndarray) -> np.ndarray:
        """预测 (推理)."""
        return self.forward(x)

    def get_parameters(self) -> Dict[str, np.ndarray]:
        """获取所有参数."""
        params = {}
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            params[f'W{i}'] = w.copy()
            params[f'b{i}'] = b.copy()
        return params
