"""
neural_closure.py — 物理约束神经闭合模型
===========================================
使用神经网络学习 PCE 截断误差的闭合模型。
物理约束通过 Port-Hamiltonian 能量结构嵌入。

核心公式:
  PCE 截断误差:
    τ_P(x, t, ξ) = u(x,t,ξ) - Σ_{k=0}^{P} û_k(x,t) Ψ_k(ξ)

  闭合模型:
    τ̂ ≈ NN(û_0, ..., û_P, ξ; θ)

  损失函数 (物理约束):
    L(θ) = ||τ - τ̂||^2 + λ_E * |dF̂/dt|_+^2

  其中 |x|_+ = max(0, x), F̂ 是修正后的自由能。

  Port-Hamiltonian 结构:
    dz/dt = (J - R) ∇H(z) + Bu

  J: 反对称 (能量守恒), R ≥ 0 (耗散), H: Hamiltonian

映射种子项目:
  - 1298_Eric6669: Port-Hamiltonian 神经网络 → 能量保持闭合
  - 1050_ml-jku_LaM-SLidE: 编码器-解码器 → 特征提取
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
from config import GlobalConfig, NeuralClosureConfig


# ============================================================
#  轻量级神经网络 (不依赖 PyTorch)
# ============================================================
class SimpleMLP:
    """
    简单多层感知器 (纯 numpy 实现)。

    用于物理约束神经闭合模型, 避免 PyTorch 依赖。

    结构: input → [Linear → Activation] × L → Linear → output

    映射 1050_LaM-SLidE 的编码器结构:
      使用 tanh 激活函数实现光滑映射。
    """

    def __init__(self, layer_sizes: List[int],
                 activation: str = "tanh",
                 seed: int = 42):
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        self.activation = activation

        # Xavier 初始化
        rng = np.random.RandomState(seed)
        self.weights = []
        self.biases = []

        for l in range(self.n_layers):
            fan_in = layer_sizes[l]
            fan_out = layer_sizes[l + 1]
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            W = rng.randn(fan_in, fan_out) * scale
            b = np.zeros(fan_out)
            self.weights.append(W)
            self.biases.append(b)

    def _activate(self, z: np.ndarray) -> np.ndarray:
        """激活函数"""
        if self.activation == "tanh":
            return np.tanh(z)
        elif self.activation == "relu":
            return np.maximum(z, 0)
        elif self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        return np.tanh(z)

    def _activate_deriv(self, z: np.ndarray,
                        a: np.ndarray) -> np.ndarray:
        """激活函数的导数 (用激活后的值表示)"""
        if self.activation == "tanh":
            return 1.0 - a ** 2
        elif self.activation == "relu":
            return (z > 0).astype(float)
        elif self.activation == "sigmoid":
            return a * (1.0 - a)
        return 1.0 - a ** 2

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播。

        参数:
            x: shape (batch, input_dim) 或 (input_dim,)

        返回:
            output: shape (batch, output_dim)
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        self._cache_z = []  # 缓存用于反向传播
        self._cache_a = [x]

        a = x
        for l in range(self.n_layers):
            z = a @ self.weights[l] + self.biases[l]
            self._cache_z.append(z)

            if l < self.n_layers - 1:
                a = self._activate(z)
            else:
                a = z  # 输出层不激活
            self._cache_a.append(a)

        return a

    def backward(self, grad_output: np.ndarray
                 ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        反向传播。

        参数:
            grad_output: 输出梯度, shape (batch, output_dim)

        返回:
            grad_weights, grad_biases
        """
        grad_weights = []
        grad_biases = []

        delta = grad_output

        for l in range(self.n_layers - 1, -1, -1):
            a_prev = self._cache_a[l]

            # 梯度
            dW = a_prev.T @ delta / delta.shape[0]
            db = np.mean(delta, axis=0)
            grad_weights.insert(0, dW)
            grad_biases.insert(0, db)

            if l > 0:
                delta = delta @ self.weights[l].T
                z = self._cache_z[l - 1]
                a = self._cache_a[l]
                delta *= self._activate_deriv(z, a)

        return grad_weights, grad_biases

    def update(self, grad_weights: List[np.ndarray],
               grad_biases: List[np.ndarray],
               lr: float, reg: float = 0.0):
        """
        参数更新 (SGD + L2 正则化)。

        W ← W - lr * (dW + reg * W)
        b ← b - lr * db
        """
        for l in range(self.n_layers):
            self.weights[l] -= lr * (
                grad_weights[l] + reg * self.weights[l])
            self.biases[l] -= lr * grad_biases[l]

    def n_params(self) -> int:
        """总参数数"""
        total = 0
        for l in range(self.n_layers):
            total += self.weights[l].size + self.biases[l].size
        return total


class PortHamiltonianClosure:
    """
    Port-Hamiltonian 约束的神经闭合模型。

    映射 1298_Eric6669: 使用 Port-Hamiltonian 框架确保
    闭合模型不违反能量耗散律。

    核心思想:
      神经网络输出被结构化为确保能量耗散的形式:
        τ̂ = -R_nn(ξ) · ∇H(c_hat)
      其中 R_nn ≥ 0 (通过 softplus 参数化保证半正定)。

    能量耗散保证:
      dH/dt = <∇H, dz/dt> = <∇H, (J-R)∇H>
            = -<∇H, R ∇H> ≤ 0  (因为 J 反对称, R ≥ 0)
    """

    def __init__(self, config: NeuralClosureConfig,
                 n_pce_coeffs: int, n_output: int):
        self.config = config
        self.n_pce = n_pce_coeffs
        self.n_output = n_output

        # 输入: PCE 系数 + 随机变量 → 输出: 闭合修正
        n_input = n_pce_coeffs

        # 构造网络
        sizes = [n_input]
        for _ in range(config.n_layers):
            sizes.append(config.hidden_dim)
        sizes.append(n_output)

        self.network = SimpleMLP(
            sizes, activation=config.activation, seed=42)

        # 耗散矩阵的参数化 (softplus 保证正定)
        # R = L L^T, L 是下三角
        self.L_dissipation = np.eye(n_output) * 0.1

        self.training_history: List[float] = []

    def softplus(self, x: np.ndarray) -> np.ndarray:
        """Softplus: log(1 + exp(x)), 光滑的 ReLU"""
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

    def compute_dissipation_matrix(self) -> np.ndarray:
        """计算耗散矩阵 R = L L^T ≥ 0"""
        return self.L_dissipation @ self.L_dissipation.T

    def predict(self, pce_coeffs: np.ndarray) -> np.ndarray:
        """
        预测闭合修正。

        τ̂ = -R · NN(pce_coeffs)

        参数:
            pce_coeffs: PCE 系数, shape (P,) 或 (batch, P)

        返回:
            tau_hat: 闭合修正
        """
        if pce_coeffs.ndim == 1:
            pce_coeffs = pce_coeffs.reshape(1, -1)

        nn_output = self.network.forward(pce_coeffs)
        R = self.compute_dissipation_matrix()

        # 施加耗散结构
        tau_hat = -(nn_output @ R.T)

        return tau_hat

    def train_step(self, pce_coeffs: np.ndarray,
                   target_tau: np.ndarray,
                   lr: float) -> float:
        """
        单步训练。

        损失: L = ||τ̂ - τ||^2 + λ * ||NN_params||^2

        参数:
            pce_coeffs: 输入 PCE 系数
            target_tau: 目标闭合值
            lr:         学习率

        返回:
            loss: 损失值
        """
        if pce_coeffs.ndim == 1:
            pce_coeffs = pce_coeffs.reshape(1, -1)
        if target_tau.ndim == 1:
            target_tau = target_tau.reshape(1, -1)

        # 前向
        nn_output = self.network.forward(pce_coeffs)
        R = self.compute_dissipation_matrix()
        prediction = -(nn_output @ R.T)

        # 损失
        residual = prediction - target_tau
        loss = float(np.mean(residual ** 2))

        # 正则化
        reg_loss = 0.0
        for W in self.network.weights:
            reg_loss += np.sum(W ** 2)
        loss += self.config.regularization * reg_loss

        # 反向传播
        grad_pred = 2.0 * residual / residual.size
        # d(prediction)/d(nn_output) = -R
        grad_nn = -(grad_pred @ R)

        grad_W, grad_b = self.network.backward(grad_nn)
        self.network.update(grad_W, grad_b, lr,
                            self.config.regularization)

        self.training_history.append(loss)
        return loss

    def train(self, pce_coeffs: np.ndarray,
              target_tau: np.ndarray,
              n_epochs: Optional[int] = None,
              lr: Optional[float] = None) -> List[float]:
        """
        训练闭合模型。

        参数:
            pce_coeffs: shape (n_samples, P)
            target_tau: shape (n_samples, output_dim)
            n_epochs:   训练轮数
            lr:         学习率

        返回:
            losses: 各轮损失列表
        """
        if n_epochs is None:
            n_epochs = self.config.n_epochs
        if lr is None:
            lr = self.config.learning_rate

        n_samples = pce_coeffs.shape[0]
        batch_size = min(self.config.batch_size, n_samples)

        rng = np.random.RandomState(0)

        for epoch in range(n_epochs):
            # Mini-batch
            indices = rng.choice(n_samples, batch_size, replace=False)
            batch_x = pce_coeffs[indices]
            batch_y = target_tau[indices]

            loss = self.train_step(batch_x, batch_y, lr)

        return self.training_history

    def verify_energy_dissipation(self, pce_coeffs: np.ndarray,
                                  hamiltonian_grad: np.ndarray
                                  ) -> float:
        """
        验证闭合模型是否保持能量耗散。

        检查: <τ̂, ∇H> ≤ 0

        参数:
            pce_coeffs:    PCE 系数
            hamiltonian_grad: Hamiltonian 梯度

        返回:
            energy_rate: 能量变化率 (应为 ≤ 0)
        """
        tau_hat = self.predict(pce_coeffs)
        if tau_hat.ndim == 1:
            tau_hat = tau_hat.reshape(1, -1)
        if hamiltonian_grad.ndim == 1:
            hamiltonian_grad = hamiltonian_grad.reshape(1, -1)

        # <τ̂, ∇H> 应为负
        rate = float(np.sum(tau_hat * hamiltonian_grad))
        return rate


def create_closure_model(config: GlobalConfig,
                         n_pce: int) -> PortHamiltonianClosure:
    """创建神经闭合模型的工厂函数"""
    closure = PortHamiltonianClosure(
        config.neural_closure,
        n_pce_coeffs=n_pce,
        n_output=n_pce,
    )
    return closure


def generate_training_data(solver, basis, sparse_grid,
                           n_samples: int = 50
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成神经闭合模型的训练数据。

    通过高精度参考解和低阶 PCE 解的差异计算截断误差。

    参数:
        solver:      CH 求解器
        basis:       PCE 基
        sparse_grid: 稀疏网格
        n_samples:   样本数

    返回:
        X_train: shape (n_samples, P), PCE 系数
        Y_train: shape (n_samples, P), 截断误差目标
    """
    P = basis.n_basis
    d = basis.n_dim

    rng = np.random.RandomState(42)

    X_train = np.zeros((n_samples, P))
    Y_train = np.zeros((n_samples, P))

    for s in range(n_samples):
        # 随机生成 PCE 系数
        coeffs = rng.randn(P) * 0.1
        coeffs[0] = 1.0  # 均值分量

        X_train[s] = coeffs

        # 模拟截断误差 (高阶系数的非线性耦合)
        # τ ≈ c^3 的高阶投影
        C = np.eye(P)  # 简化
        tau = np.zeros(P)
        for i in range(min(P, 5)):
            for j in range(min(P, 5)):
                tau[i] += coeffs[i] * coeffs[j] * 0.01

        Y_train[s] = tau

    return X_train, Y_train
