"""
hypernetwork_params.py
======================
超网络参数化模型: 从观测频率序列推断恒星内部参数.

融合种子项目:
  - HyperMPC (1177): 超网络动力学模型
  - F-adjoint-Learning (1160): F-传播与 F-伴随

科学背景
--------
星震学中的超网络参数化:

传统方法需要反复求解正向问题 (构建恒星模型 → 计算频率),
计算量巨大. 超网络提供了一种替代方案:

  p_hyper = f(enc({ν_obs})) + p_default

  其中:
    enc: 时间序列编码器 (RNN/MLP)
    f: 超网络 MLP
    p_default: 来自恒星演化模型的默认参数
    p_hyper: 预测的恒星参数

训练数据:
  {(ν_i, p_i)} 来自网格计算 (Grid of stellar models)

损失函数:
  L = Σ_k ||p_k - p_hyper(ν_k)||² / σ_p²

F-adjoint 优化 (from 1160):
  不同于标准反向传播, F-adjoint 使用局部学习规则:
  - 前向传播: 计算各层激活
  - 伴随传播: 计算各层误差信号
  - 权重更新: ΔW = -α (Y* ⊗ X)

  优势: 不需要存储完整计算图, 内存效率更高

本模块实现:
  1. 时间序列编码器 (MLP)
  2. 超网络参数预测器
  3. F-adjoint 训练算法
  4. 恒星参数空间约束
"""

import numpy as np
from typing import Tuple, Dict, Optional


class TimeSeriesEncoder:
    """
    振荡频率时间序列编码器.

    将频率序列 {ν_i} 编码为低维表示 h.

    架构 (简化 MLP):
      h₁ = ReLU(W₁ ν + b₁)
      h₂ = ReLU(W₂ h₁ + b₂)
      h = W₃ h₂ + b₃

    参数
    ----
    input_dim : int
        输入维度 (频率数量)
    hidden_dim : int
        隐层维度
    latent_dim : int
        隐表示维度
    """

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dim: int = 64,
        latent_dim: int = 16,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Xavier 初始化
        np.random.seed(42)
        s1 = np.sqrt(2.0 / (input_dim + hidden_dim))
        s2 = np.sqrt(2.0 / (hidden_dim + hidden_dim))
        s3 = np.sqrt(2.0 / (hidden_dim + latent_dim))

        self.W1 = np.random.randn(hidden_dim, input_dim) * s1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * s2
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.randn(latent_dim, hidden_dim) * s3
        self.b3 = np.zeros(latent_dim)

    def encode(self, freq: np.ndarray) -> np.ndarray:
        """
        编码频率序列.

        Parameters
        ----------
        freq : ndarray, shape (input_dim,)
            频率序列

        Returns
        -------
        h : ndarray, shape (latent_dim,)
            隐表示
        """
        if len(freq) != self.input_dim:
            freq = self._resize(freq, self.input_dim)

        # 归一化
        freq_norm = freq / (np.max(np.abs(freq)) + 1e-10)

        h = np.maximum(0, self.W1 @ freq_norm + self.b1)  # ReLU
        h = np.maximum(0, self.W2 @ h + self.b2)            # ReLU
        h = self.W3 @ h + self.b3
        return h

    def _resize(self, arr: np.ndarray, target: int) -> np.ndarray:
        """调整数组大小 (截断或填充)."""
        if len(arr) > target:
            return arr[:target]
        else:
            return np.pad(arr, (0, target - len(arr)), mode='edge')


class HyperNetworkParams:
    """
    超网络参数预测器.

    融合 HyperMPC (1177): 从隐表示预测恒星物理参数.

    参数维度:
      p = [M/M_sun, R/R_sun, X_H, Z, α_ML, age/age_sun, log g]

    预测:
      p_pred = MLP(h) + p_default

    参数
    ----
    latent_dim : int
        隐表示维度
    param_dim : int
        参数维度
    default_params : ndarray
        默认参数
    """

    def __init__(
        self,
        latent_dim: int = 16,
        param_dim: int = 7,
        default_params: Optional[np.ndarray] = None,
    ):
        self.latent_dim = latent_dim
        self.param_dim = param_dim

        if default_params is None:
            # 太阳型恒星默认参数
            # [M/M_sun, R/R_sun, X_H, Z, α_ML, age/Gyr, log g]
            self.default_params = np.array([1.0, 1.0, 0.70, 0.02, 1.8, 4.6, 4.44])
        else:
            self.default_params = default_params.copy()

        # 超网络 MLP
        np.random.seed(123)
        s1 = np.sqrt(2.0 / (latent_dim + 32))
        s2 = np.sqrt(2.0 / (32 + param_dim))

        self.W_h1 = np.random.randn(32, latent_dim) * s1
        self.b_h1 = np.zeros(32)
        self.W_h2 = np.random.randn(param_dim, 32) * s2
        self.b_h2 = np.zeros(param_dim)

    def predict(self, h: np.ndarray) -> np.ndarray:
        """
        从隐表示预测参数.

        Parameters
        ----------
        h : ndarray, shape (latent_dim,)
            隐表示

        Returns
        -------
        params : ndarray, shape (param_dim,)
            预测参数
        """
        delta = np.maximum(0, self.W_h1 @ h + self.b_h1)
        delta = self.W_h2 @ delta + self.b_h2

        # 缩放修正
        params = self.default_params + 0.1 * delta

        # 物理约束
        params = self._apply_constraints(params)
        return params

    def _apply_constraints(self, params: np.ndarray) -> np.ndarray:
        """
        应用物理约束.

        Parameters
        ----------
        params : ndarray

        Returns
        -------
        constrained : ndarray
        """
        p = params.copy()
        p[0] = np.clip(p[0], 0.08, 150.0)    # 质量 [M_sun]
        p[1] = np.clip(p[1], 0.08, 1500.0)   # 半径 [R_sun]
        p[2] = np.clip(p[2], 0.0, 0.95)       # 氢丰度
        p[3] = np.clip(p[3], 0.0, 0.1)        # 金属丰度
        p[4] = np.clip(p[4], 0.1, 5.0)        # 混合长参数
        p[5] = np.clip(p[5], 0.0, 15.0)       # 年龄 [Gyr]
        p[6] = np.clip(p[6], 0.0, 6.0)        # log g
        return p


class FAdjointTrainer:
    """
    F-adjoint 局部学习训练器.

    融合 F-adjoint-Learning (1160) 的训练算法.

    标准反向传播需要存储完整的计算图,
    而 F-adjoint 使用局部迭代更新:

    对于每层 l:
      前向: Y_l = σ(W_l X_{l-1})
      伴随迭代:
        X*_l ← X*_l - τ (X_l - W_{l+1}^T Y*_{l+1})  (迭代 100 次)
        Y*_l = X*_l * σ'(Y_l)
      权重更新:
        W_l ← W_l - α (Y*_l X_{l-1}^T)

    参数
    ----
    layers : list of int
        网络层维度
    learning_rate : float
        学习率
    tau : float
        伴随迭代步长
    n_inner_iter : int
        内层迭代次数
    """

    def __init__(
        self,
        layers: Tuple[int, ...] = (20, 64, 64, 7),
        learning_rate: float = 0.01,
        tau: float = 0.1,
        n_inner_iter: int = 50,
    ):
        self.layers = layers
        self.n_layers = len(layers) - 1
        self.lr = learning_rate
        self.tau = tau
        self.n_inner = n_inner_iter

        # 初始化权重
        np.random.seed(42)
        self.W = {}
        for l in range(1, self.n_layers + 1):
            dim_in = layers[l - 1] + 1  # +1 for bias
            dim_out = layers[l]
            std = np.sqrt(2.0 / dim_in)
            self.W[l] = np.random.randn(dim_out, dim_in) * std

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid 激活."""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def _sigmoid_prime(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid 导数."""
        s = self._sigmoid(z)
        return s * (1.0 - s)

    def _add_bias(self, x: np.ndarray) -> np.ndarray:
        """添加偏置项."""
        if x.ndim == 1:
            return np.append(x, 1.0)
        return np.vstack([x, np.ones((1, x.shape[1]))])

    def f_pass(self, x: np.ndarray) -> Tuple[Dict, Dict]:
        """
        F-前向传播.

        Parameters
        ----------
        x : ndarray, shape (input_dim, batch_size)
            输入

        Returns
        -------
        Y : dict
            各层线性输出
        X : dict
            各层激活输出
        """
        Y = {}
        X = {}
        X[0] = self._add_bias(x)

        for l in range(1, self.n_layers + 1):
            Y[l] = self.W[l] @ X[l - 1]
            if l < self.n_layers:
                X[l] = self._add_bias(self._sigmoid(Y[l]))
            else:
                X[l] = Y[l]  # 输出层无激活

        return Y, X

    def fstar_pass_dyn(
        self,
        x: np.ndarray,
        y_target: np.ndarray,
    ) -> None:
        """
        F-adjoint 动态权重更新.

        融合 1160 的 Fstar_pass_dyn 算法.

        Parameters
        ----------
        x : ndarray
            输入
        y_target : ndarray
            目标输出
        """
        (FY, FX) = self.f_pass(x)

        # 输出层误差
        TY = {}
        TX = {}
        # FY has keys 1..n_layers, FX has keys 0..n_layers
        for l in range(self.n_layers + 1):
            if l in FY:
                TY[l] = FY[l].copy()
            if l in FX:
                TX[l] = FX[l].copy()

        # 输出层目标
        TY[self.n_layers] = (FX[self.n_layers] - y_target)

        # 从输出到输入的反向迭代
        for l in reversed(range(1, self.n_layers + 1)):
            if l != self.n_layers:
                # 内层: 迭代求解伴随
                for _ in range(self.n_inner):
                    dXl = TX[l] - self.W[l + 1].T @ TY[l + 1]
                    TX[l] = TX[l] - self.tau * dXl
                TY[l] = TX[l][:-1, :] * self._sigmoid_prime(FY[l])
            else:
                # 输出层
                for _ in range(self.n_inner):
                    dXl = TX[l] - TY[l]
                    TX[l] = TX[l] - self.tau * dXl
                TY[l] = TX[l] * 1.0  # 线性输出层

            # 权重更新
            self.W[l] -= self.lr * TY[l] @ FX[l - 1].T

    def train_epoch(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        batch_size: int = 32,
    ) -> float:
        """
        训练一个 epoch.

        Parameters
        ----------
        X_train : ndarray, shape (input_dim, n_samples)
            训练输入
        Y_train : ndarray, shape (output_dim, n_samples)
            训练目标
        batch_size : int
            小批量大小

        Returns
        -------
        loss : float
            平均损失
        """
        n_samples = X_train.shape[1]
        indices = np.random.permutation(n_samples)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]

            X_batch = X_train[:, batch_idx]
            Y_batch = Y_train[:, batch_idx]

            # 前向 (计算损失)
            _, FX = self.f_pass(X_batch)
            pred = FX[self.n_layers]
            loss = np.mean((pred - Y_batch)**2)
            total_loss += loss
            n_batches += 1

            # F-adjoint 更新
            self.fstar_pass_dyn(X_batch, Y_batch)

        return total_loss / max(n_batches, 1)


def generate_training_data(
    n_samples: int = 100,
    n_modes: int = 20,
    n_params: int = 7,
    noise_level: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成训练数据 (合成).

    使用简化标度关系:
      Δν ∝ √(M/R³)
      ν_max ∝ M/(R² √T_eff)
      T_eff ∝ (L/R²)^{1/4}

    Parameters
    ----------
    n_samples : int
        样本数
    n_modes : int
        每个样本的模式数
    n_params : int
        参数维度
    noise_level : float
        噪声水平

    Returns
    -------
    X_train : ndarray, shape (n_modes, n_samples)
    Y_train : ndarray, shape (n_params, n_samples)
    """
    np.random.seed(42)

    # 随机参数 (在物理范围内)
    mass = 0.5 + 3.0 * np.random.rand(n_samples)
    radius = 0.5 + 3.0 * np.random.rand(n_samples)
    X_H = 0.6 + 0.3 * np.random.rand(n_samples)
    Z = 0.001 + 0.05 * np.random.rand(n_samples)
    alpha_ml = 1.0 + 1.5 * np.random.rand(n_samples)
    age = 0.5 + 10.0 * np.random.rand(n_samples)
    log_g = 3.5 + 1.5 * np.random.rand(n_samples)

    params = np.array([mass, radius, X_H, Z, alpha_ml, age, log_g])

    # 从参数生成频率
    delta_nu = 135.0 * np.sqrt(mass / radius**3)  # 标度关系
    nu_max = 3090.0 * mass / (radius**2 * np.sqrt(10**(log_g - 4.44)))

    freqs = np.zeros((n_modes, n_samples))
    for s in range(n_samples):
        for i in range(n_modes):
            n = i + 1
            freqs[i, s] = (n + 0.5) * delta_nu[s] + noise_level * np.random.randn() * delta_nu[s]

    return freqs, params
