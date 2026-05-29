"""
policy_network.py

策略网络 —— 基于谱基函数的参数化随机策略

基于种子项目:
  - 664_legendre_product_polynomial: Legendre 谱逼近
  - 1267_toms179 (incomplete beta): Beta 分布用于有界动作空间
  - 1006_random_data (Brownian): 动作噪声生成

科学原理:
  策略 π_θ(a|s) 使用以下参数化:

  1. 均值函数 μ_θ(s): 多元 Legendre 乘积多项式展开
         μ_i(s) = Σ_{|α|≤p} θ_{i,α} · P_α(φ(s))
     其中 φ(s) = tanh(s / s_scale) 将状态映射到 [-1,1]^d.

  2. 协方差矩阵 Σ: 使用 Toeplitz 结构刻画动作间时间相关性,
     通过 Toeplitz Cholesky 采样实现.

  3. 对于有界动作空间 [a_min, a_max], 采用 Beta 分布变换:
         a = a_min + (a_max - a_min) · Beta(α, β)
     其中 α, β 通过神经网络输出并经 softplus 保证正性.

  对数策略梯度:
      ∇_θ log π_θ(a|s) = ∇_θ μ_θ(s) · Σ^{-1} · (a - μ_θ(s))
"""

import numpy as np
from typing import Tuple
from linear_algebra import sample_from_toeplitz_covariance
from special_functions import beta_cdf


class SpectralPolicyNetwork:
    """
    基于谱基的高斯策略网络.

    参数:
        state_dim: 状态维度
        action_dim: 动作维度
        max_degree: Legendre 多项式最大次数
        action_bounds: 动作上下界 (可选)
    """

    def __init__(self, state_dim: int, action_dim: int,
                 max_degree: int = 3, action_bounds: Tuple[float, float] = (-2.0, 2.0)):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_degree = max_degree
        self.action_bounds = action_bounds
        self.state_scale = 2.0  # tanh 缩放参数

        # 构造 Legendre 基函数索引
        self.basis_indices = self._build_indices(state_dim, max_degree)
        self.num_basis = len(self.basis_indices)

        # 参数初始化: θ = [θ_mean, θ_log_std]
        # θ_mean: action_dim × num_basis
        self.theta_mean = np.random.randn(action_dim, self.num_basis) * 0.1
        self.theta_log_std = np.zeros(action_dim) - 0.5  # 初始标准差 ~0.6
        self.min_log_std = -3.0  # 最小标准差 ~0.05
        self.max_log_std = 1.0   # 最大标准差 ~2.72

        # Toeplitz 协方差的一列 (用于动作间相关性)
        self.cov_first_col = np.ones(action_dim)
        self.use_toeplitz_cov = (action_dim > 1)

        self._param_flat = self._flatten_params()

    def _build_indices(self, m: int, max_degree: int) -> list:
        """生成 multi-index 列表."""
        indices = []
        def backtrack(pos, current, remaining):
            if pos == m - 1:
                current.append(remaining)
                indices.append(tuple(current))
                current.pop()
                return
            for val in range(remaining + 1):
                current.append(val)
                backtrack(pos + 1, current, remaining - val)
                current.pop()
        for total in range(max_degree + 1):
            backtrack(0, [], total)
        return indices

    def _state_transform(self, state: np.ndarray) -> np.ndarray:
        """将状态映射到 [-1,1] 区间."""
        return np.tanh(state / self.state_scale)

    def _evaluate_basis(self, state: np.ndarray) -> np.ndarray:
        """
        在当前状态处评估所有 Legendre 基函数.
        返回 num_basis 维向量.
        """
        from spectral_basis import legendre_polynomial_1d
        z = self._state_transform(state)
        phi = np.ones(self.num_basis)
        for k, alpha in enumerate(self.basis_indices):
            for i in range(self.state_dim):
                phi[k] *= legendre_polynomial_1d(alpha[i], np.array([z[i]]))[0]
        return phi

    def mean(self, state: np.ndarray) -> np.ndarray:
        """计算策略均值 μ_θ(s)."""
        phi = self._evaluate_basis(state)
        return self.theta_mean @ phi

    def std(self) -> np.ndarray:
        """计算标准差 σ = exp(log_std)."""
        return np.exp(np.clip(self.theta_log_std, self.min_log_std, self.max_log_std))

    def sample(self, state: np.ndarray) -> np.ndarray:
        """
        从 π_θ(·|s) 中采样动作.
        """
        mu = self.mean(state)
        sigma = self.std()
        if self.use_toeplitz_cov and self.action_dim > 1:
            # 使用 Toeplitz 相关结构
            try:
                noise = sample_from_toeplitz_covariance(self.action_dim,
                                                         sigma * self.cov_first_col)
            except Exception:
                noise = np.random.randn(self.action_dim) * sigma
        else:
            noise = np.random.randn(self.action_dim) * sigma
        action = mu + noise
        # 截断到边界
        action = np.clip(action, self.action_bounds[0], self.action_bounds[1])
        return action

    def log_prob(self, state: np.ndarray, action: np.ndarray) -> float:
        """
        计算 log π_θ(a|s).
        """
        mu = self.mean(state)
        sigma = self.std()
        # 高斯对数概率
        diff = action - mu
        logp = -0.5 * np.sum((diff / sigma) ** 2) \
               - np.sum(np.log(sigma)) \
               - 0.5 * self.action_dim * np.log(2.0 * np.pi)
        return float(logp)

    def grad_log_prob(self, state: np.ndarray, action: np.ndarray) -> dict:
        """
        计算 ∇_θ log π_θ(a|s).

        解析导数:
            ∇_{θ_mean} log π = φ(s) ⊗ (Σ^{-1} (a - μ))
            ∇_{θ_log_std} log π = ( (a-μ)^2 / σ^2 - 1 )
        """
        # TODO: Hole_1 实现策略对数概率梯度
        # 需要计算并返回 {'mean': grad_mean, 'log_std': grad_log_std}
        pass

    def _flatten_params(self) -> np.ndarray:
        """将参数展平为向量 (用于优化器)."""
        return np.concatenate([
            self.theta_mean.flatten(),
            self.theta_log_std.flatten()
        ])

    def _set_flat_params(self, flat: np.ndarray):
        """从展平向量恢复参数."""
        n_mean = self.action_dim * self.num_basis
        self.theta_mean = flat[:n_mean].reshape(self.action_dim, self.num_basis)
        self.theta_log_std = flat[n_mean:]

    def get_params(self) -> np.ndarray:
        return self._flatten_params()

    def set_params(self, flat: np.ndarray):
        self._set_flat_params(flat)
        # 确保 log_std 在边界内
        n_mean = self.action_dim * self.num_basis
        self.theta_log_std = np.clip(self.theta_log_std, self.min_log_std, self.max_log_std)
        self._param_flat = self._flatten_params()

    def num_params(self) -> int:
        return self.action_dim * self.num_basis + self.action_dim
