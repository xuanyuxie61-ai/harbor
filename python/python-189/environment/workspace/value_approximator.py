"""
value_approximator.py

价值函数逼近器 —— 结合 PCA 降维与谱回归

基于种子项目:
  - 326_eigenfaces (pc_vectors): PCA 特征提取
  - 664_legendre_product_polynomial: 谱基回归
  - 1048_rref2 (rref_solve): 线性最小二乘求解

科学原理:
  状态值函数 V^π(s) 的近似:

  阶段 1: PCA 降维
      给定高维观测历史 {o_t}, 提取前 d 个主成分 (V, Ψ),
      低维状态: z_t = V^T (o_t - Ψ).

  阶段 2: 谱回归
      在 z-空间中使用 Legendre 乘积多项式基 {P_α(z)} 展开:
          V_w(z) = Σ_α w_α P_α(z) = w^T φ(z)

  阶段 3: 最小二乘求解 (RREF 或正规方程)
      给定样本 {(z_i, G_i)}, G_i 为折扣回报,
      求解:  Φ w ≈ G,  Φ_{i,α} = P_α(z_i)
      若 Φ 亏秩, 使用 RREF 求最小范数解.

  优势函数估计 (GAE):
      A_t = Σ_{l=0}^∞ (γ λ)^l δ_{t+l}
      δ_t = r_t + γ V(s_{t+1}) - V(s_t)
"""

import numpy as np
from typing import List, Tuple
from spectral_basis import pca_vectors, pca_transform, build_legendre_basis
from linear_algebra import rref_solve


class PCAStateRepresentation:
    """PCA 状态降维表示器."""

    def __init__(self, original_dim: int, reduced_dim: int):
        self.original_dim = original_dim
        self.reduced_dim = reduced_dim
        self.V = None
        self.Psi = None
        self._fitted = False

    def fit(self, observations: np.ndarray):
        """从历史观测中学习 PCA 子空间."""
        observations = np.asarray(observations, dtype=float)
        if observations.ndim == 1:
            observations = observations.reshape(-1, 1)
        self.V, self.eigvals, self.Psi = pca_vectors(
            observations.T, self.reduced_dim
        )
        self._fitted = True

    def transform(self, observation: np.ndarray) -> np.ndarray:
        if not self._fitted:
            # 未拟合时直接截断
            obs = np.asarray(observation, dtype=float)
            if len(obs) > self.reduced_dim:
                return obs[:self.reduced_dim]
            return obs
        return pca_transform(observation, self.V, self.Psi)


class SpectralValueFunction:
    """
    基于谱基的最小二乘价值函数.
    """

    def __init__(self, state_dim: int, max_degree: int = 3, gamma: float = 0.99):
        self.state_dim = state_dim
        self.max_degree = max_degree
        self.gamma = gamma
        self.w = None
        self.basis_indices = None
        self.num_basis = 0
        self._build_indices()

    def _build_indices(self):
        """构造 multi-index 列表."""
        indices = []
        def backtrack(pos, current, remaining):
            if pos == self.state_dim - 1:
                current.append(remaining)
                indices.append(tuple(current))
                current.pop()
                return
            for val in range(remaining + 1):
                current.append(val)
                backtrack(pos + 1, current, remaining - val)
                current.pop()
        for total in range(self.max_degree + 1):
            backtrack(0, [], total)
        self.basis_indices = indices
        self.num_basis = len(indices)

    def _phi(self, state: np.ndarray) -> np.ndarray:
        """评估基函数向量."""
        from spectral_basis import legendre_polynomial_1d
        z = np.tanh(state / 2.0)
        phi = np.ones(self.num_basis)
        for k, alpha in enumerate(self.basis_indices):
            for i in range(min(self.state_dim, len(z))):
                phi[k] *= legendre_polynomial_1d(alpha[i], np.array([z[i]]))[0]
        return phi

    def fit(self, states: List[np.ndarray], returns: List[float],
            reg: float = 1.0e-4, method: str = 'rref'):
        """
        用最小二乘拟合价值函数.

        参数:
            states:  状态序列
            returns: 对应的折扣回报
            reg:     正则化系数
            method:  'rref' 或 'normal'
        """
        n = len(states)
        if n == 0:
            self.w = np.zeros(self.num_basis)
            return
        Phi = np.zeros((n, self.num_basis))
        for i, s in enumerate(states):
            Phi[i, :] = self._phi(s)
        G = np.array(returns, dtype=float)

        if method == 'normal':
            # 正规方程: (Phi^T Phi + reg I) w = Phi^T G
            A = Phi.T @ Phi + reg * np.eye(self.num_basis)
            b = Phi.T @ G
            try:
                self.w = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                self.w = np.linalg.lstsq(A, b, rcond=None)[0]
        else:
            # RREF 求解 (对亏秩更鲁棒)
            A = Phi.T @ Phi + reg * np.eye(self.num_basis)
            b = Phi.T @ G
            w = rref_solve(A, b.reshape(-1, 1))
            self.w = w.flatten()
            # 若 RREF 给出异常解, 回退到最小二乘
            if not np.all(np.isfinite(self.w)):
                self.w = np.linalg.lstsq(A, b, rcond=None)[0]

    def predict(self, state: np.ndarray) -> float:
        """预测状态值 V(s)."""
        if self.w is None:
            return 0.0
        phi = self._phi(state)
        return float(phi @ self.w)

    def predict_batch(self, states: np.ndarray) -> np.ndarray:
        """批量预测."""
        if self.w is None:
            return np.zeros(len(states))
        if states.ndim == 1:
            states = states.reshape(1, -1)
        vals = np.zeros(states.shape[0])
        for i in range(states.shape[0]):
            vals[i] = self.predict(states[i])
        return vals


def compute_discounted_returns(rewards: List[float], gamma: float) -> List[float]:
    """
    计算折扣回报 G_t = Σ_{k=0}^∞ γ^k r_{t+k}.
    """
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def generalized_advantage_estimate(rewards: List[float], values: List[float],
                                    gamma: float = 0.99, lam: float = 0.95) -> List[float]:
    """
    GAE(λ) 优势估计.

    公式:
        δ_t = r_t + γ V(s_{t+1}) - V(s_t)
        Â_t = Σ_{l=0}^∞ (γ λ)^l δ_{t+l}

    参数:
        rewards: 奖励序列
        values:  值函数估计序列 (长度比 rewards 多 1, 包含终止状态)
        gamma:   折扣因子
        lam:     GAE 参数

    返回:
        优势估计序列
    """
    T = len(rewards)
    advantages = []
    adv = 0.0
    for t in reversed(range(T)):
        if t + 1 < len(values):
            delta = rewards[t] + gamma * values[t + 1] - values[t]
        else:
            delta = rewards[t] - values[t]
        adv = delta + gamma * lam * adv
        advantages.insert(0, adv)
    return advantages
