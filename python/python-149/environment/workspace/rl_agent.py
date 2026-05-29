"""
rl_agent.py
深度强化学习智能体：Actor-Critic架构

科学背景:
  在连续动作空间的随机最优控制中，Actor-Critic方法同时学习:
  - Critic: 值函数 V^π(s) 或动作值函数 Q^π(s,a)
  - Actor: 策略 π_θ(a|s)

  对于神经质量模型的控制问题，状态 s = [E, I] ∈ [0,1]^2，动作 a = u ∈ [-5,5]。

  Critic更新（TD学习）:
      δ_t = r_t + γ V(s_{t+1}) - V(s_t)
      w ← w + α_w δ_t ∇_w V(s_t)

  Actor更新（策略梯度）:
      θ ← θ + α_θ δ_t ∇_θ log π_θ(a_t|s_t)

  其中优势函数 A(s,a) ≈ δ_t 为TD误差。

  值函数采用径向基函数(RBF)网络近似:
      V_w(s) = Σ_i w_i φ_i(s)
      φ_i(s) = exp( -||s - c_i||^2 / (2σ_i^2) )

  策略采用带噪声的高斯策略:
      π_θ(a|s) = N( μ_θ(s), σ_policy^2 )
      μ_θ(s) = tanh( θ^T φ(s) ) · a_max
"""

import numpy as np
from typing import Callable, Optional, Tuple, List


class RBFValueFunction:
    """
    基于径向基函数的值函数逼近器。

        V(s) = w_0 + Σ_i w_i φ_i(s)
        φ_i(s) = exp( -||s - c_i||^2 / (2σ^2) )
    """

    def __init__(
        self,
        n_centers: int = 16,
        state_dim: int = 2,
        sigma: float = 0.15,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        if rng is None:
            rng = np.random.default_rng(seed=42)
        self.state_dim = state_dim
        self.sigma = sigma
        self.n_centers = n_centers

        if bounds is None:
            self.centers = rng.uniform(0, 1, (n_centers, state_dim))
        else:
            xmin, xmax = bounds
            self.centers = np.zeros((n_centers, state_dim))
            for i in range(n_centers):
                self.centers[i, :] = xmin + rng.uniform(0, 1, state_dim) * (xmax - xmin)

        # 权重初始化（含偏置项）
        self.w = rng.normal(0, 0.1, n_centers + 1)

    def features(self, s: np.ndarray) -> np.ndarray:
        """
        计算RBF特征向量 [1, φ_1(s), ..., φ_n(s)]。
        """
        s = np.atleast_1d(s)
        phi = np.exp(-np.sum((self.centers - s) ** 2, axis=1) / (2 * self.sigma ** 2))
        return np.concatenate(([1.0], phi))

    def value(self, s: np.ndarray) -> float:
        """V(s) = w^T φ(s)"""
        phi = self.features(s)
        return float(np.dot(self.w, phi))

    def gradient(self, s: np.ndarray) -> np.ndarray:
        """∇_w V(s) = φ(s)"""
        return self.features(s)

    def update(self, s: np.ndarray, target: float, alpha: float = 0.1) -> float:
        """
        梯度下降更新:
            w ← w + α (target - V(s)) ∇_w V(s)
        """
        phi = self.features(s)
        v = np.dot(self.w, phi)
        delta = target - v
        self.w += alpha * delta * phi
        return float(delta)


class GaussianPolicy:
    """
    参数化高斯策略。

        μ_θ(s) = a_max · tanh( θ_μ^T φ(s) )
        σ > 0 固定（探索噪声）

        π(a|s) = (1/√(2π)σ) exp( -(a-μ)^2 / (2σ^2) )

    对数梯度:
        ∇_θ log π(a|s) = (a - μ) / σ^2 · ∇_θ μ_θ(s)
        ∇_θ μ = a_max · (1 - tanh^2(·)) · φ(s)
    """

    def __init__(
        self,
        n_features: int,
        action_bound: float = 5.0,
        sigma: float = 0.5,
        rng: Optional[np.random.Generator] = None,
    ):
        if rng is None:
            rng = np.random.default_rng(seed=42)
        self.n_features = n_features
        self.action_bound = action_bound
        self.sigma = max(sigma, 1e-3)
        self.theta = rng.normal(0, 0.1, n_features)

    def mean(self, phi_s: np.ndarray) -> float:
        """确定性均值 μ_θ(s)。"""
        z = float(np.dot(self.theta, phi_s))
        # tanh饱和保护
        z = np.clip(z, -10.0, 10.0)
        return self.action_bound * np.tanh(z)

    def sample(self, phi_s: np.ndarray, rng: Optional[np.random.Generator] = None) -> float:
        """采样动作。"""
        if rng is None:
            rng = np.random.default_rng()
        mu = self.mean(phi_s)
        a = rng.normal(mu, self.sigma)
        return float(np.clip(a, -self.action_bound, self.action_bound))

    def log_prob_gradient(self, phi_s: np.ndarray, a: float) -> np.ndarray:
        """
        ∇_θ log π(a|s) 的估计。
        """
        z = float(np.dot(self.theta, phi_s))
        z = np.clip(z, -10.0, 10.0)
        mu = self.action_bound * np.tanh(z)
        # dμ/dz = a_max * (1 - tanh^2(z))
        dtanh = 1.0 - np.tanh(z) ** 2
        dmu_dtheta = self.action_bound * dtanh * phi_s
        # ∇_θ log π = (a - μ)/σ^2 * dμ/dθ
        grad = (a - mu) / (self.sigma ** 2) * dmu_dtheta
        return grad

    def update(self, phi_s: np.ndarray, a: float, advantage: float, alpha: float = 0.01) -> None:
        """
        策略梯度上升:
            θ ← θ + α · advantage · ∇_θ log π(a|s)
        """
        grad = self.log_prob_gradient(phi_s, a)
        self.theta += alpha * advantage * grad
        # 参数范数保护
        norm = np.linalg.norm(self.theta)
        if norm > 100.0:
            self.theta = self.theta * 100.0 / norm


class ActorCriticAgent:
    """
    Actor-Critic智能体，用于连续动作空间的神经最优控制。
    """

    def __init__(
        self,
        state_dim: int = 2,
        n_rbf: int = 16,
        action_bound: float = 5.0,
        gamma: float = 0.95,
        alpha_critic: float = 0.1,
        alpha_actor: float = 0.01,
        policy_sigma: float = 0.5,
        rng: Optional[np.random.Generator] = None,
    ):
        if rng is None:
            rng = np.random.default_rng(seed=42)
        self.rng = rng
        self.gamma = gamma
        self.alpha_critic = alpha_critic
        self.alpha_actor = alpha_actor

        # Critic
        self.critic = RBFValueFunction(
            n_centers=n_rbf,
            state_dim=state_dim,
            sigma=0.15,
            bounds=(np.zeros(state_dim), np.ones(state_dim)),
            rng=rng,
        )

        # Actor
        self.actor = GaussianPolicy(
            n_features=n_rbf + 1,
            action_bound=action_bound,
            sigma=policy_sigma,
            rng=rng,
        )

        self.episode_rewards: List[float] = []

    def select_action(self, state: np.ndarray) -> float:
        """根据当前策略选择动作。"""
        phi = self.critic.features(state)
        return self.actor.sample(phi, self.rng)

    def train_step(
        self,
        state: np.ndarray,
        action: float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Tuple[float, float]:
        """
        单步训练。

        Returns
        -------
        td_error : float
        advantage : float
        """
        phi_s = self.critic.features(state)
        v_s = self.critic.value(state)

        if done:
            v_next = 0.0
        else:
            v_next = self.critic.value(next_state)

        td_target = reward + self.gamma * v_next
        td_error = td_target - v_s

        # Critic更新
        self.critic.w += self.alpha_critic * td_error * phi_s

        # Actor更新
        advantage = td_error
        grad = self.actor.log_prob_gradient(phi_s, action)
        self.actor.theta += self.alpha_actor * advantage * grad

        # 参数保护
        norm = np.linalg.norm(self.actor.theta)
        if norm > 100.0:
            self.actor.theta *= 100.0 / norm

        return td_error, advantage

    def run_episode(
        self,
        env_step_fn: Callable[[np.ndarray, float], Tuple[np.ndarray, float, bool]],
        state0: np.ndarray,
        max_steps: int = 500,
    ) -> float:
        """
        运行单个训练回合。

        Parameters
        ----------
        env_step_fn : callable
            (next_state, reward, done) = env_step_fn(state, action)

        Returns
        -------
        total_reward : float
        """
        state = state0.copy()
        total_reward = 0.0

        for _ in range(max_steps):
            action = self.select_action(state)
            next_state, reward, done = env_step_fn(state, action)

            # 边界保护
            next_state = np.clip(next_state, -0.1, 1.1)

            self.train_step(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state

            if done:
                break

        self.episode_rewards.append(total_reward)
        return total_reward
