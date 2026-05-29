"""
policy_gradient_core.py

策略梯度核心算法 —— REINFORCE + 自然梯度 + GAE

科学原理:
  本模块实现完整的策略梯度训练循环, 融合以下高级技术:

  1. REINFORCE with Baseline:
         ∇J = E[ Σ_t ∇_θ log π_θ(a_t|s_t) · (G_t - b_t) ]
     基线 b_t 使用学得的值函数 V(s_t) 降低方差.

  2. 广义优势估计 GAE(λ):
         Â_t = Σ_{l=0}^∞ (γ λ)^l δ_{t+l}
         δ_t = r_t + γ V(s_{t+1}) - V(s_t)

  3. 自然策略梯度 (NPG):
         θ ← θ + α · F(θ)^{-1} ∇J
     通过 CG 近似避免显式求逆 Fisher 矩阵.

  4. 约束投影:
     动作经 LP 投影保证满足物理约束.

  5. 谱正则化:
     使用 Bessel 零点设计状态特征的带通滤波器,
     仅保留与系统动力学共振频率匹配的成分.

  训练目标 (博士级公式):
      max_θ  J(θ) = E_{τ~π_θ} [ Σ_{t=0}^T γ^t r(s_t, a_t) ]

      s.t.   D_KL( π_θ_old || π_θ ) ≤ δ          (信任区域)
             C·a_t ≤ d,  ∀t                      (动作约束)
             ||∇_θ J||_F ≤ G_max                 (梯度截断)
"""

import numpy as np
from typing import List, Tuple, Dict
from dynamical_system import ControlledNonlinearOscillator
from policy_network import SpectralPolicyNetwork
from value_approximator import SpectralValueFunction, compute_discounted_returns, generalized_advantage_estimate
from natural_gradient import NaturalPolicyGradientOptimizer, fisher_vector_product
from constrained_optimizer import lp_action_projection, CosineAnnealingScheduler
from stochastic_processes import ornstein_uhlenbeck_process


class TrajectoryBuffer:
    """轨迹数据缓存."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def add(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def size(self) -> int:
        return len(self.states)


class PolicyGradientTrainer:
    """
    策略梯度训练器.
    """

    def __init__(self,
                 state_dim: int = 4,
                 action_dim: int = 4,
                 policy_degree: int = 3,
                 value_degree: int = 3,
                 gamma: float = 0.99,
                 lam: float = 0.95,
                 lr_policy: float = 0.005,
                 lr_value: float = 0.01,
                 cg_iter: int = 30,
                 cg_damping: float = 0.1,
                 max_kl: float = 0.02,
                 entropy_coef: float = 0.01,
                 batch_episodes: int = 5,
                 max_steps_per_episode: int = 500):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lam = lam
        self.entropy_coef = entropy_coef
        self.batch_episodes = batch_episodes
        self.max_steps = max_steps_per_episode

        # 策略与值网络
        self.policy = SpectralPolicyNetwork(state_dim, action_dim,
                                             max_degree=policy_degree)
        self.value_fn = SpectralValueFunction(state_dim, max_degree=value_degree,
                                               gamma=gamma)

        # 优化器
        self.ng_optimizer = NaturalPolicyGradientOptimizer(
            learning_rate=lr_policy,
            cg_iter=cg_iter,
            cg_damping=cg_damping,
            max_kl=max_kl
        )
        self.lr_scheduler = CosineAnnealingScheduler(alpha_max=lr_policy,
                                                      alpha_min=lr_policy * 0.1,
                                                      T_period=50)

        # 环境
        self.env = ControlledNonlinearOscillator(dt=0.005)

        # 探索噪声 (OU 过程)
        self.ou_noise = ornstein_uhlenbeck_process(1000, action_dim,
                                                    theta=0.15, sigma=0.2, dt=0.01)
        self.noise_idx = 0

        # 训练历史
        self.episode_rewards = []
        self.episode_lengths = []

    def collect_episode(self, deterministic: bool = False) -> TrajectoryBuffer:
        """采集一条完整轨迹."""
        buffer = TrajectoryBuffer()
        obs = self.env.reset()
        for step in range(self.max_steps):
            # 策略采样
            action = self.policy.sample(obs)
            if not deterministic:
                # 添加 OU 探索噪声
                noise = self.ou_noise[self.noise_idx % len(self.ou_noise)]
                self.noise_idx += 1
                action = action + noise
                action = np.clip(action, -2.0, 2.0)
            # LP 投影 (此处无额外约束, 仅边界截断)
            action = lp_action_projection(action, bounds=(-2.0, 2.0))
            next_obs, reward, done, info = self.env.step(action)
            buffer.add(obs, action, reward, next_obs, done)
            obs = next_obs
            if done:
                break
        return buffer

    def _flatten_policy_grad(self, state, action) -> np.ndarray:
        """获取展平的策略梯度."""
        # TODO: Hole_2 将策略梯度字典展平为向量
        # 注意与 policy_network.grad_log_prob 的返回格式保持一致
        pass

    def update(self, buffers: List[TrajectoryBuffer]) -> Dict:
        """
        执行一次策略与值函数更新.
        """
        # 合并所有轨迹数据
        all_states = []
        all_actions = []
        all_rewards = []
        all_advantages = []
        all_returns = []

        for buf in buffers:
            rewards = buf.rewards
            states = buf.states
            actions = buf.actions

            # 计算值函数估计
            values = [self.value_fn.predict(s) for s in states]
            # 添加终止状态的值估计
            if len(buf.next_states) > 0:
                values.append(self.value_fn.predict(buf.next_states[-1]))
            else:
                values.append(0.0)

            # 折扣回报
            returns = compute_discounted_returns(rewards, self.gamma)

            # GAE
            advantages = generalized_advantage_estimate(
                rewards, values, self.gamma, self.lam
            )

            all_states.extend(states)
            all_actions.extend(actions)
            all_rewards.extend(rewards)
            all_advantages.extend(advantages)
            all_returns.extend(returns)

        if len(all_states) == 0:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'mean_reward': 0.0}

        # 标准化优势
        advs = np.array(all_advantages, dtype=float)
        if len(advs) > 1 and np.std(advs) > 1.0e-8:
            advs = (advs - np.mean(advs)) / (np.std(advs) + 1.0e-8)

        # ---------------- 策略梯度 ----------------
        theta = self.policy.get_params()
        grad = np.zeros_like(theta)
        entropy_grad = np.zeros_like(theta)

        for s, a, adv in zip(all_states, all_actions, advs):
            g = self._flatten_policy_grad(s, a)
            grad += adv * g
            # 熵奖励梯度 (鼓励探索)
            # 近似: ∇ H ≈ - E[ ∇ log π · log π ]
            logp = self.policy.log_prob(s, a)
            entropy_grad += -self.entropy_coef * logp * g

        grad = grad / len(all_states) + entropy_grad / len(all_states)

        # 梯度截断
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1.0:
            grad = grad / grad_norm

        # 自然梯度更新
        def pg_func(s, a):
            return self._flatten_policy_grad(s, a)

        if not np.all(np.isfinite(grad)):
            print("WARNING: NaN/Inf in policy gradient, skipping update")
        else:
            theta_new = self.ng_optimizer.step(
                theta, grad, all_states, all_actions, pg_func
            )
            if np.all(np.isfinite(theta_new)):
                self.policy.set_params(theta_new)
            else:
                print("WARNING: NaN/Inf in new parameters, keeping old")

        # ---------------- 值函数更新 ----------------
        self.value_fn.fit(all_states, all_returns, reg=1.0e-2, method='normal')
        # 数值稳定性: 截断值函数权重
        if self.value_fn.w is not None:
            self.value_fn.w = np.clip(self.value_fn.w, -100.0, 100.0)

        # 计算损失指标
        policy_loss = -np.mean(advs)
        value_preds = self.value_fn.predict_batch(np.array(all_states))
        value_loss = float(np.mean((value_preds - np.array(all_returns)) ** 2))
        mean_reward = float(np.mean([np.sum(b.rewards) for b in buffers]))

        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'mean_reward': mean_reward,
            'grad_norm': grad_norm,
            'num_samples': len(all_states)
        }

    def train(self, num_iterations: int = 100) -> Dict:
        """
        完整训练循环.
        """
        for it in range(num_iterations):
            buffers = []
            episode_rewards = []
            for _ in range(self.batch_episodes):
                buf = self.collect_episode(deterministic=False)
                buffers.append(buf)
                episode_rewards.append(np.sum(buf.rewards))

            metrics = self.update(buffers)
            self.episode_rewards.extend(episode_rewards)
            self.episode_lengths.extend([b.size() for b in buffers])

            lr = self.lr_scheduler.step()
            self.ng_optimizer.lr = lr

            if (it + 1) % 10 == 0 or it == 0:
                avg_reward = np.mean(episode_rewards)
                print(f"Iter {it+1}/{num_iterations} | AvgReward: {avg_reward:.3f} | "
                      f"PolicyLoss: {metrics['policy_loss']:.4f} | "
                      f"ValueLoss: {metrics['value_loss']:.4f} | "
                      f"GradNorm: {metrics['grad_norm']:.4f} | "
                      f"LR: {lr:.6f}")

        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'final_policy_params': self.policy.get_params()
        }
