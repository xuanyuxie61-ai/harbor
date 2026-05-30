
import numpy as np
from typing import List, Tuple, Dict
from dynamical_system import ControlledNonlinearOscillator
from policy_network import SpectralPolicyNetwork
from value_approximator import SpectralValueFunction, compute_discounted_returns, generalized_advantage_estimate
from natural_gradient import NaturalPolicyGradientOptimizer, fisher_vector_product
from constrained_optimizer import lp_action_projection, CosineAnnealingScheduler
from stochastic_processes import ornstein_uhlenbeck_process


class TrajectoryBuffer:

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


        self.policy = SpectralPolicyNetwork(state_dim, action_dim,
                                             max_degree=policy_degree)
        self.value_fn = SpectralValueFunction(state_dim, max_degree=value_degree,
                                               gamma=gamma)


        self.ng_optimizer = NaturalPolicyGradientOptimizer(
            learning_rate=lr_policy,
            cg_iter=cg_iter,
            cg_damping=cg_damping,
            max_kl=max_kl
        )
        self.lr_scheduler = CosineAnnealingScheduler(alpha_max=lr_policy,
                                                      alpha_min=lr_policy * 0.1,
                                                      T_period=50)


        self.env = ControlledNonlinearOscillator(dt=0.005)


        self.ou_noise = ornstein_uhlenbeck_process(1000, action_dim,
                                                    theta=0.15, sigma=0.2, dt=0.01)
        self.noise_idx = 0


        self.episode_rewards = []
        self.episode_lengths = []

    def collect_episode(self, deterministic: bool = False) -> TrajectoryBuffer:
        buffer = TrajectoryBuffer()
        obs = self.env.reset()
        for step in range(self.max_steps):

            action = self.policy.sample(obs)
            if not deterministic:

                noise = self.ou_noise[self.noise_idx % len(self.ou_noise)]
                self.noise_idx += 1
                action = action + noise
                action = np.clip(action, -2.0, 2.0)

            action = lp_action_projection(action, bounds=(-2.0, 2.0))
            next_obs, reward, done, info = self.env.step(action)
            buffer.add(obs, action, reward, next_obs, done)
            obs = next_obs
            if done:
                break
        return buffer

    def _flatten_policy_grad(self, state, action) -> np.ndarray:


        pass

    def update(self, buffers: List[TrajectoryBuffer]) -> Dict:

        all_states = []
        all_actions = []
        all_rewards = []
        all_advantages = []
        all_returns = []

        for buf in buffers:
            rewards = buf.rewards
            states = buf.states
            actions = buf.actions


            values = [self.value_fn.predict(s) for s in states]

            if len(buf.next_states) > 0:
                values.append(self.value_fn.predict(buf.next_states[-1]))
            else:
                values.append(0.0)


            returns = compute_discounted_returns(rewards, self.gamma)


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


        advs = np.array(all_advantages, dtype=float)
        if len(advs) > 1 and np.std(advs) > 1.0e-8:
            advs = (advs - np.mean(advs)) / (np.std(advs) + 1.0e-8)


        theta = self.policy.get_params()
        grad = np.zeros_like(theta)
        entropy_grad = np.zeros_like(theta)

        for s, a, adv in zip(all_states, all_actions, advs):
            g = self._flatten_policy_grad(s, a)
            grad += adv * g


            logp = self.policy.log_prob(s, a)
            entropy_grad += -self.entropy_coef * logp * g

        grad = grad / len(all_states) + entropy_grad / len(all_states)


        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1.0:
            grad = grad / grad_norm


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


        self.value_fn.fit(all_states, all_returns, reg=1.0e-2, method='normal')

        if self.value_fn.w is not None:
            self.value_fn.w = np.clip(self.value_fn.w, -100.0, 100.0)


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
