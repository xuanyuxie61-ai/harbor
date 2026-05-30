
import numpy as np
from typing import Callable, Optional, Tuple, List


class RBFValueFunction:

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


        self.w = rng.normal(0, 0.1, n_centers + 1)

    def features(self, s: np.ndarray) -> np.ndarray:
        s = np.atleast_1d(s)
        phi = np.exp(-np.sum((self.centers - s) ** 2, axis=1) / (2 * self.sigma ** 2))
        return np.concatenate(([1.0], phi))

    def value(self, s: np.ndarray) -> float:
        phi = self.features(s)
        return float(np.dot(self.w, phi))

    def gradient(self, s: np.ndarray) -> np.ndarray:
        return self.features(s)

    def update(self, s: np.ndarray, target: float, alpha: float = 0.1) -> float:
        phi = self.features(s)
        v = np.dot(self.w, phi)
        delta = target - v
        self.w += alpha * delta * phi
        return float(delta)


class GaussianPolicy:

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
        z = float(np.dot(self.theta, phi_s))

        z = np.clip(z, -10.0, 10.0)
        return self.action_bound * np.tanh(z)

    def sample(self, phi_s: np.ndarray, rng: Optional[np.random.Generator] = None) -> float:
        if rng is None:
            rng = np.random.default_rng()
        mu = self.mean(phi_s)
        a = rng.normal(mu, self.sigma)
        return float(np.clip(a, -self.action_bound, self.action_bound))

    def log_prob_gradient(self, phi_s: np.ndarray, a: float) -> np.ndarray:
        z = float(np.dot(self.theta, phi_s))
        z = np.clip(z, -10.0, 10.0)
        mu = self.action_bound * np.tanh(z)

        dtanh = 1.0 - np.tanh(z) ** 2
        dmu_dtheta = self.action_bound * dtanh * phi_s

        grad = (a - mu) / (self.sigma ** 2) * dmu_dtheta
        return grad

    def update(self, phi_s: np.ndarray, a: float, advantage: float, alpha: float = 0.01) -> None:
        grad = self.log_prob_gradient(phi_s, a)
        self.theta += alpha * advantage * grad

        norm = np.linalg.norm(self.theta)
        if norm > 100.0:
            self.theta = self.theta * 100.0 / norm


class ActorCriticAgent:

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


        self.critic = RBFValueFunction(
            n_centers=n_rbf,
            state_dim=state_dim,
            sigma=0.15,
            bounds=(np.zeros(state_dim), np.ones(state_dim)),
            rng=rng,
        )


        self.actor = GaussianPolicy(
            n_features=n_rbf + 1,
            action_bound=action_bound,
            sigma=policy_sigma,
            rng=rng,
        )

        self.episode_rewards: List[float] = []

    def select_action(self, state: np.ndarray) -> float:
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
        phi_s = self.critic.features(state)
        v_s = self.critic.value(state)

        if done:
            v_next = 0.0
        else:
            v_next = self.critic.value(next_state)

        td_target = reward + self.gamma * v_next
        td_error = td_target - v_s


        self.critic.w += self.alpha_critic * td_error * phi_s


        advantage = td_error
        grad = self.actor.log_prob_gradient(phi_s, action)
        self.actor.theta += self.alpha_actor * advantage * grad


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
        state = state0.copy()
        total_reward = 0.0

        for _ in range(max_steps):
            action = self.select_action(state)
            next_state, reward, done = env_step_fn(state, action)


            next_state = np.clip(next_state, -0.1, 1.1)

            self.train_step(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state

            if done:
                break

        self.episode_rewards.append(total_reward)
        return total_reward
