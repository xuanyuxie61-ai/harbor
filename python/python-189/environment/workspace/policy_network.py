
import numpy as np
from typing import Tuple
from linear_algebra import sample_from_toeplitz_covariance
from special_functions import beta_cdf


class SpectralPolicyNetwork:

    def __init__(self, state_dim: int, action_dim: int,
                 max_degree: int = 3, action_bounds: Tuple[float, float] = (-2.0, 2.0)):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_degree = max_degree
        self.action_bounds = action_bounds
        self.state_scale = 2.0


        self.basis_indices = self._build_indices(state_dim, max_degree)
        self.num_basis = len(self.basis_indices)



        self.theta_mean = np.random.randn(action_dim, self.num_basis) * 0.1
        self.theta_log_std = np.zeros(action_dim) - 0.5
        self.min_log_std = -3.0
        self.max_log_std = 1.0


        self.cov_first_col = np.ones(action_dim)
        self.use_toeplitz_cov = (action_dim > 1)

        self._param_flat = self._flatten_params()

    def _build_indices(self, m: int, max_degree: int) -> list:
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
        return np.tanh(state / self.state_scale)

    def _evaluate_basis(self, state: np.ndarray) -> np.ndarray:
        from spectral_basis import legendre_polynomial_1d
        z = self._state_transform(state)
        phi = np.ones(self.num_basis)
        for k, alpha in enumerate(self.basis_indices):
            for i in range(self.state_dim):
                phi[k] *= legendre_polynomial_1d(alpha[i], np.array([z[i]]))[0]
        return phi

    def mean(self, state: np.ndarray) -> np.ndarray:
        phi = self._evaluate_basis(state)
        return self.theta_mean @ phi

    def std(self) -> np.ndarray:
        return np.exp(np.clip(self.theta_log_std, self.min_log_std, self.max_log_std))

    def sample(self, state: np.ndarray) -> np.ndarray:
        mu = self.mean(state)
        sigma = self.std()
        if self.use_toeplitz_cov and self.action_dim > 1:

            try:
                noise = sample_from_toeplitz_covariance(self.action_dim,
                                                         sigma * self.cov_first_col)
            except Exception:
                noise = np.random.randn(self.action_dim) * sigma
        else:
            noise = np.random.randn(self.action_dim) * sigma
        action = mu + noise

        action = np.clip(action, self.action_bounds[0], self.action_bounds[1])
        return action

    def log_prob(self, state: np.ndarray, action: np.ndarray) -> float:
        mu = self.mean(state)
        sigma = self.std()

        diff = action - mu
        logp = -0.5 * np.sum((diff / sigma) ** 2) \
               - np.sum(np.log(sigma)) \
               - 0.5 * self.action_dim * np.log(2.0 * np.pi)
        return float(logp)

    def grad_log_prob(self, state: np.ndarray, action: np.ndarray) -> dict:


        pass

    def _flatten_params(self) -> np.ndarray:
        return np.concatenate([
            self.theta_mean.flatten(),
            self.theta_log_std.flatten()
        ])

    def _set_flat_params(self, flat: np.ndarray):
        n_mean = self.action_dim * self.num_basis
        self.theta_mean = flat[:n_mean].reshape(self.action_dim, self.num_basis)
        self.theta_log_std = flat[n_mean:]

    def get_params(self) -> np.ndarray:
        return self._flatten_params()

    def set_params(self, flat: np.ndarray):
        self._set_flat_params(flat)

        n_mean = self.action_dim * self.num_basis
        self.theta_log_std = np.clip(self.theta_log_std, self.min_log_std, self.max_log_std)
        self._param_flat = self._flatten_params()

    def num_params(self) -> int:
        return self.action_dim * self.num_basis + self.action_dim
