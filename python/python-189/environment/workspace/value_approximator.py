
import numpy as np
from typing import List, Tuple
from spectral_basis import pca_vectors, pca_transform, build_legendre_basis
from linear_algebra import rref_solve


class PCAStateRepresentation:

    def __init__(self, original_dim: int, reduced_dim: int):
        self.original_dim = original_dim
        self.reduced_dim = reduced_dim
        self.V = None
        self.Psi = None
        self._fitted = False

    def fit(self, observations: np.ndarray):
        observations = np.asarray(observations, dtype=float)
        if observations.ndim == 1:
            observations = observations.reshape(-1, 1)
        self.V, self.eigvals, self.Psi = pca_vectors(
            observations.T, self.reduced_dim
        )
        self._fitted = True

    def transform(self, observation: np.ndarray) -> np.ndarray:
        if not self._fitted:

            obs = np.asarray(observation, dtype=float)
            if len(obs) > self.reduced_dim:
                return obs[:self.reduced_dim]
            return obs
        return pca_transform(observation, self.V, self.Psi)


class SpectralValueFunction:

    def __init__(self, state_dim: int, max_degree: int = 3, gamma: float = 0.99):
        self.state_dim = state_dim
        self.max_degree = max_degree
        self.gamma = gamma
        self.w = None
        self.basis_indices = None
        self.num_basis = 0
        self._build_indices()

    def _build_indices(self):
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
        from spectral_basis import legendre_polynomial_1d
        z = np.tanh(state / 2.0)
        phi = np.ones(self.num_basis)
        for k, alpha in enumerate(self.basis_indices):
            for i in range(min(self.state_dim, len(z))):
                phi[k] *= legendre_polynomial_1d(alpha[i], np.array([z[i]]))[0]
        return phi

    def fit(self, states: List[np.ndarray], returns: List[float],
            reg: float = 1.0e-4, method: str = 'rref'):
        n = len(states)
        if n == 0:
            self.w = np.zeros(self.num_basis)
            return
        Phi = np.zeros((n, self.num_basis))
        for i, s in enumerate(states):
            Phi[i, :] = self._phi(s)
        G = np.array(returns, dtype=float)

        if method == 'normal':

            A = Phi.T @ Phi + reg * np.eye(self.num_basis)
            b = Phi.T @ G
            try:
                self.w = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                self.w = np.linalg.lstsq(A, b, rcond=None)[0]
        else:

            A = Phi.T @ Phi + reg * np.eye(self.num_basis)
            b = Phi.T @ G
            w = rref_solve(A, b.reshape(-1, 1))
            self.w = w.flatten()

            if not np.all(np.isfinite(self.w)):
                self.w = np.linalg.lstsq(A, b, rcond=None)[0]

    def predict(self, state: np.ndarray) -> float:
        if self.w is None:
            return 0.0
        phi = self._phi(state)
        return float(phi @ self.w)

    def predict_batch(self, states: np.ndarray) -> np.ndarray:
        if self.w is None:
            return np.zeros(len(states))
        if states.ndim == 1:
            states = states.reshape(1, -1)
        vals = np.zeros(states.shape[0])
        for i in range(states.shape[0]):
            vals[i] = self.predict(states[i])
        return vals


def compute_discounted_returns(rewards: List[float], gamma: float) -> List[float]:
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def generalized_advantage_estimate(rewards: List[float], values: List[float],
                                    gamma: float = 0.99, lam: float = 0.95) -> List[float]:
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
