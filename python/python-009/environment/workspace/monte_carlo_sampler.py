
import numpy as np
from typing import Callable, Tuple, Optional, List


class SimplexSampler:

    @staticmethod
    def sample_unit_simplex(m: int, n: int, seed: Optional[int] = None) -> np.ndarray:
        if m <= 0 or n <= 0:
            raise ValueError("维度和样本数必须为正")
        if seed is not None:
            np.random.seed(seed)


        y = np.random.exponential(scale=1.0, size=(n, m))
        s = y.sum(axis=1, keepdims=True)
        s = np.maximum(s, 1e-30)
        return y / s

    @staticmethod
    def sample_general_simplex(vertices: np.ndarray, n: int,
                                seed: Optional[int] = None) -> np.ndarray:
        vertices = np.asarray(vertices, dtype=np.float64)
        m, n_vert = vertices.shape
        if n_vert != m + 1:
            raise ValueError(f"M维单纯形应有 M+1 个顶点，得到 {n_vert} 个顶点")

        u = SimplexSampler.sample_unit_simplex(m + 1, n, seed)
        return u @ vertices.T

    @staticmethod
    def simplex_volume(vertices: np.ndarray) -> float:
        vertices = np.asarray(vertices, dtype=np.float64)
        m = vertices.shape[0]
        M = vertices[:, 1:] - vertices[:, 0:1]
        return abs(np.linalg.det(M)) / np.math.factorial(m)


class PinkNoiseGenerator:

    def __init__(self, beta: float = 1.0):
        if beta < 0 or beta > 2:
            raise ValueError("β 应在 [0, 2] 范围内")
        self.beta = beta

    def generate(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)

        white = np.random.normal(0.0, 1.0, size=n)
        spectrum = np.fft.rfft(white)
        freq = np.fft.rfftfreq(n)


        freq[0] = freq[1] if len(freq) > 1 else 1e-10


        filter_resp = freq**(-self.beta / 2.0)
        filter_resp[0] = 0.0

        spectrum *= filter_resp
        pink = np.fft.irfft(spectrum, n=n)


        std = np.std(pink)
        if std > 1e-15:
            pink = pink / std
        return pink


class MetropolisHastingsSampler:

    def __init__(self, log_posterior: Callable[[np.ndarray], float],
                 proposal_cov: np.ndarray,
                 bounds: Optional[List[Tuple[float, float]]] = None):
        self.log_posterior = log_posterior
        self.proposal_cov = np.asarray(proposal_cov, dtype=np.float64)
        self.bounds = bounds
        self.dim = self.proposal_cov.shape[0]

        if self.proposal_cov.shape != (self.dim, self.dim):
            raise ValueError("提议协方差矩阵必须为方阵")


        try:
            self.L = np.linalg.cholesky(self.proposal_cov + 1e-12 * np.eye(self.dim))
        except np.linalg.LinAlgError:

            self.L = np.diag(np.sqrt(np.maximum(np.diag(self.proposal_cov), 1e-12)))

    def _propose(self, current: np.ndarray) -> np.ndarray:
        noise = self.L @ np.random.normal(0.0, 1.0, size=self.dim)
        candidate = current + noise

        if self.bounds is not None:
            for i, (low, high) in enumerate(self.bounds):
                candidate[i] = np.clip(candidate[i], low, high)
        return candidate

    def sample(self, x0: np.ndarray, n_samples: int,
                burn_in: int = 1000, thin: int = 10,
                seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        if seed is not None:
            np.random.seed(seed)

        x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
        if x0.shape[0] != self.dim:
            raise ValueError(f"初始参数维度 {x0.shape[0]} 与提议分布维度 {self.dim} 不匹配")

        total_steps = burn_in + n_samples * thin
        current = x0.copy()
        current_logp = self.log_posterior(current)

        samples = np.zeros((n_samples, self.dim), dtype=np.float64)
        log_probs = np.zeros(n_samples, dtype=np.float64)

        accepted = 0
        sample_idx = 0

        for step in range(total_steps):
            candidate = self._propose(current)
            cand_logp = self.log_posterior(candidate)


            log_alpha = cand_logp - current_logp

            alpha = np.exp(min(log_alpha, 0.0))

            if np.random.uniform() < alpha:
                current = candidate
                current_logp = cand_logp
                accepted += 1


            if step >= burn_in and (step - burn_in) % thin == 0:
                samples[sample_idx] = current
                log_probs[sample_idx] = current_logp
                sample_idx += 1

        acceptance_rate = accepted / total_steps
        return samples, log_probs, acceptance_rate


class NestedSampler:

    def __init__(self, log_likelihood: Callable[[np.ndarray], float],
                 prior_transform: Callable[[np.ndarray], np.ndarray],
                 n_live: int = 100):
        self.log_likelihood = log_likelihood
        self.prior_transform = prior_transform
        self.n_live = n_live

    def run(self, dim: int, max_iter: int = 10000,
            log_z_tol: float = 0.1, seed: Optional[int] = None) -> dict:
        if seed is not None:
            np.random.seed(seed)


        live_u = np.random.uniform(0.0, 1.0, size=(self.n_live, dim))
        live_v = np.array([self.prior_transform(u) for u in live_u])
        live_logl = np.array([self.log_likelihood(v) for v in live_v])

        logZ = -1e300
        samples = []
        logls = []
        logws = []

        logX = 0.0

        for it in range(max_iter):

            min_idx = np.argmin(live_logl)
            min_logl = live_logl[min_idx]


            log_dX = -it / self.n_live - np.log(self.n_live)
            log_dZ = min_logl + log_dX


            logZ = np.logaddexp(logZ, log_dZ)

            samples.append(live_v[min_idx].copy())
            logls.append(min_logl)
            logws.append(log_dZ)


            other_idx = np.random.choice([i for i in range(self.n_live) if i != min_idx])
            proposal = live_v[other_idx].copy()


            max_attempts = 1000
            new_point = None
            for _ in range(max_attempts):

                scale = 0.1
                trial_u = np.clip(np.random.normal(0.5, scale, size=dim), 0.0, 1.0)
                trial_v = self.prior_transform(trial_u)
                trial_logl = self.log_likelihood(trial_v)
                if trial_logl > min_logl:
                    new_point = trial_v
                    new_logl = trial_logl
                    break

            if new_point is None:

                new_point = live_v[np.argmax(live_logl)].copy()
                new_logl = np.max(live_logl)

            live_v[min_idx] = new_point
            live_logl[min_idx] = new_logl


            logLmax = np.max(live_logl)
            log_remainder = logLmax + logX - it / self.n_live
            if log_remainder < np.log(log_z_tol) + logZ:
                break

        return {
            'logZ': float(logZ),
            'samples': np.array(samples),
            'log_likelihoods': np.array(logls),
            'log_weights': np.array(logws)
        }
