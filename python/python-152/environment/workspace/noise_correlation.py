import numpy as np
from scipy.linalg import cholesky, eigh
from scipy.fft import fft, ifft
from utils import chop_array


class CorrelatedPauliNoise:

    def __init__(self, n_qubits: int, base_rate: float, sigma: float = 0.1,
                 correlation_length: float = 1.0, nu: float = 0.5, rng=None):
        if n_qubits <= 0:
            raise ValueError("n_qubits must be positive.")
        if not (0 <= base_rate <= 1):
            raise ValueError("base_rate must be in [0,1].")
        self.n_qubits = n_qubits
        self.base_rate = base_rate
        self.sigma = sigma
        self.correlation_length = correlation_length
        self.nu = nu
        self.rng = rng or np.random.default_rng()
        self.covariance = None
        self._build_covariance()

    def _matern_correlation(self, r: np.ndarray) -> np.ndarray:
        from scipy.special import kv, gamma
        l = self.correlation_length
        nu = self.nu
        if nu == float('inf') or nu > 50:

            return np.exp(-(r / l) ** 2 / 2)
        r = np.maximum(r, 1e-12)
        sqrt_term = np.sqrt(2 * nu) * r / l

        if abs(nu - 0.5) < 1e-12:
            return np.exp(-r / l)
        prefactor = (2.0 ** (1.0 - nu)) / gamma(nu)
        corr = prefactor * (sqrt_term ** nu) * kv(nu, sqrt_term)

        corr[r < 1e-12] = 1.0
        return corr

    def _build_covariance(self):
        coords = np.arange(self.n_qubits).reshape(-1, 1)
        dists = np.abs(coords - coords.T)
        self.covariance = (self.sigma ** 2) * self._matern_correlation(dists)

        w = np.linalg.eigvalsh(self.covariance)
        if np.min(w) < -1e-10:

            shift = abs(np.min(w)) + 1e-10
            self.covariance += shift * np.eye(self.n_qubits)

    def sample_rates_cholesky(self) -> np.ndarray:
        L = cholesky(self.covariance, lower=True, check_finite=False)
        z = self.rng.standard_normal(self.n_qubits)
        rates = self.base_rate + L @ z
        rates = np.clip(rates, 1e-6, 1.0 - 1e-6)
        return rates

    def sample_rates_eigen(self) -> np.ndarray:
        w, v = eigh(self.covariance)
        w = np.maximum(w, 0)
        z = self.rng.standard_normal(self.n_qubits)
        rates = self.base_rate + v @ np.diag(np.sqrt(w)) @ z
        rates = np.clip(rates, 1e-6, 1.0 - 1e-6)
        return rates

    def sample_rates_fft(self, n_periodic: int = None) -> np.ndarray:
        if n_periodic is None:
            n_periodic = 2 * self.n_qubits

        c = np.zeros(n_periodic)
        for j in range(n_periodic):
            r = min(j, n_periodic - j)
            c[j] = (self.sigma ** 2) * self._matern_correlation(np.array([r]))[0]

        lam = np.real(fft(c))
        if np.min(lam) < -1e-10:

            return self.sample_rates_cholesky()
        lam = np.maximum(lam, 0)
        z = self.rng.standard_normal(n_periodic) + 1j * self.rng.standard_normal(n_periodic)
        y = ifft(np.sqrt(lam) * z)
        rates = self.base_rate + np.real(y[:self.n_qubits])
        rates = np.clip(rates, 1e-6, 1.0 - 1e-6)
        return rates

    def sample_error_instance(self, rates: np.ndarray = None, error_type: str = "depolarizing") -> np.ndarray:
        if rates is None:
            rates = self.sample_rates_cholesky()
        x_err = np.zeros(self.n_qubits, dtype=int)
        z_err = np.zeros(self.n_qubits, dtype=int)
        for i in range(self.n_qubits):
            p = rates[i]
            if error_type == "depolarizing":


                if self.rng.random() < p:
                    choice = self.rng.integers(1, 4)
                    if choice == 1:
                        x_err[i] = 1
                    elif choice == 2:
                        x_err[i] = 1
                        z_err[i] = 1
                    else:
                        z_err[i] = 1
            elif error_type == "bitflip":
                if self.rng.random() < p:
                    x_err[i] = 1
            elif error_type == "phaseflip":
                if self.rng.random() < p:
                    z_err[i] = 1
            else:
                raise ValueError(f"Unknown error_type: {error_type}")
        return np.concatenate([x_err, z_err])

    def generate_brc_like_data(self, n_samples: int = 1000) -> np.ndarray:

        data = np.zeros((n_samples, self.n_qubits))
        for s in range(n_samples):
            z = self.rng.standard_normal(self.n_qubits)
            rates = self.base_rate + self.sigma * z
            data[s, :] = np.clip(rates, 0.0, 1.0)
        return data

    def covariance_to_correlation(self) -> np.ndarray:
        d = np.sqrt(np.diag(self.covariance))
        Dinv = np.diag(1.0 / np.maximum(d, 1e-12))
        return Dinv @ self.covariance @ Dinv

    def exponential_correlation(self, r: np.ndarray) -> np.ndarray:
        return np.exp(-r / np.maximum(self.correlation_length, 1e-12))

    def gaussian_correlation(self, r: np.ndarray) -> np.ndarray:
        return np.exp(-(r / np.maximum(self.correlation_length, 1e-12)) ** 2 / 2.0)

    def rational_quadratic_correlation(self, r: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        return (1.0 + (r / np.maximum(self.correlation_length, 1e-12)) ** 2 / (2 * alpha)) ** (-alpha)


class NonMarkovianNoise(CorrelatedPauliNoise):

    def __init__(self, n_qubits: int, base_rate: float, memory_lambda: float = 0.3, **kwargs):
        super().__init__(n_qubits, base_rate, **kwargs)
        if not (0 <= memory_lambda <= 1):
            raise ValueError("memory_lambda must be in [0,1].")
        self.memory_lambda = memory_lambda

    def sample_temporal_sequence(self, n_steps: int, error_type: str = "depolarizing") -> np.ndarray:
        errors = np.zeros((n_steps, 2 * self.n_qubits), dtype=int)
        prev = self.sample_error_instance(error_type=error_type)
        errors[0, :] = prev
        for t in range(1, n_steps):
            if self.rng.random() < self.memory_lambda:

                errors[t, :] = prev.copy()
            else:

                fresh = self.sample_error_instance(error_type=error_type)
                errors[t, :] = fresh
                prev = fresh
        return errors
