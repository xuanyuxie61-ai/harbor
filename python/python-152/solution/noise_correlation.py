"""
Correlated Pauli noise model for quantum error correction threshold analysis.

Incorporates:
- 117_brc_data: random data generation with Gaussian perturbation
- 220_correlation: stationary correlation functions, Cholesky/Eigen/FFT sampling
"""
import numpy as np
from scipy.linalg import cholesky, eigh
from scipy.fft import fft, ifft
from utils import chop_array


class CorrelatedPauliNoise:
    """
    Correlated Pauli noise model for surface codes.

    The noise is described by a spatially correlated random field over the
    qubit lattice. For each qubit i, the total error probability is drawn
    from a Gaussian random field with specified correlation structure.

    Correlation kernel (Matérn-like, from 220_correlation):
        C(r) = σ² * (2^{1-ν} / Γ(ν)) * (r√(2ν)/l)^{ν} K_{ν}(r√(2ν)/l)

    where K_ν is the modified Bessel function of the second kind.
    """

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
        """Matérn correlation function."""
        from scipy.special import kv, gamma
        l = self.correlation_length
        nu = self.nu
        if nu == float('inf') or nu > 50:
            # Gaussian limit
            return np.exp(-(r / l) ** 2 / 2)
        r = np.maximum(r, 1e-12)
        sqrt_term = np.sqrt(2 * nu) * r / l
        # For nu=0.5, this reduces to exponential
        if abs(nu - 0.5) < 1e-12:
            return np.exp(-r / l)
        prefactor = (2.0 ** (1.0 - nu)) / gamma(nu)
        corr = prefactor * (sqrt_term ** nu) * kv(nu, sqrt_term)
        # Handle r=0 limit
        corr[r < 1e-12] = 1.0
        return corr

    def _build_covariance(self):
        """Build covariance matrix for qubit error rates (1D chain embedding)."""
        coords = np.arange(self.n_qubits).reshape(-1, 1)
        dists = np.abs(coords - coords.T)
        self.covariance = (self.sigma ** 2) * self._matern_correlation(dists)
        # Ensure positive semidefinite
        w = np.linalg.eigvalsh(self.covariance)
        if np.min(w) < -1e-10:
            # Shift to PSD
            shift = abs(np.min(w)) + 1e-10
            self.covariance += shift * np.eye(self.n_qubits)

    def sample_rates_cholesky(self) -> np.ndarray:
        """
        Sample correlated error rates via Cholesky factorization.
        p_i = base_rate + z_i, where z ~ N(0, Σ).
        Clipped to [0, 1].
        """
        L = cholesky(self.covariance, lower=True, check_finite=False)
        z = self.rng.standard_normal(self.n_qubits)
        rates = self.base_rate + L @ z
        rates = np.clip(rates, 1e-6, 1.0 - 1e-6)
        return rates

    def sample_rates_eigen(self) -> np.ndarray:
        """Sample via eigenvalue decomposition."""
        w, v = eigh(self.covariance)
        w = np.maximum(w, 0)
        z = self.rng.standard_normal(self.n_qubits)
        rates = self.base_rate + v @ np.diag(np.sqrt(w)) @ z
        rates = np.clip(rates, 1e-6, 1.0 - 1e-6)
        return rates

    def sample_rates_fft(self, n_periodic: int = None) -> np.ndarray:
        """
        Sample using FFT-based circulant embedding (Dietrich & Newsam, 1997).
        Requires embedding covariance into circulant matrix.
        """
        if n_periodic is None:
            n_periodic = 2 * self.n_qubits
        # First row of circulant matrix
        c = np.zeros(n_periodic)
        for j in range(n_periodic):
            r = min(j, n_periodic - j)
            c[j] = (self.sigma ** 2) * self._matern_correlation(np.array([r]))[0]
        # Eigenvalues via FFT
        lam = np.real(fft(c))
        if np.min(lam) < -1e-10:
            # Not embeddable; fall back to Cholesky
            return self.sample_rates_cholesky()
        lam = np.maximum(lam, 0)
        z = self.rng.standard_normal(n_periodic) + 1j * self.rng.standard_normal(n_periodic)
        y = ifft(np.sqrt(lam) * z)
        rates = self.base_rate + np.real(y[:self.n_qubits])
        rates = np.clip(rates, 1e-6, 1.0 - 1e-6)
        return rates

    def sample_error_instance(self, rates: np.ndarray = None, error_type: str = "depolarizing") -> np.ndarray:
        """
        Sample a Pauli error instance given per-qubit rates.
        Returns binary error vector of length 2*n_qubits: (x_errors | z_errors).
        """
        if rates is None:
            rates = self.sample_rates_cholesky()
        x_err = np.zeros(self.n_qubits, dtype=int)
        z_err = np.zeros(self.n_qubits, dtype=int)
        for i in range(self.n_qubits):
            p = rates[i]
            if error_type == "depolarizing":
                # With prob p: uniform random Pauli {X,Y,Z}
                # With prob 1-p: I
                if self.rng.random() < p:
                    choice = self.rng.integers(1, 4)  # 1=X, 2=Y, 3=Z
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
        """
        Generate synthetic noise data analogous to 1BRC (117_brc_data).
        Returns array of shape (n_samples, n_qubits) with perturbed error rates.
        """
        # Base rates as "cities", Gaussian noise as "temperature perturbation"
        data = np.zeros((n_samples, self.n_qubits))
        for s in range(n_samples):
            z = self.rng.standard_normal(self.n_qubits)
            rates = self.base_rate + self.sigma * z
            data[s, :] = np.clip(rates, 0.0, 1.0)
        return data

    def covariance_to_correlation(self) -> np.ndarray:
        """Convert covariance matrix to correlation matrix."""
        d = np.sqrt(np.diag(self.covariance))
        Dinv = np.diag(1.0 / np.maximum(d, 1e-12))
        return Dinv @ self.covariance @ Dinv

    def exponential_correlation(self, r: np.ndarray) -> np.ndarray:
        """Exponential correlation: C(r) = exp(-r / l)."""
        return np.exp(-r / np.maximum(self.correlation_length, 1e-12))

    def gaussian_correlation(self, r: np.ndarray) -> np.ndarray:
        """Gaussian correlation: C(r) = exp(-(r/l)²/2)."""
        return np.exp(-(r / np.maximum(self.correlation_length, 1e-12)) ** 2 / 2.0)

    def rational_quadratic_correlation(self, r: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """Rational quadratic correlation."""
        return (1.0 + (r / np.maximum(self.correlation_length, 1e-12)) ** 2 / (2 * alpha)) ** (-alpha)


class NonMarkovianNoise(CorrelatedPauliNoise):
    """
    Non-Markovian noise with temporal correlations.
    The error process has memory described by a quasi-Markov kernel:
        p(e_t | e_{t-1}) = (1-λ) p_0(e_t) + λ δ_{e_t, e_{t-1}}
    where λ ∈ [0,1] is the memory parameter.
    """

    def __init__(self, n_qubits: int, base_rate: float, memory_lambda: float = 0.3, **kwargs):
        super().__init__(n_qubits, base_rate, **kwargs)
        if not (0 <= memory_lambda <= 1):
            raise ValueError("memory_lambda must be in [0,1].")
        self.memory_lambda = memory_lambda

    def sample_temporal_sequence(self, n_steps: int, error_type: str = "depolarizing") -> np.ndarray:
        """
        Sample a temporal sequence of errors with memory.
        Returns array of shape (n_steps, 2*n_qubits).
        """
        errors = np.zeros((n_steps, 2 * self.n_qubits), dtype=int)
        prev = self.sample_error_instance(error_type=error_type)
        errors[0, :] = prev
        for t in range(1, n_steps):
            if self.rng.random() < self.memory_lambda:
                # Copy previous error
                errors[t, :] = prev.copy()
            else:
                # Fresh sample
                fresh = self.sample_error_instance(error_type=error_type)
                errors[t, :] = fresh
                prev = fresh
        return errors
