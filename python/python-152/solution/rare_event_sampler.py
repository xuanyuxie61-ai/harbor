"""
Rare event sampling and importance sampling for logical error estimation.

Incorporates:
- 725_matlab_map: rejection sampling (inpolygon / bounding box)
- 113_box_distance: distance metrics for importance regions
- 140_caustic: caustic patterns for degenerate syndrome analysis
"""
import numpy as np
from scipy.special import erfc


class RejectionSampler:
    """
    Rejection sampling for logical error events above threshold.
    Analogous to 725_matlab_map (rejection sampling inside Florida polygon).
    """

    def __init__(self, code_distance: int, threshold: float, rng=None):
        self.code_distance = code_distance
        self.threshold = threshold
        self.rng = rng or np.random.default_rng()

    def _in_logical_error_region(self, p: float) -> bool:
        """
        Indicator for being in the rare logical error region.
        For p > p_th, logical errors are non-negligible.
        """
        return p > self.threshold

    def sample_above_threshold(self, base_rate: float, n_samples: int,
                               proposal_sigma: float = 0.05) -> np.ndarray:
        """
        Sample error rates conditioned on being above threshold
        using rejection sampling with Gaussian proposal.
        """
        samples = []
        max_attempts = n_samples * 1000
        attempts = 0
        while len(samples) < n_samples and attempts < max_attempts:
            attempts += 1
            p = self.rng.normal(base_rate, proposal_sigma)
            if 0 <= p <= 1 and self._in_logical_error_region(p):
                # Accept with probability proportional to distance from threshold
                # (importance weighting)
                weight = (p - self.threshold) / (1.0 - self.threshold)
                if self.rng.random() < weight:
                    samples.append(p)
        if len(samples) < n_samples:
            # Fallback: sample uniformly above threshold
            samples = list(self.rng.uniform(self.threshold, 1.0, n_samples))
        return np.array(samples)

    def estimate_logical_rate_importance(self, p_func, n_samples: int = 10000) -> float:
        """
        Estimate logical error rate using importance sampling:
            P_L = E_q[ P_L(p) * w(p) ]
        where w(p) = f(p)/q(p) is the importance weight.
        """
        samples = self.sample_above_threshold(0.0, n_samples)
        rates = np.array([p_func(s) for s in samples])
        # Uniform proposal above threshold -> weight = 1 / (1 - threshold)
        weights = 1.0 / max(1.0 - self.threshold, 1e-12)
        return np.mean(rates) * weights * (1.0 - self.threshold)


class CausticDegeneracyAnalyzer:
    """
    Analyze degenerate error syndromes using caustic-like interference patterns.
    Adapted from 140_caustic (caustic patterns in a circle).

    In quantum error correction, degenerate errors produce the same syndrome.
    The "caustic" pattern emerges when plotting syndrome manifolds in
    parameter space.
    """

    def __init__(self, n_points: int = 1000):
        self.n_points = n_points

    def caustic_syndrome_pattern(self, n_defects: int, multiplier: int = 3) -> tuple:
        """
        Generate caustic-like pattern from syndrome connections.
        Points on circle = possible syndrome values,
        lines connect syndrome i to syndrome (i*m mod n).
        """
        theta = np.linspace(0, 2 * np.pi, n_defects, endpoint=False)
        x = np.cos(theta)
        y = np.sin(theta)
        lines = []
        for j in range(n_defects):
            k = (j * multiplier) % n_defects
            lines.append(((x[j], y[j]), (x[k], y[k])))
        return x, y, lines

    def degeneracy_interference(self, error_weights: np.ndarray, n_qubits: int) -> np.ndarray:
        """
        Compute interference pattern from competing degenerate errors:
            I(θ) = | Σ_k w_k exp(i k θ) |²
        where w_k are weights of degenerate error configurations.
        """
        theta = np.linspace(0, 2 * np.pi, self.n_points)
        interference = np.zeros_like(theta)
        for i, th in enumerate(theta):
            val = np.sum(error_weights * np.exp(1j * np.arange(len(error_weights)) * th))
            interference[i] = np.abs(val) ** 2
        return theta, interference

    def box_distance_importance_region(self, p_center: float, box_size: float) -> tuple:
        """
        Define importance sampling region as a box in parameter space.
        From 113_box_distance: probability of falling in region.
        """
        a = max(p_center - box_size / 2, 0.0)
        b = min(p_center + box_size / 2, 1.0)
        volume = b - a
        # Mean distance in box = (b+a)/2 for uniform distribution
        mean_p = (a + b) / 2.0
        variance = ((b - a) ** 2) / 12.0
        return {
            "lower": a,
            "upper": b,
            "volume": volume,
            "mean": mean_p,
            "variance": variance
        }


class MonteCarloLogicalError:
    """
    Monte Carlo estimation of logical error rate P_L(p).
    """

    def __init__(self, code, decoder, rng=None):
        self.code = code
        self.decoder = decoder
        self.rng = rng or np.random.default_rng()

    def estimate(self, p: float, n_shots: int = 10000, error_type: str = "depolarizing") -> dict:
        """
        Estimate logical error rate at physical error probability p.
        """
        n_logical_errors = 0
        n_qubits = self.code.n_qubits
        for _ in range(n_shots):
            # Sample error
            if error_type == "depolarizing":
                x_err = (self.rng.random(n_qubits) < p).astype(int)
                z_err = (self.rng.random(n_qubits) < p).astype(int)
                # Y errors where both X and Z occur
                y_mask = (x_err & z_err)
                e_vec = np.concatenate([x_err, z_err])
            elif error_type == "bitflip":
                x_err = (self.rng.random(n_qubits) < p).astype(int)
                e_vec = np.concatenate([x_err, np.zeros(n_qubits, dtype=int)])
            elif error_type == "phaseflip":
                z_err = (self.rng.random(n_qubits) < p).astype(int)
                e_vec = np.concatenate([np.zeros(n_qubits, dtype=int), z_err])
            else:
                raise ValueError(f"Unknown error_type: {error_type}")

            syndrome = self.code.syndrome_of_error(e_vec)
            # Decode
            if hasattr(self.decoder, 'decode'):
                recovery = self.decoder.decode(syndrome)
            else:
                recovery = np.zeros(2 * n_qubits, dtype=int)
            # Check if recovery · error is trivial
            combined = (recovery + e_vec) % 2
            logical_ind = self.code.logical_error_indicator(recovery, e_vec)
            if np.any(logical_ind):
                n_logical_errors += 1

        P_L = n_logical_errors / n_shots
        variance = P_L * (1.0 - P_L) / n_shots
        return {
            "P_L": P_L,
            "variance": variance,
            "std_error": np.sqrt(variance),
            "n_shots": n_shots
        }

    def estimate_correlated(self, noise_model, n_shots: int = 5000) -> dict:
        """
        Estimate logical error rate under correlated noise.
        """
        n_logical_errors = 0
        n_qubits = self.code.n_qubits
        for _ in range(n_shots):
            rates = noise_model.sample_rates_cholesky()
            e_vec = noise_model.sample_error_instance(rates=rates)
            syndrome = self.code.syndrome_of_error(e_vec)
            if hasattr(self.decoder, 'decode'):
                recovery = self.decoder.decode(syndrome)
            else:
                recovery = np.zeros(2 * n_qubits, dtype=int)
            logical_ind = self.code.logical_error_indicator(recovery, e_vec)
            if np.any(logical_ind):
                n_logical_errors += 1

        P_L = n_logical_errors / n_shots
        return {
            "P_L": P_L,
            "std_error": np.sqrt(P_L * (1.0 - P_L) / n_shots),
            "n_shots": n_shots
        }
