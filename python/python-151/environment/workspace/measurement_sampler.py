
import numpy as np
from typing import Tuple, Optional


def chebyshev2_sample(n: int) -> Tuple[np.ndarray, int]:
    pdfmax = 2.0 / np.pi
    samples = np.zeros(n)
    i = 0
    n_trials = 0
    while i < n:
        x = -1.0 + 2.0 * np.random.rand()
        y = pdfmax * np.random.rand()
        z = (2.0 / np.pi) * np.sqrt(max(1.0 - x ** 2, 0.0))
        n_trials += 1
        if y <= z:
            samples[i] = x
            i += 1
    return samples, n_trials


def cvt_density_sample(n: int, alpha: float = 1.0 / 6.0) -> np.ndarray:
    samples = np.zeros(n)
    i = 0

    pdfmax = 1.0 / np.sqrt(np.pi) / (1.0 ** alpha)
    while i < n:
        x = 2.0 * np.random.rand() - 1.0
        y = pdfmax * np.random.rand()
        z = 1.0 / np.sqrt(np.pi) / (max(1.0 - x ** 2, 1e-10) ** alpha)
        if y <= z:
            samples[i] = x
            i += 1
    return samples


class QuantumMeasurementSampler:
    def __init__(self, n_qubits: int, n_shots: int = 8192):
        self.n_qubits = n_qubits
        self.n_shots = n_shots
        self.dim = 2 ** n_qubits

    def sample_pauli_expectation(self, statevector: np.ndarray,
                                  pauli_string: str) -> float:
        statevector = np.asarray(statevector, dtype=complex)
        if statevector.shape[0] != self.dim:
            raise ValueError("态向量维度不匹配")


        psi = statevector.copy()
        for q, p in enumerate(pauli_string):
            if p == 'X':
                psi = self._apply_hadamard(q, psi)
            elif p == 'Y':
                psi = self._apply_hy(q, psi)


        probs = np.abs(psi) ** 2

        shots = np.random.multinomial(self.n_shots, probs)

        expectation = 0.0
        for outcome in range(self.dim):

            eigenval = 1.0
            for q, p in enumerate(pauli_string):
                if p == 'I':
                    continue
                bit = (outcome >> q) & 1
                eigenval *= (-1.0) ** bit
            expectation += eigenval * shots[outcome]
        expectation /= self.n_shots
        return float(expectation)

    def _apply_hadamard(self, q: int, psi: np.ndarray) -> np.ndarray:
        dim = psi.shape[0]
        psi_out = np.zeros(dim, dtype=complex)
        stride = 2 ** q
        for i in range(dim):
            partner = i ^ stride
            if i <= partner:
                psi_out[i] = (psi[i] + psi[partner]) / np.sqrt(2)
                psi_out[partner] = (psi[i] - psi[partner]) / np.sqrt(2)
        return psi_out

    def _apply_hy(self, q: int, psi: np.ndarray) -> np.ndarray:
        dim = psi.shape[0]
        psi_out = np.zeros(dim, dtype=complex)
        stride = 2 ** q
        for i in range(dim):
            partner = i ^ stride
            if i <= partner:

                psi_out[i] = (psi[i] + 1j * psi[partner]) / np.sqrt(2)
                psi_out[partner] = (psi[i] - 1j * psi[partner]) / np.sqrt(2)
        return psi_out

    def estimate_with_chebyshev_sampling(self, observable_func: callable,
                                          n_samples: int = 1000) -> Tuple[float, float]:
        samples, trials = chebyshev2_sample(n_samples)
        vals = np.array([observable_func(x) for x in samples])
        mean_est = float(np.mean(vals))
        var_est = float(np.var(vals, ddof=1))
        stderr = np.sqrt(var_est / n_samples)
        return mean_est, stderr

    def sample_bitstrings(self, statevector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        probs = np.abs(statevector) ** 2
        probs = np.clip(probs, 0, 1)
        probs /= np.sum(probs)
        shots = np.random.multinomial(self.n_shots, probs)
        mask = shots > 0
        bitstrings = np.arange(self.dim)[mask]
        frequencies = shots[mask] / self.n_shots
        return bitstrings, frequencies
