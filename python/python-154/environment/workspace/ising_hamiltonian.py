
import numpy as np
from typing import Tuple, Optional


class IsingHamiltonian:

    def __init__(self, n_spins: int, seed: int = 154):
        if n_spins <= 0:
            raise ValueError("n_spins must be positive")
        self.n_spins = n_spins
        self.rng = np.random.default_rng(seed)
        self.J: Optional[np.ndarray] = None
        self.h: Optional[np.ndarray] = None
        self.offset: float = 0.0
        self.qubo_matrix: Optional[np.ndarray] = None

    def build_from_qubo(self, Q: np.ndarray) -> None:
        if Q.shape != (self.n_spins, self.n_spins):
            raise ValueError("QUBO matrix shape mismatch")
        self.qubo_matrix = Q.astype(float)


        Qs = (Q + Q.T) / 2.0
        np.fill_diagonal(Qs, np.diag(Q))

        n = self.n_spins
        self.J = np.zeros((n, n))
        self.h = np.zeros(n)

        for i in range(n):
            for j in range(i + 1, n):
                self.J[i, j] = Qs[i, j] / 4.0
                self.J[j, i] = self.J[i, j]

        for i in range(n):
            self.h[i] = Qs[i, i] / 2.0
            for j in range(n):
                if j != i:
                    self.h[i] += Qs[i, j] / 4.0

        self.offset = 0.0
        for i in range(n):
            self.offset += Qs[i, i] / 4.0
            for j in range(i + 1, n):
                self.offset += Qs[i, j] / 4.0

    def build_knapsack_qubo(self, weights: np.ndarray, values: np.ndarray,
                            capacity: float, penalty: float = 5.0) -> None:
        n = self.n_spins
        if len(weights) != n or len(values) != n:
            raise ValueError("weights/values length must equal n_spins")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if any(w < 0 for w in weights):
            raise ValueError("weights must be non-negative")

        Q = np.zeros((n, n))
        for i in range(n):
            Q[i, i] = -values[i] + penalty * (weights[i] ** 2 - 2 * capacity * weights[i])
            for j in range(i + 1, n):
                Q[i, j] = penalty * weights[i] * weights[j]
                Q[j, i] = Q[i, j]
        self.build_from_qubo(Q)

    def build_random_ensemble(self, connectivity: float = 0.3,
                              j_mean: float = 0.0, j_std: float = 1.0,
                              h_mean: float = 0.0, h_std: float = 0.5) -> None:
        n = self.n_spins
        if not (0.0 <= connectivity <= 1.0):
            raise ValueError("connectivity must be in [0,1]")

        self.J = np.zeros((n, n))
        mask = self.rng.random((n, n)) < connectivity

        triu_mask = np.triu(mask, k=1)
        self.J[triu_mask] = self.rng.normal(j_mean, j_std, size=np.count_nonzero(triu_mask))
        self.J = self.J + self.J.T
        self.h = self.rng.normal(h_mean, h_std, size=n)
        self.offset = 0.0

    def energy(self, spin_config: np.ndarray) -> float:


        raise NotImplementedError("Hole 3: 请补全伊辛哈密顿量能量计算公式")

    def exact_ground_state_brute_force(self) -> Tuple[np.ndarray, float]:
        n = self.n_spins
        if n > 20:
            raise ValueError("Brute force only feasible for n_spins <= 20")


        bits = np.zeros(n, dtype=int)
        spins = 2 * bits - 1
        e_min = self.energy(spins)
        s_opt = spins.copy()

        e_current = e_min

        for g in range(1, 2 ** n):

            gray_g = g ^ (g >> 1)
            gray_prev = (g - 1) ^ ((g - 1) >> 1)
            diff = gray_g ^ gray_prev
            k = int(np.log2(diff))


            delta_e = -2.0 * spins[k] * (np.dot(self.J[k, :], spins) + self.h[k])


            delta_e = -2.0 * spins[k] * (
                np.dot(self.J[k, :], spins) - self.J[k, k] * spins[k] + self.h[k]
            )
            e_current += delta_e
            spins[k] *= -1

            if e_current < e_min:
                e_min = e_current
                s_opt = spins.copy()

        return s_opt, e_min

    def qubo_energy(self, binary_config: np.ndarray) -> float:
        if self.qubo_matrix is None:
            raise RuntimeError("QUBO matrix not initialized")
        x = binary_config.astype(float)
        return float(x @ self.qubo_matrix @ x)
