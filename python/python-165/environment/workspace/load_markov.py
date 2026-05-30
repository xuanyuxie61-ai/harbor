
import numpy as np
from typing import Optional, Tuple


class LoadMarkovModel:

    def __init__(self, n_states: int):
        self.n_states = n_states
        self.P = np.eye(n_states, dtype=np.float64)
        self.steady_state: Optional[np.ndarray] = None

    def fit(self, load_series: np.ndarray) -> None:
        load_series = np.array(load_series, dtype=np.float64)
        if len(load_series) < 2:
            raise ValueError("load_series must have at least 2 points")


        sorted_load = np.sort(load_series)
        n = len(sorted_load)
        self.state_edges = np.zeros(self.n_states + 1)
        for k in range(self.n_states + 1):
            idx = int(np.clip(k * n / self.n_states, 0, n - 1))
            self.state_edges[k] = sorted_load[idx]
        self.state_edges[-1] = sorted_load[-1] + 1e-6


        states = np.digitize(load_series, self.state_edges) - 1
        states = np.clip(states, 0, self.n_states - 1)

        count = np.zeros((self.n_states, self.n_states), dtype=np.float64)
        for t in range(len(states) - 1):
            i, j = int(states[t]), int(states[t + 1])
            count[i, j] += 1.0


        count += 1e-3
        row_sums = count.sum(axis=1, keepdims=True)
        row_sums[row_sums < 1e-12] = 1.0
        self.P = count / row_sums


        self._compute_steady_state()

    def _compute_steady_state(self) -> None:
        w, v = np.linalg.eig(self.P.T)
        idx = np.argmin(np.abs(w - 1.0))
        pi = np.real(v[:, idx])
        pi = np.abs(pi)
        pi = pi / np.sum(pi)
        self.steady_state = pi

    def predict(self, current_state: int, n_steps: int) -> np.ndarray:
        if current_state < 0 or current_state >= self.n_states:
            raise ValueError("current_state out of range")
        p_n = np.eye(self.n_states, dtype=np.float64)
        P_power = self.P.copy()

        while n_steps > 0:
            if n_steps % 2 == 1:
                p_n = p_n @ P_power
            P_power = P_power @ P_power
            n_steps //= 2
        return p_n[current_state]

    def generate_trajectory(self, initial_state: int, n_steps: int,
                            rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng(seed=42)
        traj = np.zeros(n_steps, dtype=np.int32)
        traj[0] = initial_state
        for t in range(1, n_steps):
            traj[t] = rng.choice(self.n_states, p=self.P[traj[t - 1]])
        return traj

    def entropy_rate(self) -> float:
        if self.steady_state is None:
            self._compute_steady_state()
        H = 0.0
        for i in range(self.n_states):
            for j in range(self.n_states):
                p = self.P[i, j]
                if p > 1e-12:
                    H -= self.steady_state[i] * p * np.log2(p)
        return float(H)

    def n_step_correlation(self, n: int) -> float:
        Pn = np.linalg.matrix_power(self.P, n)

        diff = Pn - self.steady_state[np.newaxis, :]
        return float(np.linalg.norm(diff, 'fro'))


def load_forecast_example() -> dict:

    t = np.linspace(0, 24, 48)
    base = 100.0
    peak1 = 60.0 * np.exp(-0.5 * ((t - 12.0) / 3.0) ** 2)
    peak2 = 40.0 * np.exp(-0.5 * ((t - 19.0) / 2.0) ** 2)
    noise = np.random.default_rng(seed=7).normal(0, 5.0, len(t))
    load = base + peak1 + peak2 + noise
    load = np.maximum(load, 20.0)

    model = LoadMarkovModel(n_states=6)
    model.fit(load)
    return {
        "load_series": load,
        "model": model
    }
