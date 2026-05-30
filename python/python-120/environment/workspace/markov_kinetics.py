
import numpy as np
from typing import Tuple, List, Optional


class SurfaceReactionNetwork:

    def __init__(self, n_sites: int, max_occupancy: int = 1):
        if n_sites < 1:
            raise ValueError("n_sites >= 1")
        self.n_sites = n_sites
        self.max_occupancy = max_occupancy


        self.states = self._enumerate_representative_states()
        self.n_states = len(self.states)
        self.W = np.zeros((self.n_states, self.n_states))

    def _enumerate_representative_states(self) -> List[np.ndarray]:
        states = []

        states.append(np.zeros(self.n_sites, dtype=int))

        for i in range(min(self.n_sites, 4)):
            s = np.zeros(self.n_sites, dtype=int)
            s[i] = 1
            states.append(s.copy())
            s[i] = 2
            states.append(s.copy())

        for i in range(min(self.n_sites - 1, 3)):
            s = np.zeros(self.n_sites, dtype=int)
            s[i] = 1
            s[i + 1] = 2
            states.append(s.copy())
            s[i] = 2
            s[i + 1] = 1
            states.append(s.copy())

        unique = []
        seen = set()
        for s in states:
            key = tuple(s.tolist())
            if key not in seen:
                seen.add(key)
                unique.append(s)
        return unique

    def build_transition_matrix(self, rate_ads_co: float = 1.0,
                                rate_des_co: float = 0.1,
                                rate_ads_o: float = 0.5,
                                rate_des_o: float = 0.05,
                                rate_rxn: float = 0.2):
        self.W = np.zeros((self.n_states, self.n_states))

        for i in range(self.n_states):
            for j in range(self.n_states):
                if i == j:
                    continue
                s_i = self.states[i]
                s_j = self.states[j]
                diff = s_j - s_i
                n_diff = np.sum(diff != 0)

                if n_diff == 1:

                    idx = np.where(diff != 0)[0][0]
                    if diff[idx] == 1 and s_i[idx] == 0:

                        self.W[i, j] = rate_ads_co
                    elif diff[idx] == 2 and s_i[idx] == 0:

                        self.W[i, j] = rate_ads_o
                    elif diff[idx] == -1 and s_i[idx] == 1:

                        self.W[i, j] = rate_des_co
                    elif diff[idx] == -2 and s_i[idx] == 2:

                        self.W[i, j] = rate_des_o
                elif n_diff == 2:

                    idxs = np.where(diff != 0)[0]
                    if (s_i[idxs[0]] == 1 and s_i[idxs[1]] == 2 and
                        s_j[idxs[0]] == 0 and s_j[idxs[1]] == 0):
                        self.W[i, j] = rate_rxn
                    elif (s_i[idxs[0]] == 2 and s_i[idxs[1]] == 1 and
                          s_j[idxs[0]] == 0 and s_j[idxs[1]] == 0):
                        self.W[i, j] = rate_rxn


        for i in range(self.n_states):
            self.W[i, i] = -np.sum(self.W[i, :])

    def solve_master_equation_ode(self, p0: np.ndarray, t_end: float,
                                  n_steps: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        p = np.asarray(p0, dtype=float)
        if len(p) != self.n_states:
            raise ValueError("p0 长度必须等于状态数")
        if abs(np.sum(p) - 1.0) > 1e-6:
            p = p / np.sum(p)

        dt = t_end / n_steps
        trajectory = [p.copy()]
        times = [0.0]


        I = np.eye(self.n_states)
        M = I - dt * self.W.T
        for _ in range(n_steps):
            p = np.linalg.solve(M, p)
            p = np.maximum(p, 0.0)
            p = p / np.sum(p)
            trajectory.append(p.copy())
            times.append(times[-1] + dt)

        return np.array(trajectory), np.array(times)

    def steady_state_distribution(self) -> np.ndarray:
        A = self.W.T.copy()
        A[-1, :] = 1.0
        b = np.zeros(self.n_states)
        b[-1] = 1.0
        try:
            p_ss = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            p_ss = np.linalg.lstsq(A, b, rcond=None)[0]
        p_ss = np.maximum(p_ss, 0.0)
        p_ss = p_ss / np.sum(p_ss)
        return p_ss

    def compute_turnover_frequency(self, p_ss: np.ndarray) -> float:
        tof = 0.0
        for i in range(self.n_states):
            for j in range(self.n_states):
                if i != j:
                    tof += self.W[i, j] * p_ss[i]
        return tof

    def mean_first_passage_time(self, target_state: int,
                                start_state: int) -> float:
        n = self.n_states
        if target_state < 0 or target_state >= n:
            raise ValueError("target_state 超出范围")
        if start_state < 0 or start_state >= n:
            raise ValueError("start_state 超出范围")


        mask = np.ones(n, dtype=bool)
        mask[target_state] = False
        W_sub = self.W[np.ix_(mask, mask)]
        b = -np.ones(n - 1)

        try:
            tau = np.linalg.solve(W_sub, b)
        except np.linalg.LinAlgError:
            tau = np.linalg.lstsq(W_sub, b, rcond=None)[0]


        idx = int(np.sum(~mask[:start_state]))
        if start_state == target_state:
            return 0.0
        return float(tau[start_state - idx])

    def entropy_production_rate(self, p_ss: np.ndarray) -> float:
        sigma = 0.0
        for i in range(self.n_states):
            for j in range(i + 1, self.n_states):
                j_i = self.W[i, j] * p_ss[i]
                i_j = self.W[j, i] * p_ss[j]
                if j_i > 1e-300 and i_j > 1e-300:
                    sigma += (j_i - i_j) * np.log(j_i / i_j)
        return sigma

    def dump_transition_matrix(self):
        print("=" * 60)
        print("马尔可夫转移速率矩阵 W (s^{-1})")
        print("=" * 60)
        print(f"状态数: {self.n_states}")
        print(f"总跃迁速率: {np.sum(self.W[self.W > 0]):.4e}")
        for i in range(self.n_states):
            row_sum = np.sum(self.W[i, :]) - self.W[i, i]
            print(f"  状态 {i}: 离开速率 = {row_sum:.4e}")
        print("=" * 60)
