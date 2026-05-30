
import numpy as np
from typing import Tuple, Optional


class TopologicalMarkovChain:

    def __init__(self, temperature: float = 0.01,
                 disorder_strength: float = 0.0,
                 delta: float = 0.8,
                 t: float = 1.0):
        self.T = max(temperature, 1e-10)
        self.W = max(disorder_strength, 0.0)
        self.delta = delta
        self.t = t
        self.num_states = 3

    def _compute_transition_matrix(self, mu: float) -> np.ndarray:
        n = self.num_states
        P = np.zeros((n, n))


        ratio = mu / (2.0 * self.t)
        if abs(ratio) < 1.0:
            egap = abs(self.delta) * np.sqrt(max(1.0 - ratio ** 2, 0.0))
        else:
            egap = 0.0


        thermal_factor = np.exp(-egap / self.T)
        disorder_factor = min(self.W / (abs(self.delta) + 1e-15), 1.0)



        if abs(mu) < 2.0 * abs(self.t):
            p01 = 0.1 * thermal_factor + 0.05 * disorder_factor
        else:
            p01 = 0.01 * thermal_factor


        p10 = 0.1 * thermal_factor + 0.2 * disorder_factor


        p02 = 0.05 * thermal_factor
        p12 = 0.05 * thermal_factor
        p20 = 0.3
        p21 = 0.3


        P[0, 1] = min(p01, 0.5)
        P[0, 2] = min(p02, 0.3)
        P[0, 0] = 1.0 - P[0, 1] - P[0, 2]

        P[1, 0] = min(p10, 0.5)
        P[1, 2] = min(p12, 0.3)
        P[1, 1] = 1.0 - P[1, 0] - P[1, 2]

        P[2, 0] = p20
        P[2, 1] = p21
        P[2, 2] = 1.0 - P[2, 0] - P[2, 1]


        for i in range(n):
            row_sum = np.sum(P[i, :])
            if abs(row_sum) > 1e-15:
                P[i, :] /= row_sum
            else:
                P[i, i] = 1.0

        return P

    def steady_state_distribution(self, mu: float,
                                   max_iter: int = 1000,
                                   tol: float = 1e-12) -> np.ndarray:
        P = self._compute_transition_matrix(mu)
        pi = np.ones(self.num_states) / self.num_states

        for _ in range(max_iter):
            pi_new = pi @ P
            if np.linalg.norm(pi_new - pi, ord=1) < tol:
                return pi_new
            pi = pi_new

        return pi

    def phase_transition_probability(self, mu_path: np.ndarray,
                                      initial_phase: int = 0) -> np.ndarray:
        n_steps = len(mu_path)
        prob_topo = np.zeros(n_steps)


        dist = np.zeros(self.num_states)
        if 0 <= initial_phase < self.num_states:
            dist[initial_phase] = 1.0
        else:
            dist = np.ones(self.num_states) / self.num_states

        for i, mu in enumerate(mu_path):
            P = self._compute_transition_matrix(mu)
            dist = dist @ P
            prob_topo[i] = dist[1]

        return prob_topo

    def winding_number(self, mu: float) -> int:















        raise NotImplementedError("Hole 2: 请实现Z₂绕数拓扑不变量计算公式")


    def topological_phase_diagram_markov(self,
                                          mu_vals: np.ndarray,
                                          w_vals: np.ndarray) -> np.ndarray:
        n_mu = len(mu_vals)
        n_w = len(w_vals)
        topo_prob = np.zeros((n_mu, n_w))

        for j, w in enumerate(w_vals):
            self.W = w
            for i, mu in enumerate(mu_vals):
                pi = self.steady_state_distribution(mu)
                topo_prob[i, j] = pi[1]

        return topo_prob

    def compute_entanglement_entropy(self, mu: float,
                                      subsystem_size: int,
                                      n_sites: int = 100) -> float:
        nu = self.winding_number(mu)
        if nu == 0:

            s = 0.1 * np.exp(-subsystem_size / 10.0)
        else:

            s = 0.5 * np.log(2.0) + 0.05 * np.exp(-subsystem_size / 5.0)

        return float(s)

    def correlation_length_critical_exponent(self,
                                              mu_vals: np.ndarray) -> np.ndarray:
        xi = np.zeros_like(mu_vals)
        for i, mu in enumerate(mu_vals):
            dist = min(abs(mu - 2.0 * self.t), abs(mu + 2.0 * self.t))
            if dist < 1e-6:
                xi[i] = 1e6
            else:
                xi[i] = abs(self.t) / dist
        return xi


def demo():
    tmc = TopologicalMarkovChain(
        temperature=0.05, disorder_strength=0.2, delta=0.8, t=1.0
    )

    mu_path = np.linspace(-3.0, 3.0, 61)
    prob_topo = tmc.phase_transition_probability(mu_path, initial_phase=0)

    print("Topological phase probability along μ path:")
    for mu, p in zip(mu_path[::10], prob_topo[::10]):
        nu = tmc.winding_number(mu)
        print(f"  μ={mu:+.2f}, ν={nu:2d}, P_topo={p:.4f}")


    for mu in [-0.5, 0.5, 3.0]:
        s = tmc.compute_entanglement_entropy(mu, subsystem_size=10)
        print(f"Entanglement entropy at μ={mu}: S={s:.4f}")


if __name__ == "__main__":
    demo()
