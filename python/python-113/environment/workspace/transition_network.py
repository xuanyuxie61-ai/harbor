
import numpy as np


class TransitionNetwork:
    def __init__(self, n_states, state_names=None):
        self.n_states = n_states
        self.state_names = state_names or [f"S{i}" for i in range(n_states)]

        self.K = np.zeros((n_states, n_states))

        self.pi = np.ones(n_states) / n_states

    def add_transition(self, i, j, rate):
        if i < 0 or i >= self.n_states or j < 0 or j >= self.n_states:
            raise IndexError("状态索引越界")
        self.K[i, j] = rate

    def compute_degrees(self):
        indegree = np.sum(self.K, axis=0)
        outdegree = np.sum(self.K, axis=1)
        return indegree, outdegree

    def is_eulerian_path(self):
        indegree, outdegree = self.compute_degrees()
        diff = indegree - outdegree

        n_plus = np.sum(diff == 1)
        n_minus = np.sum(diff == -1)
        n_zero = np.sum(diff == 0)

        if n_plus == 0 and n_minus == 0:
            return 2
        elif n_plus == 1 and n_minus == 1:
            return 1
        else:
            return 0

    def steady_state_probability(self, max_iter=1000, tol=1e-12):

        P = np.zeros_like(self.K)
        for i in range(self.n_states):
            row_sum = np.sum(self.K[i, :])
            if row_sum > 0:
                P[i, :] = self.K[i, :] / row_sum
            else:
                P[i, i] = 1.0

        pi = np.ones(self.n_states) / self.n_states
        for _ in range(max_iter):
            pi_new = pi @ P
            pi_new = pi_new / np.sum(pi_new)
            if np.max(np.abs(pi_new - pi)) < tol:
                break
            pi = pi_new

        self.pi = pi
        return pi

    def mean_first_passage_time(self, target):
        n = self.n_states

        Q = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    Q[i, j] = self.K[j, i]
            Q[i, i] = -np.sum(self.K[:, i])


        idx = [i for i in range(n) if i != target]
        Q_reduced = Q[np.ix_(idx, idx)]
        b = -np.ones(n - 1)

        tau_reduced = np.linalg.solve(Q_reduced, b)
        tau = np.zeros(n)
        for k, i in enumerate(idx):
            tau[i] = tau_reduced[k]
        return tau

    def conductivity(self, entry_state, exit_state):
        pi = self.steady_state_probability()
        rate_out = np.sum(self.K[exit_state, :])

        flux = pi[entry_state] * np.sum(self.K[entry_state, :])
        e_charge = 1.602176634e-19

        V = 0.1
        I = e_charge * flux
        G = I / V
        return G * 1e12


def build_kcsa_k_channel_network(k_on=1e8, k_off=1e7, k_hop=5e7):
    net = TransitionNetwork(6, ["S0", "S1", "S2", "S3", "S4", "S5"])


    net.add_transition(0, 1, k_on)
    net.add_transition(1, 0, k_off)


    for i in range(1, 5):
        net.add_transition(i, i + 1, k_hop)
        net.add_transition(i + 1, i, k_hop * 0.5)


    net.add_transition(5, 4, k_hop * 0.3)

    return net


def build_na_leaky_network(k_on=1e8, k_off=1e8, k_hop=1e6):
    net = TransitionNetwork(6, ["S0", "S1", "S2", "S3", "S4", "S5"])
    net.add_transition(0, 1, k_on)
    net.add_transition(1, 0, k_off)
    for i in range(1, 5):
        net.add_transition(i, i + 1, k_hop)
        net.add_transition(i + 1, i, k_hop)
    net.add_transition(5, 4, k_hop)
    return net
