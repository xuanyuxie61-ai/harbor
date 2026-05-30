
import numpy as np


class CheckpointMDP:

    STATES = ["Compute", "Checkpoint", "Verify", "Recover", "Done"]
    ACTIONS = ["Memory", "Local", "Remote"]

    def __init__(self, p_fault: float, p_fault_during_ckpt: float,
                 recover_probs: np.ndarray, step_costs: np.ndarray):
        self.n_states = 5
        self.n_actions = 3
        self.p_fault = max(0.0, min(1.0, p_fault))
        self.p_fault_during_ckpt = max(0.0, min(1.0, p_fault_during_ckpt))
        self.recover_probs = np.asarray(recover_probs, dtype=float)
        self.step_costs = np.asarray(step_costs, dtype=float)
        self._build_transition_matrices()

    def _build_transition_matrices(self):
        self.P = np.zeros((self.n_actions, self.n_states, self.n_states))
        for a in range(self.n_actions):
            P = self.P[a]

            P[0, 0] = 1.0 - self.p_fault
            P[0, 3] = self.p_fault

            P[1, 2] = 1.0 - self.p_fault_during_ckpt
            P[1, 3] = self.p_fault_during_ckpt

            P[2, 0] = 0.9
            P[2, 4] = 0.1

            succ = self.recover_probs[a]
            P[3, 0] = succ
            P[3, 3] = 1.0 - succ

            P[4, 4] = 1.0

    def value_iteration(self, gamma: float = 0.95, tol: float = 1.0e-8, max_iter: int = 10000):
        V = np.zeros(self.n_states)
        policy = np.zeros(self.n_states, dtype=int)
        for _ in range(max_iter):
            V_old = V.copy()
            for s in range(self.n_states):
                q_vals = np.zeros(self.n_actions)
                for a in range(self.n_actions):
                    q_vals[a] = self.step_costs[s, a] + gamma * np.dot(self.P[a, s, :], V_old)
                V[s] = np.min(q_vals)
                policy[s] = np.argmin(q_vals)
            if np.max(np.abs(V - V_old)) < tol:
                break
        return V, policy

    def stationary_distribution(self, action: int = 0) -> np.ndarray:
        P = self.P[action][:4, :4]

        row_sums = P.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        P = P / row_sums

        w, v = np.linalg.eig(P.T)
        idx = np.argmin(np.abs(w - 1.0))
        pi = np.real(v[:, idx])
        pi = np.abs(pi)
        pi = pi / np.sum(pi)
        return pi

    def expected_time_to_done(self, action: int = 0, max_steps: int = 10000) -> float:
        P = self.P[action]
        state = 0
        total_cost = 0.0
        for step in range(max_steps):
            if state == 4:
                break
            total_cost += self.step_costs[state, action]
            state = np.random.choice(self.n_states, p=P[state, :])
        return total_cost
