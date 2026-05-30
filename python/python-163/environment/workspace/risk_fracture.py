
import numpy as np


class FractureMarkovModel:

    def __init__(self, num_states=42):
        self.num_states = int(num_states)
        self.adjacency = self._build_adjacency_matrix()
        self.transition = self._build_transition_matrix()

    def _build_adjacency_matrix(self):
        n = self.num_states
        A = np.zeros((n, n))

        for i in range(n):
            if i > 0:
                A[i, i - 1] = 1.0
            if i < n - 1:
                A[i, i + 1] = 1.0
            A[i, i] = 1.0
        return A

    def _build_transition_matrix(self):
        A = self.adjacency
        P = np.zeros_like(A)
        for i in range(A.shape[0]):
            row_sum = np.sum(A[i, :])
            if row_sum == 0:
                P[i, :] = 1.0 / A.shape[1]
            else:
                P[i, :] = A[i, :] / row_sum
        return P

    def steady_state(self, max_iter=10000, tol=1.0e-12):
        n = self.num_states
        pi = np.ones(n) / n
        P = self.transition

        for _ in range(max_iter):
            pi_new = P.T @ pi
            if np.linalg.norm(pi_new - pi, ord=1) < tol:
                break
            pi = pi_new

        pi = pi_new / np.sum(pi_new)
        return pi

    def evolve(self, initial_dist, num_steps):
        initial_dist = np.asarray(initial_dist, dtype=np.float64)
        if initial_dist.size != self.num_states:
            raise ValueError("Initial distribution size must match num_states.")
        if abs(np.sum(initial_dist) - 1.0) > 1.0e-10:
            initial_dist = initial_dist / np.sum(initial_dist)

        P = self.transition
        dist = initial_dist.copy()
        history = [dist.copy()]

        for _ in range(num_steps):
            dist = P.T @ dist
            history.append(dist.copy())

        return dist, np.array(history)

    def mean_first_passage_time(self, target_state):
        n = self.num_states
        target = int(target_state)
        if target < 0 or target >= n:
            raise ValueError("Invalid target state.")


        transient = [i for i in range(n) if i != target]
        Q = self.transition[np.ix_(transient, transient)]
        I = np.eye(len(transient))
        N = np.linalg.solve(I - Q, I)
        ones = np.ones(len(transient))
        mfpt = N @ ones


        full_mfpt = np.zeros(n)
        for idx, state in enumerate(transient):
            full_mfpt[state] = mfpt[idx]
        full_mfpt[target] = 0.0
        return full_mfpt


def fracture_aperture_markov_evolution(a_initial, thermal_cycles, delta_a=1.0e-5,
                                       closure_prob=0.3, opening_prob=0.4):
    if delta_a <= 0:
        raise ValueError("delta_a must be positive.")
    if closure_prob < 0 or opening_prob < 0 or closure_prob + opening_prob > 1.0:
        raise ValueError("Probabilities must be non-negative and sum <= 1.")

    num_states = 20
    state = int(np.clip(a_initial / delta_a, 0, num_states - 1))


    P = np.zeros((num_states, num_states))
    for i in range(num_states):
        if i > 0:
            P[i, i - 1] = closure_prob
        if i < num_states - 1:
            P[i, i + 1] = opening_prob
        P[i, i] = 1.0 - closure_prob - opening_prob

        P[i, :] /= np.sum(P[i, :])

    a_history = np.zeros(thermal_cycles + 1)
    a_history[0] = state * delta_a

    for t in range(thermal_cycles):
        state = np.random.choice(num_states, p=P[state, :])
        a_history[t + 1] = state * delta_a

    return a_history


def effective_permeability_from_fracture_network(apertures, fracture_density,
                                                  matrix_perm, max_perm=1.0e-12):
    if fracture_density < 0 or matrix_perm < 0:
        raise ValueError("Densities and permeabilities must be non-negative.")

    a_mean = np.mean(apertures)

    k_frac = fracture_density * (a_mean ** 3) / 12.0
    k_eff = matrix_perm + k_frac
    k_eff = min(k_eff, max_perm)
    return k_eff
